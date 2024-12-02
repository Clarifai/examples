import base64
import itertools
import os
import subprocess
import sys
import threading
from typing import Iterator

from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from openai import OpenAI


def construct_messages(input_data: resources_pb2.Data) -> list[dict]:
  """Constructs the prompt and messages based on input data."""
  DEFAULT_PROMPT = "please describe the image."
  prompts = []
  images = []

  if input_data.parts:
    prompts = [part.data.text.raw for part in input_data.parts
               if part.data.text.raw] or [DEFAULT_PROMPT]
    images = [part.data.image.base64 for part in input_data.parts if part.data.image.base64]

    if not prompts:
      prompts.append(DEFAULT_PROMPT)
  else:
    prompts.append(input_data.text.raw or DEFAULT_PROMPT)
    images.append(input_data.image.base64)

  content = []
  for prompt, image_bytes in itertools.zip_longest(prompts, images):
    if prompt:
      content.append({"type": "text", "text": prompt})
    if image_bytes:
      image = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
      content.append({"type": "image_url", "image_url": {"url": image}})

    messages = [{"role": "user", "content": content}]

  return messages


def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)

    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params


class VLLMServerManager:

  def __init__(self, port, gpu_memory_utilization, tensor_parallel_size, max_model_len, dtype):
    self.port = port
    self.gpu_memory_utilization = gpu_memory_utilization
    self.tensor_parallel_size = tensor_parallel_size
    self.max_model_len = max_model_len
    self.dtype = dtype
    self.server_started_event = threading.Event()
    self.process = None

  def start_server(self, python_executable, checkpoints):
    try:
      self.process = subprocess.Popen(
          [
              python_executable,
              '-m',
              'vllm.entrypoints.openai.api_server',
              '--model',
              checkpoints,
              '--dtype',
              str(self.dtype),
              '--tensor-parallel-size',
              str(self.tensor_parallel_size),
              '--quantization',
              'awq',
              '--gpu-memory-utilization',
              str(self.gpu_memory_utilization),
              '--port',
              str(self.port),
              '--host',
              'localhost',
              '--max-model-len',
              str(self.max_model_len),
              "--trust-remote-code",
          ],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )
      for line in self.process.stderr:
        logger.info(line.strip())
        if "Uvicorn running on http://localhost:" in line.strip():
          self.server_started_event.set()
          break
    except Exception as e:
      if self.process:
        self.process.terminate()
      raise RuntimeError(f"Failed to start vLLM server: {e}")

  def wait_for_startup(self):
    self.server_started_event.wait()


def stream_completion(model, client, input_data, inference_params):
  """Stream iteratively generates completions for the input data."""

  temperature = inference_params.get("temperature", 0.7)
  max_tokens = inference_params.get("max_tokens", 512)
  top_p = inference_params.get("top_p", 1.0)

  messages = construct_messages(input_data)
  kwargs = dict(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      extra_body={"stop_token_ids": [151645, 151643]},
      stream=True,
  )
  stream = client.chat.completions.create(**kwargs)

  return stream


class MyRunner(ModelRunner):
  """
  A custom runner that integrates with the Clarifai platform and uses vLLM inference
  to process inputs, including text and images.
  """

  def load_model(self):
    """Load the model here and start the vllm server."""

    # vLLM parameters
    self.gpu_memory_utilization = 0.9
    self.tensor_parallel_size = 1
    self.max_model_len = 2048
    self.dtype = "float16"
    self.port = 7000

    self.server_manager = VLLMServerManager(self.port, self.gpu_memory_utilization,
                                            self.tensor_parallel_size, self.max_model_len,
                                            self.dtype)

    # Initialize the OpenAI client
    openai_api_base = f"http://localhost:{self.port}/v1"
    openai_api_key = "Not Required"
    self.client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    python_executable = sys.executable

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")

    try:
      # Start the vllm server in a separate thread
      vllm_server_thread = threading.Thread(
          target=self.server_manager.start_server, args=(python_executable, checkpoints))
      vllm_server_thread.start()

      # Wait for the server to start
      self.server_manager.wait_for_startup()
    except Exception as e:
      logger.error(f"Error starting vLLM server: {e}")
      raise Exception(f"Error starting vLLM server: {e}")

    # Get the model ID from the OpenAI API
    models = self.client.models.list()
    self.model = models.data[0].id

  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    inference_params = get_inference_params(request)

    streams = []
    for input in request.inputs:

      # it contains the input data for the model
      input_data = input.data

      stream = stream_completion(self.model, self.client, input_data, inference_params)

      streams.append(stream)

    outputs = [resources_pb2.Output() for _ in request.inputs]
    for output in outputs:
      output.status.code = status_code_pb2.SUCCESS

    for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
      for idx, chunk in enumerate(chunk_batch):
        outputs[idx].data.text.raw += chunk.choices[0].delta.content if (
            chunk and chunk.choices[0].delta.content) is not None else ''

    return service_pb2.MultiOutputResponse(outputs=outputs,)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    # Get the inference params from the request
    inference_params = get_inference_params(request)

    streams = []
    for input in request.inputs:

      # it contains the input data for the model
      input_data = input.data

      stream = stream_completion(self.model, self.client, input_data, inference_params)

      streams.append(stream)
    for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
      resp = service_pb2.MultiOutputResponse()

      for chunk in chunk_batch:
        output = resp.outputs.add()
        output.data.text.raw = (chunk.choices[0].delta.content
                                if (chunk and chunk.choices[0].delta.content) is not None else '')
        output.status.code = status_code_pb2.SUCCESS
      yield resp

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    NotImplementedError("Stream method is not implemented for the models.")
