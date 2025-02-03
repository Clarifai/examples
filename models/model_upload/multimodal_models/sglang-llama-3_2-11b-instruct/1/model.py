import base64
import itertools
import sys
from typing import Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format
from openai import OpenAI
from sglang.utils import (execute_shell_command, terminate_process, wait_for_server)


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


# All CLI arguments for the sglang server are defined here.
# https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py#L210
class SGLangServerManager:

  def __init__(self,
               mem_fraction_static,
               tensor_parallel_size,
               dtype="auto",
               context_length=None,
               quantization=None,
               chat_template=None,
               port=8761,
               host="localhost"):
    self.mem_fraction_static = mem_fraction_static
    self.tensor_parallel_size = tensor_parallel_size
    self.context_length = context_length
    self.quantization = quantization
    self.dtype = dtype
    self.chat_template = chat_template
    self.host = host
    self.port = port

    self.command = None
    self.process = None

  def start_server(self, python_executable, checkpoints):
    """Start the sglang server."""
    self.command = f"{python_executable} -m sglang.launch_server --model-path {checkpoints}"
    if self.dtype:
      self.command += f" --dtype {self.dtype}"
    if self.quantization:
      self.command += f" --quantization {self.quantization}"
    if self.tensor_parallel_size:
      self.command += f" --tensor-parallel-size {self.tensor_parallel_size}"
    if self.mem_fraction_static:
      self.command += f" --mem-fraction-static {self.mem_fraction_static}"
    if self.context_length:
      self.command += f" --context-length {self.context_length}"
    if self.chat_template:
      self.command += f" --chat-template {self.chat_template}"
    if self.host:
      self.command += f" --host {self.host}"
    if self.port:
      self.command += f" --port {self.port}"

    try:
      self.process = execute_shell_command(self.command)
      wait_for_server(f'http://localhost:{self.port}')
    except Exception as e:
      if self.process:
        logger.error("Terminating the sglang server process.")
        terminate_process(self.process)
      raise RuntimeError("Failed to start sglang server: " + str(e))


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


class MyModel(ModelClass):
  """A custom model that loads the model and generates text using SGLang Inference.
  """

  def load_model(self):
    """Load the model here and start the openai sglang server."""

    # SGLang parameters
    self.mem_fraction_static = 0.9
    self.tensor_parallel_size = 1
    self.dtype = "half"
    self.context_length = 4096
    self.quantization = None
    self.chat_template = "llama_3_vision"
    self.host = "localhost"
    self.port = 8761

    self.server_manager = SGLangServerManager(self.mem_fraction_static, self.tensor_parallel_size,
                                              self.dtype, self.context_length, self.quantization,
                                              self.chat_template, self.port, self.host)

    # Initialize the OpenAI client
    openai_api_base = f"http://localhost:{self.port}/v1"
    openai_api_key = "Not Required"
    self.client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    python_executable = sys.executable

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    # checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")

    checkpoints = "unsloth/Llama-3.2-11B-Vision-Instruct"

    try:
      # Start the sglang server
      self.server_manager.start_server(python_executable, checkpoints)
    except Exception as e:
      logger.error(f"Error starting sglang server: {e}")
      raise Exception(f"Error starting sglang server: {e}")

    # Get the model ID from the OpenAI API
    models = self.client.models.list()
    self.model = models.data[0].id

  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the model is run. It takes in an input and
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

    return service_pb2.MultiOutputResponse(
        outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))

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
      resp = service_pb2.MultiOutputResponse(
          status=status_pb2.Status(code=status_code_pb2.SUCCESS))

      for chunk in chunk_batch:
        output = resp.outputs.add()
        output.data.text.raw = (chunk.choices[0].delta.content
                                if (chunk and chunk.choices[0].delta.content) is not None else '')
        output.status.code = status_code_pb2.SUCCESS
      yield resp

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    NotImplementedError("Stream method is not implemented for the models.")
