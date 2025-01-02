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
  DEFAULT_PROMPT = None
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


class ServerManager:

  def __init__(
    self, 
    port=23333, 
    backend:str="turbomind",
    gpu_memory_utilization=0.5, 
    tensor_parallel_size = 1, 
    max_model_len=4096, 
    dtype='float16',
    quantization_format: str = None,
    quant_policy: int = 0,
    chat_template: str = None,
    max_batch_size=16,
    host="localhost",
    device="cuda",
    ):
    self.host = host
    self.port = port
    self.backend = backend
    self.gpu_memory_utilization = gpu_memory_utilization
    self.tensor_parallel_size = tensor_parallel_size
    self.max_model_len = max_model_len
    self.dtype = dtype
    self.server_started_event = threading.Event()
    self.process = None
    
    self.quantization_format = quantization_format
    self.quant_policy = quant_policy
    self.chat_template = chat_template
    self.max_batch_size = max_batch_size
    self.device = device

  def start_server(self, python_executable, checkpoints):
    try:
      # lmdeploy serve api_server $MODEL_DIR --backend $LMDEPLOY_BE --server-port 23333
      cmds = [
        python_executable,
        '-m',
        'lmdeploy',
        'serve',
        'api_server',
        checkpoints,
        '--dtype',
        str(self.dtype),
        '--backend',
        str(self.backend),
        '--tp',
        str(self.tensor_parallel_size),
        '--server-port',
        str(self.port),
        '--server-name',
        str(self.host),
        '--cache-max-entry-count',
        str(self.gpu_memory_utilization),
        '--quant-policy',
        str(self.quant_policy),
        '--device',
        str(self.device),
      ]
      
      if self.quantization_format:
        cmds += ['--model-format', str(self.quantization_format)]
      
      if self.chat_template:
        cmds += [ '--chat-template', str(self.chat_template)]
      
      if self.max_batch_size:
        cmds += [ '--max-batch-size', str(self.max_batch_size)]
      
      if self.max_model_len:
        cmds += [ '--max-prefill-token-num', str(self.max_model_len)]
        
      
      self.process = subprocess.Popen(
          cmds,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )
      for line in self.process.stderr:
        logger.info(line.strip())
        if f"Uvicorn running on http://{self.host}:" in line.strip():
          self.server_started_event.set()
          break
    except Exception as e:
      if self.process:
        self.process.terminate()
      raise RuntimeError(f"Failed to start Server server: {e}")

  def wait_for_startup(self):
    self.server_started_event.wait()


def completion(
  model, client, input_data, inference_params, extra_body={}, stream=False):
  """Create completions for the input data."""

  messages = construct_messages(input_data)
  kwargs = dict(
      model=model,
      messages=messages,
      **inference_params,
      extra_body=extra_body,
      stream=stream,
  )
  stream = client.chat.completions.create(**kwargs)

  return stream


class MyRunner(ModelRunner):
  """
  A custom runner that integrates with the Clarifai platform and uses Server inference
  to process inputs, including text and images.
  """

  def load_model(self):
    """Load the model here and start the  server."""

    # Server parameters
    self.gpu_memory_utilization = 0.5
    self.tensor_parallel_size = 1
    self.max_model_len = 2048
    self.dtype = "float16"
    self.port = 23333
    self.host = "localhost"
    
    self.server_manager = ServerManager(
      port=self.port, 
      host=self.host,
      backend="turbomind",
      gpu_memory_utilization=self.gpu_memory_utilization,
      tensor_parallel_size=self.tensor_parallel_size, 
      max_model_len=self.max_model_len,
      dtype=self.dtype,
      chat_template="qwen",
      #quantization_format="awq"
    )

    # Initialize the OpenAI client
    openai_api_base = f"http://{self.host}:{self.port}/v1"
    openai_api_key = "Not Required"
    self.client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    python_executable = sys.executable

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")

    try:
      # Start the  server in a separate thread
      server_thread = threading.Thread(
          target=self.server_manager.start_server, args=(python_executable, checkpoints),
          daemon=True)
      server_thread.start()

      # Wait for the server to start
      self.server_manager.wait_for_startup()
    except Exception as e:
      logger.error(f"Error starting Server server: {e}")
      raise Exception(f"Error starting Server server: {e}")

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
      stream = completion(self.model, self.client, input_data, inference_params, stream=True)
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
    for each_input in request.inputs:

      # it contains the each_input data for the model
      input_data = each_input.data

      stream = completion(self.model, self.client, input_data, inference_params)

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
