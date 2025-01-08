import itertools
import sys
from typing import Iterator

from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from openai import OpenAI
from sglang.utils import (execute_shell_command, terminate_process, wait_for_server)


def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)

    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params


def stream_completion(model, client, input_data, inference_params):
  """Stream iteratively generates completions for the input data."""

  temperature = inference_params.get("temperature", 0.7)
  max_tokens = inference_params.get("max_tokens", 512)
  top_p = inference_params.get("top_p", 1.0)

  prompt = input_data.text.raw
  messages = [{"role": "user", "content": prompt}]
  kwargs = dict(
      model=model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      stream=True,
  )
  stream = client.chat.completions.create(**kwargs)

  return stream


class SGLangServerManager:

  def __init__(self,
               port,
               mem_fraction_static,
               tensor_parallel_size,
               dtype="half",
               context_length=4096,
               quantization="awq"):
    self.port = port
    self.mem_fraction_static = mem_fraction_static
    self.tensor_parallel_size = tensor_parallel_size
    self.context_length = context_length
    self.quantization = quantization

    self.dtype = dtype
    self.process = None

  def start_server(self, python_executable, checkpoints):
    try:
      self.process = execute_shell_command(
          f"{python_executable} -m sglang.launch_server --model-path {checkpoints} --dtype {self.dtype} --tensor-parallel-size {self.tensor_parallel_size} --quantization {self.quantization} --mem-fraction-static {self.mem_fraction_static} --context-length {self.context_length} --port {self.port} --host localhost"
      )
      wait_for_server(f'http://localhost:{self.port}')
    except Exception as e:
      if self.process:
        logger.error("Terminating the sglang server process.")
        terminate_process(self.process)
      raise RuntimeError("Failed to start sglang server: " + str(e))


class MyRunner(ModelRunner):
  """A custom runner that loads the model and generates text using SGLang Inference.
  """

  def load_model(self):
    """Load the model here and start the openai sglang server."""

    # SGLang parameters
    self.mem_fraction_static = 0.9
    self.tensor_parallel_size = 1
    self.dtype = "float16"
    self.port = 8761
    self.context_length = 4096
    self.quantization = "awq"

    self.server_manager = SGLangServerManager(self.port, self.mem_fraction_static,
                                              self.tensor_parallel_size, self.dtype,
                                              self.context_length, self.quantization)

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
    checkpoints = "casperhansen/llama-3.3-70b-instruct-awq"

    try:
      # Start the sglang server
      self.server_manager.start_server(python_executable, checkpoints)
    except Exception as e:
      logger.error(f"Error starting sglang server: {e}")
      raise Exception(f"Error starting sglang server: {e}")

    # Get the model ID from the OpenAI API
    models = self.client.models.list()
    self.model = models.data[0].id

  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method generates outputs text for the given inputs using SGLang Inference."""

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
    """This method generates stream of outputs for the given inputs using SGLang Inference."""

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
