import itertools
from typing import Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format
from openai import OpenAI

# Set your OpenAI API key here
API_KEY = 'YOUR_OPENAI_API_KEY'


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
  system_prompt = "You'r a helpful assistant"
  system_prompt = inference_params.get("system_prompt", system_prompt)

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


class MyModel(ModelClass):
  """A custom runner that wraps the Openai GPT-4 model and generates text using it.
  """

  def load_model(self):
    """Load the model here."""
    self.client = OpenAI(api_key=API_KEY)
    self.model = "gpt-4-1106-preview"

  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    inference_params = get_inference_params(request)
    streams = []
    for input in request.inputs:
      output = resources_pb2.Output()

      # it contains the input data for the model
      input_data = input.data
      stream = stream_completion(self.model, self.client, input_data, inference_params)
      streams.append(stream)

    outputs = [resources_pb2.Output() for _ in request.inputs]
    for output in outputs:
      output.status.code = status_code_pb2.SUCCESS
    try:
      for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
        for idx, chunk in enumerate(chunk_batch):
          outputs[idx].data.text.raw += chunk.choices[0].delta.content if (
              chunk and chunk.choices[0].delta.content) is not None else ''
      response = service_pb2.MultiOutputResponse(
          outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))
    except Exception as e:
      for output in outputs:
        output.status.code = status_code_pb2.MODEL_PREDICTION_FAILED
        output.status.description = str(e)
      response = service_pb2.MultiOutputResponse(
          outputs=outputs, status=status_pb2.Status(code=status_code_pb2.MODEL_PREDICTION_FAILED))

    return response

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method generates stream of outputs for the given inputs in the request."""

    inference_params = get_inference_params(request)
    streams = []
    for input in request.inputs:
      # it contains the input data for the model
      input_data = input.data
      stream = stream_completion(self.model, self.client, input_data, inference_params)
      streams.append(stream)
    try:
      for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
        resp = service_pb2.MultiOutputResponse()
        resp.status.code = status_code_pb2.SUCCESS
        for chunk in chunk_batch:
          output = resp.outputs.add()
          output.data.text.raw = (chunk.choices[0].delta.content if
                                  (chunk and chunk.choices[0].delta.content) is not None else '')
          output.status.code = status_code_pb2.SUCCESS
        yield resp
    except Exception as e:
      outputs = [resources_pb2.Output() for _ in request.inputs]
      for output in outputs:
        output.status.code = status_code_pb2.MODEL_PREDICTION_FAILED
        output.status.description = str(e)
      yield service_pb2.MultiOutputResponse(
          outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    NotImplementedError("Stream is not implemented for this model.")
