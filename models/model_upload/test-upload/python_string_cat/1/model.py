from typing import Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format


class PythonStringCat(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text and replaces the domain of the
  image URL as an example.
  """

  def __init__(self, prefix='Hello'):
    self.prefix = prefix

  def load_model(self):
    """Load the model here."""

  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
          model.model_version.output_info, preserving_proto_field_name=True)

    outputs = []
    # TODO: parallelize this over inputs in a single request.
    for inp in request.inputs:
      output = resources_pb2.Output()

      data = inp.data

      # Optional use of output_info
      params_dict = {}
      if "params" in output_info:
        params_dict = output_info["params"]

      output.data.text.raw = self.prefix + " " + data.text.raw + " " + params_dict.get(
          "testparam", "")

      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
    return service_pb2.MultiOutputResponse(outputs=outputs,)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    for i in range(10):  # Generate 10 outputs.
      outputs = []
      for inp in request.inputs:
        output = resources_pb2.Output()
        output.data.text.raw = self.prefix + " " + inp.data.text.raw + f" generate {i}"
        output.status.code = status_code_pb2.SUCCESS
        outputs.append(output)
      yield service_pb2.MultiOutputResponse(outputs=outputs,)

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    for i, request in enumerate(request_iterator):
      outputs = []
      for inp in request.inputs:
        output = resources_pb2.Output()
        output.data.text.raw = self.prefix + " " + inp.data.text.raw + f" stream {i}"
        output.status.code = status_code_pb2.SUCCESS
        outputs.append(output)
      yield service_pb2.MultiOutputResponse(outputs=outputs,)
