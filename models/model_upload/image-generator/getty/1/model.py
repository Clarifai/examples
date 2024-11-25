import os
import sys
sys.path.append(os.path.dirname(__file__))
import itertools
from typing import Iterator

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

from clarifai.runners.models.model_runner import ModelRunner
from getty_gen_image import GettyImagesAPI

API_KEY = ''
ACCESS_TOKEN = ''

class MyRunner(ModelRunner):
  """A custom runner that wraps the Openai GPT-4 model and generates text using it.
  """

  def load_model(self):
    """Load the model here."""
    pass

  def predict(
    self, request: service_pb2.PostModelOutputsRequest
  ) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
        model.model_version.output_info, preserving_proto_field_name=True
      )

    outputs = []
    for inp in request.inputs:
      output = resources_pb2.Output()
      data = inp.data
      # Optional use of output_info
      inference_params = {}
      if "params" in output_info:
        inference_params = output_info["params"]

      prompt = data.text.raw
      getty_caller = GettyImagesAPI(api_key=API_KEY, access_token=ACCESS_TOKEN)
      base64_images = getty_caller.generate_images(prompt=prompt, **inference_params)
      # Return only one?
      output.data.image.base64 = base64_images[0]
      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
    
    return service_pb2.MultiOutputResponse(
      outputs=outputs,
    )

  def generate(
    self, request: service_pb2.PostModelOutputsRequest
  ) -> Iterator[service_pb2.MultiOutputResponse]:
    raise NotImplementedError("Not supported")

  def stream(
    self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
  ) -> Iterator[service_pb2.MultiOutputResponse]:
    raise NotImplementedError("Not supported")