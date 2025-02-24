import base64
import random
from copy import deepcopy
import json
import os
import re
import string
from typing import Iterator, List
from collections import Counter
from itertools import zip_longest

from clarifai.runners.models.model_class import ModelClass
from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format

from PIL import Image
from io import BytesIO
import yaml

DEFAULT_N_TRIES = 1
CHAT_TEMPLATE = 'minicpmv-2d6'

# Must be name (not id) of concepts
ROOT = os.path.dirname(__file__)

def parse_text_request(
  request: service_pb2.PostModelOutputsRequest,
):
  
  messages = []
  for inp in request.inputs:
    text = inp.data.text.raw
    messages.append(text)

  return messages


class MyRunner(ModelClass):
  """A custom runner that loads the model and generates text using lmdeploy inference.
  """

  def load_model(self):
    """Load the model here"""
    os.path.join(os.path.dirname(__file__))
    
  
  def predict_and_convert_to_concepts(self, messages, infer_kwargs, n_tries=1):
    pass
    
  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    #concept_protos = request.model.model_version.output_info.data.concepts
    messages = parse_text_request(request)
    outputs = []
    for msg in messages:
      final_output_protos = []
      for text in msg.split(" "):
        concept_id = "id-" + text.lower()
        concept = resources_pb2.Concept(name=text, id=concept_id, value=random.uniform(0, 1.))
        final_output_protos.append(concept)
      output = resources_pb2.Output()
      output.data.concepts.extend(final_output_protos)
      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
      
    return service_pb2.MultiOutputResponse(outputs=outputs,)
  
  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    pass
        

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    pass
