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

from clarifai.runners.models.model_runner import ModelRunner
from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig
from transformers import AutoTokenizer

from PIL import Image
from io import BytesIO
import yaml

DEFAULT_N_TRIES = 1
CHAT_TEMPLATE = 'minicpmv-2d6'

# Must be name (not id) of concepts
ROOT = os.path.dirname(__file__)

DEFAULT_PROMPT = """Identify all objects you see in this image."""
DELIMITER = ","
STRUCTURE_PROMPT = f"Use character '{DELIMITER}' to separate your response."


def bytes_to_pillow(image_data: bytes) -> Image.Image:
    image_bytes = BytesIO(image_data)
    return Image.open(image_bytes)
  
def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)
    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params

def parse_image_request(
  request: service_pb2.PostModelOutputsRequest,
  image_data_type = "image_data"
):
  
  inference_params = get_inference_params(request)
  temperature = inference_params.get("temperature", 0.1)
  max_tokens = inference_params.get("max_tokens", 128)
  top_p = inference_params.get("top_p", .9)
  
  messages = []
  for inp in request.inputs:
    image = inp.data.image.base64
    query = inp.data.text.raw or DEFAULT_PROMPT
    query = f"{query} {STRUCTURE_PROMPT}"
    print(f"Query: {query}")
    queries = []
    if image_data_type == "image_data":
      image_data = {
            'type': 'image_data', 
            'image_data': {
              "data": bytes_to_pillow(image)
              }
          }
    elif image_data_type == "image_url":
      image_data = {
            'type': 'image_url', 
            'image_url': {
              "url": "data:image/jpeg;base64," + base64.b64encode(image).decode("utf-8")
              }
          }
    else:
      raise ValueError
      
    prompt = [{
      'role': 'user',
      'content': [
          {
            'type': 'text', 
            'text': query
          },
          image_data
      ]
    }]
    queries.append(prompt)
    
    messages.append(queries)

  gen_config = dict(
    temperature=temperature,
    max_new_tokens=max_tokens,
    top_p=top_p
  )

  return messages, gen_config

########## Post process

def concept_name_to_id(name:str):
  # Remove special characters (keeping only alphanumeric and spaces)
  name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
  # Replace spaces with "-"
  name = name.replace(' ', '-')
  
  return "id-" + name.lower()

class MyRunner(ModelRunner):
  """A custom runner that loads the model and generates text using lmdeploy inference.
  """

  def load_model(self):
    """Load the model here"""
    os.path.join(os.path.dirname(__file__))
    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")
    # Note that if the model is not supported by turbomind yet, lmdeploy will auto switch to pytorch engine
    backend_config = TurbomindEngineConfig(
      tp=1, quant_policy=0, 
      max_batch_size=16, dtype='float16', 
      cache_max_entry_count=0.5, max_prefill_token_num=4096,
      model_format="awq",
    )
    self.pipe = pipeline(
      checkpoints,
      backend_config=backend_config,
      chat_template_config=ChatTemplateConfig(model_name=CHAT_TEMPLATE),
      trust_remote_code=True
    )
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoints, trust_remote_code=True)
  
  def predict_and_convert_to_concepts(self, messages, infer_kwargs):
    outputs = []
    
    gen_config = GenerationConfig(**infer_kwargs)
    messages = [each[0] for each in messages]
    gen_text = self.pipe(messages, gen_config=gen_config)
    list_generated_text = [each.text for each in gen_text]
    print("Output: ", list_generated_text)
    for out in list_generated_text:
      final_output_protos = []
      for label in out.split(DELIMITER):
        label_name = label.strip()
        concept_id = concept_name_to_id(label_name)
        concept = resources_pb2.Concept(name=label_name, id=concept_id, value=1)
        final_output_protos.append(concept)
        
      output = resources_pb2.Output()
      output.data.concepts.extend(final_output_protos)
      output.data.text.raw = out
      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
    
    return outputs
    
  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    messages, gen_config = parse_image_request(request)

    output_concepts = self.predict_and_convert_to_concepts(messages, gen_config)    
      
    return service_pb2.MultiOutputResponse(outputs=output_concepts)
  
  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    pass
        

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    pass
