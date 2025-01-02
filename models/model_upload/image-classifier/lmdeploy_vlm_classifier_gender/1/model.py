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
with open(os.path.join(ROOT, "../config.yaml"), "r") as f:
  CONCEPTS = [each["name"] for each in yaml.safe_load(f)["concepts"]]


COMMAND = """Identify the gender of the person in this photo. 
Select one option from {concepts}. Do not say anything else except for the option you choose."""

def get_command(concepts: list):
  return COMMAND.format(
  concepts=f'{{{", ".join(concepts)}}}'
)

def shuffle_list(classes: list) -> list:
  _clss = deepcopy(classes)
  random.shuffle(_clss)
  return _clss

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
  n_tries = int(inference_params.get("n_tries", DEFAULT_N_TRIES))
  max_tokens = inference_params.get("max_tokens", 128)
  top_p = inference_params.get("top_p", .9)
  
  messages = []
  for inp in request.inputs:
    image = inp.data.image.base64
    queries = []
    for i in range(n_tries):
      if i == 0:
        _concepts = CONCEPTS
      else:
        _concepts = shuffle_list(CONCEPTS)
        
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
              'text': get_command(_concepts)
            },
            image_data
        ]
      }]
      queries.append(prompt)
    
    messages.append(queries)

  gen_config = dict(temperature=temperature,
    max_new_tokens=max_tokens,
    top_p=top_p)

  return messages, gen_config

########## Post process

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def ensure_output_concepts(batch_outputs: List[List[str]]):
  res = []
  for batch in batch_outputs:
    answers = {k: v/len(batch) for k, v in Counter(batch).items()}
    answers = {k: v for k, v in answers.items() if k in CONCEPTS}
    for concept in CONCEPTS:
      if not concept in answers:
        answers.update({concept: 0.})

    res.append(answers)
  
  return res



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
  
  def predict_and_convert_to_concepts(self, messages, infer_kwargs, n_tries=1):
    list_generated_text = []
    gen_config = GenerationConfig(**infer_kwargs)
    if n_tries == 1:
      messages = [each[0] for each in messages]
      gen_text = self.pipe(messages, gen_config=gen_config)
      list_generated_text = [[each.text] for each in gen_text]
    else:
      for each in messages:
        list_generated_text.append([out.text for out in self.pipe(each, gen_config=gen_config)])
      
    print("N inputs", len(messages))
    print("N outs", len(list_generated_text))
    output_concepts = ensure_output_concepts(list_generated_text)
    print("Gen text", list_generated_text)
    print("Scores", output_concepts)
    
    return output_concepts
    
  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    concept_protos = request.model.model_version.output_info.data.concepts
    messages, gen_config = parse_image_request(request)
    n_tries = gen_config.pop("n_tries", DEFAULT_N_TRIES)

    output_concepts = self.predict_and_convert_to_concepts(messages, gen_config, n_tries)    
    outputs = []
    for each_out_con in output_concepts:
      output = resources_pb2.Output()
      final_output_protos = []
      for concept in concept_protos:
        concept_name = concept.name
        concept.value = each_out_con[concept_name]
        final_output_protos.append(concept)
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
