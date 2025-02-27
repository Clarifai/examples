import base64
import itertools
import json
from typing import Iterator, List, Union
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format
from openai import OpenAI

SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  output_info = request.model.model_version.output_info
  output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)
  if "params" in output_info:
    inference_params = output_info["params"]
      
  return inference_params


def image_proto_to_chat(image: resources_pb2.Image)-> Union[str, None]:
  if image.base64:
    image = "data:image/jpeg;base64," + base64.b64encode(image.base64).decode("utf-8")
  elif image.url:
    image = image.url
  else:
    image = None
  
  return image


def proto_to_chat(inp: resources_pb2.Input) -> List:
  prompt = inp.data.text.raw
  # extract role and content in text if possible
  try:
    prompt = json.loads(prompt)
    role = prompt.get("role", USER).lower()
    prompt = prompt.get("content", "")
  except:
    role = USER
  image = image_proto_to_chat(inp.data.image)
  if image and role != SYSTEM:
    content = [
          {'type': 'text', 'text': prompt},
          {"type": "image_url", "image_url": {"url": image}},
      ]
    # each turn could have more than 1 image
    for each_part in inp.data.parts:
      sub_img = image_proto_to_chat(each_part.data.image)
      if sub_img:
        content.append({"type": "image_url", "image_url": {"url": sub_img}})
    msg = {
      'role': role,
      'content': content
    }
    logger.info(f"N images = {len(content) - 1}")
  elif prompt:
    msg = {"role": role, "content": prompt}
  else:
    msg = None
  
  return msg
  

def parse_request(request: service_pb2.PostModelOutputsRequest):
  inference_params = get_inference_params(request)
  logger.info(f"inference_params: {inference_params}")
  temperature = inference_params.pop("temperature", 0.7)
  max_tokens = inference_params.pop("max_tokens", 256)
  top_p = inference_params.pop("top_p", .95)
  _ = inference_params.pop("stream", None)
  chat_history = inference_params.pop("chat_history", False)
  
  batch_messages = []
  for input_proto in request.inputs:
    # Treat 'parts' as history [0:-1) chat + new chat [-1]
    # And discard everything in input_proto.data
    messages = []
    if chat_history:
      for each_part in input_proto.data.parts:
        extrmsg = proto_to_chat(each_part)
        if extrmsg:
          messages.append(extrmsg)
    # If not chat_history, input_proto.data as input
    # And parts as sub data e.g. image1, image2
    else:  
      new_message = proto_to_chat(input_proto)
      if new_message:
        messages.append(new_message)
    batch_messages.append(messages)

  gen_config = dict(
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    **inference_params)
  
  return batch_messages, gen_config


class OpenAIWrapper():
  
  def __init__(self, client: OpenAI, **kwargs):
    self.client = client
    models = self.client.models.list()
    self.model_id = models.data[0].id

  @staticmethod
  def make_api_url(host, port, ver="v1"):
    return f"http://{host}:{port}/{ver}"
  
  def predict(
    self, 
    request: service_pb2.PostModelOutputsRequest, 
    extra_body:dict = {}
  ) -> service_pb2.MultiOutputResponse:

    messages, inference_params = parse_request(request)
    list_kwargs = [
        dict(
          model=self.model_id,
          messages=msg,
          **inference_params,
          extra_body=extra_body,
          stream=True,
      ) for msg in messages
    ]
    
    streams = [
      self.client.chat.completions.create(**kwargs) for kwargs in list_kwargs]
    
    outputs = [resources_pb2.Output() for _ in request.inputs]
    for output in outputs:
      output.status.code = status_code_pb2.SUCCESS
      
    for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
      for idx, chunk in enumerate(chunk_batch):
        if chunk:
          outputs[idx].data.text.raw += chunk.choices[0].delta.content if (
              chunk and chunk.choices[0].delta.content) is not None else ''

    return service_pb2.MultiOutputResponse(outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))
  
  def generate(
    self, 
    request: service_pb2.PostModelOutputsRequest,
    extra_body:dict = {}
  ) -> Iterator[service_pb2.MultiOutputResponse]:
    
    messages, inference_params = parse_request(request)
    list_kwargs = [
        dict(
          model=self.model_id,
          messages=msg,
          **inference_params,
          extra_body=extra_body,
          stream=True,
      ) for msg in messages
    ]
    
    streams = [
      self.client.chat.completions.create(**kwargs) 
      for kwargs in list_kwargs
    ]
    
    for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
      resp = service_pb2.MultiOutputResponse(status=status_pb2.Status(code=status_code_pb2.SUCCESS))
      for chunk in chunk_batch:
        output = resp.outputs.add()
        if chunk:
          text = (chunk.choices[0].delta.content
                                  if (chunk and chunk.choices[0].delta.content) is not None else '')
          output.data.text.raw = text
          output.status.code = status_code_pb2.SUCCESS
          
      yield resp
    