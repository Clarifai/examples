import base64
import itertools
import json
from typing import Iterator
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format
from openai import OpenAI

def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)

    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params

def parse_request(request: service_pb2.PostModelOutputsRequest):
  prompts = [inp.data.text.raw for inp in request.inputs]
  bytes_images  = [inp.data.image.base64 for inp in request.inputs]
  
  inference_params = get_inference_params(request)
  temperature = inference_params.pop("temperature", 0.7)
  max_tokens = inference_params.pop("max_tokens", 256)
  top_p = inference_params.pop("top_p", .95)
  system_prompt = inference_params.pop("system_prompt", "You are a helpful assistant.")
  _ = inference_params.pop("stream", None)
  
  messages = []
  for prompt, bytes_image in itertools.zip_longest(prompts, bytes_images):
    try:
      prompt = json.loads(prompt)
    except:
      if not bytes_image:
        logger.info("No image")
        logger.info(f"system: {system_prompt}")
        logger.info(f"user: {prompt}")
        prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
      else:
        image = "data:image/jpeg;base64," + base64.b64encode(bytes_image).decode("utf-8")
        logger.info("With image")
        logger.info(f"system: {system_prompt}")
        logger.info(f"user: {prompt}")
        prompt = [
          {"role": "system", "content": system_prompt},
          {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        }]
    finally:
      messages.append(prompt)

  gen_config = dict(
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    **inference_params
  )
  
  return messages, gen_config

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
    