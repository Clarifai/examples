from copy import deepcopy
import json
import os
from typing import Iterator

from clarifai.runners.models.model_runner import ModelRunner
from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from transformers import AutoTokenizer

  
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
  inference_params = get_inference_params(request)
  temperature = inference_params.get("temperature", 0.7)
  max_tokens = inference_params.get("max_tokens", 256)
  top_p = inference_params.get("top_p", .9)
  
  messages = []
  for prompt in prompts:
    try:
      prompt = json.loads(prompt)
    except:
      prompt = [{"role": "user", "content": prompt}]
    finally:
      messages.append(prompt)
  
  gen_config = dict(temperature=temperature,
    max_new_tokens=max_tokens,
    top_p=top_p)

  return messages, gen_config

def set_output(texts: list):
  assert isinstance(texts, list)
  output_protos = []
  for text in texts:
    output_protos.append(
      resources_pb2.Output(
        data=resources_pb2.Data(text=resources_pb2.Text(raw=text)),
        status=status_pb2.Status(code=status_code_pb2.SUCCESS)
      )
    )
  return output_protos
  

class MyRunner(ModelRunner):
  """A custom runner that loads the model and generates text using lmdeploy inference.
  """

  def load_model(self):
    """Load the model here"""
    os.path.join(os.path.dirname(__file__))
    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")
    # Note that if the model is not supported by turbomind yet, lmdeploy will auto switch to pytorch engine
    backend_config = TurbomindEngineConfig(tp=1, quant_policy=0)
    self.pipe = pipeline(checkpoints,
                    backend_config=backend_config)
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoints)
    
  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    messages, gen_config = parse_request(request)
    messages = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gen_config = GenerationConfig(**gen_config)
    list_generated_text = self.pipe(messages, gen_config=gen_config)
    raw_texts = [each.text for each in list_generated_text]
    output_protos = set_output(raw_texts)

    return service_pb2.MultiOutputResponse(outputs=output_protos)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    messages, gen_config = parse_request(request)
    messages = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gen_config = GenerationConfig(**gen_config)
    
    batch_size = len(messages)
    outputs = [
        resources_pb2.Output(
          data=resources_pb2.Data(text=resources_pb2.Text(raw="")),
          status=status_pb2.Status(code=status_code_pb2.SUCCESS)
        ) for _ in range(batch_size)
      ]
    self.pipe.running_session_ids
    current_session_id = next(deepcopy(self.pipe._session_id))
    for item in self.pipe.stream_infer(messages, gen_config=gen_config):
      text = item.text
      running_idx = item.session_id
      idx = running_idx - current_session_id
      outputs[idx].data.text.raw = text
      yield service_pb2.MultiOutputResponse(outputs=outputs,)
        

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    pass