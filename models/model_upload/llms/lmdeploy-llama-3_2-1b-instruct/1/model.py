import os
from typing import Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from transformers import AutoTokenizer


class MyModel(ModelClass):
  """A custom runner that loads the model and generates text using lmdeploy inference.
  """

  def load_model(self):
    """Load the model here"""
    os.path.join(os.path.dirname(__file__))
    # Load checkpoints
    model_path = os.path.dirname(os.path.dirname(__file__))
    builder = ModelBuilder(model_path, download_validation_only=True)
    checkpoints = builder.download_checkpoints(stage="runtime")
    # Note that if the model is not supported by turbomind yet, lmdeploy will auto switch to pytorch engine
    backend_config = TurbomindEngineConfig(tp=1)
    self.pipe = pipeline(checkpoints, backend_config=backend_config)
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoints)

  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
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
      inference_params = {}
      if "params" in output_info:
        inference_params = output_info["params"]

      temperature = inference_params.get("temperature", 0.7)
      max_tokens = inference_params.get("max_tokens", 256)
      top_p = inference_params.get("top_p", .9)

      if data.text.raw != "":
        prompt = data.text.raw
        messages = [{"role": "user", "content": prompt}]
        gen_config = GenerationConfig(
            temperature=temperature, max_new_tokens=max_tokens, top_p=top_p)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        res = self.pipe(prompt, gen_config=gen_config)
        text = res.text.replace("<|start_header_id|>assistant<|end_header_id|>", "")
        text = text.replace("\n\n", "")
        output.data.text.raw = text

      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
    return service_pb2.MultiOutputResponse(outputs=outputs,)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
          model.model_version.output_info, preserving_proto_field_name=True)

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
          model.model_version.output_info, preserving_proto_field_name=True)

    # TODO: parallelize this over inputs in a single request.
    for inp in request.inputs:
      output = resources_pb2.Output()

      data = inp.data

      # Optional use of output_info
      inference_params = {}
      if "params" in output_info:
        inference_params = output_info["params"]

      messages = []
      temperature = inference_params.get("temperature", 0.7)
      max_tokens = inference_params.get("max_tokens", 256)
      top_p = inference_params.get("top_p", .9)

      if data.text.raw != "":
        prompt = data.text.raw
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
        )
        gen_config = GenerationConfig(**kwargs)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        for item in self.pipe.stream_infer(prompt, gen_config=gen_config):
          text = item.text.replace("<|start_header_id|>assistant<|end_header_id|>", "")
          text = text.replace("\n\n", "")
          output.data.text.raw = text
          output.status.code = status_code_pb2.SUCCESS
          yield service_pb2.MultiOutputResponse(outputs=[output],)

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    for ri, request in enumerate(request_iterator):
      output_info = None
      if ri == 0:  # only first request has model information.
        model = request.model
        if request.model.model_version.id != "":
          output_info = json_format.MessageToDict(
              model.model_version.output_info, preserving_proto_field_name=True)
          # Optional use of output_info
          inference_params = {}
          if "params" in output_info:
            inference_params = output_info["params"]
      # TODO: parallelize this over inputs in a single request.
      for inp in request.inputs:
        output = resources_pb2.Output()

        data = inp.data

        system_prompt = "You are a helpful assistant"

        messages = [{"role": "system", "content": system_prompt}]
        temperature = inference_params.get("temperature", 0.7)
        max_tokens = inference_params.get("max_tokens", 100)
        top_p = inference_params.get("top_p", 1.0)

        if data.text.raw != "":
          prompt = data.text.raw
          messages.append({"role": "user", "content": prompt})
          kwargs = dict(
              model=self.model,
              messages=messages,
              temperature=temperature,
              max_tokens=max_tokens,
              top_p=top_p,
              stream=True,
          )
          stream = self.client.chat.completions.create(**kwargs)
          for chunk in stream:
            if chunk.choices[0].delta.content is None:
              continue
            output.data.text.raw = chunk.choices[0].delta.content
            output.status.code = status_code_pb2.SUCCESS
            yield service_pb2.MultiOutputResponse(outputs=[output],)
