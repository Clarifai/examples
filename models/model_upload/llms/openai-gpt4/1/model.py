import itertools
from typing import Iterator

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from openai import OpenAI

from clarifai.runners.models.model_runner import ModelRunner

# model name
MODEL = "gpt-4-1106-preview"

API_KEY = 'OPENAI_API_KEY'


class MyRunner(ModelRunner):
  """A custom runner that wraps the Openai GPT-4 model and generates text using it.
  """

  def load_model(self):
    """Load the model here."""
    self.client = OpenAI(api_key=API_KEY)

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
    # TODO: parallelize this over inputs in a single request.
    for inp in request.inputs:
      output = resources_pb2.Output()

      data = inp.data

      # Optional use of output_info
      inference_params = {}
      if "params" in output_info:
        inference_params = output_info["params"]

      system_prompt = "You are a helpful assistant"

      messages = [{"role": "system", "content": system_prompt}]
      temperature = inference_params.get("temperature", 0.7)
      max_tokens = inference_params.get("max_tokens", 100)
      top_p = inference_params.get("top_p", 1.0)

      kwargs = dict(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p
      )

      if data.text.raw != "":
        prompt = data.text.raw
        messages.append({"role": "user", "content": prompt})

        res = self.client.chat.completions.create(**kwargs)
        res = res.choices[0].message.content

        output.data.text.raw = res

      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
    return service_pb2.MultiOutputResponse(
      outputs=outputs,
    )

  def generate(
    self, request: service_pb2.PostModelOutputsRequest
  ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
        model.model_version.output_info, preserving_proto_field_name=True
      )

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
        model.model_version.output_info, preserving_proto_field_name=True
      )
      # Optional use of output_info
      inference_params = {}
      if "params" in output_info:
        inference_params = output_info["params"]

    # TODO: parallelize this over inputs in a single request.
    streams = []
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
          model=MODEL,
          messages=messages,
          temperature=temperature,
          max_tokens=max_tokens,
          top_p=top_p,
          stream=True,
        )
        stream = self.client.chat.completions.create(**kwargs)

        streams.append(stream)
    for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
      resp = service_pb2.MultiOutputResponse()

      for chunk in chunk_batch:
        output = resp.outputs.add()
        output.data.text.raw = (
          chunk.choices[0].delta.content
          if (chunk and chunk.choices[0].delta.content) is not None
          else ''
        )
        output.status.code = status_code_pb2.SUCCESS
      yield resp

  def stream(
    self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
  ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    for ri, request in enumerate(request_iterator):
      output_info = None
      if ri == 0:  # only first request has model information.
        model = request.model
        if request.model.model_version.id != "":
          output_info = json_format.MessageToDict(
            model.model_version.output_info, preserving_proto_field_name=True
          )
          # Optional use of output_info
          inference_params = {}
          if "params" in output_info:
            inference_params = output_info["params"]

      streams = []
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
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
          )
          stream = self.client.chat.completions.create(**kwargs)

          streams.append(stream)
      for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
        resp = service_pb2.MultiOutputResponse()

        for chunk in chunk_batch:
          output = resp.outputs.add()
          output.data.text.raw = (
            chunk.choices[0].delta.content
            if (chunk and chunk.choices[0].delta.content) is not None
            else ''
          )
          output.status.code = status_code_pb2.SUCCESS
        yield resp
