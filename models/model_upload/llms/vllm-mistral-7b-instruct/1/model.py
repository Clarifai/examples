import os
import subprocess
import sys
import threading
from typing import Iterator

from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from openai import OpenAI


class MyRunner(ModelRunner):
  """A custom runner that loads the model and generates text using vLLM Inference.
  """

  def load_model(self):
    """Load the model here and start the vllm server."""
    openai_api_base = "http://localhost:9000/v1"
    openai_api_key = "Not Required"
    self.client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    self.server_started_event = threading.Event()

    self.python_executable = sys.executable

    # Start the VLLM server in a separate thread
    vllm_server_thread = threading.Thread(target=self.vllm_openai_server)
    vllm_server_thread.start()

    # Wait for the server to be ready
    self.server_started_event.wait()

    models = self.client.models.list()
    self.model = models.data[0].id

  def vllm_openai_server(self):
    os.path.join(os.path.dirname(__file__))

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")
    process = subprocess.Popen(
        [
            self.python_executable,
            '-m',
            'vllm.entrypoints.openai.api_server',
            '--model',
            checkpoints,
            '--dtype',
            'float16',
            '--tensor-parallel-size',
            '1',
            '--quantization',
            'awq',
            '--gpu-memory-utilization',
            '0.4',
            '--port',
            '9000',
            '--host',
            'localhost',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Check the server output to confirm it's running
    for line in process.stderr:
      line = line.strip()
      logger.info(line)
      if "Uvicorn running on http://localhost:" in line:
        self.server_started_event.set()
        logger.info("vLLM Server started at http://localhost:9000")
        break

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

      messages = []
      temperature = inference_params.get("temperature", 0.7)
      max_tokens = inference_params.get("max_tokens", 100)
      top_p = inference_params.get("top_p", 1.0)

      kwargs = dict(
          model=self.model,
          temperature=temperature,
          max_tokens=max_tokens,
          top_p=top_p,
      )

      if data.text.raw != "":
        prompt = data.text.raw
        messages.append({"role": "user", "content": prompt})
        kwargs["messages"] = messages
        res = self.client.chat.completions.create(**kwargs)
        res = res.choices[0].message.content

        output.data.text.raw = res

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
