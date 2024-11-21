import base64
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
    self.port = 8000
    openai_api_base = f"http://localhost:{self.port}/v1"
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
            '0.8',
            '--port',
            str(self.port),
            '--host',
            'localhost',
            '--max-model-len',
            '2048',
            "--trust-remote-code",
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
        logger.info(f"vLLM Server started at http://localhost:{self.port}")
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

      prompt = None
      image_url = None
      image_bytes = None
      if data.text.raw != "":
        prompt = data.text.raw
      if data.image.url != "":
        image_url = data.image.url
      elif data.image.base64 != b"":
        image_bytes = data.image.base64

      if prompt is None:
        prompt = "please describe the image."

      if not image_url and not image_bytes:
        output.status.code = status_code_pb2.MODEL_PREDICTION_FAILED
        output.status.description = "No image provided, Model requires an image input."
        outputs.append(output)
        continue

      if image_url:
        message = {
            "role":
                "user",
            "content": [{
                "type": "text",
                "text": prompt,
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                }
            }]
        }
      elif image_bytes:
        image = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
        message = {
            "role":
                "user",
            "content": [{
                "type": "text",
                "text": prompt,
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image,
                }
            }]
        }

      messages.append(message)

      chat_response = self.client.chat.completions.create(
          model=self.model,
          messages=messages,
          temperature=temperature,
          max_tokens=max_tokens,
          top_p=top_p,
          extra_body={"stop_token_ids": [151645, 151643]},
      )

      res = chat_response.choices[0].message.content

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

      prompt = None
      image_url = None
      image_bytes = None
      if data.text.raw != "":
        prompt = data.text.raw
      if data.image.url != "":
        image_url = data.image.url
      elif data.image.base64 != b"":
        image_bytes = data.image.base64

      if prompt is None:
        prompt = "please describe the image."

      if not image_url and not image_bytes:
        output.status.code = status_code_pb2.MODEL_PREDICTION_FAILED
        output.status.description = "No image provided, Model requires an image input."
        yield service_pb2.MultiOutputResponse(outputs=[output],)

      if image_url:
        message = {
            "role":
                "user",
            "content": [{
                "type": "text",
                "text": prompt,
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                }
            }]
        }
      elif image_bytes:
        image = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
        message = {
            "role":
                "user",
            "content": [{
                "type": "text",
                "text": prompt,
            }, {
                "type": "image_url",
                "image_url": {
                    "url": image,
                }
            }]
        }

      messages.append(message)

      stream = self.client.chat.completions.create(
          model=self.model,
          messages=messages,
          temperature=temperature,
          max_tokens=max_tokens,
          top_p=top_p,
          extra_body={"stop_token_ids": [151645, 151643]},
          stream=True,
      )

      for chunk in stream:
        if chunk.choices[0].delta.content is None:
          continue
        output.data.text.raw = chunk.choices[0].delta.content
        output.status.code = status_code_pb2.SUCCESS
        yield service_pb2.MultiOutputResponse(outputs=[output],)

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    NotImplementedError("Stream method is not implemented for the models.")


if __name__ == '__main__':
  # Make sure you set these env vars before running the example.
  # CLARIFAI_PAT
  # CLARIFAI_USER_ID
  # CLARIFAI_API_BASE
  # CLARIFAI_RUNNER_ID
  # CLARIFAI_NODEPOOL_ID
  # CLARIFAI_COMPUTE_CLUSTER_ID

  # You need to first create a runner in the Clarifai API and then use the ID here.
  MyRunner(
      runner_id=os.environ["CLARIFAI_RUNNER_ID"],
      nodepool_id=os.environ["CLARIFAI_NODEPOOL_ID"],
      compute_cluster_id=os.environ["CLARIFAI_COMPUTE_CLUSTER_ID"],
      base_url=os.environ["CLARIFAI_API_BASE"],
      num_parallel_polls=int(os.environ.get("CLARIFAI_NUM_THREADS", 1)),
  ).start()
