import os
from io import BytesIO
from typing import Iterator

import requests
import torch
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def preprocess_image(image_url=None, image_base64=None):
  if image_base64:
    img = Image.open(BytesIO(image_base64))
  elif image_url:
    img = Image.open(BytesIO(requests.get(image_url).content))
  return img


class MyRunner(ModelRunner):
  """A custom runner that loads the Llama model and generates text using it.
  """

  def load_model(self):
    """Load the model here."""
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on device: {self.device}")

    revision = "2024-08-26"

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")

    self.model = AutoModelForCausalLM.from_pretrained(
        checkpoints,
        trust_remote_code=True,
        revision=revision,
    )

    self.tokenizer = AutoTokenizer.from_pretrained(checkpoints, revision=revision)

  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an outputs the response using llama model.
    """

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = {}
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
          model.model_version.output_info, preserving_proto_field_name=True)

    outputs = []
    # TODO: parallelize this over inputs in a single request.
    for inp in request.inputs:
      data = inp.data

      # Optional use of output_info
      inference_params = {}
      if "params" in output_info:
        inference_params = output_info["params"]

      temperature = inference_params.get("temperature", 0.7)
      max_tokens = inference_params.get("max_tokens", 100)
      max_tokens = int(max_tokens)

      top_k = inference_params.get("top_k", 40)
      top_k = int(top_k)
      top_p = inference_params.get("top_p", 1.0)

      kwargs = dict(temperature=temperature, top_p=top_p, max_new_tokens=max_tokens, top_k=top_k)

      prompt = ""
      image = None
      if data.image.base64 != b"":
        image = preprocess_image(image_base64=data.image.base64)
      elif data.image.url != "":
        image = preprocess_image(image_url=data.image.url)

      if image is not None:
        enc_image = self.model.encode_image(image)
      else:
        output = resources_pb2.Output()
        output.status.code = status_code_pb2.MODEL_PREDICTION_FAILED
        output.status.description = "No image provided, Moondream Model requires an image input."
        outputs.append(output)
        continue

      if data.text.raw != "":
        prompt = data.text.raw
      else:
        prompt = "Describe this image."

      model_response = self.model.answer_question(enc_image, prompt, self.tokenizer, **kwargs)

      output = resources_pb2.Output()
      output.data.text.raw = model_response

      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
    return service_pb2.MultiOutputResponse(outputs=outputs,)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""
    NotImplementedError("Generate is not implemented for this model.")

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""
    NotImplementedError("Stream is not implemented for this model.")
