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


def preprocess_image(image_url=None, image_bytes=None) -> Image.Image:
  """Preprocess an image from a URL or byte input."""
  try:
    if image_bytes:
      img = Image.open(BytesIO(image_bytes))
    elif image_url:
      img = Image.open(BytesIO(requests.get(image_url).content))
    else:
      raise ValueError("No valid image source provided.")
    return img
  except Exception as e:
    raise ValueError(f"Error processing image: {str(e)}")


def process_inputs(request, inference_params):
  """Process inputs into batches of images and prompts."""
  batch_images, batch_prompts = [], []
  for input in request.inputs:
    input_data = input.data
    image = preprocess_image(
        image_bytes=input_data.image.base64) if input_data.image.base64 else None
    prompt = input_data.text.raw if input_data.text.raw else "Describe this image."
    batch_images.append(image)
    batch_prompts.append(prompt)
  return batch_images, batch_prompts


def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)

    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params


def validate_inputs(batch_images, batch_prompts, num_inputs):
  if len(batch_images) == 0 and len(batch_prompts) == 0:
    raise ValueError(
        "No image or prompt provided. Moondream Model requires both image and prompt in all inputs to generate text."
    )
  if len(batch_images) != len(batch_prompts):
    raise ValueError("Mismatch between the number of images and prompts provided.")

  return True


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
    """Generate predictions for the given inputs."""
    inference_params = get_inference_params(request)
    try:
      batch_images, batch_prompts = process_inputs(request, inference_params)
      validate_inputs(batch_images, batch_prompts, len(request.inputs))

      kwargs = {
          "max_new_tokens": inference_params.get("max_length", 128),
          "temperature": inference_params.get("temperature", 0.8),
          "top_k": inference_params.get("top_k", 50),
          "top_p": inference_params.get("top_p", 0.95),
          "repetition_penalty": inference_params.get("repetition_penalty", 1.0),
          "num_beams": inference_params.get("num_beams", 1),
          "do_sample": inference_params.get("do_sample", True),
      }

      model_responses = self.model.batch_answer(batch_images, batch_prompts, self.tokenizer,
                                                **kwargs)

      outputs = []
      for response in model_responses:
        output = resources_pb2.Output()
        output.data.text.raw = response
        output.status.code = status_code_pb2.SUCCESS
        outputs.append(output)
      return service_pb2.MultiOutputResponse(outputs=outputs)
    except ValueError as e:
      output = resources_pb2.Output()
      output.status.code = status_code_pb2.MODEL_PREDICTION_FAILED
      output.status.description = str(e)
      return service_pb2.MultiOutputResponse(outputs=[output] * len(request.inputs))

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""
    NotImplementedError("Generate is not implemented for this model.")

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""
    NotImplementedError("Stream is not implemented for this model.")
