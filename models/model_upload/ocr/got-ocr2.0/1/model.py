import os
from io import BytesIO
from typing import Iterator

import torch
from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def preprocess_image(image_bytes=None):
  """convert image bytes to temp image file and return path"""
  image = Image.open(BytesIO(image_bytes))
  image_path = "/tmp/temp_image.jpg"
  image.save(image_path)
  return image_path


class MyModel(ModelClass):
  """A custom runner that loads the OCR model and runs it on the input image.
  """

  def load_model(self):
    """Load the model here."""

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on device: {self.device}")

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints")

    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    self.model = AutoModel.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        use_safetensors=True,
        device_map=self.device,
        low_cpu_mem_usage=True,
        pad_token_id=self.tokenizer.eos_token_id)
    logger.info("Done loading Model checkpoints!")

  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    ocr_type = "format"  # default ocr type
    outputs = []
    # TODO: parallelize this over inputs in a single request.
    for inp in request.inputs:
      output = resources_pb2.Output()

      data = inp.data

      if data.image.base64 != b"":
        img = preprocess_image(image_bytes=data.image.base64)
      elif data.image.url != "":
        img = data.image.url
      res = self.model.chat(self.tokenizer, img, ocr_type=ocr_type)

      output.data.text.raw = res

      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
    return service_pb2.MultiOutputResponse(outputs=outputs,)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    raise NotImplementedError("Stream method is not implemented for this models.")

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    ## raise NotImplementedError
    raise NotImplementedError("Stream method is not implemented for this models.")
