import os
from io import BytesIO
from typing import Iterator

import torch
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor


def preprocess_image(image_bytes):
  """Fetch and preprocess image data from bytes"""
  return Image.open(BytesIO(image_bytes)).convert("RGB")


class MyRunner(ModelRunner):
  """A custom runner that loads the model and classifies images using it.
  """

  def load_model(self):
    """Load the model here."""

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on device: {self.device}")

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints")

    self.model = AutoModelForImageClassification.from_pretrained(checkpoint_path,).to(self.device)
    self.processor = ViTImageProcessor.from_pretrained(checkpoint_path)
    logger.info("Done loading!")

  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    # Get the concept protos from the model.
    concept_protos = request.model.model_version.output_info.data.concepts

    outputs = []
    images = []
    for input in request.inputs:
      input_data = input.data
      image = preprocess_image(image_bytes=input_data.image.base64)
      images.append(image)

    with torch.no_grad():
      inputs = self.processor(images=images, return_tensors="pt")
      inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
      logits = self.model(**inputs).logits

      for logit in logits:
        output_concepts = []
        probs = torch.softmax(logit, dim=-1)
        sorted_indices = torch.argsort(probs, dim=-1, descending=True)
        for idx in sorted_indices:
          concept_protos[idx.item()].value = probs[idx.item()].item()
          output_concepts.append(concept_protos[idx.item()])

        output = resources_pb2.Output()
        output.data.concepts.extend(output_concepts)
        output.status.code = status_code_pb2.SUCCESS
        outputs.append(output)

    return service_pb2.MultiOutputResponse(outputs=outputs,)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    raise NotImplementedError("Stream method is not implemented for image classification models.")

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    ## raise NotImplementedError
    raise NotImplementedError("Stream method is not implemented for image classification models.")
