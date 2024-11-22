import os
from io import BytesIO
from typing import Iterator

import requests
import torch
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor


def open_images(image):
  if image.data.image.base64 != b"":
    img = Image.open(BytesIO(image_base64))
  elif data.image.url != "":
    img = Image.open(BytesIO(requests.get(image_url).content))
  return img


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
    #self.threadpool = ThreadPoolExecutor(8)
    self.bsize = 4  # TODO get this from config
    logger.info("Done loading!")

  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    # Get the concept protos from the model.
    # TODO: this should come from the model config or model directory --- NOT the request
    concept_protos = request.model.model_version.output_info.data.concepts

    images = list(map(open_images, request.inputs))
    # TODO put in threadpool.map, do we need to call wait() on futures or does this return imgs already?

    outputs = []
    batch = []
    for i, (inp, img) in enumerate(zip(request.inputs, images)):
      is_last = (i == len(images) - 1)

      batch.append(img)

      if len(batch) < bsize and not is_last:
        continue

      with torch.no_grad():
        inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
        model_output = self.model(**inputs)
        logits = model_output.logits

      # TODO: double-check dim=-1 means last dimension (not flattened)
      batch_probs = torch.softmax(logits, dim=-1)

      for probs in batch_probs:
        output = service_pb2.Output()

        sorted_labels = torch.argsort(probs, dim=-1, descending=True)

        for label in sorted_labels:
          c = output.data.concepts.add()
          c.CopyFrom(concept_protos[label])
          c.value = probs[label]

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
