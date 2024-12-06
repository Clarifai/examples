import os
import tempfile
from io import BytesIO
from typing import Iterator

import cv2
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


def video_to_frames(video_bytes):
  """Convert video bytes to frames"""
  # Write video bytes to a temporary file
  with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
    temp_video_file.write(video_bytes)
    temp_video_path = temp_video_file.name

  video = cv2.VideoCapture(temp_video_path)
  print("video opened")
  logger.info(f"video opened: {video.isOpened()}")
  while video.isOpened():
    ret, frame = video.read()
    if not ret:
      break
    # Convert the frame to byte format
    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    yield frame_bytes


def classify_image(images, model, processor, device):
  """Classify an image using the model and processor."""
  inputs = processor(images=images, return_tensors="pt")
  inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
  logits = model(**inputs).logits
  return logits


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

    outputs = []
    images = []
    for input in request.inputs:
      input_data = input.data
      image = preprocess_image(image_bytes=input_data.image.base64)
      images.append(image)

    with torch.no_grad():
      logits = classify_image(images, self.model, self.processor, self.device)

      for logit in logits:
        output_concepts = []
        probs = torch.softmax(logit, dim=-1)
        sorted_indices = torch.argsort(probs, dim=-1, descending=True)
        for idx in sorted_indices:
          output_concepts.append(
              resources_pb2.Concept(
                  id=str(idx.item()),
                  name=str(idx.item()),
                  value=probs[idx].item(),
              ))

        output = resources_pb2.Output()
        output.data.concepts.extend(output_concepts)
        output.status.code = status_code_pb2.SUCCESS
        outputs.append(output)

    return service_pb2.MultiOutputResponse(outputs=outputs,)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:

    if len(request.inputs) != 1:
      raise ValueError("Only one input is allowed for image classification models.")
    for input in request.inputs:
      input_data = input.data

      if input_data.video.base64:
        video_bytes = input_data.video.base64
      elif input_data.image.base64:
        logger.info("Image input")
        logger.info(
            f"len(input_data.image.base64): {len(input_data.image.base64)} start of image: {input_data.image.base64[:10]}"
        )
        video_bytes = input_data.image.base64

      frame_generator = video_to_frames(video_bytes)
      for frame in frame_generator:
        image = preprocess_image(frame)
        images = [image]

        with torch.no_grad():
          logits = classify_image(images, self.model, self.processor, self.device)

          for logit in logits:
            output_concepts = []
            probs = torch.softmax(logit, dim=-1)
            sorted_indices = torch.argsort(probs, dim=-1, descending=True)
            for idx in sorted_indices:
              output_concepts.append(
                  resources_pb2.Concept(
                      id=str(idx.item()),
                      name=str(idx.item()),
                      value=probs[idx].item(),
                  ))

            output = resources_pb2.Output()
            output.data.image.base64 = frame
            output.data.concepts.extend(output_concepts)
            output.status.code = status_code_pb2.SUCCESS
            yield service_pb2.MultiOutputResponse(outputs=[
                output,
            ])

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    for request in request_iterator:
      for output in self.generate(request):
        yield output


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
