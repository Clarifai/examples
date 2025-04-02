import os
import uuid
from typing import List

import torch
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Image, Region
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2
from transformers import AutoProcessor, OmDetTurboForObjectDetection


def detect_objects(images, classes, model, processor, device, threshold):
  # classes is a list of lists.
  tasks = ["Detect {}.".format(", ".join(cl)) for cl in classes]
  model_inputs = processor(images=images, text=classes, task=tasks, return_tensors="pt")
  model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
  model_output = model(**model_inputs)
  sizes = [image.size[::-1] for i, image in enumerate(images)]
  results = processor.post_process_grounded_object_detection(
      model_output,
      classes=classes,
      target_sizes=sizes,
      score_threshold=threshold,
      nms_threshold=threshold)
  return results


def process_process_regions(results, images, threshold):
  regions = []  # convert back into Region objects from the protos.
  for i, result in enumerate(results):
    image = images[i]
    width, height = image.size  # PIL image size is (width, height)
    output_regions = []
    for score, clas, box in zip(result["scores"], result["classes"], result["boxes"]):
      if score > threshold:
        output_region = resources_pb2.Region(
            id=uuid.uuid4().hex,
            region_info=resources_pb2.RegionInfo(
                bounding_box=resources_pb2.BoundingBox(
                    top_row=box[1],
                    left_col=box[0],
                    bottom_row=box[3],
                    right_col=box[2],
                ),))
        concept_proto = resources_pb2.Concept(id=clas, name=clas, value=score.item())
        output_region.data.concepts.append(concept_proto)
        # wrap the proto in the Region class.
        output_regions.append(Region(output_region))
    regions.append(output_regions)
  return regions


class MyModel(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text and replaces the domain of the
  image URL as an example.
  """

  def load_model(self):
    """Load the model here."""
    model_path = os.path.dirname(os.path.dirname(__file__))
    builder = ModelBuilder(model_path, download_validation_only=True)
    checkpoint_path = builder.download_checkpoints(stage="runtime")

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on device: {self.device}")

    self.model = OmDetTurboForObjectDetection.from_pretrained(checkpoint_path,).to(self.device)
    self.processor = AutoProcessor.from_pretrained(checkpoint_path,)
    self.model.eval()
    logger.info("Done loading!")

  @ModelClass.method
  def f(self, images: List[Image] = None, classes: List[str] = None,
        threshold: float = 0.5) -> List[List[Region]]:
    all_images = [image.to_pil() for image in images]
    all_classes = [classes]  # list of lists.
    with torch.no_grad():
      results = detect_objects(all_images, all_classes, self.model, self.processor, self.device,
                               threshold)
      regions = process_process_regions(results, all_images, threshold)
      logger.info(regions)
      return regions
