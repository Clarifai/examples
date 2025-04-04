import base64
from io import BytesIO
import tempfile
from typing import Dict, List, Tuple

from google.protobuf import json_format
import cv2
from PIL import Image
import cv2
import numpy as np
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.utils.logging import logger
from clarifai.runners.utils.data_handler import InputDataHandler, OutputDataHandler
from clarifai_grpc.grpc.api import resources_pb2

def numpy_image_to_bytes(image_array: np.ndarray, format: str = 'PNG') -> str:
    if len(image_array.shape) == 2:  # Grayscale image
        mode = 'L'
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB image
        mode = 'RGB'
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA image
        mode = 'RGBA'
    else:
        raise ValueError("Unsupported array shape for image conversion.")

    image = Image.fromarray(image_array.astype('uint8'), mode)

    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    return buffer.getvalue()

def preprocess_image(image_bytes):
  """Fetch and preprocess image data from bytes"""
  return Image.open(BytesIO(image_bytes)).convert("RGB")


def video_to_frames(video_bytes):
  """Convert video bytes to frames"""
  # Write video bytes to a temporary file
  with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
    temp_video_file.write(video_bytes)
    temp_video_path = temp_video_file.name
    logger.info(f"temp_video_path: {temp_video_path}")

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
    video.release()

def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)
    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params

def create_polygon(mask: np.ndarray) -> List[List[Tuple[float, float]]]:
  """Create polygons from np mask

  Args:
      mask (np.ndarray): gray scale binary mask

  Returns:
      List[List[Tuple[float, float]]]: List of list polygon coordinates
  """
  polygons = []
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  for obj in contours:
    coords = []
    for point in obj:
      coords.append((int(point[0][0]), int(point[0][1])))
    polygons.append(coords)
  return polygons

def make_concept_name(concept):
  return concept if not concept.startswith("id-") else concept[3:]

class OutputDataHandlerV2(OutputDataHandler):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.set_status(status_code_pb2.SUCCESS)
  
  def set_polygons(self, w, h, dict_data: Dict[str, List[np.ndarray]], concepts_name2id: dict = {}, from_type="mask"):
    types = ["mask", "polygon"]
    assert from_type in types, ValueError(f"Only accept from_type in {types}. 'mask' is 2d np.array and 'polygon' is 1d array of [x,y,x1,y1..]")
    regions = []
    for concept, data in dict_data.items():
      concept_id = concepts_name2id.get(concept) or make_concept_name(concept)
      concept_proto = resources_pb2.Concept(
                    name=concept,
                    id=concept_id,
                    value=1.
                )
      if from_type == "mask":
        #mask_h, mask_w = each_data.shape[:2]
        #assert mask_w == w and mask_h == h, ValueError('Mask size must equal to input image size')
        polygons = create_polygon(data)
      elif from_type == "polygon":
        polygons = data
      for polygon in polygons:
        points = []
        for (x, y) in polygon:
          x = x / w
          y = y / h
          point = resources_pb2.Point(row=x, col=y)
          points.append(point)
        region = resources_pb2.Region(
            region_info=resources_pb2.RegionInfo(polygon=resources_pb2.Polygon(points=points),),
            data=resources_pb2.Data(concepts=[
                concept_proto
            ]))
        regions.append(region)
    
    self._proto.data.regions.extend(regions)

  def set_masks(self, w, h, dict_data: Dict[str, List[np.ndarray]], concepts_name2id: dict = {}):
   
    regions = []
    for concept, np_mask in dict_data.items():
      concept_id = concepts_name2id.get(concept) or make_concept_name(concept)
      concept_proto = resources_pb2.Concept(
                    name=concept,
                    id=concept_id,
                    value=1.
                )
      bytes_mask = numpy_image_to_bytes(np_mask, "JPEG")
      # with open(f".venv/tmp/{concept}.jpg", "wb") as f:
      #   f.write(bytes_mask)
      region = resources_pb2.Region(
          region_info = resources_pb2.RegionInfo(
            mask = resources_pb2.Mask(
              image = resources_pb2.Image(base64=bytes_mask)
            )
          ),
          data = resources_pb2.Data(concepts=[
              concept_proto
          ]))
      regions.append(region)
    
    self._proto.data.regions.extend(regions)




###############
import yaml
from transformers import AutoConfig

def create_concepts_in_yaml(path, model_short_name, repo_id=None):
  with open(path, "r") as f:
    data = yaml.safe_load(f)
  
  if not repo_id:
    repo_id = data["checkpoints"]["repo_id"]
  model_config = AutoConfig.from_pretrained(repo_id)
  concepts = []
  for _id, lb in model_config.id2label.items():
    concepts.append({
      "id": f"id-{model_short_name}-{_id}",
      "name": lb
    })

  data.update(dict(concepts=concepts))
  with open(path, "w") as f:
    yaml.dump(data, f, sort_keys=False)
  
  return data