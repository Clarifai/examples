# import local module
import io
import os
import sys
sys.path.append(os.path.dirname(__file__))

import re
import os
import tempfile
from io import BytesIO
from typing import Iterator, List

import cv2
from clarifai.utils.logging import logger
from PIL import Image, ImageDraw


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

def preprocess_image(image_bytes):
  """Fetch and preprocess image data from bytes"""
  return Image.open(BytesIO(image_bytes)).convert("RGB")

from clarifai_grpc.grpc.api import resources_pb2

def render_mask(image: Image.Image, mask_bytes: bytes):
  mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
  new_image = image.convert("RGBA")
  blue_overlay = Image.new("RGBA", image.size, (0, 0, 255, 255))  # Solid blue
  blue_result = Image.composite(blue_overlay, new_image, mask)
  
  return blue_result

import cv2
import numpy as np

def render_box(image: Image.Image, regions: resources_pb2.Region):
  img_draw = np.array(image).astype("uint8")
  width, height = image.size
  if width < 500:
    thickness = 1
    fontScale = 1
  elif 500 <= width <= 1500:
    thickness = 2
    fontScale = 2
  else:
    fontScale = 3
    thickness = 3
    
  for reg in regions:
    box = reg.region_info.bounding_box
    ymin = box.top_row
    xmin = box.left_col
    ymax = box.bottom_row
    xmax = box.right_col
    xmin, ymin, xmax, ymax = [int(each) for each in (xmin * width, ymin * height, xmax * width, ymax * height)]
    label = reg.data.concepts[0].name if len(reg.data.concepts) else reg.data.text.raw
    cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=thickness, lineType=1)
    cv2.putText(img_draw, label, (xmin, ymin), color=(0, 0, 255), thickness=thickness, fontScale=fontScale, fontFace=1)
  
  img_draw = Image.fromarray(img_draw)
  
  return img_draw