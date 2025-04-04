# import local module
import os
import sys
sys.path.append(os.path.dirname(__file__))
import constant as const
############

import re
import io
from typing import List, Union

import cv2
import torch
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from google.protobuf import json_format
import numpy as np

def box_to_prompt(xmin, ymin, xmax, ymax) -> str:
  prompt = [f"<loc_{int(each)}>"  for each in [xmin, ymin, xmax, ymax]]
  prompt = "".join(prompt)
  return prompt


def region_to_prompt(
  regions: List[resources_pb2.Region], width: int = None, height: int = None, denoramlized_box: bool = True) -> str:
  prompt = ""
  if len(regions) > 0:
    first_region = regions[0]
    box = first_region.region_info.bounding_box
    ymin = box.top_row
    xmin = box.left_col
    ymax = box.bottom_row
    xmax = box.right_col
    if denoramlized_box:
      xmin, ymin, xmax, ymax = (xmin * width, ymin * height, xmax * width, ymax * height)
    prompt = box_to_prompt(xmin, ymin, xmax, ymax)
  return prompt


def concept_name_to_id(name:str):
  name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
  name = name.replace(' ', '-')
  return "id-" + name.lower()


def xyxy_to_region_proto(xmin, ymin, xmax, ymax) -> resources_pb2.Region:
  
  return resources_pb2.Region(
    region_info=resources_pb2.RegionInfo(
      bounding_box=resources_pb2.BoundingBox(
        top_row=ymin,
        left_col=xmin,
        bottom_row=ymax,
        right_col=xmax,
      )))


def result_to_bboxes_proto(
  result: dict,
  width: int = None,
  height: int = None,
  threshold:float=0.01,
  normalized_bboxes = True
) -> resources_pb2.Output:

  output_regions = []
  for score, label_name, box in zip(result["scores"], result["labels"], result["boxes"]):
    if score > threshold:
      xmin, ymin, xmax, ymax = box
      if normalized_bboxes:
        assert width and height, ValueError("require width and height for normalization")
        xmin = xmin/ width
        xmax = xmax/ width
        ymin = ymin/ height
        ymax = ymax/ height
      output_region = xyxy_to_region_proto(xmin, ymin, xmax, ymax)
      label_name = label_name or "placeholder_concept"
      concept_id = concept_name_to_id(label_name)
      concept = resources_pb2.Concept(name=label_name, id=concept_id, value=score)
      output_region.data.concepts.append(concept)
      output_regions.append(output_region)
  output = resources_pb2.Output()
  output.data.regions.extend(output_regions)

  return output


def result_to_text_proto(text:str) -> resources_pb2.Output:
  output = resources_pb2.Output()
  output.data.text.raw = text
  return output


def result_to_masks_proto(result: dict, width: int, height: int):
  mask_proto_list = []
  for polygons, label in zip(result['polygons'], result['labels']):
    mask = np.zeros((height, width), dtype=np.uint8)
    for _polygon in polygons:
      if len(_polygon) < 3:  
          logger.error('Invalid polygon:', _polygon)  
          continue  
      pts = np.array(_polygon, dtype=np.int32).reshape(-1, 2)
      cv2.fillPoly(mask, [pts], color=255)
    
    # Convert mask to binary image format
    pil_img = Image.fromarray(mask)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    label = label or "placeholder_concept"
    concept_id = concept_name_to_id(label)
    concept_proto = resources_pb2.Concept(
      name=label, 
      id=concept_id, 
      value=1 # always 1.
    )
    mask_proto = resources_pb2.Region(
      region_info=resources_pb2.RegionInfo(
        mask=resources_pb2.Mask(
          image=resources_pb2.Image(base64=img_byte_arr))),
      data=resources_pb2.Data(concepts=[concept_proto])
      )
    mask_proto_list.append(mask_proto)
  output = resources_pb2.Output()
  output.data.regions.extend(mask_proto_list)
  return output

def result_to_ocr_proto(result:dict, width: int = None, height: int = None, normalized_bboxes = True) -> resources_pb2.Output:
  output_regions = []
  for quad_box, text in zip(result["quad_boxes"], result["labels"]):
    if len(quad_box) != 8:
      logger.error(f"quad_box must have length of 8 but got {len(quad_box)}, text: {text}")
      continue
    x1, y1, x2, y2, x3, y3, x4, y4 = quad_box
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    if normalized_bboxes:
      assert width and height, ValueError("require width and height for normalization")
      xmin, ymin, xmax, ymax = xmin / width, ymin / height, xmax / width, ymax / height
    output_region = xyxy_to_region_proto(xmin, ymin, xmax, ymax)
    output_region.data.text.raw = text
    output_regions.append(output_region)
  output_proto = resources_pb2.Output()
  output_proto.data.regions.extend(output_regions)

  return output_proto

class Florence2():
  
  DEFAULT_TASK = const.OD
  
  def __init__(
    self,
    model_id
  ):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    self.model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=self.torch_dtype).to(self.device)
    self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    self.model.eval()
  
  @staticmethod
  def parse_task_name(x) -> str:
    pattern = r'^<.*?>'
    task = re.match(pattern, x)
    if task:
      task = task.group()
      assert task in const.TASKS, ValueError(f"Expected task is one of '{const.TASKS}', got {task} from prompt '{x}'")
    else:
      task = Florence2.DEFAULT_TASK
    
    return task
  
  @staticmethod
  def convert_output(
    task:str,
    raw_result:Union[dict, str], 
    image_width:int=None, 
    image_height:int=None,
    normalized_bboxes:bool=True,
  ) -> resources_pb2.Output:
    assert task in const.TASKS, ValueError(f"task must be one of {const.TASKS}, got '{task}'")
    
    result = raw_result[task]
    output = resources_pb2.Output()

    # If no result
    if not result:
      output.status.description = "No result"
    else:
      if task in const.CAPTION_TASKS:
        output = result_to_text_proto(result)
      elif task in const.DETECTION_TASKS:
        boxes = result.get("bboxes", [])
        classes = result.get("labels") or result.get("bboxes_labels")
        classes = classes or []
        # The model does not output scores, so set them as 1. as dummies
        scores = [1.]*len(boxes)
        output = result_to_bboxes_proto(
          dict(boxes=boxes, labels=classes, scores=scores), 
          width=image_width, height=image_height, 
          normalized_bboxes=normalized_bboxes
        )
      elif task in const.SEGMENTATION_TASKS:
        output = result_to_masks_proto(result, width=image_width, height=image_height, )
      elif task in const.REGION_OCR_TASKS:
        output = result_to_ocr_proto(result, width=image_width, height=image_height, normalized_bboxes=normalized_bboxes)
      else:
        logger.warning("No Task")
    output.status.code = status_code_pb2.SUCCESS
    
    return output
  
  def predict(
    self,
    prompts: List[str],
    images: List[Image.Image],
    max_new_tokens=1024,
    early_stopping=False,
    do_sample=False,
    num_beams=3,
    normalized_bboxes:bool=True,
    **kwargs
  ) -> List[resources_pb2.Output]:
    tasks = [self.parse_task_name(x) for x in prompts]
    inputs = self.processor(text=prompts, images=images, return_tensors="pt").to(self.device, self.torch_dtype)
    with torch.inference_mode():
      generated_ids = self.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        early_stopping=early_stopping,
        do_sample=do_sample,
        num_beams=num_beams,
      )
    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
    outputs = []
    for i, each_generated_text in enumerate(generated_text):
      image = images[i]
      task = tasks[i]
      image_width, image_height = image.size
      answer = self.processor.post_process_generation(
          each_generated_text, 
          task=task, 
          image_size=(image_width, image_height)
      )
      proto_answer = self.convert_output(
        task, answer, image_width=image_width, image_height=image_height, normalized_bboxes=normalized_bboxes)
      outputs.append(proto_answer)
    
    return outputs