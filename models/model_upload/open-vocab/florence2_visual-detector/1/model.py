import re
import os
import tempfile
from io import BytesIO
from typing import Iterator, List

import cv2
import torch
from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from google.protobuf import json_format


def concept_name_to_id(name:str):
  # Remove special characters (keeping only alphanumeric and spaces)
  name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
  # Replace spaces with "-"
  name = name.replace(' ', '-')
  
  return "id-" + name.lower()

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


def detect_objects(images, model, processor, device):
  model_inputs = processor(images=images, return_tensors="pt").to(device)
  model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
  model_output = model(**model_inputs)
  results = processor.post_process_object_detection(model_output)
  return results


def process_bounding_boxes(results, images, threshold:float=0.01):
  outputs = []
  for i, result in enumerate(results):
    image = images[i]
    width, height = image.size
    output_regions = []
    for score, label_name, box in zip(result["scores"], result["labels"], result["boxes"]):
      if score > threshold:
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = xmin / width, ymin / height, xmax / width, ymax / height
        output_region = resources_pb2.Region(region_info=resources_pb2.RegionInfo(
            bounding_box=resources_pb2.BoundingBox(
                top_row=ymin,
                left_col=xmin,
                bottom_row=ymax,
                right_col=xmax,
            ),))
        #concept_protos[label_name.item()].value = score.item()
        concept_id = concept_name_to_id(label_name)
        concept = resources_pb2.Concept(name=label_name, id=concept_id, value=score)
        output_region.data.concepts.append(concept)
        output_regions.append(output_region)
    output = resources_pb2.Output()
    output.data.regions.extend(output_regions)
    output.status.code = status_code_pb2.SUCCESS
    outputs.append(output)
  return outputs


def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)
    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params

class MyRunner(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text and replaces the domain of the
  image URL as an example.
  """

  def load_model(self):
    """Load the model here."""
    checkpoint_path = "microsoft/Florence-2-large"
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Running on device: {self.device}")

    self.model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, torch_dtype=self.torch_dtype).to(self.device)
    self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    self.model.eval()
    self.default_task = "<OD>"
    
    logger.info("Done loading!")
  
  def _model_predict(self, images: List[Image.Image], prompts: List[str], max_new_tokens=1024) -> dict:
    pattern = r'^<.*?>'
    tasks = []
    print(f"Promtps: {prompts}")
    for prompt in prompts:
      task = re.match(pattern, prompt)
      if task:
        tasks.append(task.group())
      else:
        msg = f"Not found task in prompt, task must be defined in '<>'. Got prompt: {prompt}"
        print(msg)
        tasks.append(self.default_task)
    
    inputs = self.processor(text=prompts, images=images, return_tensors="pt").to(self.device, self.torch_dtype)
    generated_ids = self.model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=max_new_tokens,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
    outputs = []
    for i, each_generated_text in enumerate(generated_text):
      image = images[i]
      task = tasks[i]
      parsed_answer = self.processor.post_process_generation(
          each_generated_text, 
          task=task, 
          image_size=(image.width, image.height)
      )
      if parsed_answer:
        results = parsed_answer[task]
        boxes = results.get("bboxes", [])
        classes = results.get("labels") or results.get("bboxes_labels")
        # The model does not output scores, so set them as 1. as dummies
        scores = [1.]*len(boxes)
        
        outputs.append(dict(boxes=boxes, labels=classes, scores=scores))
      else:
        outputs.append(dict(boxes=[], labels=[], scores=[]))
    return outputs
  
  def get_default_infer_kwargs(self, request):
    infer_kwargs = get_inference_params(request)
    if not "threshold" in infer_kwargs:
      infer_kwargs["threshold"] = 0.7
    #if not "task_prompt" in infer_kwargs:
    #  infer_kwargs["task_prompt"] = "<OD>"
    if not "max_new_tokens" in infer_kwargs:
      infer_kwargs["max_new_tokens"] = 1024
    else:
      infer_kwargs["max_new_tokens"] = int(infer_kwargs["max_new_tokens"])
      
    return infer_kwargs

  
  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    outputs = []
    images = []
    prompts = []

    infer_kwargs = self.get_default_infer_kwargs(request)
    threshold = infer_kwargs.get("threshold")
    max_new_tokens = infer_kwargs.get("max_new_tokens")
    
    for input in request.inputs:
      image = preprocess_image(image_bytes=input.data.image.base64)
      images.append(image)
      prompts.append(input.data.text.raw or self.default_task)
    
    with torch.no_grad():
      results = self._model_predict(images, prompts, max_new_tokens)
      outputs = process_bounding_boxes(results, images, threshold)
      return service_pb2.MultiOutputResponse(
          outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    if len(request.inputs) != 1:
      raise ValueError("Only one input is allowed for image models for this method.")
    infer_kwargs = self.get_default_infer_kwargs(request)
    threshold = infer_kwargs.get("threshold")
    max_new_tokens = infer_kwargs.get("max_new_tokens")
    
    for input in request.inputs:
      input_data = input.data
      video_bytes = None
      prompts = [input.data.text.raw] or [self.default_task]
      if input_data.video.base64:
        video_bytes = input_data.video.base64
      if video_bytes:
        frame_generator = video_to_frames(video_bytes)
        for frame in frame_generator:
          image = preprocess_image(frame)
          images = [image]
          with torch.no_grad():
            results = self._model_predict(images, prompts, max_new_tokens)
            outputs = process_bounding_boxes(results, images, threshold)
            yield service_pb2.MultiOutputResponse(
                outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))
      else:
        raise ValueError("Only video input is allowed for this method.")

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    for request in request_iterator:
      if request.inputs[0].data.video.base64:
        for output in self.generate(request):
          yield output
      elif request.inputs[0].data.image.base64:
        yield self.predict(request)