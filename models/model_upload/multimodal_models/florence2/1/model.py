# import local module
import os
import sys
sys.path.append(os.path.dirname(__file__))
import constant as const
from utils import *
from florence_wrapper import Florence2, region_to_prompt
#####
from typing import Iterator

import cv2
import torch
from PIL import Image

from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format

class MyRunner(ModelClass):

  def load_model(self):
    """Load the model here."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints")
    self.model = Florence2(checkpoint_path)
    logger.info("Done loading!")
  
  
  def preprocess_kwargs(self, request):
    inference_params = {}
    if request.model.model_version.id != "":
      output_info = request.model.model_version.output_info
      output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)
      if "params" in output_info:
        inference_params = output_info["params"]
    if not "threshold" in inference_params:
      inference_params["threshold"] = 0.7
    if not "normalized_bboxes" in inference_params:
      inference_params["normalized_bboxes"] = True
    if not "max_new_tokens" in inference_params:
      inference_params["max_new_tokens"] = 1024
    else:
      inference_params["max_new_tokens"] = int(inference_params["max_new_tokens"])
    print("inference_params: ", inference_params)
    return inference_params

  def preprocess_text_data(self, data: resources_pb2.Data, w:int, h:int, denormalized_bboxes = True):
    prompt = data.text.raw
    task = self.model.parse_task_name(prompt)
    if task in const.HAS_REGION_INPUT_TASK:
      region_prompt = region_to_prompt(data.regions, width=w, height=h, denoramlized_box=denormalized_bboxes)
      logger.debug(f"data region: {data.regions} to region_prompt: {region_prompt}")
      if region_prompt:
        prompt = f"{task}{region_prompt}"
        print("Region prompt", prompt)
    return prompt
  
  def predict(
    self, request: service_pb2.PostModelOutputsRequest
  ) -> service_pb2.MultiOutputResponse:
  
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    infer_kwargs = self.preprocess_kwargs(request)
    images = []
    prompts = []
    for _input in request.inputs:
      image = preprocess_image(image_bytes=_input.data.image.base64)
      images.append(image)
      w, h = image.size
      prompt = self.preprocess_text_data(_input.data, w, h, denormalized_bboxes=infer_kwargs.get("normalized_bboxes"))
      prompts.append(prompt)
    list_output_protos = self.model.predict(prompts, images, **infer_kwargs)
    
    return service_pb2.MultiOutputResponse(
                outputs=list_output_protos, status=status_pb2.Status(code=status_code_pb2.SUCCESS))

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    if len(request.inputs) != 1:
      raise ValueError("Only one input is allowed for image models for this method.")
    infer_kwargs = self.preprocess_kwargs(request)
    for input in request.inputs:
      input_data = input.data
      video_bytes = None
      if input_data.video.base64:
        video_bytes = input_data.video.base64
      if video_bytes:
        frame_generator = video_to_frames(video_bytes)
        for frame in frame_generator:
          image = preprocess_image(frame)
          w, h = image.size
          prompts = [self.preprocess_text_data(input_data, w=w, h=h)]
          images = [image]
          with torch.no_grad():
            list_output_protos = self.model.predict(prompts, images, **infer_kwargs)
            yield service_pb2.MultiOutputResponse(
                outputs=list_output_protos, status=status_pb2.Status(code=status_code_pb2.SUCCESS))
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