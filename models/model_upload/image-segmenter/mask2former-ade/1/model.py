import re
import os
import sys

sys.path.append(os.path.dirname(__file__))
from typing import Iterator, List, Tuple

from google.protobuf import json_format
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from clarifai_grpc.grpc.api import resources_pb2

import yaml
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation 

from utils import *

ROOT = os.path.dirname(__file__)



class MyRunner(ModelRunner):
  """A custom runner that adds "Hello World" to the end of the text and replaces the domain of the
  image URL as an example.
  """
  
  def _load_concepts(self, config_path, name, model_path):
    
    with open(config_path, "r") as f:
      data = yaml.safe_load(f)
    if not data.get("concepts"):
      data = create_concepts_in_yaml(config_path, name, model_path)
    
    # Map Clarifai concept name to id and reverse
    self.conceptid2name = {each["id"] : each["name"] for each in data.get("concepts", [])}
    self.conceptname2id = {each["name"] : each["id"] for each in data.get("concepts", [])}
        
  
  def load_model(self):
    """Load the model here."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints")
    self.device = 'cuda' #if torch.cuda.is_available() else 'cpu'
    #self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Running on device: {self.device}")
    self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint_path, trust_remote_code=True).to(self.device)
    self.processor = AutoImageProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    self.model.eval()
    # Load clarifai concept
    config_path = os.path.join(ROOT, "../config.yaml")
    self._load_concepts(config_path, "mask2former-ade", checkpoint_path)
        
    logger.info("Done loading!")
  
  
  def get_default_infer_kwargs(self, request):
    infer_kwargs = get_inference_params(request)
    
    return infer_kwargs


  def _model_predict(self, images: List[Image.Image]) -> dict:
    inputs = self.processor(images=images, return_tensors="pt").to(self.device)
    with torch.no_grad():
      outputs = self.model(**inputs)
    target_sizes = [image.size[::-1] for image in images]
    results = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    outputs = []
    for i, all_masks_tensor in enumerate(results):
      masks = {}
      h, w = target_sizes[i]
      
      for clss_id in all_masks_tensor.unique().tolist():
        label = self.model.config.id2label[clss_id]
        mask = torch.zeros_like(all_masks_tensor)
        mask[all_masks_tensor == clss_id] = 255
        mask = mask.cpu().numpy() if self.device == "cuda" else mask.numpy()
        mask = mask.astype("uint8")
        masks.update({
          label : mask
        })
      out_hdl = OutputDataHandlerV2()
      out_hdl.set_masks(w=w, h=h, dict_data=masks, concepts_name2id=self.conceptname2id)
      out_hdl.set_status(status_code_pb2.SUCCESS)
      outputs.append(out_hdl._proto)
    
    return outputs
  
  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    images = []
    infer_kwargs = self.get_default_infer_kwargs(request)
    for input in request.inputs:
      image = preprocess_image(image_bytes=input.data.image.base64)
      images.append(image)
    outputs = self._model_predict(images)
    
    return service_pb2.MultiOutputResponse(
        outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS)
        )

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    if len(request.inputs) != 1:
      raise ValueError("Only one input is allowed for image models for this method.")
    infer_kwargs = self.get_default_infer_kwargs(request)
    
    for input in request.inputs:
      input_data = input.data
      video_bytes = None
      if input_data.video.base64:
        video_bytes = input_data.video.base64
      if video_bytes:
        frame_generator = video_to_frames(video_bytes)
        for frame in frame_generator:
          images = [preprocess_image(frame)]
          outputs = self._model_predict(images)
    
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