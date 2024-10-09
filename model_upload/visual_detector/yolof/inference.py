# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules

# Initialize the DetInferencer
register_all_modules()

from clarifai.models.model_serving.model_config import *  # noqa


class InferenceModel(VisualDetector):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    self.checkpoint = os.path.join(self.base_path,
                                   "config/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth")
    self.config_path = os.path.join(self.base_path, "config/yolof_r50_c5_8x8_1x_coco.py")
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model = init_detector(self.config_path, self.checkpoint, device=self.device)


  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int, bool]] = {}) -> list:
    """ Custom prediction function for `visual-detector` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[str, float, int, bool]]): your inference parameters

    Returns:
      list of VisualDetectorOutput
    
    """
    outputs = []

    predictions = inference_detector(self.model, input_data)
    for inp_data, preds in zip(input_data, predictions):

      labels = preds.pred_instances.labels.cpu().numpy()
      bboxes = preds.pred_instances.bboxes.cpu().numpy()
      scores = preds.pred_instances.scores.cpu().numpy()
      h, w, _ = inp_data.shape  # input image shape
      # convert model output to clarifai detection output format
      output = VisualDetector.postprocess(
        width=w, height=h, labels=labels, xyxy_boxes=bboxes, scores=scores, max_bbox_count=300)
      outputs.append(output)
    
    # return list of VisualDetectorOutput
    return outputs
