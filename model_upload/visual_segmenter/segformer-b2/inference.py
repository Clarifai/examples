# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import torch
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from clarifai.models.model_serving.model_config import *  # noqa


class InferenceModel(VisualSegmenter):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    self.huggingface_model_path = os.path.join(self.base_path, "checkpoint")
    self.processor = SegformerImageProcessor.from_pretrained(self.huggingface_model_path)
    self.model = AutoModelForSemanticSegmentation.from_pretrained(self.huggingface_model_path)

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `visual-segmenter` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of MasksOutput
    
    """
    outputs = []

    inputs = self.processor(images=input_data, return_tensors="pt")
    with torch.no_grad():
      output = self.model(**inputs)
    logits = output.logits.cpu()
    for logit in logits:
      mask = logit.argmax(dim=0).numpy()
      outputs.append(MasksOutput(predicted_mask=mask))

    return outputs
