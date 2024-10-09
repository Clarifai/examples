# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import torch
from scipy.special import softmax
from transformers import AutoImageProcessor, ViTForImageClassification

from clarifai.models.model_serving.model_config import *


class InferenceModel(VisualClassifier):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    model_path = os.path.join(self.base_path, "checkpoint")
    self.transforms = AutoImageProcessor.from_pretrained(model_path)
    self.model = ViTForImageClassification.from_pretrained(model_path)
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `visual-classifier` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of ClassifierOutput
    
    """

    # Transform image and pass it to the model
    inputs = self.transforms(input_data, return_tensors='pt')
    with torch.no_grad():
      preds = self.model(**inputs).logits
    outputs = []
    for pred in preds:
      pred_scores = softmax(
          pred.detach().numpy())  # alt: softmax(output.logits[0].detach().numpy())
      outputs.append(ClassifierOutput(predicted_scores=pred_scores))

    return outputs
