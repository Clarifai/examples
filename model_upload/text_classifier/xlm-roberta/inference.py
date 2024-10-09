# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union
from clarifai.models.model_serving.model_config import *  # noqa

import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class InferenceModel(TextClassifier):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    self.checkpoint_path: Path = os.path.join(self.base_path, "checkpoint")
    self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_path)
    self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `text-classifier` model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of ClassifierOutput
    
    """

    outputs = []
    for inp in input_data:
      encoded_input = self.tokenizer(inp, return_tensors='pt')
      output = self.model(**encoded_input)
      scores = output[0][0].detach().numpy()
      scores = softmax(scores)
      outputs.append(ClassifierOutput(predicted_scores=scores))

    return outputs

if __name__ == "__main__":
  
  # Dummy test
  model = InferenceModel()
  input = "How are you today?"
  
  output = model.predict([input])
  print(output[0])