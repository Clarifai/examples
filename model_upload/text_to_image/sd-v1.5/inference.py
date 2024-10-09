# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from clarifai.models.model_serving.model_config import *  # noqa


class InferenceModel(TextToImage):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    self.huggingface_model_path = os.path.join(self.base_path, "checkpoint")
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.pipeline = StableDiffusionPipeline.from_pretrained(
        self.huggingface_model_path, torch_dtype=torch.float16)
    self.pipeline = self.pipeline.to(self.device)

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `text-to-image` model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of ImageOutput
    
    """

    outputs = []
    num_inference_steps = int(inference_parameters.pop("num_inference_steps", 50))
    for inp in input_data:
      out_image = self.pipeline(
        inp, num_inference_steps=num_inference_steps, **inference_parameters).images[0]
      out_image = np.asarray(out_image)
      outputs.append(ImageOutput(image=out_image))

    return outputs

if __name__ == "__main__":
  
  # Dummy test
  from PIL import Image
  
  model = InferenceModel()
  input = "A cat"
  output = model.predict([input], inference_parameters=dict(num_inference_steps=30))
  
  Image.fromarray(output[0].image).save("tmp.jpg")