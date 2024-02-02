# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union
from clarifai.models.model_serving.model_config import *  # noqa
import torch
from transformers import CLIPModel, CLIPProcessor

class InferenceModel(MultiModalEmbedder):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    # local checkpoint for openai/clip-vit-base-patch32
    self.model = CLIPModel.from_pretrained(os.path.join(self.base_path, "checkpoint"))
    self.model.eval()
    self.processor = CLIPProcessor.from_pretrained(os.path.join(self.base_path, "checkpoint"))

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int, bool]] = {}) -> list:
    """ Custom prediction function for `multimodal-embedder` model.

    Args:
      input_data (List[_MultiModalInputTypeDict]): List of dict of key-value: `image`(np.ndarray) and `text` (str)
      inference_parameters (Dict[str, Union[str, float, int, bool]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    
    """

    outputs = []
    for inp in input_data:
      image, text = inp.get("image", None), inp.get("text", None)
      with torch.no_grad():
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        if text is not None:
          inputs = self.processor(text=text, return_tensors="pt", padding=True)
          embeddings = self.model.get_text_features(**inputs)
        else:
          inputs = self.processor(images=image, return_tensors="pt", padding=True)
          embeddings = self.model.get_image_features(**inputs)
      embeddings = embeddings.squeeze().cpu().numpy()
      outputs.append(EmbeddingOutput(embedding_vector=embeddings))

    return outputs


if __name__ == "__main__":
  
  # Dummy test
  model = InferenceModel()
  input = dict(text="Hi")
  
  output = model.predict([input])
  print(output[0])