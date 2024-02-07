# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import torch
from transformers import AutoModel, ViTImageProcessor

from clarifai.models.model_serving.model_config import *  # noqa


class InferenceModel(VisualEmbedder):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    self.huggingface_model_path = os.path.join(self.base_path, "checkpoint")
    self.processor = ViTImageProcessor.from_pretrained(self.huggingface_model_path)
    self.model = AutoModel.from_pretrained(self.huggingface_model_path)

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `visual-embedder` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    
    """
    outputs = []
    inputs = self.processor(images=input_data, return_tensors="pt")
    with torch.no_grad():
      embedding_vectors = self.model(**inputs).last_hidden_state[:, 0].cpu().numpy()
    for embedding_vector in embedding_vectors:
      outputs.append(EmbeddingOutput(embedding_vector=embedding_vector))

    return outputs
