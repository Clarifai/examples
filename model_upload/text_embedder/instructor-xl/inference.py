# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

# Set up env for huggingface
ROOT_PATH = os.path.join(os.path.dirname(__file__))
PIPELINE_PATH = os.path.join(ROOT_PATH, 'checkpoint')

os.environ['TORCH_HOME'] = PIPELINE_PATH
os.environ['TRANSFORMERS_CACHE'] = PIPELINE_PATH  # noqa
#os.environ["TRANSFORMERS_OFFLINE"] = "1"  # noqa

import torch  # noqa
from InstructorEmbedding import INSTRUCTOR  # noqa
from clarifai.models.model_serving.model_config import *  # noqa


class InferenceModel(TextEmbedder):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model = INSTRUCTOR('hkunlp/instructor-xl')

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `text-embedder` model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    
    """

    batch_preds = self.model.encode(input_data, device=self.device)

    return [EmbeddingOutput(each) for each in batch_preds]

if __name__ == "__main__":
  
  # Dummy test
  model = InferenceModel()
  input = "How are you today?"
  
  output = model.predict([input])
  print(output[0])
