# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

from vllm import LLM, SamplingParams

from clarifai.models.model_serving.model_config import *  # noqa


class InferenceModel(TextToText):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    path = os.path.join(self.base_path, "weights")
    self.model = LLM(
        model=path,
        dtype="float16",
        gpu_memory_utilization=0.9,
        #quantization="awq"
    )


  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int, bool]] = {}) -> list:
    """ Custom prediction function for `text-to-text` (also called as `text generation`) model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[str, float, int, bool]]): your inference parameters

    Returns:
      list of TextOutput
    
    """

    sampling_params = SamplingParams(**inference_parameters)
    preds = self.model.generate(input_data, sampling_params)
    outputs = [TextOutput(each.outputs[0].text) for each in preds]

    return outputs


if __name__ == "__main__":
  # dummy test
  model = InferenceModel()
  output = model.predict(["Test"])
  print(output)
