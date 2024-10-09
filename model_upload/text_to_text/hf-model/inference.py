# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    # where you save hf checkpoint in your working dir e.i. `your_model`
    model_path = os.path.join(self.base_path, "checkpoint")
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    nf4_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
      model_path,
      # uncomment to use 4bit
      #quantization_config =nf4_config,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
      device_map="auto"
    )
    model.eval()
    
    self.pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=self.tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `text-to-text` (also called as `text generation`) model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of TextOutput
    
    """
    top_k = int(inference_parameters.pop("top_k", 50))

    output_sequences = self.pipeline(
        input_data, 
        eos_token_id=self.tokenizer.eos_token_id, 
        top_k=top_k,
        **inference_parameters)
    
    # wrap outputs in Clarifai defined output
    return [TextOutput(each[0]) for each in output_sequences]
