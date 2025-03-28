import os
import sys

sys.path.append(os.path.dirname(__file__))
from threading import Thread
from typing import List

# In this example, we use the HuggingFace transformers library to build a text generation model.
import torch
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Stream
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer)

##################


class MyRunner(ModelClass):
  """
  A custom runner that integrates with the Clarifai platform and uses Server inference
  to process inputs, including text and images.
  """

  def load_model(self):
    """Load the model here and start the  server."""
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints")
    self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    #
    self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token
    self.model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=self.device,
    )
    self.streamer = TextIteratorStreamer(tokenizer=self.tokenizer,)
    # Use downloaded checkpoints.

  @ModelClass.method
  def predict(self,
              prompt: str,
              chat_history: List[dict] = None,
              max_tokens: int = 512,
              temperature: float = 0.7,
              top_p: float = 0.8) -> str:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    inputs = self.tokenizer(
        prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(self.device)

    generation_kwargs = {
        "input_ids": input_ids,
        "do_sample": True,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "eos_token_id": self.tokenizer.eos_token_id,
    }

    output = self.model.generate(**generation_kwargs)
    return self.tokenizer.decode(output[0], skip_special_tokens=True)

  @ModelClass.method
  def generate(self,
               prompt: str,
               chat_history: List[dict] = None,
               max_tokens: int = 512,
               temperature: float = 0.7,
               top_p: float = 0.8) -> Stream[str]:
    """Example yielding a whole batch of streamed stuff back."""

    inputs = self.tokenizer(
        prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to("cuda")

    generation_kwargs = {
        "input_ids": input_ids,
        "do_sample": True,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "eos_token_id": self.tokenizer.eos_token_id,
        "streamer": self.streamer
    }

    # Start generation in a separate thread
    thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
    thread.start()

    for output in self.streamer:
      yield output

    thread.join()

  # This method is needed to test the model with the test-locally CLI command.
  def test(self):
    """Test the model here."""
    try:
      print("Testing predict...")
      # Test predict
      print(self.predict(prompt="Hello, how are you?",))
    except Exception as e:
      print("Error in predict", e)

    try:
      print("Testing generate...")
      # Test generate
      for each in self.generate(prompt="Hello, how are you?",):
        print(each, end=" ")
    except Exception as e:
      print("Error in generate", e)
