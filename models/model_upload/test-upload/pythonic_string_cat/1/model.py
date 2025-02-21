import time
from typing import Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_handler import Output


class PythonicStringCat(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text and replaces the domain of the
  image URL as an example.
  """

  def load_model(self):
    """Load the model here."""

    print("Model loaded")

  def predict(
      self,
      text1: str,
      text2: str = "default text2",
  ):
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    text_result = text1 + text2 + " predict"

    return {"text_result": text_result, "text2": text2}

  def generate(
      self,
      text1: str,
  ) -> Iterator[Output]:
    """Example yielding a whole batch of streamed stuff back."""

    for i in range(10):  # Generate 10 outputs.
      text_result = text1 + f" stream {i}"
      # output = Output(text_result= text1 + f" stream {i}")
      time.sleep(1)
      yield text_result

  def stream(self, text1: Iterator[str]) -> Iterator[Output]:
    """Example yielding a whole batch of streamed stuff back."""
    NotImplementedError("Stream not implemented")
