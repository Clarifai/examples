import time

import requests
import torch
from clarifai_grpc.grpc.api import resources_pb2
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from clarifai.client.runner import Runner

# This example requires to run the following before running this example:
# pip install transformers optimum auto-gptq

# https://huggingface.co/microsoft/phi-2
model_name_or_path = "microsoft/phi-2"

use_triton = False

torch.set_default_device("cuda")

class Phi2Runner(Runner):
  """A custom runner that runs the Phi2 LLM."""

  def __init__(self, *args, **kwargs):
    super(Phi2Runner, self).__init__(*args, **kwargs)
    self.logger.info("Starting to load the model...")
    st = time.time()
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", trust_remote_code=True)

    self.logger.info("Loading model complete in (%f seconds), ready to loop for requests." %
                     (time.time() - st))

  def run_input(self, input: resources_pb2.Input,
                output_info: resources_pb2.OutputInfo) -> resources_pb2.Output:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    output = resources_pb2.Output()
    data = input.data
    if data.text.raw != "":
      input_text = data.text.raw
    elif data.text.url != "":
      input_text = str(requests.get(data.text.url).text)
    else:
      raise Exception("Need to include data.text.raw or data.text.url in your inputs.")

    if "params" in output_info:
      params_dict = output_info["params"]
      self.logger.info("params_dict: %s", params_dict)

    # # Method 1
    inputs = self.tokenizer(input_text, return_tensors="pt", return_attention_mask=False)
    out = self.model.generate(**inputs, max_length=512)
    out_text = self.tokenizer.batch_decode(out)[0]
    output.data.text.raw = out_text

    return output


if __name__ == '__main__':
  # Make sure you set these env vars before running the example.
  # CLARIFAI_PAT
  # CLARIFAI_USER_ID

  # You need to first create a runner in the Clarifai API and then use the ID here.
  Phi2Runner(runner_id="sdk-phi2-runner").start()
