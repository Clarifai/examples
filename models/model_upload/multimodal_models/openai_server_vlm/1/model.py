import os
import sys

sys.path.append(os.path.dirname(__file__))
from openai_client_wrapper import OpenAIWrapper
from openai_server_starter import OpenAI_APIServer
##################

from typing import Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from openai import OpenAI

class MyRunner(ModelClass):
  """
  A custom runner that integrates with the Clarifai platform and uses Server inference
  to process inputs, including text and images.
  """

  def load_model(self):
    """Load the model here and start the  server."""
    os.path.join(os.path.dirname(__file__))
    # Use downloaded checkpoints.
    # Or if you intend to download checkpoint at runtime, set hf id instead. For example:
    # checkpoints = "Qwen/Qwen2-7B-Instruct"
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")
    
    # Start server
    self.port = 23333
    self.host = "localhost"
    
    self.server = OpenAI_APIServer.from_lmdeploy_backend(
      checkpoints=checkpoints,
      backend="turbomind",
      server_name=self.host,
      server_port=self.port,
      chat_template="qwen"
    )
    
    # Example of vllm
    #self.server = OpenAI_APIServer.from_vllm_backend(
    #  checkpoints=checkpoints,
    #  host=self.host,
    #  port=self.port,
    #)
    
    # Example of Sglang
    #self.server = OpenAI_APIServer.from_sglang_backend(
    #  checkpoints=checkpoints,
    #  chat_template="qwen2-vl",
    #  host=self.host,
    #  port=self.port,
    #  additional_list_args=["--disable-cuda-graph"],
    #)

    # Create client
    self.client = OpenAIWrapper(
      client=OpenAI(
        api_key="notset",
        base_url=OpenAIWrapper.make_api_url(self.host, self.port)
      )
    )
    
  def predict(
    self,
    request: service_pb2.PostModelOutputsRequest
  ) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    return self.client.predict(request)

  def generate(
    self, request: service_pb2.PostModelOutputsRequest
  ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""
    for each in self.client.generate(request):
      yield each
    
  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    NotImplementedError("Stream method is not implemented for the models.")
