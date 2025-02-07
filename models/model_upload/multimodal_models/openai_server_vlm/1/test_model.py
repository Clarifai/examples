# Runner test your implementation locally
#
import os
import sys

sys.path.append(os.path.dirname(__file__))
from model import MyRunner
##

from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

if __name__ == "__main__":
  model = MyRunner()
  model.load_model()
  prompt = "Write 2000 word story?"
  prompt2 = "Explain why cat can't fly"
  image = b"00012555"
  cl_request = service_pb2.PostModelOutputsRequest(
        model=resources_pb2.Model(model_version=resources_pb2.ModelVersion(
        pretrained_model_config=resources_pb2.PretrainedModelConfig(),
    )),
        inputs=[
            resources_pb2.Input(data=resources_pb2.Data(
                text=resources_pb2.Text(raw=prompt),
                image=resources_pb2.Image(base64=image)
            )),
            resources_pb2.Input(data=resources_pb2.Data(
                text=resources_pb2.Text(raw=prompt2)
            ))
        ],
    )
  
  resp = model.generate(cl_request)
  for res in resp:
    if res.status.code == status_code_pb2.SUCCESS:
      text = res.outputs[0].data.text.raw
      print(text, end="", flush=False)