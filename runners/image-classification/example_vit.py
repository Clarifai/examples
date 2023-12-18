from clarifai_grpc.grpc.api import resources_pb2
from clarifai.client.runner import Runner
from mmpretrain import ImageClassificationInferencer
import mmcv
import uuid


checkpoint = '/mmpretrain/checkpoints/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth'
inferencer = ImageClassificationInferencer(
   model='vit-base-p32_in21k-pre_3rdparty_in1k-384px',
   device='cpu',
   pretrained=checkpoint)

class VITRunner(Runner):
    """A custom runner that detects objects in an image and returns the bounding boxes using Yolov8"""
    def run_input(self, input: resources_pb2.Input,
                output_info: resources_pb2.OutputInfo) -> resources_pb2.Output:
        """This is the method that will be called when the runner is run. It takes in an input and
        returns an output.
        """

        output_proto = resources_pb2.Output()
        data = input.data

        if data.image.url == "":
            raise Exception("Need to include data.image.url in your inputs.")

        image_url = input.data.image.url
        array = mmcv.imread(image_url)

        results = inferencer(array)[0]

        data = resources_pb2.Data(
                                concepts=[
                                    resources_pb2.Concept(
                                        id=uuid.uuid4().hex,
                                        name=results["pred_class"],
                                        value=float(results["pred_score"]),
                                        app_id="runner-app",
                                    )
                                ])

        output_proto.data.CopyFrom(data)
        return output_proto

if __name__ == '__main__':
  # Make sure you set these env vars before running the example.
  # CLARIFAI_PAT
  # CLARIFAI_USER_ID

  # You need to first create a runner in the Clarifai API and then use the ID here.
  VITRunner(runner_id="vit-cls-runner").start()
