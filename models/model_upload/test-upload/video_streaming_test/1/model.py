from typing import Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils import video_utils
from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2


class VideoStreamModel(ModelClass):
  """
  Example model that processes a video stream and returns the time and shape of each frame.
  """

  def load_model(self):
    pass

  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    raise NotImplementedError("This model does not support predict().")

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    '''
    Generate outputs for a video stream from a URL.
    '''
    assert len(request.inputs) == 1, "This model only supports one input."
    input = request.inputs[0].data
    video = input.video
    url = video.url
    assert url, "Video URL is required."

    for frame in video_utils.stream_frames_from_url(url, download_ok=True):
      yield self._predict_frame(frame)

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    '''
    Generate outputs for a video stream uploaded as bytes in a sequence of requests.
    '''

    def _bytes_iterator():
      for request in request_iterator:
        assert len(request.inputs) == 1, "This model only supports one input."
        input = request.inputs[0].data
        video_bytes = input.video.base64  # not actually base64, but the raw bytes
        assert video_bytes, "Video bytes are required."
        yield video_bytes

    for frame in video_utils.stream_frames_from_bytes(_bytes_iterator()):
      yield self._predict_frame(frame)

  def _predict_frame(self, frame):
    '''
    Predict the output for a single frame.
    '''
    # Get frame timestamps
    frame_pts = frame.pts  # Presentation timestamp (codec-specific)
    frame_time = frame.time  # Frame time in seconds (calculated using time_base)

    # Convert the frame to a NumPy array (RGB format)
    frame_array = frame.to_ndarray(format="rgb24")

    text = f"Frame PTS: {frame_pts}, Frame Time: {frame_time:.3f} s, Frame Shape: {frame_array.shape}"

    resp = service_pb2.MultiOutputResponse()
    output = resp.outputs.add()
    output.data.text.raw = text
    output.status.code = status_code_pb2.SUCCESS

    return resp
