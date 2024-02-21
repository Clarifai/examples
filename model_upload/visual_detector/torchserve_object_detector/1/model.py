# Copyright 2023 Clarifai, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton inference server Python Backend Model."""

import os
import sys
import numpy as np
import json

try:
  import triton_python_backend_utils as pb_utils
except ModuleNotFoundError:
  pass


def parse_req_parameters(req_params: str):
  req_params = json.loads(req_params)
  for k, v in req_params.items():
    if isinstance(v, str):
      for t in (int, float):
        try:
          v = t(v)
        except ValueError:
          pass
    req_params.update({k: v})
  return req_params


class TritonPythonModel:
  """
  Triton Python BE Model.
  """

  def initialize(self, args):
    """
    Triton server init.
    """
    args["model_repository"] = args["model_repository"].replace("/1/model.py", "")
    sys.path.append(os.path.dirname(__file__))
    from inference import InferenceModel

    self.inference_model = InferenceModel()

  def execute(self, requests):
    """
    Serve model inference requests.
    """
    responses = []

    for request in requests:
      parameters = request.parameters()
      parameters = parse_req_parameters(parameters) if parameters else {}

      try:
        in_batch = pb_utils.get_input_tensor_by_name(request, 'image')
        in_batch = in_batch.as_numpy()
        outputs = self.inference_model.predict(in_batch, **parameters)
        responses.append(_visual_detector_reponse(outputs))
      except Exception as ex:
        responses.append(
            pb_utils.InferenceResponse(
              output_tensors=[],
              error=pb_utils.TritonError(
                  f"{ex.__class__.__name__}: {str(ex)}"
              )),
        )

    return responses

def _visual_detector_reponse(outputs):
  """
  Visual detector type output parser.
  """
  out_bboxes = []
  out_labels = []
  out_scores = []

  for pred in outputs:
    out_bboxes.append(pred.predicted_bboxes)
    out_labels.append(pred.predicted_labels)
    out_scores.append(pred.predicted_scores)

  if len(out_bboxes) < 1 or len(out_labels) < 1:
    out_tensor_bboxes = pb_utils.Tensor("predicted_bboxes", np.zeros((0, 4), dtype=np.float32))
    out_tensor_labels = pb_utils.Tensor("predicted_labels", np.zeros((0, 1), dtype=np.int32))
    out_tensor_scores = pb_utils.Tensor("predicted_scores", np.zeros((0, 1), dtype=np.float32))
  else:
    out_tensor_bboxes = pb_utils.Tensor("predicted_bboxes",
                                        np.asarray(out_bboxes, dtype=np.float32))
    out_tensor_labels = pb_utils.Tensor("predicted_labels",
                                        np.asarray(out_labels, dtype=np.int32))
    out_tensor_scores = pb_utils.Tensor("predicted_scores",
                                        np.asarray(out_scores, dtype=np.float32))

  inference_response = pb_utils.InferenceResponse(
      output_tensors=[out_tensor_bboxes, out_tensor_labels, out_tensor_scores])

  return inference_response

