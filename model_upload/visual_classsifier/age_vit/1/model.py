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

try:
  import triton_python_backend_utils as pb_utils
except ModuleNotFoundError:
  pass
from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig
from clarifai.models.model_serving.model_config.inference_parameter import parse_req_parameters
from clarifai.models.model_serving.models.model_types import outputs_to_triton_response


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

    self.inference_obj = InferenceModel()

    # Read input_name from config file
    self.config_msg = ModelConfig()
    with open(os.path.join(args["model_repository"], "config.pbtxt"), "r") as f:
      cfg = f.read()
    text_format.Merge(cfg, self.config_msg)
    self.input_names = [inp.name for inp in self.config_msg.input]

  def execute(self, requests):
    """
    Serve model inference requests.
    """
    responses = []

    for request in requests:
      parameters = request.parameters()
      parameters = parse_req_parameters(parameters) if parameters else {}

      inputs_dict = {}
      for input_name in self.input_names:
        batch = pb_utils.get_input_tensor_by_name(request, input_name).as_numpy()
        inputs_dict[input_name] = batch

      if len(self.input_names) == 1:
        input_data = inputs_dict[self.input_names[0]]
        inference_response = self.inference_obj.get_predictions(input_data, **parameters)
      else:
        inference_response = self.inference_obj.get_predictions(**input_dict, **parameters)

      responses.append(outputs_to_triton_response(inference_response, 'visual-classifier'))

    return responses
