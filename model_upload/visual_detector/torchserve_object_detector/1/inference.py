# This file contains boilerplate code to allow users write their model
# inference code that will then interact with the Triton Inference Server
# Python backend to serve end user requests.
# The module name, module path, class name & get_predictions() method names MUST be maintained as is
# but other methods may be added within the class as deemed fit provided
# they are invoked within the main get_predictions() inference method
# if they play a role in any step of model inference
"""User model inference script."""

import os
import sys
import io
from PIL import Image
from collections import namedtuple

from pathlib import Path  # noqa: E402
import json
import numpy as np  # noqa: E402
import re  # noqa: E402
import torch  # noqa: E402
import zipfile
from ts.model_loader import ModelLoaderFactory
from ts.context import Context
from ts.service import Service

from inference_format import TorchserveDataConverter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL_MAR = os.path.join(ROOT_DIR, "model.mar")
# insert the path to the model code to the sys.path
sys.path.insert(0, MODEL_DIR)

class _MetricsCacheNoop:
    def __getattr__(self, attr):
        return lambda *args, **kwargs: None

VisualDetectorOutput = namedtuple('VisualDetectorOutput',
                                  ['predicted_bboxes', 'predicted_labels', 'predicted_scores'])


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """

    self.ts_converter = TorchserveDataConverter()

    manifest_file = os.path.join(MODEL_DIR, "MAR-INF", "MANIFEST.json")

    if not os.path.exists(manifest_file):
      assert os.path.exists(MODEL_MAR), f"model.mar archive or directory does not exist: {MODEL_MAR}"
      os.makedirs(MODEL_DIR, exist_ok=True)
      with zipfile.ZipFile(MODEL_MAR, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    with open(manifest_file) as f:
      manifest = json.load(f)

    assert manifest['runtime'] == 'python', 'Only python runtime is supported'

    model_loader = ModelLoaderFactory.get_model_loader()

    self.service = model_loader.load(
        model_name=manifest['model']['modelName'],
        model_dir=MODEL_DIR,
        handler=manifest['model']['handler'],
        gpu_id=0,
        metrics_cache=_MetricsCacheNoop(),
    )
    

  def predict(self, input_data: list, max_boxes=1000, **kwargs) -> list:
    """
    Main model inference method.

    Args:
    -----
      input_data: A list of input data item to predict on.
        Input data can be an image or text, etc depending on the model type.

      **kwargs: your inference parameters.

    Returns:
    --------
      List of one of the `clarifai.models.model_serving.models.output types` or `config.inference.return_type(your_output)`. Refer to the README/docs
    """
    outputs = []

    if isinstance(input_data, np.ndarray) and len(input_data.shape) == 4:
      input_data = list(input_data)

    predictions = self.ts_predict(input_data, kwargs)

    for inp_data, preds in zip(input_data, predictions):
      preds_dicts = [self.ts_converter.from_torchserve_output(pred) for pred in preds]
      bboxes = [result['bbox'] for result in preds_dicts]
      labels = [result['label'] for result in preds_dicts]
      scores = [result['score'] for result in preds_dicts]
      h, w, _ = inp_data.shape  # input image shape
      bboxes = [[x[1] / h, x[0] / w, x[3] / h, x[2] / w]
                for x in bboxes]  # normalize the bboxes to yxyx relative
      bboxes = np.asarray(bboxes).astype(np.float32)
      bboxes = np.clip(bboxes, 0, 1)
      scores = np.asarray(scores).astype(np.float32)
      if scores.ndim == 1:
        scores = scores[:, np.newaxis]
      labels = np.asarray(labels).astype(np.int32)
      if labels.ndim == 1:
        labels = labels[:, np.newaxis]
      if len(bboxes) != 0:
        bboxes = np.concatenate((bboxes, np.zeros((max_boxes - len(bboxes), 4))))
        scores = np.concatenate((scores, np.zeros((max_boxes - len(scores), 1))))
        labels = np.concatenate((labels, np.zeros(
            (max_boxes - len(labels), 1), dtype=np.int32)))
      else:
        bboxes = np.zeros((max_boxes, 4), dtype=np.float32)
        scores = np.zeros((max_boxes, 1), dtype=np.float32)
        labels = np.zeros((max_boxes, 1), dtype=np.int32)

      outputs.append(
          VisualDetectorOutput(
              predicted_bboxes=bboxes, predicted_labels=labels, predicted_scores=scores))

    return outputs

  def ts_predict(self, input_data, kwargs):
    batch = []
    for i, inp_data in enumerate(input_data):
      request_input = {
          "requestId": str(i).encode(),
          "parameters": self.ts_converter.to_torchserve_input_parameters(inp_data, kwargs),
      }
      batch.append(request_input)

    headers, input_batch, req_id_map = Service.retrieve_data_for_inference(batch)

    self.service.context.request_ids = req_id_map
    self.service.context.request_processor = headers

    # TODO  do we need to catch PredictException and change to our own inference exception?
    ret = self.service._entry_point(input_batch, self.service.context)

    return ret


