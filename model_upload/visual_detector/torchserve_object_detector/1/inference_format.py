import io
import numpy as np
from PIL import Image
from typing import List, Dict

class TorchserveDataConverter:

  def to_torchserve_input_parameters(self, image_array: np.ndarray, parameters: dict) -> List[Dict]:
    img = Image.fromarray(image_array)
    buf = io.BytesIO()
    img.save(buf, format='tiff')
    assert not parameters
    return [{
        "name": "data",
        "contentType": "image/tiff",
        "value": buf.getvalue()
    }]

  def from_torchserve_output(self, pred: List) -> Dict:
    # parse the torchserve output to inference model format for a single image output
    # from: [x1, y1, x2, y2, conf, class_idx, label_str]
    # to: {'bbox': [x1, y1, x2, y2], 'score': conf, 'label': label
    return {'bbox': pred[:4], 'score': pred[4], 'label': pred[5]}
