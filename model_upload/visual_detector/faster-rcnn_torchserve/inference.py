import os

ROOT = os.path.dirname(__file__)
os.environ['TORCH_HOME'] = os.path.join(ROOT, "model_store")

from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import models, transforms  # noqa: E402

from pathlib import Path
from typing import Dict, Union
from clarifai.models.model_serving.model_config import *  # noqa


class InferenceModel(VisualDetector):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)
    #self.checkpoint = os.path.join(ROOT, "model_store/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    self.transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    self.model = self.model.to(self.device)
    self.model.eval()

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int, bool]] = {}) -> list:
    """ Custom prediction function for `visual-detector` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[str, float, int, bool]]): your inference parameters

    Returns:
      list of VisualDetectorOutput
    
    """
    outputs = []

    input_tensor = [self.transform(Image.fromarray(each)) for each in input_data]
    input_tensor = torch.stack(input_tensor).to(self.device)

    with torch.no_grad():
      predictions = self.model(input_tensor)

    for inp_data, preds in zip(input_data, predictions):
      boxes = preds["boxes"].cpu().numpy()
      labels = preds["labels"].detach().cpu().numpy()
      scores = preds["scores"].detach().cpu().numpy()
      h, w = inp_data.shape[:2]
      # convert model output to clarifai detection output format
      output = VisualDetector.postprocess(width=w, height=h, labels=labels, scores=scores, xyxy_boxes=boxes)
      outputs.append(output)
    
    # return list of VisualDetectorOutput
    return outputs
  