import os

from clarifai.datasets.upload.base import ClarifaiDataLoader
from clarifai.datasets.upload.features import VisualDetectionFeatures



class MultiFolderDetectionDataLoader(ClarifaiDataLoader):
  """Multifolder Images Detection Dataset."""

  def __init__(self, root_dir):
    """
    Args:
      root_dir: Directory containing multiple subdirectories, each with images and corresponding annotation files.
    """
    self.root_dir = root_dir
    self.data = []
    self.load_data()

  @property
  def task(self):
    return "visual_detection"

  def load_data(self):
    for subdir in os.listdir(self.root_dir):
      subdir_path = os.path.join(self.root_dir, subdir)
      if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
          if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(subdir_path, filename)
            annot_path = os.path.join(subdir_path, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(annot_path):
              self.data.append((image_path, annot_path))

  def __getitem__(self, index: int):
    image_path, annot_path = self.data[index]
    annots = []  # bboxes
    concept_ids = []

    with open(annot_path, 'r') as f:
      for line in f.readlines():
        parts = line.strip().split()
        if len(parts) != 5:
          continue
        class_id, x_center, y_center, width, height = map(float, parts)
        concept_id = str(int(class_id))
        
        # Convert YOLO format (x_center, y_center, width, height) to Clarifai format (left_col, top_row, right_col, bottom_row)
        left_col = x_center - (width / 2)
        top_row = y_center - (height / 2)
        right_col = x_center + (width / 2)
        bottom_row = y_center + (height / 2)

        # Ensure bounding boxes are within bounds
        left_col = max(0, left_col)
        top_row = max(0, top_row)
        right_col = min(1, right_col)
        bottom_row = min(1, bottom_row)

        if left_col >= right_col or top_row >= bottom_row:
          continue

        annots.append([left_col, top_row, right_col, bottom_row])
        concept_ids.append(concept_id)

    assert len(concept_ids) == len(annots), f"Num concepts must match num bbox annotations for a single image. Found {len(concept_ids)} concepts and {len(annots)} bboxes."

    # Generate a valid ID based on the image filename
    image_filename = os.path.basename(image_path)
    id_str = os.path.splitext(image_filename)[0].replace(' ', '_').replace('-', '_')
    id_str = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in id_str)
    id_str = id_str.strip('_')

    return VisualDetectionFeatures(image_path, concept_ids, annots, id=id_str)

  def __len__(self):
    return len(self.data)