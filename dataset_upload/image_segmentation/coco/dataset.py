#! COCO 2017 Image Segmentation dataset

import gc
import os
from functools import reduce

import cv2
import numpy as np
from clarifai.datasets.upload.base import ClarifaiDataLoader
from clarifai.datasets.upload.features import VisualSegmentationFeatures
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


class COCOSegmentationDataLoader(ClarifaiDataLoader):
  """COCO 2017 Image Segmentation Dataset. url: https://cocodataset.org/#download
  """

  def __init__(self, split: str = "train"):
    """Inititalize dataset params.
    Args:
      split: "train" or "test"
    """
    self.split = split
    self.image_dir = {"train": os.path.join(os.path.dirname(__file__), "images")}
    self.annotations_file = {
        "train":
            os.path.join(os.path.dirname(__file__), "annotations/instances_val2017_subset.json")
    }

    self.load_data()

  def load_data(self):
    self.coco = COCO(self.annotations_file[self.split])
    categories = self.coco.loadCats(self.coco.getCatIds())
    self.cat_id_map = {category["id"]: category["name"] for category in categories}
    self.cat_img_ids = {}
    for cat_id in list(self.cat_id_map.keys()):
      self.cat_img_ids[cat_id] = self.coco.getImgIds(catIds=[cat_id])

    img_ids = []
    for i in list(self.cat_img_ids.values()):
      img_ids.extend(i)

    # Get the image information for the specified image IDs
    image_info = self.coco.loadImgs(img_ids)
    # Extract the file names from the image information
    self.image_filenames = {img_id: info['file_name'] for info, img_id in zip(image_info, img_ids)}
    self.img_ids = list(set(img_ids))

  def __len__(self):
    return len(self.img_ids)

  def __getitem__(self, idx):

    _id = self.img_ids[idx]
    annots = []  # polygons
    class_names = []
    labels = [i for i in list(filter(lambda x: _id in self.cat_img_ids[x], self.cat_img_ids))]
    image_path = os.path.join(self.image_dir[self.split], self.image_filenames[_id])

    image_height, image_width = cv2.imread(image_path).shape[:2]
    for cat_id in labels:
      annot_ids = self.coco.getAnnIds(imgIds=_id, catIds=[cat_id])

      if len(annot_ids) > 0:
        img_annotations = self.coco.loadAnns(annot_ids)
        for ann in img_annotations:
          # get polygons
          if type(ann['segmentation']) == list:
            for seg in ann['segmentation']:
              poly = np.array(seg).reshape((int(len(seg) / 2), 2))
              poly[:, 0], poly[:, 1] = poly[:, 0] / image_width, poly[:, 1] / image_height
              annots.append(poly.tolist())  #[[x=col, y=row],...]
              class_names.append(self.cat_id_map[cat_id])
          else:  # seg: {"counts":[...]}
            if type(ann['segmentation']['counts']) == list:
              rle = maskUtils.frPyObjects([ann['segmentation']], image_height, image_width)
            else:
              rle = ann['segmentation']
            mask = maskUtils.decode(rle)  #binary mask
            #convert mask to polygons and add to annots
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for cont in contours:
              if cont.size >= 6:
                polygons.append(cont.astype(float).flatten().tolist())
            # store polygons in (x,y) pairs
            polygons_flattened = reduce(lambda x, y: x + y, polygons)
            del polygons
            del contours
            del mask
            gc.collect()

            polygons = np.array(polygons_flattened).reshape((int(len(polygons_flattened) / 2), 2))
            polygons[:, 0] = polygons[:, 0] / image_width
            polygons[:, 1] = polygons[:, 1] / image_height

            annots.append(polygons.tolist())  #[[x=col, y=row],...,[x=col, y=row]]
            class_names.append(self.cat_id_map[cat_id])
      else:  # if no annotations for given image_id-cat_id pair
        continue
    assert len(class_names) == len(annots), f"Num classes must match num annotations\
    for a single image. Found {len(class_names)} classes and {len(annots)} polygons."

    return VisualSegmentationFeatures(
        image_path, class_names, annots, id=self.image_filenames[_id].split(".")[0])
