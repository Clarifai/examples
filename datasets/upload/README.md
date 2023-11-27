# Dataset upload
Examples of how to upload your datasets into clarifai app using features from `Dataset`.



---
## Requirement
```bash
pip install -U clarifai
```

**Note:**

- Ensure that the `CLARIFAI_PAT` environment variable is set.
- Ensure that the appropriate base workflow is being set for indexing respective input type.
---

## upload_dataset feature

- Examples of how to upload your local directory datasets into clarifai app using `module_dir` feature from `Dataset`.

#### Image Classification - Cifar10
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")

from image_classification.cifar10.dataset import Cifar10DataLoader
cifar_dataloader = Cifar10DataLoader()
dataset.upload_dataset(dataloader=cifar_dataloader)
```

#### Text Classification - IMDB Reviews
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")

#create a custom dataloader for the dataset and pass it in this function.
dataset.upload_dataset(dataloader=imdb_dataloader)
```

#### Object Detection - VOC - 2012
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")

from image_detection.voc.dataset import VOCDetectionDataLoader
voc_dataloader = VOCDetectionDataLoader()
dataset.upload_dataset(dataloader=voc_dataloader)
```

#### Image Segmentation - COCO
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")

from image_segmentation.coco.dataset import COCOSegmentationDataLoader
coco_dataloader = COCOSegmentationDataLoader()
dataset.upload_dataset(dataloader=coco_dataloader)
```

## upload_from_folder

- Uploading textfiles, imagefiles from local directory to Clarifai App.
- Quick injection of data into the app with or without annotations.
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_from_folder(folder_path='path_to_textfiles_folder', input_type='text', labels=True)
```

## upload_from_CSV

-  Uploading text data from CSV to Clarifai App.
- Quick injection of data into the app with or without annotations.
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_from_csv(csv_path='path_to_csv_file',input_type='text', labels=True)
```
