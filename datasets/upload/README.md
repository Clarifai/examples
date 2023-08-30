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
dataset.upload_dataset(task="visual_classification", split="train", module_dir="path_to_cifar10_module")
```

#### Image Classification - [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="visual_classification", split="train", module_dir="path_to_food-101_module")
```

#### Text Classification - IMDB Reviews
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="text_clf", split="train", module_dir="path_to_imdb_reviews_module")
```

#### Object Detection - VOC - 2012
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="visual_detection", split="train", module_dir="path_to_voc_module")
```

#### Image Segmentation - COCO
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="visual_segmentation", split="train", module_dir="path_to_coco_module")
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
