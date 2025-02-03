# [Clarifai Data Utils](https://github.com/Clarifai/clarifai-python-datautils)

Clarifai Data Utils offers various types of multimedia data utilities. Enhance your experience by seamlessly integrating these utilities with the Clarifai Python SDK.

---


## Installation


Install from PyPi:

```bash
pip install clarifai-datautils
```

Install from Source:

```bash
git clone https://github.com/Clarifai/clarifai-python-datautils
cd clarifai-python-datautils
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Features

### Image Utils
- #### Annotation Loader
  - Load various annotated image datasets and export to clarifai Platform
  - Convert from one annotation format to other supported annotation formats

### Data Ingestion Pipeline
  - Easy to use pipelines to load data from files and ingest into clarifai platfrom.
  - Load text files(pdf, doc, etc..) , transform, chunk and upload to the Clarifai Platform

## Quick Usage
### Image Annotation Loader
```python
from clarifai_datautils.image import ImageAnnotations
#import from folder
coco_dataset = ImageAnnotations.import_from(path='folder_path',format= 'coco_detection')

#Using clarifai SDK to upload to Clarifai Platform
#export CLARIFAI_PAT={your personal access token}  # set PAT as env variable
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(dataloader=coco_dataset.dataloader)

#info about loaded dataset
coco_dataset.get_info()


#exporting to other formats
coco_dataset.export_to('voc_detection')
```

#### [Annotation Loader Notebook](./Image%20Annotation/image_annotation_loader.ipynb)

### Data Ingestion Pipelines

#### Setup
To use Data Ingestion Pipeline, please run
```python
pip install -r requirements-dev.txt
```


```python
from clarifai_datautils.multimodal import Pipeline, PDFPartition
from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace

# Define the pipeline
pipeline = Pipeline(
    name='pipeline-1',
    transformations=[
        PDFPartition(chunking_strategy = "by_title",max_characters = 1024),
        Clean_extra_whitespace()
    ]
)


# Using SDK to upload
from clarifai.client import Dataset
dataset = Dataset(dataset_url)
dataset.upload_dataset(pipeline.run(files = file_path, loader = True))

```

#### [Data Ingestion Notebooks](./Ingestion%20pipelines/)
