# Data Ingestion Pipeline
Load text files(pdf, doc, etc..) , transform, chunk and upload to the Clarifai Platform

## Features

- File Partitioning
- Cleaning Chunks
- Metadata Extraction


## Setup
To use Data Ingestion Pipeline, please run
```python
pip install -r requirements-dev.txt
```

## Notebooks
- [Ready to Use Pipelines](./Ready_to_use_foundational_pipelines.ipynb)
- [Multimodal dataloader](Multimodal_dataloader.ipynb)

## Quick Usage

```python
from clarifai_datautils.text import Pipeline, PDFPartition
from clarifai_datautils.text.pipeline.cleaners import Clean_extra_whitespace

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

## Supported File Formats
- PDF
- Text(.txt)
- Docx
- Markdown