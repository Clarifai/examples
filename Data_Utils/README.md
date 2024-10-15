![Clarifai logo](docs/logo.png)

# Clarifai Python Data Utils


[![Discord](https://img.shields.io/discord/1145701543228735582)](https://discord.gg/M32V7a7a)
[![codecov](https://img.shields.io/pypi/dm/clarifai)](https://pypi.org/project/clarifai-datautils)


This is a collection of utilities for handling various types of multimedia data. Enhance your experience by seamlessly integrating these utilities with the Clarifai Python SDK. This powerful combination empowers you to address both visual and textual use cases effortlessly through the capabilities of Artificial Intelligence. Unlock new possibilities and elevate your projects with the synergy of versatile data utilities and the robust features offered by the [Clarifai Python SDK](https://github.com/Clarifai/clarifai-python). Explore the fusion of these tools to amplify the intelligence in your applications! 🌐🚀

[Website](https://www.clarifai.com/) | [Schedule Demo](https://www.clarifai.com/company/schedule-demo) | [Signup for a Free Account](https://clarifai.com/signup) | [API Docs](https://docs.clarifai.com/) | [Clarifai Community](https://clarifai.com/explore) | [Python SDK Docs](https://docs.clarifai.com/python-sdk/api-reference) | [Examples](https://github.com/Clarifai/examples) | [Colab Notebooks](https://github.com/Clarifai/colab-notebooks) | [Discord](https://discord.gg/XAPE3Vtg)

---
## Table Of Contents

* **[Installation](#installation)**
* **[Getting Started](#getting-started)**
* **[Features](#features)**
  * [Data Ingestion Pipeline](#ingestion-pipeline)


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

### Data Ingestion Pipelines

#### Setup
To use Data Ingestion Pipeline, please run
```python
pip install -r requirements-dev.txt
```


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