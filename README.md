# Clarifai Python SDK Examples


This is a collection of examples for Clarifai-python. Use these examples to learn Clarifai and build your own robust and scalable AI applications.

Experience the power of Clarifai in building Computer Vision , Natual Language processsing , Generative AI applications.

## Setup
* Sign up for a free account at [clarifai](https://clarifai.com/signup) and set your PAT key.

* Install the [Clarifai python sdk.](https://github.com/Clarifai/clarifai-python/tree/master)

* Export your PAT as an environment variable.
    ```cmd
    export CLARIFAI_PAT={your personal access token}
    ```

* Explore and run the examples  in this  repo.

## Usage

The examples are organized into several folders based on their category. A quick example below,

```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="text_clf", split="train", module_dir="path_to_imdb_reviews_module")
```


## Notebooks

 - [Dataset Upload](datasets/upload/dataset_upload.ipynb)


## Note

Although these scripts are run on your local machine, they'll communicate with Clarifai and run in our cloud on demand.

Examples provide a guided tour through Clarifai's concepts and capabilities.
contains uncategorized, miscellaneous examples.