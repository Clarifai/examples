# Model Training
Examples of how to train different Model Types in Clarifai Platform.



---
## Requirement
```bash
pip install -U clarifai
```

**Note:**

- Ensure that the `CLARIFAI_PAT` environment variable is set.
---

## Model Training



#### Create model with trainable model_type
```python
from clarifai.client.app import App
from clarifai.client.model import Model

app = App(user_id="user_id", app_id="app_id")
model = app.create_model(model_id="model_id", model_type_id="visual-classifier")
               (or)
model = Model('url')
```

#### List training templates for the model_type
```python
templates = model.list_training_templates()
print(templates)
```

#### Get parameters for the model.
```python
params = model.get_params(template='classification_basemodel_v1', yaml_file='model_params.yaml')
```

#### Update the model params yaml and pass it to model.train()
```python
model_version_id = model.train('model_params.yaml')
```

#### Training status and saving logs
```python
status = model.training_status(version_id=model_version_id,training_logs=True)
print(status)
```


## Notebooks
| Notebook | Open in Colab |
| ----------- | ----------- |
| [Image Classificaion Training](image-classification_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-classification_training.ipynb) |
| [Text Classificaion Training](text-classification_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/text-classification_training.ipynb) |
| [Image Detection Training](image-detection_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-detection_training.ipynb) |
| [Image Segmentation Training](image-segmentation_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-segmentation_training.ipynb) |
