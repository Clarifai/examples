## Clarifai Model Upload Examples

Clarifai provides a simplest way to serve AI/ML models in production.

There are collection of pre-built model examples for different tasks.

## Installation

Install latest `clarifai`

```bash
pip install --upgrade clarifai
```

Get the repository with:

```
git clone https://github.com/Clarifai/examples.git
```

## Model Upload

#### Note:
- Ensure that the CLARIFAI_PAT environment variable is set.

### Structure of Model Folder

There are only three necessary files:
 - `requirements.txt` - model requirements python libraries
 - `model.py` - the implementation of the model.
 - `config.yaml` - contains compute resource information and other details needed to build the Docker image and upload the model to the to Clarifai Platform.

### Define config.yaml
- Define a appropriate model id, and your user_id and app_id in the config.yaml of model example for where the model should be uploaded to Clarifia Platform

- Edit the model.py to do as you’d like for the implementation of your model

- Now that your custom model is ready, you can upload it to the API.

```
python -m clarifai.runners.models.model_upload --model_path {model_path}
```
