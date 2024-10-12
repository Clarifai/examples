# Clarifai Model Upload

Clarifai provides an easy-to-use platform to serve AI/ML models in production.

> This feature is currently in Private Preview. We'd love for you to try it out and provide your feedback. To do so please sign up for Private Preview [here](https://forms.gle/MSx7QNxmug2oFZYD6).

This guide will walk you through the process of uploading custom models to the Clarifai platform, leveraging pre-built model examples for different tasks. You'll also learn to customize models and adjust configurations for deployment.

### Available Model Examples
Clarifai provides a collection of pre-built model examples designed for different tasks. You can leverage these examples to streamline the model upload and deployment process.

## Installation

To begin, install the latest version of the `clarifai` Python package:

```bash
pip install --upgrade clarifai
```

Next, clone the repository containing example models:

```bash
git clone https://github.com/Clarifai/examples.git
```

### Environment Setup

Before proceeding, ensure that the `CLARIFAI_PAT` (Personal Access Token) environment variable is set. This token authenticates your connection to the Clarifai platform.

```bash
export CLARIFAI_PAT="your_personal_access_token"
```

## Model Folder Structure

Your model folder should contain three key files for uploading a model to Clarifai:

* **requirements.txt** - Lists the Python libraries required for your model to run.
* **model.py** - Contains the implementation of your model (e.g., loading and running inference).
* **config.yaml** - Holds metadata and configuration needed for building the Docker image, defining compute resources, and uploading the model.

### Example Directory Structure:

```plaintext
your_model_directory/
├── 1/
    ├── model.py
├── requirements.txt
└── config.yaml
```

## Clarifai Model Upload Guide

### Step 1: Define the `config.yaml` File

The `config.yaml` file is crucial for specifying model metadata, compute requirements, and model checkpoints. Here's a breakdown of key sections:

#### Model Info

This section defines your model’s ID, user ID, and app ID. These fields determine where your model will be uploaded in the Clarifai platform.

```yaml
model:
  id: "model_id"
  user_id: "user_id"
  app_id: "app_id"
  model_type_id: "text-to-text" # Change this according to your model type (e.g., image-classifier, text-to-text).
```

#### Compute Resources

Define the compute resources your model requires to run. This includes CPU, memory, and optional GPU resources.
```yaml
inference_compute_info:
  cpu_limit: "2"
  cpu_memory: "13Gi"
  num_accelerators: 1
  accelerator_type: ["NVIDIA-A10G"] # Specify GPU type if needed
  accelerator_memory: "15Gi"
```
This is the minimum required compute resource for this model for inference

- **cpu_limit:** The number of CPUs to allocate (follows Kubernetes notation, e.g., "1").
- **cpu_memory:** Minimum amount of CPU memory (follows Kubernetes notation, e.g., "1Gi", "1500Mi", "3Gi").
- **num_accelerators**: Number of GPU/TPUs to use
- **accelerator_memory**: Minimum amount of memory for the GPU/TPU.
- **accelerator_type**: Supported accelerators that the model can run on.

#### Model Checkpoints

If you are using a model from Hugging Face, you can automatically download the model checkpoints by specifying `checkpoint` section in the `config.yaml`.

> For private or restricted Hugging Face repositories, an HF access token must be provided.

```yaml
checkpoints:
  type: "huggingface"
  repo_id: "meta-llama/Meta-Llama-3-8B-Instruct"
  hf_token: "your_hf_token" # Required for private models
```

#### Model concepts/ labels
> Important: This section is necessary if your model outputs concepts/labels (e.g., for classification or detection) and is not directly loaded from Hugging Face.

Models that outputs concepts/lables like classification or detection models need a concepts section defined in `config.yaml`

```yaml
concepts:
- id: '0'
  name: bus
- id: '1'
  name: person
- id: '2'
  name: bicycle
- id: '3'
  name: car
```
> Note: If using a model from Hugging Face and the `checkpoints` section is defined, the platform will automatically infer concepts, and there’s no need to specify them manually.


### Step 2: Define dependencies in requirements.txt
List all required Python dependencies for your model in `requirements.txt` file. This ensures that all necessary libraries are installed in the runtime environment

### Step 3: Prepare the `model.py` File

The `model.py` file contains the logic for your model, including how the model is loaded and how predictions are made. This file must implement a class that inherits from `ModelRunner`, which should define the following methods:

#### Example Class Structure:

```python
from clarifai.runners.models.model_runner import ModelRunner

class YourCustomModelRunner(ModelRunner):
def load_model(self):
  '''Initialize and load the model here'''
  pass

def predict(self, request):
  '''Define logic to handle input and return output'''
  return output_data

def generate(self, request):
'''Define streaming output logic if needed'''
  pass

def stream(self, request):
'''Handle both streaming input and output'''
  pass
```

* **load_model()**: Loads the model, similar to an initialization step.
* **predict(input_data)**: Handles the core logic for making a prediction with the model. It processes input data and returns a output reponse.
* **generate(input_data)**: Returns output in a streaming manner (if applicable).
* **stream(input_data)**: Manages streaming input and output (for more advanced use cases).

### Step 4: Test Model locall
Before uploading your model to the Clarifai platform, test it locally to avoid upload failures due to typos or misconfigurations.

```bash
python -m clarifai.runners.models.model_run_locally --model_path <model_path>

This prevents from model upload failure due to typos in model.py or wrong implementation of model
```
> Note: Ensure your local setup has enough memory to load and run the model for testing.

### Step 5: Upload the Model to Clarifai

Now that your custom model is ready, you can serve the model in production using:

```bash
python -m clarifai.runners.models.model_upload --model_path {model_directory_path}
```

This command builds the model Docker image based on the specified compute resources and uploads it to the Clarifai platform.
