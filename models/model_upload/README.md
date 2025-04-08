# Clarifai Model Upload

[Clarifai](https://www.clarifai.com/) provides an easy-to-use platform to serve AI/ML models in production.

> This feature is currently in Private Preview. We'd love for you to try it out and provide your feedback. To do so please sign up for Private Preview [here](https://forms.gle/MSx7QNxmug2oFZYD6).

This guide will walk you through the process of uploading custom models to the Clarifai platform, leveraging pre-built model examples for different tasks. You'll also learn to customize models and adjust configurations for deployment.

## Contents

- [Clarifai Model Upload](#clarifai-model-upload)
  - [Contents](#contents)
    - [Available Model Examples](#available-model-examples)
  - [Installation](#installation)
    - [Environment Setup](#environment-setup)
  - [Model Folder Structure](#model-folder-structure)
    - [Example Directory Structure:](#example-directory-structure)
  - [Model Upload Guide](#model-upload-guide)
    - [Step 1: Define the `config.yaml` File](#step-1-define-the-configyaml-file)
      - [Model Info](#model-info)
      - [Compute Resources](#compute-resources)
      - [Model Checkpoints](#model-checkpoints)
      - [Model concepts/ labels](#model-concepts-labels)
    - [Step 2: Define dependencies in requirements.txt](#step-2-define-dependencies-in-requirementstxt)
    - [Step 3: Prepare the `model.py` File](#step-3-prepare-the-modelpy-file)
      - [Core Model Class Structure\*\*](#core-model-class-structure)
      - [Example Class Structure:](#example-class-structure)
      - [load\_model() (Optional):](#load_model-optional)
      - [Method Decorators](#method-decorators)
      - [Supported Input and Output Data Types](#supported-input-and-output-data-types)
    - [Step 4: Test Model locally](#step-4-test-model-locally)
      - [Testing the Model in a Container](#testing-the-model-in-a-container)
      - [Testing the Model in a Virtual Environment](#testing-the-model-in-a-virtual-environment)
      - [Running the Model in a Docker Container](#running-the-model-in-a-docker-container)
      - [Running the Model in a Virtual Environment](#running-the-model-in-a-virtual-environment)
      - [Making Inference Requests to the Locally Running Model](#making-inference-requests-to-the-locally-running-model)
      - [CLI flags](#cli-flags)
    - [Step 5: Upload the Model to Clarifai](#step-5-upload-the-model-to-clarifai)
    - [Step 6: Model Prediction](#step-6-model-prediction)
      - [Prediction Method Structure](#prediction-method-structure)
      - [Initializing the Model Client](#initializing-the-model-client)
      - [Unary Prediction](#unary-prediction)
      - [Unary-Stream Prediction](#unary-stream-prediction)
      - [Stream-Stream Prediction](#stream-stream-prediction)
      - [Dynamic Batch Prediction Handling](#dynamic-batch-prediction-handling)

### Available Model Examples
Clarifai provides a collection of pre-built model examples designed for different tasks. You can leverage these examples to streamline the model upload and deployment process.

## Installation

To begin, install the latest version of the `clarifai` Python package:

```bash
python -m pip install --upgrade clarifai
```

Next, clone the repository containing example models:

```bash
git clone https://github.com/Clarifai/examples.git
```

### Environment Setup

Before proceeding, ensure that the `CLARIFAI_PAT` (Personal Access Token) environment variable is set.

* You can generate PAT key in the Personal settings -> [Security section](https://clarifai.com/settings/security)
* This token authenticates your connection to the Clarifai platform.

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

## Model Upload Guide

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
  accelerator_type: ["NVIDIA-T4", "NVIDIA-A10G","NVIDIA-L4","NVIDIA-L40S","NVIDIA-A100","NVIDIA-H100"] # Specify GPU types if needed
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
  when: "runtime"
```

> `when` in the checkpoints section defines when checkpoints for the model should be downloaded and stored. The `when` parameter must be one of `['upload', 'build', 'runtime']`. 
>
> * **runtime**: Downloads checkpoints at runtime when loading the model in the `load_model` method.
> * **build**: Downloads checkpoints at build time, while the image is being built.
> * **upload**: Downloads checkpoints before uploading the model.
>
> For larger models, we recommend downloading checkpoints at `runtime`. Downloading them during the `upload` or `build` stages can significantly increase the Docker image size, leading to longer upload times and increased cold start time for the model. Downloading checkpoints at `runtime` has some advantages:
>
> * Smaller image sizes
> * Decreases the model-build duration
> * Faster pushes and inference on Clarifai platform

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


> If your model requires `Torch`, we provide optimized pre-built Torch images as the base for machine learning and inference tasks. These images include all necessary dependencies, ensuring efficient execution. 
>
>
> The available pre-built Torch images are:
> • `2.4.1-py3.11-cuda124`: Based on PyTorch 2.4.1, Python 3.11, and CUDA 12.4.
> • `2.5.1-py3.11-cuda124`: Based on PyTorch 2.5.1, Python 3.11, and CUDA 12.4.
> • `2.4.1-py3.12-cuda124`: Based on PyTorch 2.4.1, Python 3.12, and CUDA 12.4.
> • `2.5.1-py3.12-cuda124`: Based on PyTorch 2.5.1, Python 3.12, and CUDA 12.4.
>
> To use a specific Torch version, define it in your `requirements.txt` file like this: `torch==2.5.1` This ensures the correct pre-built image is pulled from Clarifai's container registry, ensuring the correct environment is used. This minimizes cold start times and speeds up model uploads and runtime execution — avoiding the overhead of building images from scratch or pulling and configuring them from external sources.
> We recommend using either `torch==2.5.1` or `torch==2.4.1`. If your model requires a different Torch version, you can specify it in requirements.txt, but this may slightly increase the model upload time.

### Step 3: Prepare the `model.py` File

The `model.py` file contains the logic for your model, including how the model is loaded and how predictions are made. This file must implement a class that inherits from `ModelClass`.

#### Core Model Class Structure**

To define a custom model, you need to create a class that inherits from **ModelClass** and implements the **load\_model** method and at least one method decorated with **@ModelClass.method** to define prediction endpoints

> Each parameter of the Class method must be annotated with a type. The method's return type must also be annotated. The supported types are described [here](SUPPORTED_DATATYPE.md):

#### Example Class Structure:

```python
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Stream, Text


class MyModel(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text."""

  def load_model(self):
    """Load the model here."""

  @ModelClass.method
  def predict(self, text1: Text = "") -> Text:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    output_text = text1.text + "Hello World"

    return Text(output_text)

  @ModelClass.method
  def generate(self, text1: Text = Text("")) -> Stream[Text]:
    """Example yielding a whole batch of streamed stuff back."""

    for i in range(10):  # fake something iterating generating 10 times.
      output_text = text1.text + f"Generate Hello World {i}"
      yield Text(output_text)

  @ModelClass.method
  def stream(self, input_iterator: Stream[Text]) -> Stream[Text]:
    """Example yielding a whole batch of streamed stuff back."""

    for i, input in enumerate(input_iterator):
      output_text = input.text + f"Stream Hello World {i}"
      yield Text(output_text)
```

#### load\_model() (Optional):

Use this _optional_ method to include any expensive one-off operations in here like loading trained models, instantiate data transformations, etc.

* One-time initialization of heavy resources
* Called during model container startup
* Ideal for loading model:

```python
def load_model(self):
  self.tokenizer = AutoTokenizer.from_pretrained("model/")
  self.pipeline = transformers.pipeline(...)
```

#### Method Decorators

* **@ModelClass.method** registers prediction endpoints
* Supports method types via type hints:

```python
# Unary-Unary (Standard request-response)
@ModelClass.method
def predict(self, input: Image) -> Text

# Unary-Stream (Server-side streaming)
@ModelClass.method
def generate(self, prompt: Text) -> Stream[Text]

# Stream-Stream (Bidirectional streaming)
@ModelClass.method
def analyze_video(self, frames: Stream[Image]) -> Stream[str]
```

#### [Supported Input and Output Data Types](SUPPORTED_DATATYPE.md)
Clarifai's model framework supports rich data typing for both inputs and outputs. [Here](SUPPORTED_DATATYPE.md) is a comprehensive guide to supported types with usage examples.

### Step 4: Test Model locally

Before uploading your model to the Clarifai platform, it’s important to test it locally. This ensures that your model works as expected, avoids upload failures caused by typos, misconfigurations, or implementation errors, and saves valuable time during deployment.

You can test your model locally using **two methods**:

1. **Test with the single CLI command:**

This method runs the model locally and sends a sample request to verify that the model responds successfully.

2. **Run the model locally and perform inference using the Clarifai client SDK:**

This method runs the model locally and starts a local gRPC server at \`https://localhost:{port}\`, Once the server is running, you can perform inference on the model via the Clarifai client SDK

You can test the model within a Docker container or a Python virtual environment.

> **Recommendation:** If Docker is installed on your system, it is highly recommended to use Docker for testing or running the model, as it provides better isolation and avoids dependency conflicts.

1. ### Testing the Model

To test the model, you need to implement a `test` method in the `model.py` file. This method should call other model methods for validation. When you run the below `test-locally` CLI command, it will execute the `test` method to perform the model testing.

Below is a sample `model.py` file with an example implementation of the `test` method

```python
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Stream, Text


class MyModel(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text."""

  def load_model(self):
    """Load the model here."""

  @ModelClass.method
  def predict(self, text1: Text = "") -> Text:
    output_text = text1.text + "Hello World"

    return Text(output_text)

  @ModelClass.method
  def generate(self, text1: Text = Text("")) -> Stream[Text]:
    """Example yielding a whole batch of streamed stuff back."""

    for i in range(10):  # fake something iterating generating 10 times.
      output_text = text1.text + f"Generate Hello World {i}"
      yield Text(output_text)

  def test(self):
    res = self.predict(Text("test"))
    assert res.text == "testHello World"

    res = self.generate(Text("test"))
    for i, r in enumerate(res):
      assert r.text == f"testGenerate Hello World {i}"
```

#### Testing the Model in a Container

```python
clarifai model test-locally {model_path} --mode container
```

#### Testing the Model in a Virtual Environment

```python
clarifai model test-locally {model_path} --mode env
```

#### Running the Model in a Docker Container

```python
clarifai model run-locally {model_path} --mode container --port 8000
```

#### Running the Model in a Virtual Environment

```python
clarifai model run-locally {model_path} --mode container --port 8000
```

#### Making Inference Requests to the Locally Running Model

Once the model is running locally, you need to configure the `CLARIFAI_API_BASE` environment variable to point to the localhost and the port where the gRPC server is running

```python
export CLARIFAI_API_BASE="localhost:{port}"
```

Then make `unary-unary`, `unary-stream` and `stream-stream` predict calls to the model based on [Step 6: Model Prediction](#step-6-model-prediction)

#### CLI flags

* `--model_path`: Path to the model directory.
* `--mode`: Specify how to run the model: "env" for virtual environment or "container" for Docker container. Defaults to "env". \[default: env\]
* `-p` or `--port`: The port to host the gRPC server for running the model locally. Defaults to 8000..
* `--keep_env`: Keep the virtual environment after testing the model locally (applicable for `env` mode). Defaults to False.
* `--keep_image`: Keep the Docker image after testing the model locally (applicable for container mode). Defaults to False.

This prevents from model upload failure due to typos in model.py or wrong implementation of model.

> Note: Ensure your local setup has enough memory to load and run the model for testing.

### Step 5: Upload the Model to Clarifai

Now that your custom model is ready, you can serve the model in production using:

```bash
clarifai model upload {model_directory_path}
```

This command builds the model Docker image based on the specified compute resources and uploads it to the Clarifai platform.


### Step 6: Model Prediction

Once the model is uploaded, you can easily make the prediction to the model using Clarifai SDK.

> Make sure to create compute cluster and nodepool before making predict call

#### Prediction Method Structure

The client **exactly mirrors** the method signatures defined in your model's **model.py**:

| Model Implementation | Client Usage Pattern |
| --- | --- |
| **@ModelClass.method def func(...)** | **model.func(...)** |
| **@ModelClass.method def generate(...)** | **model.generate(...)** |
| **@ModelClass.method def analyze(...)** | **model.analyze(...)** |

**Key Characteristics:**

* Method names match exactly what's defined in **model.py**
* Arguments/parameters preserve the same names and types
* Return types mirror the model's output definitions

#### Initializing the Model Client
First, instantiate your model with proper credentials:

```python
from clarifai.client.model import Model

# Initialize with explicit IDs
model = Model(
    user_id="model_user_id",
    app_id="model_app_id",
    model_id="model_id",
    compute_cluster_id="cluster_id",
    nodepool_id="nodepool_id"
)

# Or initialize with model URL
model = Model(
    model_url="https://clarifai.com/model_user_id/model_app_id/models/model_id",
    compute_cluster_id="your_cluster_id",
    nodepool_id="your_nodepool_id"
)
```
> Make sure to create compute cluster and nodepool before making predict call and if you don't provide `compute_cluster_id` and `nodepool_id` or `deployment_id` while initializing the Model Client, model will use the Clarifai Shared Nodepool.

#### Unary Prediction

_**Model Method Signature (server-side):**_

```python
@ModelClass.method
def predict_image(self, image: Image) -> Dict[str, float]:
```

_**unary-unary predict call (Client Usage:)**_

```python
# Single input
result = model.predict_image(
    image=Image(url="https://example.com/pet1.jpg")
)
print(f"Cat confidence: {result['cat']:.2%}")

# Batch processing (automatically handled)
batch_results = model.predict_image([
    {"image": Image(url="https://example.com/pet1.jpg")},
    {"image": Image(url="https://example.com/pet2.jpg")},
    ])
for i, pred in enumerate(batch_results):
print(f"Image {i+1} cat confidence: {pred['cat']:.2%}")
```

#### Unary-Stream Prediction

_**Model Method Signature (server-side):**_

```python
@ModelClass.method
def generate(self, prompt: Text) -> Stream[Text]:
```

_**Client Usage:**_

```python
response_stream = model.generate(
    prompt=Text("Explain quantum computing in simple terms")
)

for text_chunk in response_stream:
    print(text_chunk.text, end="", flush=True)
```

#### Stream-Stream Prediction

_**Model Method Signature (server-side):**_

```python
@ModelClass.method
def transcribe_audio(self, audio: Stream[Audio]) -> Stream[Text]:
```

_**Client Usage:**_

```python
# client-side streaming

for text_chunk in model.transcribe_audio(audio=iter(Audio(bytes=b''))):
    print(text_chunk.text, end="", flush=True)
```


#### Dynamic Batch Prediction Handling

Clarifai's model framework automatically handles both single and batch predictions through a unified interface, dynamically adapting to input formats without requiring code changes.

**Automatic Input Detection**

* Single input: Processed as singleton batch
* Multiple inputs in a list: Handled as parallel batch

Therefore, when a user passes multiple inputs as a list, the system automatically handles them as a batch. This means users can pass a single input or a list, and the system adapts accordingly.

**Model Implementation:**

```python
class TextClassifier(ModelClass):
  @ModelClass.method
  def predict(self, text: Text) -> float:
    """Single text classification (automatically batched)"""
    return self.model(text.text)
```

**Client Usage:**

```python
# Single prediction
single_result = model.predict(Text("Positive review"))

# Batch prediction
batch_results = model.predict([
    {"text": Text("Positive review")},
    {"text": Text("Positive review")},
    {"text": Text("Positive review")},
  ])
```
