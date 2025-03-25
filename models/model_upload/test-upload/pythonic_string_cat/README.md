### Model Prediction

Once the model is uploaded, you can easily make the prediction to the model using Clarifai SDK.

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
)

# Or initialize with model URL
model = Model(model_url="https://clarifai.com/model_user_id/model_app_id/models/model_id",)
```

#### Unary-Stream Prediction

```python
# Single input prediction
result = model.predict("Write 2000 word story?")
print(f"unary-unary response: {result['cat']:.2%}")

# Batch processing (automatically handled)
batch_results = model.predict([
    {"text": "Write 2000 word story?"},
    {"text": "What is API?"},
])
for i, pred in enumerate(batch_results):
    print(f"unary-unary response {i}: {pred}")
```

#### Unary-Stream Prediction

```python
response_stream = model.generate(text="Write 2000 word story?")

for text_chunk in response_stream:
    print(text_chunk, end="", flush=True)
```

#### Stream-Stream Prediction

```python
# client-side streaming

for text_chunk in model.stream(iter(["Write 2000 word story?"] * 5)):
    print(text_chunk, end="", flush=True)
```
