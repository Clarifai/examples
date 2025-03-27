### Model Prediction

Once the model is uploaded, you can easily make the prediction to the model using Clarifai SDK.

#### Prediction Method Structure

The client **exactly mirrors** the method signatures defined in your model's **model.py**:

| Model Implementation | Client Usage Pattern |
| --- | --- |
| **@ModelClass.method def func(self, prompt: str, image: Image = None, images: List[Image] = None, chat_history: List[dict] = None, max_tokens: int = 512, temperature: int = 0.7, top_p: float = 0.8)** | **model.func(prompt="Write 2000 word story")** |
| **@ModelClass.method def generate(self, prompt: str, image: Image = None, images: List[Image] = None, chat_history: List[dict] = None, max_tokens: int = 512, temperature: int = 0.7, top_p: float = 0.8)** | **model.generate(prompt="Write 2000 word story")** |
| **@ModelClass.method def chat(self, messages: List[dict] = None, max_tokens: int = 512, temperature: int = 0.7, top_p: float = 0.8)** | **model.chat(messages={'role': 'user', 'content': "Write 2000 word story", })** |

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

#### Unary-Unary Prediction

```python
# Single input prediction
result = model.predict("Write 2000 word story?")
print(f"unary-unary response: {result}")

# Batch processing (automatically handled)
batch_results = model.predict([
    {"text": "Write 2000 word story?"},
    {"text": "What is API?"},
])
for i, pred in enumerate(batch_results):
    print(f"unary-unary response {i}: {pred}")
```

#### Unary-Stream Prediction

#### Using `generate` Method

```python
response_stream = model.generate(text="Write 2000 word story", temperature=0.4, max_tokens=100)

for text_chunk in response_stream:
    print(text_chunk, end="", flush=True)
```

#### Using `chat` Method

```python
messages=[{
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": "What are in these images? Is there any difference between them?",
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://samples.clarifai.com/metro-north.jpg",
                  },
              },
              {
                  "type": "image_url",
                  "image_url": {
                      "url": "https://samples.clarifai.com/metro-north.jpg",
                  },
              },
          ],
      },
      ]

response_stream = model.chat(messages= messages, max_tokens = 256)

for text_chunk in response_stream:
    print(text_chunk, end="", flush=True)
```
