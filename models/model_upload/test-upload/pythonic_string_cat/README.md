## **Clarifai Python Model Interface Guide**

This will guild you how to use Clarifai's new Python Model Interface for building and prediction from custom models.

## **Key Features** 

The Clarifai Python Model Interface simplifies model integration by:

* Abstracting Protobuf complexity
* Pythonic interface with native type support
* Supporting multi-modal inputs/outputs
* Enabling both single and batch predictions
* Streaming input/output support

### **Model Structure**

my\_model/

├── 1/

          ├── model.py

├── requirements.txt

└── config.yaml

## **Defining a Custom Model** 

### **Basic Template**

```plaintext
from clarifai.runners.models import ModelClass
from clarifai.runners.utils import Image, Text, Output

class CustomModel(ModelClass):
def load_model(self):
"""Initialize model weights and resources"""
self.model = load_your_model()

def predict(self, text: Text = None, image: Image = None) -> Output:
"""Single prediction handling"""
# Process inputs
prediction_text = self.model.predict(
text=text.text if text else None,
image=image.to_pil() if image else None
)
return prediction_text
```

### **Required Methods**

| Method | Description |
| --- | --- |
| **load\_model** | Initialize model resources |
| **predict** | Handle single input prediction |
| **generate** | Stream output predictions (optional) |
| **stream** | Handle streaming input/output (optional) |

## **Input/Output Handling** 

## **Supported Input/Output Types** 

### **Input Types**

| Type | Example | Proto Mapping |
| --- | --- | --- |
| **str** | **"hello world"** | Text.raw |
| **bytes** | **b"image\_bytes"** | Data.bytes\_value |
| **int** | **42** | Data.int\_value |
| **float** | **3.14** | Data.float\_value |
| **bool** | **True** | Data.bool\_value |
| **Text** | **Text("input text")** | Data.text |
| **Image** | **Image.from\_url("image.jpg")** | Data.image |
| PIL.Image | PIL.Image.Image.open(image) | Data.image |
| **Audio** | **Audio.from\_url(audio.wav)** | Data.audio |
| **List\[Image\], List\[Audio\], List\[Video\]** | **\[Image(...), Image(...)\]** | Repeated Data.parts |
| **np.ndarray** | **np.array(\[\[1,2\],\[3,4\]\])** | Data.ndarray |
| **Dict** (metadata) | **{"param": "value"}** | Data.metadata |

### **Output Types**

| Type | Example |
| --- | --- |
| **str** | **"hello world"** |
| **bytes** | **b"image\_bytes"** |
| **int** | **42** |
| **float** | **3.14** |
| **bool** | **True** |
| **Text** | **Text("input text")** |
| **Image** | **Image.from\_url("image.jpg")** |
| **PIL.Image** | **PIL.Image.Image.open(image)** |
| **Audio** | **Audio.from\_url(audio.wav)** |
| **Video** | **Video.from\_url(video.mp4)** |
| **List\[Image\], List\[Audio\], List\[Video\]** | **\[Image(...), Image(...)\]** |
| **np.ndarray** | **np.array(\[\[1,2\],\[3,4\]\])** |
| **Dict (metadata)** | **{"param": "value"}** |
| **Output** | **Output(text1= “result”, image=bytes, metadata={"key": value})** |


 

### **Output object Construction**

```python

# Multi-modal output
Output(
    text="Result text",
    confidence=0.95,
    heatmap=Image.from_pil(heatmap_image)
)

```

## **Edge Cases Error Handling**

1. **Missing Parameters**
* I've done proper error handling of Missing required  Parameters
2. **Type Mismatches or Incorrect data type in predict method**

i.e def predict(x: str) -> str: return x and the client calls with model.predict(x=Image.open('test.jpg'))

```plaintext
Exception: Model Predict failed with response code: FAILURE
details: "expected str datatype but the provided input is not a str"
req_id: "sdk-python-11.1.5-f79bd7b9739a4879b9f2abcd774012f8"
```

3. client calls with the wrong parameter argument

i.e. def predict(x: str) -> str: return x and the client calls with model.predict(y='string  input')


```plaintext
Exception: Model Predict failed with response code: FAILURE
details: "Unknown parameter: `text3` in predict method, available parameters: odict_keys([\'text1\', \'text2\'])"
req_id: "sdk-python-11.1.5-dec9f9995bac4d53ba30cbadac651ae2"
```


 

## **Examples**

### **1\. Text Classification**

```python
from clarifai.runners.models.model_class import ModelClass
class TextClassifier(ModelClass):
def load_model(self):
    from transformers import pipeline
    self.classifier = pipeline("text-classification")

def predict(self, text: str) -> Output:
    result = self.classifier(text.text)[0]
    return Output(
    label=result['label'],
    confidence=result['score']
    )

# Usage
from clarifai.client import Model
model = Model(url)
print(model.predict(text="I love this product!"))
# Output: label="POSITIVE", confidence=0.99
```

**Text streaming Model**

```python
from clarifai.runners.models.model_class import ModelClass
class StreamingModel(ModelClass):
    def generate(self, prompt: str, image: Image) -> Iterator[Output]:
        for token in stream_llm_response(prompt.text, image.to_pil()):
            yield token

# Client usage
from clarifai.client import Model
model = Model(url)
for partial in model.generate(prompt="Explain AI:", Image.from_url('https://samples.clarifai.com/metro-north.jpg')):
print(token, end="", flush=True)
```

### **3\. Image to text Model**

```python
from clarifai.runners.models.model_class import ModelClass
class CaptionGenerator(ModelClass):
def predict(self, image: Image, context: Text = None) -> Output:
    caption = generate_caption(image.to_pil())
    if context:
        caption = f"{context.text}: {caption}"
    return caption

# Usage
from clarifai.client import Model
model = Model(url)
print(model.predict(image=Image.from_url("https://example.com/image.jpg"),context=Text("about the image"),))
```
