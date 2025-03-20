
# Supported Input and Output Data Types

## Contents

- [Supported Input and Output Data Types](#supported-input-and-output-data-types)
  - [Contents](#contents)
  - [Core Primitive Types](#core-primitive-types)
  - [Supported Data Types: Python Primitive \& Generic Types](#supported-data-types-python-primitive--generic-types)
    - [Core Primitive Types](#core-primitive-types-1)
      - [Primitive Type Usage](#primitive-type-usage)
    - [Generic Container Types](#generic-container-types)
      - [1. List\[T\]](#1-listt)
      - [2. Dict\[K, V\]](#2-dictk-v)
      - [3. Tuple\[T1, T2, ...\]](#3-tuplet1-t2-)
  - [NamedFields DataType ](#namedfields-datatype)
    - [Streaming with NamedFields Usage Datatype](#streaming-with-namedfields-usage-datatype)
      - [Client Usage:](#client-usage)

Clarifai's model framework supports rich data typing for both inputs and outputs. Below is a comprehensive guide to supported types with usage examples:
> Each parameter of the Model Class method must be annotated with a type. The method's return type must also be annotated. The supported types are described below:
## Core Primitive Types

| Type | Python Class | Description | Initialization Examples |
| --- | --- | --- | --- |
| **Text** | **Text** | UTF-8 encoded text | **Text("Hello World")** **Text(url="https://example.com/text.txt")** |
| **Image** | **Image** | RGB images (PNG/JPG format) | **Image(bytes=b"")** **Image(url="https://example.com/image.jpg")** **Image.from\_pil(pil\_image)** |
| **Audio** | **Audio** | Audio data (WAV/MP3 format) | **Audio(bytes=b"")** **Audio(url="https://example.com/audio.mp3")** |
| **Video** | **Video** | Video data (MP4/AVI format) | **Video(bytes=b"")** **Video(url="https://example.com/video.mp4")** |
| **Frame** | **Frame** | Video frame with metadata | **Frame(time=1.5, image=Image(...))** |
| **Concept** | **Concept** | Label with confidence score | **Concept("cat", 0.97)** **Concept(name="dog", value=0.92)** |
| **Region** | **Region** |box and list of Concepts where as box is a list of x1, y1, x2, y2| **Region(box=\[0.7, 0.3, 0.9, 0.7\], \[Concept("cat", 0.7)** **Concept(name="dog", value=0.2)\])** |
| **NameFields** | **dict** | Structured data | **{"scores": \[0.9, 0.1\]}** |

## Supported Data Types: Python Primitive & Generic Types

Clarifai's model framework supports standard Python generic types for flexible data handling. These enable type-safe processing of complex structures while maintaining compatibility with Python's native type system.

### Core Primitive Types

These fundamental types are supported as both inputs and outputs:

| Type | Example Inputs | Example Outputs |
| --- | --- | --- |
| **int** | **42**, **user\_age: int = 30** | **return 100** |
| **float** | **0.95**, **temperature: float = 36.6** | **return 3.14159** |
| **str** | **"Hello"**, **prompt: str = "Generate..."** | **return "success"** |
| **bool** | **True**, **flag: bool = False** | **return is\_valid** |
| **bytes** | **b'raw\_data'**, **file\_bytes: bytes** | **return processed\_bytes** |
| **None** | **None** | **return None** |

#### Primitive Type Usage

```python
class MyModel(ModelClass):

  @ModelClass.method
  def calculate_bmi(
    self,
    height_cm: float,
    weight_kg: float
  ) -> float:
    """Calculate Body Mass Index"""
    return weight_kg / (height_cm/100) ** 2
```

### Generic Container Types

- **List\[T\]**
- **Dict\[K, V\]**
- **Tuple\[T1, T2, ...\]**

#### 1\. List\[T\]

Handles homogeneous collections of any supported type.

```python
class MyModel(ModelClass):

  def load_model(self):
    self.model = ...

  @ModelClass.method
  def predict_images(self, images: List[Image]) -> List[str]:
    """Process multiple images simultaneously"""
    return [self.model(img) for img in images]
```
Client Usage:

```python
images = [
  Image(file_path="img1.jpg"),
  Image(url="https://example.com/img2.png")
]
predictions = model.predict_images(images=images)
```

#### 2\. Dict\[K, V\]

Supports JSON-like structures with string keys.

Example: Configuration Handling

```python
class MyModel(ModelClass):

  @ModelClass.method
  def configure_model(
    self,
    params: Dict[str, float]
  ) -> Dict[str, str]:
    """Update model parameters"""
    self.threshold = params.get('threshold', 0.5)
    return {"status": "success", "new_threshold": str(self.threshold)}
```

#### 3\. Tuple\[T1, T2, ...\]

Handles fixed-size heterogeneous data.

Example: Multi-Output Model

```python
class MyModel(ModelClass):

  @ModelClass.method
  def analyze_document(
    self,
    doc: List[Text]
  ) -> Tuple[List[Text], Dict[str, float]]:
    """Return keywords and sentiment scores"""
    return (doc, {"docs": len(doc)})
```

## NamedFields DataType 

**NamedFields** class enables creation of custom structured data types for handling complex inputs and outputs. This is particularly useful for models requiring multi-field data or producing compound results.

```python
DocumentMetadata = NamedFields(
author=str,
title=str,
page_count=int,
keywords=List[str]
)
class MyModel(ModelClass):

  @ModelClass.method
  def process_document(
    self,
    content: Text,
    metadata: DocumentMetadata
  ) -> NamedFields(
    summary=Text,
    sentiment=float,
    topics=List[str]):
...
```

### Streaming with NamedFields Usage Datatype

```python
class RealTimeAnalytics(ModelClass):
  @ModelClass.method
  def monitor_sensors(
    self,
    sensor_stream: Stream[NamedFields(
    temperature=float,
    pressure=float,
    timestamp=float
  )]) -> Stream[NamedFields(
    status=str,
    anomaly_score=float
  )]:
    for reading in sensor_stream:
      yield self._analyze_reading(reading)
```

#### Client Usage:

```python
sensor_data = [
  {"temperature": 25.6, "pressure": 1013, "timestamp": 1625097600},
  {"temperature": 26.1, "pressure": 1012, "timestamp": 1625097610},
  {"temperature": 27.5, "pressure": 1011, "timestamp": 1625097620}
]

for status in model.monitor_sensors(iter(sensor_data)):
  if status.anomaly_score > 0.9:
    return True
```
