# Model Inference Examples

This directory contains examples of how to perform model inference using different SDKs and model types. The examples demonstrate both streaming and non-streaming modes, as well as tool calling capabilities.

## Supported SDKs

1. **Clarifai SDK**
2. **OpenAI Client**
3. **LiteLLM**

## Installation

```bash
# Install Clarifai SDK
pip install clarifai

# Install OpenAI SDK
pip install openai

# Install LiteLLM
pip install litellm
```

## Environment Setup

Set your Clarifai Personal Access Token (PAT) as an environment variable:

```bash
export CLARIFAI_PAT=your_pat_here
```

## Model Types and Examples

### 1. LLMs (e.g., QwQ-32B-AWQ)

#### Clarifai SDK
- `clarifai_llm.py`: Basic inference
- `clarifai_llm_stream.py`: Streaming inference
- `clarifai_llm_tools.py`: Tool calling example

#### OpenAI Client
- `openai_llm.py`: Basic inference
- `openai_llm_stream.py`: Streaming inference
- `openai_llm_tools.py`: Tool calling example

#### LiteLLM SDK
- `litellm_llm.py`: Basic inference
- `litellm_llm_stream.py`: Streaming inference
- `litellm_llm_tools.py`: Tool calling example

### 2. Multimodal Models (e.g., GPT-4_1)

#### Clarifai SDK
- `clarifai_multimodal.py`: Basic inference
- `clarifai_multimodal_stream.py`: Streaming inference
- `clarifai_multimodal_tools.py`: Tool calling example

#### OpenAI Client
- `openai_multimodal.py`: Basic inference
- `openai_multimodal_stream.py`: Streaming inference
- `openai_multimodal_tools.py`: Tool calling example

#### LiteLLM SDK
- `litellm_multimodal.py`: Basic inference
- `litellm_multimodal_stream.py`: Streaming inference
- `litellm_multimodal_tools.py`: Tool calling example

## Usage Examples

### Basic Inference

```python
# Using Clarifai SDK
from clarifai.client import Model

model = Model(url="https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ")
response = model.predict("What is the capital of France?")
print(response)

# Using OpenAI Client
from openai import OpenAI

client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key="YOUR_PAT"
)
response = client.chat.completions.create(
    model="CLARIFAI_MODEL_URL",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response.choices[0].message.content)

# Using LiteLLM
import litellm

response = litellm.completion(
    model="openai/CLARIFAI_MODEL_URL",
    api_key="YOUR_PAT",
    api_base="https://api.clarifai.com/v2/ext/openai/v1",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response.choices[0].message.content)
```

### Streaming Inference

```python
# Using Clarifai SDK
response_stream = model.generate("Tell me a story")
for chunk in response_stream:
    print(chunk, end="", flush=True)

# Using OpenAI Client
stream = client.chat.completions.create(
    model="CLARIFAI_MODEL_URL",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="", flush=True)

# Using LiteLLM
for chunk in litellm.completion(
    model="openai/CLARIFAI_MODEL_URL",
    api_key="YOUR_PAT",
    api_base="https://api.clarifai.com/v2/ext/openai/v1",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Tool Calling

```python
# Example tool definition
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Tokyo, Japan"
                }
            },
            "required": ["location"]
        }
    }
}]

# Using Clarifai SDK
response = model.predict(
    prompt="What's the weather in Tokyo?",
    tools=tools,
    tool_choice='auto'
)

# Using OpenAI Client
response = client.chat.completions.create(
    model="CLARIFAI_MODEL_URL",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

# Using LiteLLM
response = litellm.completion(
    model="openai/CLARIFAI_MODEL_URL",
    api_key="YOUR_PAT",
    api_base="https://api.clarifai.com/v2/ext/openai/v1",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)
```

## Notes

- Always ensure your Clarifai PAT is set in the environment variables
- For multimodal models, provide both text and image inputs as required
- Tool calling support may vary depending on the model capabilities
- Streaming responses are token-by-token and may have different formatting across SDKs
- Error handling and retry logic should be implemented in production environments

## Additional Resources

- [Clarifai Documentation](https://docs.clarifai.com/)
- [Clarifai Model Inference Docs](https://docs.clarifai.com/compute/models/inference/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [LiteLLM Documentation](https://github.com/BerriAI/litellm) 