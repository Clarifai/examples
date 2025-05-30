![image](https://github.com/user-attachments/assets/b22c9807-f5e7-49eb-b00d-598e400781af)

# Clarifai - Google ADK 

This guide demonstrates how to use Clarifai's OpenAI-compatible endpoint with Google's Agent Development Kit (ADK) framework. Clarifai provides a wide range of powerful LLM models that can be accessed through an OpenAI-compatible API interface.

## Prerequisites

- Python 3.8+
- A Clarifai account with API access
- Clarifai Personal Access Token (PAT)
- Google ADK installed

## Setup

1. Install required dependencies:
```bash
pip instal google-adk clarifai openai litellm
```

2. Set your Clarifai PAT as an environment variable:
```bash
export CLARIFAI_PAT=your_personal_access_token_here
```

## Configuration

Configure OpenAI client to use Clarifai's endpoint:

```python
import openai
import os

openai.api_key = os.getenv("CLARIFAI_PAT")
openai.api_base = "https://api.clarifai.com/v2/ext/openai/v1"
```

## Using Clarifai Models

Clarifai models can be accessed using LiteLLM in the below model URL format:
Starts with prefix `openai` - 

`openai/{user_id}/{app_id}/models/{model_id}`

**Example using Gemini-2.5 Flash**:\
Google ADK utilises Litellm to call models internally, since it supports the openAI compatible endpoints directly, we are creating LLM model class using it.

```python
from google.adk.models.lite_llm import LiteLlm

# Configure the chat model with Clarifai endpoint

MODEL = LiteLlm(model="openai/deepseek-ai/deepseek-chat/models/DeepSeek-R1-Distill-Qwen-7B",
                      base_url="https://api.clarifai.com/v2/ext/openai/v1")

```

## Available Models

You can explore available models on the [Clarifai Community](https://clarifai.com/explore) platform. Some popular models include:

- GPT-4o: `openai/chat-completion/models/gpt-4o`
- Gemini 2.5 Flash: `gcp/generate/models/gemini-2_5-flash`
- Gemma-3-4b-it: `gcp/generate/models/gemma-3-4b-it`
- Phi4: `microsoft/text-generation/models/phi-4`

## Example Agent Implementation

Here's a simple example of implementing a Google ADK agent using Clarifai's endpoint:

```python
from google.adk import Agent

agent = Agent(
    model=MODEL,
    name='reviser_agent',
    instruction=prompt.REVISER_PROMPT,
    after_model_callback=_remove_end_of_edit_mark,
)
```

## Resources

- [Clarifai Documentation](https://docs.clarifai.com/)
- [Clarifai Python SDK](https://docs.clarifai.com/resources/api-references/python)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [LiteLLM OpenAI API Compatibility Guide](https://docs.litellm.ai/docs/providers/openai_compatible#usage---completion)

## Notes

- Ensure your Clarifai PAT has the necessary permissions
- Monitor your API usage through the Clarifai dashboard
- Consider implementing rate limiting for production use
- Test your implementation with smaller models before scaling up
