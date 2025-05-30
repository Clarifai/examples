![image](https://github.com/user-attachments/assets/b22c9807-f5e7-49eb-b00d-598e400781af)

# LLM Evaluation Agent

This project implements an AI agent system using CrewAI to generate and evaluate Shakespearean-style X (Twitter) posts using Clarifai's LLM inference.

## Project Structure

```
llm_evaluation_agent/
├── requirements.txt        # Project dependencies
├── readme.md               # readme
├── run_evaluation_agent.py # Entry point script
├── src/
│   ├── llm.py            # LLM configuration and initialization
│   ├── main.py           # Main flow implementation
│   ├── crews/            # Different CrewAI crews
│   │   ├── shakespeare_crew/    # Crew for generating Shakespeare-style posts
│   │   └── x_post_review_crew/  # Crew for reviewing and validating posts
│   └── tools/            # Custom tools for the crews
│       └── CharacterCounterTool.py
```

## How It Works

### LLM Configuration

The project uses Clarifai's OpenaAI compatibility API for LLM calling. The LLM is initialized in `src/llm.py`:

### Clarifai Model name convention
CrewAI uses litellm underhood for routing the requests to inference provider, since we are using OpenAI compatible API endpoint we have to start model name with prefix `openai`, and follow the clarifai model URL parameters.
`user_id/app_id/models/model_id` .

**Clarifai model URL for gpt-4o** - https://clarifai.com/openai/chat-completion/models/gpt-4o  

Similarly you can find other Top LLM models links from the [Clarifai community](https://clarifai.com/explore) page.

```python
from crewai import LLM
import os

clarifai_pat = os.getenv("CLARIFAI_PAT")
# LLM configuration using Clarifai
def get_llm(model_name: str = "openai/openai/chat-completion/models/gpt-4o") -> LLM:
    return LLM(
        model=model_name,
        base_url="https://api.clarifai.com/v2/ext/openai/v1",
        api_key=clarifai_pat
    )
```

### Clarifai-CrewAI System

To call clarifai LLM models for the agent it is important to pass the `llm=CLARIFAI_LLM` class in order to enable agents to access the models.

1. **Shakespeare Crew** (`crews/shakespeare_crew/`)
   - Generates X posts in Shakespearean style
   - Uses the configured LLM for creative text generation
   - Example usage in crews:

   ```python
   from src.llm import get_llm
   
   class ShakespeareanXPostCrew:
       def crew(self):
           agent = Agent(
               llm=get_llm(),
               # ... agent configuration
           )
   ```

2. **Review Crew** (`crews/x_post_review_crew/`)
   - Evaluates generated posts for quality and authenticity
   - Provides feedback for improvement
   - Uses the same LLM configuration for evaluation

### Flow Control

The main flow (`src/main.py`) orchestrates the process:
1. Generates a Shakespearean X post
2. Reviews the post for quality
3. Either accepts the post or requests revisions
4. Saves successful posts to `x_post.txt`

## Setup and Usage

1. Set up environment variables:
   ```bash
   export CLARIFAI_PAT="your_pat_here"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the evaluation agent:
   ```bash
   python src/main.py
   ```

## Tools

The project includes custom tools in the `tools/` directory:
- `CharacterCounterTool.py`: Validates post length and character constraints

## State Management

The flow uses Pydantic models for state management:
- Tracks post content, validation status, and retry attempts
- Maintains feedback for improvement iterations

## Configuration

Each crew has its own configuration in respective `config/` directories:
- Shakespeare crew: Style and topic configurations
- Review crew: Validation criteria and constraints