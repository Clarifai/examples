![image](https://github.com/user-attachments/assets/b22c9807-f5e7-49eb-b00d-598e400781af)
# Marketing Agent CrewAI

This project implements an AI marketing team using CrewAI to generate and analyze marketing strategies using Clarifai's LLM integration.

## Project Structure

```
marketing_agent_crewAI/
├── requirements.txt              # Project dependencies
├── readme.md                     # readme
├── src/
│   ├── run_marketing_agent.ipynb # Jupyter notebook entry point
│   ├── marketing_agent/         # Main agent module
│   │   ├── __init__.py
│   │   ├── crew.py             # CrewAI setup and agent definitions
│   │   ├── main.py             # Main execution logic
│   │   └── config/             # Configuration files
│   │       ├── agents.yaml     # Agent configurations
│   │       └── tasks.yaml      # Task configurations
```

## How It Works

### LLM Configuration

The project uses Clarifai's OpenAI compatibility API for LLM capabilities. The LLM is initialized in `crew.py`:

```python
from crewai import LLM

llm = LLM(
    model="openai/openai/chat-completion/models/gpt-4o",
    base_url="https://api.clarifai.com/v2/ext/openai/v1"
)
```
### Clarifai Model name convention
CrewAI uses litellm underhood for routing the requests to inference provider, since we are using OpenAI compatible API endpoint we have to start model name with prefix `openai`, and follow the clarifai model URL parameters.
`user_id/app_id/models/model_id` .

**Clarifai model URL for gpt-4o** - https://clarifai.com/openai/chat-completion/models/gpt-4o  
Similarly you can find other Top LLM models links from the [Clarifai community](https://clarifai.com/explore) page.

### Marketing Agent System

The system consists of three specialized agents working together:

1. **Lead Market Analyst**
   - Researches market trends and competition
   - Uses SerperDev and Web Scraping tools
   ```python
   @agent
   def lead_market_analyst(self) -> Agent:
       return Agent(
           config=self.agents_config["lead_market_analyst"],
           tools=[SerperDevTool(), ScrapeWebsiteTool()],
           llm=llm
       )
   ```

2. **Chief Marketing Strategist**
   - Develops marketing strategies
   - Analyzes research data
   ```python
   @agent
   def chief_marketing_strategist(self) -> Agent:
       return Agent(
           config=self.agents_config["chief_marketing_strategist"],
           tools=[SerperDevTool(), ScrapeWebsiteTool()],
           llm=llm
       )
   ```

3. **Creative Content Creator**
   - Generates marketing content
   - Creates campaign ideas
   ```python
   @agent
   def creative_content_creator(self) -> Agent:
       return Agent(
           config=self.agents_config["creative_content_creator"],
           llm=llm
       )
   ```

### Task Flow

Tasks are executed in sequence:
1. Market Research
2. Project Understanding
3. Marketing Strategy Development
4. Campaign Idea Generation
5. Content Creation

## Setup and Usage

1. Set up environment variables:
   ```bash
   export CLARIFAI_PAT="your_pat_here"
   export SERPER_API_KEY="your_serper_key_here"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the marketing agent:
  
  - Use the `run_marketing_agent.ipynb` to run the agent in notebook, also you can convert it into script and run it in shell.

## Configuration

The project uses YAML files for configuration:

1. `agents.yaml`: Defines agent roles and characteristics
   - Role descriptions
   - Goals
   - Backstories

2. `tasks.yaml`: Defines task parameters
   - Task descriptions
   - Expected outputs
   - Dependencies

## Tools Integration

The project utilizes external tools:
- `SerperDevTool`: For market research
- `ScrapeWebsiteTool`: For web content analysis

## Process Flow

The marketing crew operates in a sequential process:
```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential
    )
```

## State Management

The project uses Pydantic models for structured data:
- `MarketStrategy`: For strategy outputs
- `CampaignIdea`: For campaign concepts
- Custom BaseModels for other structured data