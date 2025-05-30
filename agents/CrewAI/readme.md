![image](https://github.com/user-attachments/assets/b22c9807-f5e7-49eb-b00d-598e400781af)

# 🚀 Clarifai CrewAI Agents

This repository demonstrates how to build an AI-powered agent crew using [CrewAI](https://github.com/joaomdmoura/crewAI) and [Clarifai's LLM](https://www.clarifai.com/) capabilities. The project showcases automation of various tasks through collaborative AI agents.

---

## 🔧 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Clarifai-Crewai.git
cd Clarifai-Crewai
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```
export CLARIFAI_PAT="your_clarifai_pat_here"
```

## 🧠 Key Components

### 1. Clarifai LLM Integration
The project uses a custom ClarifaiLLM class for model interactions:
```
from crewai import LLM

llm = LLM(
    model="openai/openai/chat-completion/models/gpt-4o",
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key="CLARIFAI_PAT"   
)

```
#### Clarifai Model name convention
CrewAI uses litellm underhood for routing the requests to inference provider, since we are using OpenAI compatible API endpoint we have to start model name with prefix `openai`, and follow the clarifai model URL parameters.
`user_id/app_id/models/model_id` .

**Clarifai model URL for gpt-4o** - https://clarifai.com/openai/chat-completion/models/gpt-4o  
Similarly you can find other Top LLM models links from the [Clarifai community](https://clarifai.com/explore) page.
### 2. Agent Configuration
Agents are defined in agents.yaml and initialized in crew.py:

For example -
```
@agent
def lead_market_analyst(self) -> Agent:
    return Agent(
        config=self.agents_config["lead_market_analyst"],
        tools=[SerperDevTool(), ScrapeWebsiteTool()],
        verbose=True,
        memory=False,
        llm=llm,
        max_iter=3
    )
```

### 3. Task Configuration
Tasks are defined in tasks.yaml and implemented in crew.py:
```
@task
def marketing_strategy_task(self) -> Task:
    return Task(
        config=self.tasks_config["marketing_strategy_task"],
        agent=self.chief_marketing_strategist(),
        output_pydantic=MarketStrategy
    )
```

## ▶️ Usage
Update the configuration files:

agents.yaml: Define agent roles and goals

tasks.yaml: Define task descriptions and expected outputs

Run the marketing crew:
```
from marketing_agent.main import run

inputs = {
    "customer_domain": "your-domain.com",
    "project_description": "Your project description"
}

run()
```

## 👥 Agents and Their Roles
**Lead Market Analyst**: Conducts market research and competitor analysis

**Chief Marketing Strategist**: Develops marketing strategies

**Creative Content Creator**: Creates campaign content and copy

**Chief Creative Director**: Oversees and approves creative work

## 🙌 Acknowledgments
- [CrewAI](https://docs.crewai.com/introduction)

- [Clarifai](https://docs.clarifai.com/compute/models/inference/api)

- [SerperDev](https://serper.dev/)

For more information, refer to the component documentation in the source code.
