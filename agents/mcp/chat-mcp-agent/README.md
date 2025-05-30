# Clarifai Chat MCP Example

This example demonstrates how to create an interactive AI agent using Clarifai's LLM (Language Learning Model) with MCP (Model Control Protocol) tools. The agent can understand user queries, use appropriate tools to solve tasks, and provide natural language responses.

## Features

- Integration with Clarifai's LLM models (GPT-4o, Gemini, Phi-4, etc.)
- MCP tool execution and management
- Natural language processing of tool responses
- Robust error handling and logging
- Interactive chat interface
- Support for multiple MCP servers
- Clean and modular tool execution architecture
- Automatic handling of chained tool calls
- Proper message history management

## Prerequisites

- A Clarifai account with API access
- A Clarifai Personal Access Token (PAT)

## Setup

1. Clone the repository and navigate to this directory:
   ```bash
   cd agents/mcp/chat-mcp-agent
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the parent directory with your Clarifai PAT:
   ```
   CLARIFAI_PAT=your_personal_access_token_here
   ```

## Usage

Run the example:
```bash
python client.py
```

The script will:
1. Connect to the Clarifai MCP server
2. List available tools
3. Start an interactive chat session
4. Process user queries using the LLM and available tools

### Example Interaction

```
You: What is 2+9?
Assistant: The sum of 2 and 9 is 11.

You: What is 1+2+3?
Assistant: The sum of 1, 2, and 3 is 6.

You: exit
Exiting...
```

## Architecture

The example consists of two main components:

1. **LLMClient**: Manages communication with Clarifai's LLM models
   - Supports multiple model options (GPT-4o, Gemini, Phi-4)
   - Handles message formatting and response parsing
   - Manages tool calls and responses
   - Configurable temperature and other parameters
   - Automatic handling of chained tool calls
   - Proper message history management

2. **ChatSession**: Orchestrates the interaction between user, LLM, and tools
   - Manages conversation history
   - Processes user input
   - Formats responses for natural conversation
   - Handles session state and cleanup

## Key Features

### Tool Handling
- Automatic detection and execution of tool calls
- Support for chained tool calls (e.g., multiple calculations)
- Proper message history management for tool calls and responses
- Robust error handling for tool execution

### Message Flow
- Clean message history management
- Proper formatting of tool calls and responses
- Natural language responses after tool execution
- Support for complex multi-step operations


## Configuration

The script can be configured by modifying:

- Model selection in `LLMClient.__init__`
- MCP server URL in `main()`
- System message in `ChatSession.start()`
- Temperature and other LLM parameters in `LLMClient.get_response()`
- Tool execution parameters in `execute_tool_call()`

## Notes

- The example uses Clarifai's GPT-4o model by default, but you can easily switch to other models
- Make sure your Clarifai PAT has the necessary permissions
- The script uses async/await for better performance
- Tool responses are processed to provide natural language output