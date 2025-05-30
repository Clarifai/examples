# Clarifai LLM MCP Agent Example

This example demonstrates how to use Clarifai's LLM with MCP (Model Control Protocol) to create an interactive AI agent that can use tools to solve tasks.

## Overview

The example shows how to:
- Connect to Clarifai's MCP server
- Use Clarifai's LLM with OpenAI-compatible API
- Handle tool calls and responses
- Process user queries with tool assistance

## Prerequisites

- A Clarifai account with API access
- A Clarifai Personal Access Token (PAT)

## Setup

1. Clone the repository and navigate to this directory:
   ```bash
   cd agents/mcp/simple\ llm+mcp
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

4. Set your Clarifai PAT in the environment variable:
   ```
   export CLARIFAI_PAT=your_personal_access_token_here
   ```

## Usage

Run the example:
```bash
python client.py
```

The script will:
1. Connect to the Clarifai MCP server
2. List available tools
3. Process a sample query using the LLM and MCP tools
4. Display the response

## How It Works

1. **MCP Connection**: The script connects to Clarifai's MCP server using the provided transport configuration.

2. **Tool Discovery**: It retrieves the list of available tools from the MCP server and converts them to OpenAI-compatible format.

3. **Query Processing**:
   - Sends the user query to Clarifai's LLM
   - Handles any tool calls requested by the LLM
   - Processes tool responses
   - Returns the final response to the user

4. **Error Handling**: The script includes comprehensive error handling for:
   - Missing environment variables
   - Tool execution errors
   - API communication issues

## Configuration

The script uses several configurable parameters:

- `MODEL`: The Clarifai model to use (default: GPT-4o)
- `MCP_TRANSPORT`: Configuration for the MCP server connection
- `CLARIFAI_PAT`: Your Clarifai Personal Access Token

## Dependencies

- `fastmcp`: For MCP client functionality
- `openai`: For OpenAI-compatible API access

## Notes

- The example uses Clarifai's GPT-4o model, but you can use any Clarifai model that supports OpenAI chat-completions and tool calling.
- Make sure your Clarifai PAT has the necessary permissions to access the MCP server and models.
- The script uses async/await for better performance and resource management.

