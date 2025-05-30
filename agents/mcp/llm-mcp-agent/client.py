import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Configuration
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
if not CLARIFAI_PAT:
    raise ValueError("CLARIFAI_PAT environment variable is not set")

# Initialize clients
openai_client = AsyncOpenAI(
    api_key=CLARIFAI_PAT,
    base_url="https://api.clarifai.com/v2/ext/openai/v1"
)

# Any Clarifai model that supports OpenAI chat-completions and tool calling
MODEL = "https://clarifai.com/openai/chat-completion/models/gpt-4o"

# MCP transport configuration
MCP_TRANSPORT = StreamableHttpTransport(
    url="https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai-demos/apps/demo-runner-models/models/zeiler-mcp-example",
    headers={"Authorization": f"Bearer {CLARIFAI_PAT}"},
)

async def get_mcp_tools(client: Client) -> List[Dict[str, Any]]:
    """Get available tools from the MCP server in OpenAI format.

    Args:
        client: The MCP client instance.

    Returns:
        A list of tools in OpenAI format.
    """
    tools_result = await client.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools_result
    ]

async def process_query(
    query: str,
    mcp_client: Client,
    openai_client: AsyncOpenAI,
    model: str
) -> str:
    """Process a query using OpenAI and available MCP tools.

    Args:
        query: The user query.
        mcp_client: The MCP client instance.
        openai_client: The OpenAI client instance.
        model: The model to use for chat completions.

    Returns:
        The response from OpenAI.
    """
    # Get available tools
    tools = await get_mcp_tools(mcp_client)

    # Initial OpenAI API call
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
    )

    # Get assistant's response
    assistant_message = response.choices[0].message
    messages = [
        {"role": "user", "content": query},
        assistant_message,
    ]

    # Handle tool calls if present
    while assistant_message.tool_calls:
        print(f"Processing tool calls: {assistant_message.tool_calls}")
        
        for tool_call in assistant_message.tool_calls:
            try:
                # Execute tool call
                result = await mcp_client.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # Add tool response to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result[0].text,
                })
            except Exception as e:
                print(f"Error executing tool {tool_call.function.name}: {str(e)}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: {str(e)}",
                })

        # Get response from OpenAI with tool results
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

    return assistant_message.content

async def main() -> None:
    """Main entry point for the client."""
    async with Client(MCP_TRANSPORT) as mcp_client:
        # List available tools
        tools_result = await mcp_client.list_tools()
        print("\nConnected to Clarifai MCP server with tools:")
        for tool in tools_result:
            print(f"  - {tool.name}: {tool.description}")

        # Process a query
        query = "what is 2+9?"
        print(f"\nQuery: {query}")

        try:
            response = await process_query(
                query=query,
                mcp_client=mcp_client,
                openai_client=openai_client,
                model=MODEL
            )
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())