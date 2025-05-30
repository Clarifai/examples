import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from clarifai.utils.logging import logger

# Load environment variables
load_dotenv("../.env")

class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        """Initialize the LLM client.

        Args:
            api_key: The Clarifai Personal Access Token.
        """
        self.api_key = api_key
        self.client = AsyncOpenAI(
            base_url="https://api.clarifai.com/v2/ext/openai/v1",
            api_key=api_key,
        )
        # You can choose different models:
        # self.model_url = "https://clarifai.com/gcp/generate/models/gemini-2_5-pro"
        # self.model_url = "https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ"
        self.model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4o"

    async def get_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.
            tools: Optional list of tool definitions.

        Returns:
            The final response from the LLM.
        """
        logger.debug("LLM Request")
        logger.debug("Messages: %s", messages)
        if tools:
            logger.debug("Tools: %s", tools)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_url,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            logger.debug("LLM Response")
            logger.debug(assistant_message)

            if not assistant_message:
                logger.error("Empty response message from LLM")
                return "I apologize, but I received an empty response. Could you please try again?"

            # Handle tool calls if present
            while assistant_message.tool_calls:
                logger.debug("Tool calls detected: %s", assistant_message.tool_calls)
                

                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    # Execute tool call
                    result = await self.execute_tool_call(tool_call)
                    # Add tool response to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                    
                # Get response from LLM with tool results
                response = await self.client.chat.completions.create(
                    model=self.model_url,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7,
                )
                assistant_message = response.choices[0].message
                messages.append(assistant_message)

            return assistant_message.content or "I apologize, but I couldn't generate a response. Could you please try again?"

        except Exception as e:
            logger.error("Error getting LLM response: %s", str(e))
            return "I apologize, but I encountered an error. Could you please try again?"

    async def execute_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> Optional[str]:
        """Execute a tool call.

        Args:
            tool_call: The tool call to execute.

        Returns:
            The result of the tool execution, or None if it failed.
        """
        if not tool_call or not tool_call.function:
            logger.error("Invalid tool call received")
            return None

        tool_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error("Error parsing tool arguments: %s", str(e))
            return None

        for server in self.servers:
            try:
                tools = await server.list_tools()
                if any(tool.name == tool_name for tool in tools):
                    logger.debug(
                        "EXECUTING TOOL: %s with args: %s",
                        tool_name,
                        arguments
                    )
                    result = await server.call_tool(
                        tool_name,
                        arguments=arguments
                    )
                    if not result or not result[0]:
                        return None
                    return str(result[0].text)
            except Exception as e:
                logger.error("Error executing tool %s: %s", tool_name, str(e))
                continue

        logger.error("No server found with tool: %s", tool_name)
        return None

def format_tool_for_openai(tool: Any) -> Dict[str, Any]:
    """Format tool information for OpenAI's tool format.

    Args:
        tool: The tool object from MCP server.

    Returns:
        A dictionary in OpenAI's tool format.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
    }

async def get_available_tools(servers: List[Client]) -> List[Any]:
    """Get all available tools from all servers.

    Args:
        servers: List of MCP clients.

    Returns:
        List of available tools.
    """
    all_tools = []
    for server in servers:
        try:
            tools = await server.list_tools()
            all_tools.extend(tools)
        except Exception as e:
            logger.error("Error getting tools from server: %s", str(e))
    return all_tools

class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Client], llm_client: LLMClient) -> None:
        """Initialize the chat session.

        Args:
            servers: List of MCP clients.
            llm_client: The LLM client instance.
        """
        self.servers = servers
        self.llm_client = llm_client
        llm_client.servers = servers  # Make servers available to LLMClient

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            logger.info("Starting chat session. Listing tools...")
            all_tools = await get_available_tools(self.servers)
            if not all_tools:
                logger.error("No tools available")
                return
                
            logger.info("Available tools: %d", len(all_tools))
            
            # Convert tools to OpenAI format
            openai_tools = [format_tool_for_openai(tool) for tool in all_tools]

            system_message = (
                "You are a helpful assistant with access to various tools. "
                "Use these tools when appropriate to help users with their tasks. "
                "When using tools, provide clear and concise responses based on the results. "
                "If no tool is needed, respond directly to the user's query."
            )

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ["quit", "exit"]:
                        logger.info("Exiting...")
                        break

                    if not user_input:
                        continue

                    messages.append({"role": "user", "content": user_input})
                    response = await self.llm_client.get_response(messages, openai_tools)
                    logger.info("Assistant: %s", response)
                    messages.append({"role": "assistant", "content": response})

                except KeyboardInterrupt:
                    logger.info("Exiting...")
                    break
                except Exception as e:
                    logger.error("Error in chat loop: %s", str(e))
                    messages.append({"role": "assistant", "content": "I apologize, but I encountered an error. Could you please try again?"})
                    continue

        except Exception as e:
            logger.error("Fatal error in chat session: %s", str(e))
            raise

async def main() -> None:
    """Initialize and run the chat session."""
    # Get Clarifai PAT from environment
    pat = os.getenv("CLARIFAI_PAT")
    if not pat:
        raise ValueError("CLARIFAI_PAT environment variable is not set")

    # Initialize MCP transport
    transport = StreamableHttpTransport(
        url="https://api.clarifai.com/v2/ext/mcp/v1/users/clarifai-demos/apps/demo-runner-models/models/zeiler-mcp-example",
        headers={"Authorization": f"Bearer {pat}"},
    )

    # Initialize clients and start chat session
    llm_client = LLMClient(pat)
    async with Client(transport) as client:
        chat_session = ChatSession([client], llm_client)
        await chat_session.start()

if __name__ == "__main__":
    asyncio.run(main()) 