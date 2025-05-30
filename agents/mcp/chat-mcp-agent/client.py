import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI
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
        self.client = OpenAI(
            base_url="https://api.clarifai.com/v2/ext/openai/v1",
            api_key=api_key,
        )
        # You can choose different models:
        # self.model_url = "https://clarifai.com/gcp/generate/models/gemini-2_5-pro"
        # self.model_url = "https://clarifai.com/microsoft/text-generation/models/phi-4"
        self.model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4o"

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.
        """
        logger.debug("LLM Request")
        logger.debug("Messages: %s", messages)

        response = self.client.chat.completions.create(
            model=self.model_url,
            messages=messages,
            temperature=0.7,
        )
        response_text = response.choices[0].message.content

        logger.debug("LLM Response")
        logger.debug(response_text)
        return response_text

def format_tool_for_llm(tool: Any) -> str:
    """Format tool information for LLM.

    Args:
        tool: The tool object from MCP server.

    Returns:
        A formatted JSON string describing the tool.
    """
    args = {}
    if "properties" in tool.inputSchema:
        for param_name, param_info in tool.inputSchema["properties"].items():
            args[param_name] = param_info.get('description', 'No description')

    return json.dumps({
        "tool": tool.name,
        "arguments": args
    })

async def get_available_tools(servers: List[Client]) -> List[Any]:
    """Get all available tools from all servers.

    Args:
        servers: List of MCP clients.

    Returns:
        List of available tools.
    """
    all_tools = []
    for server in servers:
        tools = await server.list_tools()
        all_tools.extend(tools)
    return all_tools

async def execute_tool_call(
    servers: List[Client],
    tool_name: str,
    arguments: Dict[str, Any]
) -> Tuple[bool, str]:
    """Execute a tool call on the appropriate server.

    Args:
        servers: List of MCP clients.
        tool_name: Name of the tool to execute.
        arguments: Arguments for the tool.

    Returns:
        Tuple of (success, result_message)
    """
    for server in servers:
        tools = await server.list_tools()
        if any(tool.name == tool_name for tool in tools):
            try:
                logger.debug(
                    "EXECUTING TOOL: %s with args: %s",
                    tool_name,
                    arguments
                )
                result = await server.call_tool_mcp(tool_name, arguments)
                result_data = result.model_dump()['content'][0]

                if isinstance(result_data, dict) and "progress" in result_data:
                    progress = result_data["progress"]
                    total = result_data["total"]
                    percentage = (progress / total) * 100
                    logger.info(
                        "Progress: %d/%d (%.1f%%)",
                        progress,
                        total,
                        percentage
                    )

                return True, f"Tool execution result: {result_data}"
            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}"
                logger.error(error_msg)
                return False, error_msg

    return False, f"No server found with tool: {tool_name}"

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
        self.parser = JsonOutputParser()

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        try:
            # Try to parse as JSON
            try:
                tool_call = self.parser.parse(llm_response)
                if not isinstance(tool_call, dict) or "tool" not in tool_call or "arguments" not in tool_call:
                    return llm_response
            except Exception as e:
                logger.debug("Response is not a tool call: %s", str(e))
                return llm_response

            logger.debug("TOOL CALL: %s", tool_call)
            success, result = await execute_tool_call(
                self.servers,
                tool_call["tool"],
                tool_call["arguments"]
            )
            return result

        except Exception as e:
            logger.error("Error processing LLM response: %s", str(e))
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            logger.info("Starting chat session. Listing tools...")
            all_tools = await get_available_tools(self.servers)
            logger.info("Available tools: %d", len(all_tools))
            
            tools_description = "\n".join([format_tool_for_llm(tool) for tool in all_tools])

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                '{"tool": "tool-name", "arguments": {"argument-name": "value"}}\n\n'
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ["quit", "exit"]:
                        logger.info("Exiting...")
                        break

                    messages.append({"role": "user", "content": user_input})
                    llm_response = self.llm_client.get_response(messages)
                    logger.info("Assistant: %s", llm_response)

                    tool_result = await self.process_llm_response(llm_response)
                    if tool_result != llm_response:
                        messages.append({"role": "assistant", "content": tool_result})
                        final_response = self.llm_client.get_response(messages)
                        logger.info("Final response: %s", final_response)
                        messages[-1]['content'] = f"{llm_response}\n\n{final_response}"
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logger.info("Exiting...")
                    break
                except Exception as e:
                    logger.error("Error in chat loop: %s", str(e))
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