import os
import litellm

def main():
    # Define tools
    tools = [
        {
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
                        },
                        "units": {
                            "type": "string",
                            "description": "Temperature units, e.g. Celsius or Fahrenheit",
                            "enum": ["Celsius", "Fahrenheit"]
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given ticker symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol e.g. AAPL, GOOGL"
                        }
                    },
                    "required": ["ticker"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    ]
    
    # Example prompts that should trigger tool calls
    prompts = [
        "What's the weather like in Tokyo today?",
        "What is the current stock price of Tesla?",
        "Tell me the temperature in Paris in Fahrenheit."
    ]
    
    # Process each prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        try:
            # Get completion with tools
            response = litellm.completion(
                model="openai/https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ",
                api_key=os.getenv("CLARIFAI_PAT"),
                api_base="https://api.clarifai.com/v2/ext/openai/v1",
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1024
            )
            
            # Print response and tool calls
            print(f"Response: {response.choices[0].message.content}")
            if response.choices[0].message.tool_calls:
                print("\nTool Calls:")
                for tool_call in response.choices[0].message.tool_calls:
                    print(f"- Function: {tool_call.function.name}")
                    print(f"  Arguments: {tool_call.function.arguments}")
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 