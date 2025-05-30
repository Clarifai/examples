import os
from clarifai.client import Model

def main():
    # Initialize the model
    model = Model(url="https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ")
    
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
            # Get prediction with tools
            response = model.predict(
                prompt=prompt,
                tools=tools,
                tool_choice='auto',
                max_tokens=1024,
                temperature=0.7
            )
            
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 