import os
import base64
import requests
from openai import OpenAI

def get_image_base64(image_url):
    """Download image and convert to base64."""
    response = requests.get(image_url)
    return base64.b64encode(response.content).decode('utf-8')

def main():
    # Initialize the client
    client = OpenAI(
        base_url="https://api.clarifai.com/v2/ext/openai/v1",
        api_key=os.getenv("CLARIFAI_PAT")
    )
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_image_details",
                "description": "Get detailed information about an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "aspect_ratio": {
                            "type": "string",
                            "description": "The aspect ratio of the image"
                        },
                        "dominant_colors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The dominant colors in the image"
                        }
                    },
                    "required": ["aspect_ratio", "dominant_colors"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_location_info",
                "description": "Get information about a location shown in an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location shown in the image"
                        },
                        "landmarks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Notable landmarks in the image"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    ]
    
    # Example prompts and images
    examples = [
        {
            "prompt": "Analyze this image and tell me about its composition and colors.",
            "image_url": "https://samples.clarifai.com/cat1.jpeg"
        },
        {
            "prompt": "What location is shown in this image and what landmarks can you identify?",
            "image_url": "https://samples.clarifai.com/metro-north.jpg"
        },
        {
            "prompt": "Describe this image and analyze its visual elements.",
            "image_url": "https://samples.clarifai.com/dog1.jpeg"
        }
    ]
    
    # Process each example
    for example in examples:
        print(f"\nPrompt: {example['prompt']}")
        print(f"Image: {example['image_url']}")
        print("-" * 50)
        
        try:
            # Get image as base64
            image_base64 = get_image_base64(example["image_url"])
            
            # Get completion with tools
            response = client.chat.completions.create(
                model="https://clarifai.com/openai/chat-completion/models/gpt-4_1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": example["prompt"]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
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
            print(f"Error processing example: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 