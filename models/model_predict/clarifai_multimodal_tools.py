import os
from clarifai.client import Model
from clarifai.runners.utils.data_types import Image

def main():
    # Initialize the model
    model = Model(url="https://clarifai.com/openai/chat-completion/models/gpt-4_1")
    
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
            # Get prediction with tools
            response = model.predict(
                prompt=example["prompt"],
                image=Image(url=example["image_url"]),
                tools=tools,
                tool_choice='auto',
                max_tokens=1024,
                temperature=0.7
            )
            
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 