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
    
    # Example prompts and images
    examples = [
        {
            "prompt": "Describe what you see in this image in detail.",
            "image_url": "https://samples.clarifai.com/cat1.jpeg"
        },
        {
            "prompt": "Tell me a story about what's happening in this scene.",
            "image_url": "https://samples.clarifai.com/metro-north.jpg"
        },
        {
            "prompt": "Analyze this image and describe what you observe, step by step.",
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
            
            # Get streaming completion
            stream = client.chat.completions.create(
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
                temperature=0.7,
                max_tokens=1024,
                stream=True
            )
            
            # Print streamed response
            print("Response (streaming): ", end="", flush=True)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 