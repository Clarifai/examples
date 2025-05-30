import os
from clarifai.client import Model
from clarifai.runners.utils.data_types import Image

def main():
    # Initialize the model
    model = Model(url="https://clarifai.com/openai/chat-completion/models/gpt-4_1")
    
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
            # Get streaming response
            response_stream = model.generate(
                prompt=example["prompt"],
                image=Image(url=example["image_url"])
            )
            
            # Print streamed response
            print("Response (streaming): ", end="", flush=True)
            for chunk in response_stream:
                print(chunk, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 