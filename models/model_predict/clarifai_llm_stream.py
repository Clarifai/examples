import os
from clarifai.client import Model

def main():
    # Initialize the model
    model = Model(url="https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ")
    
    # Example prompts
    prompts = [
        "Tell me a short story about a robot learning to paint.",
        "Explain how neural networks work, step by step.",
        "Write a haiku about the future of AI."
    ]
    
    # Process each prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        try:
            # Get streaming response
            response_stream = model.generate(prompt)
            
            # Print streamed response
            print("Response (streaming): ", end="", flush=True)
            for chunk in response_stream:
                print(chunk, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 