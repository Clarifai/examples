import os
from clarifai.client import Model

def main():
    # Initialize the model
    model = Model(url="https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ")
    
    # Example prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence."
    ]
    
    # Process each prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        try:
            # Get prediction
            response = model.predict(prompt)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 