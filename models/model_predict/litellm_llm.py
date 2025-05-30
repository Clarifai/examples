import os
import litellm

def main():
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
            # Get completion
            response = litellm.completion(
                model="openai/https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ",
                api_key=os.getenv("CLARIFAI_PAT"),
                api_base="https://api.clarifai.com/v2/ext/openai/v1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            
            print(f"Response: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 