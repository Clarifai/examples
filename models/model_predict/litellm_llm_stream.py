import os
import litellm

def main():
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
            # Get streaming completion
            for chunk in litellm.completion(
                model="openai/https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ",
                api_key=os.getenv("CLARIFAI_PAT"),
                api_base="https://api.clarifai.com/v2/ext/openai/v1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
                stream=True
            ):
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 