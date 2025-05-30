import os
from openai import OpenAI

def main():
    # Initialize the client
    client = OpenAI(
        base_url="https://api.clarifai.com/v2/ext/openai/v1",
        api_key=os.getenv("CLARIFAI_PAT")
    )
    
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
            stream = client.chat.completions.create(
                model="https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ",
                messages=[{"role": "user", "content": prompt}],
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
            print(f"Error processing prompt: {str(e)}")

if __name__ == "__main__":
    # Ensure PAT is set
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
        
    main() 