import os
import asyncio
from clarifai.client.model import Model

async def main():
    #Initialize the model with the appropriate URL
    model = Model(url="https://clarifai.com/openai/chat-completion/models/o4-mini")
    
    model_prediction = await model.async_predict(prompt= "what is the value of pi?",
                                 max_tokens=500)
    return await model_prediction

# Run the main function and print the result
if __name__ == "__main__":
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
    print(asyncio.run(main()))