import os
import asyncio
from clarifai.client.model import Model

async def main():
    # Initialize the model with the appropriate URL
    model = Model(url="https://clarifai.com/qwen/qwenLM/models/QwQ-32B-AWQ")
    
    generate_response = await model.async_generate(
        prompt="what is the value of pi?",
        max_tokens=100
    )
    
   
     # Use async for to iterate over the async generator
    async for response in await generate_response:
        print(response)   


#Run the main function and print the result
if __name__ == "__main__":
    if not os.getenv("CLARIFAI_PAT"):
        print("Please set your CLARIFAI_PAT environment variable")
        exit(1)
    
    # Run the main function
    # asyncio.run is used to run the async main function
    # and print the result
    asyncio.run(main())
