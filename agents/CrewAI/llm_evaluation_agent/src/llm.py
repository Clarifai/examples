from crewai import LLM
import os

# Your Clarifai PAT will be used in alias of OPENAI_API_KEY
clarifai_pat = os.getenv("CLARIFAI_PAT")
if clarifai_pat is None:
    print("Error: CLARIFAI_PAT environment variable not set.")
    exit()
else:
    os.environ["OPENAI_API_KEY"] = clarifai_pat 
    
def get_llm(model_name: str = "openai/openai/chat-completion/models/gpt-4o") -> LLM:
    """
    Returns an instance of the LLM configured for Clarifai.
    """
    return LLM(
        model= model_name,
        base_url="https://api.clarifai.com/v2/ext/openai/v1",
        api_key=clarifai_pat  # Set your Clarifai API key here
    )
    
