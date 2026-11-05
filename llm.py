from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    seed=42,
    api_key=OPENAI_API_KEY
)

# Select the model to use
llm = openai_llm
