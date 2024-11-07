### Generate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os

load_dotenv()
api_key=os.getenv("Gemini_API")

genai.configure(api_key=api_key)

def No_record(state):
    print("---Other---")
    response="No Result Available"
    count=state["nodes_count"]-1+1
    state["error"] = state.get("error", 0) - 1

    return {"generation": response,"nodes_count":count,"error":state["error"]}