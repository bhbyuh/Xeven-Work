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

def gemini_generation(state):
    print("---Gemini---")
    question = state["question"]

    model=genai.GenerativeModel("gemini-1.5-flash")
    response=model.generate_content(question)

    return { "question": question, "generation": response.text}