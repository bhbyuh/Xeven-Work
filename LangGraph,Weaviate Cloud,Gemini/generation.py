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
# Prompt
prompt = hub.pull("rlm/rag-prompt")


def result_generation(state):

    print("---Gemini with Context---")
    # Gemini API setup
    genai.configure(api_key=api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=api_key)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    
    docs=state['documents']
    
    docs="\n\n".join(docs)

    question=state["question"]
    # Run
    generation = rag_chain.invoke({"context": docs, "question": question})
    
    return {"documents": state['documents'], "question": question, "generation": generation}