from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key="AIzaSyBwJ5c_ioGevPfxl-wmiciSFueOzBHypKU"

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["Uniinfo","uservectorstore","both"] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore or Mistral",
    )

def route_quest(state):

    print("---Route Question---")
    # Gemini API setup
    genai.configure(api_key=api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=api_key)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    user_doc=state["user_doc"]

    system = f"""You are an expert in routing user questions to the appropriate vectorstores: 
    Uniinfo (containing documents related to university information) and uservectorstore 
    (containing documents about {user_doc}). If the question explicitly refers to one store, 
    route the query accordingly. Otherwise, route the query to both vectorstores to retrieve 
    documents from each."""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    
    question_router = route_prompt | structured_llm_router

    source=question_router.invoke({"question":state["question"]})

    
    state["vector"]=source.datasource
    
    if(state["error"]==7):
        return "Other"
    else:
        if source.datasource == "Uniinfo":
            return "Uniinfo"
        elif source.datasource == "uservectorstore":
            return "uservectorstore"
        elif source.datasource == "both":
            return "both"