from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key=os.getenv("Gemini_API")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["Uniinfo","NoxVision","English","Other"] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore or Mistral",
    )

def route_quest(state):

    print("---Route Question---")
    # Gemini API setup
    genai.configure(api_key=api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=api_key)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """You are an expert at routing a user question to vectorstores Uniinfo or NoxVision or 
    English or Gemini. The vectorstore1 contains document related to University information. 
    The vectorstore2 contains document related to FYP report ,project named Noxvion whose basic work is to trandform night images to Day image using deep models.
    The vectorstore3 contains English book of grade 5 which contains different chapters.    
    Use the vectorstores for their related questions. Otherwise, use Other"""
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
        elif source.datasource == "NoxVision":
            return "NoxVision"
        elif source.datasource == "English":
            return "English"
        elif source.datasource == "Other":
            return "Other"
        