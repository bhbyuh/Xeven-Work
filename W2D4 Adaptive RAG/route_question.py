from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import os

load_dotenv()

api_key=os.getenv("Mistral_API")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "Mistral"] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore or Mistral",
    )

def router_node(state):
    llm = ChatMistralAI(model="mistral-large-latest", api_key=api_key)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """You are an expert at routing a user question to a vectorstore or Gemini.
    The vectorstore contains documents related to University information. Use the vectorstore for 
    their related questions. Otherwise, use Mistral"""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    
    question_router = route_prompt | structured_llm_router
    
    question = state["question"]

    source=question_router.invoke({"question":state["question"]})

    if source.datasource == "vectorstore":
        return "vectorstore"
    elif source.datasource == "Mistral":
        return "Mistral"