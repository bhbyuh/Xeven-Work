### Hallucination Grader
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os

load_dotenv()
### Retrieval Grader
api_key=os.getenv("Gemini_API")

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def halluconation_node(state):
    print("---halluconation checkker---")

    # Gemini API setup
    genai.configure(api_key=api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=api_key)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    
    hallucination_grader = hallucination_prompt | structured_llm_grader
    docs=state['documents']
    
    docs="\n\n".join(docs)
    generation=state["generation"]
    result=hallucination_grader.invoke({"documents": docs, "generation": generation})
    
    return result