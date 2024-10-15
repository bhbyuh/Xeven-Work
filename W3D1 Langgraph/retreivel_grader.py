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
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def grade_documents(state):
    
    print("---Docs Retreival Grader---")
    # Gemini API setup
    genai.configure(api_key=api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=api_key)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    
    question=state["question"]
    docs=state['documents']
    filtered_docs=[]

    for doc  in docs:
        score=retrieval_grader.invoke({"question": question, "document": docs})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(doc)
        else:
            continue
    
    state['documents']=filtered_docs
    count=state['nodes_count']
    count=count+1
    if(state["error"]==7):
        state["generation"]="No record Available"
    return {"documents": filtered_docs,"generation":state["generation"], "question": question,"nodes_count":count,"error":state["error"]}