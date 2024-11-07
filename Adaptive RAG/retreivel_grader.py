from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import os

load_dotenv()
### Retrieval Grader
api_key=os.getenv("Mistral_API")

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def document_relevency(state):

    # LLM with function call
    llm = ChatMistralAI(model="mistral-large-latest", api_key=api_key)
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
    docs=state['doc']
    filtered_docs=[]

    for doc  in docs:
        score=retrieval_grader.invoke({"question": question, "document": docs})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(doc)
        else:
            continue

    
    return {"documents": filtered_docs, "question": question}