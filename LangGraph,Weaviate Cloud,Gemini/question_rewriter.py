### Question Re-writer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os

load_dotenv()
api_key=os.getenv("Gemini_API")

def transform_query(state):
    print("---Query Transformation---")

    # Gemini API setup
    genai.configure(api_key=api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=api_key)

    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    question=state["question"]
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    
    documents = state["documents"]
    better_question=question_rewriter.invoke({"question": question})

    return {"documents": documents, "question": better_question}