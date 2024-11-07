### Question Re-writer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import os

load_dotenv()
api_key=os.getenv("Mistral_API")

def quest_rewriter(state):
    # LLM
    llm = ChatMistralAI(model="mistral-large-latest", api_key=api_key)

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