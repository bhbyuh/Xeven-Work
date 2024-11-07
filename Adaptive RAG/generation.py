### Generate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import os

load_dotenv()

api_key=os.getenv("Mistral_API")
# Prompt
prompt = hub.pull("rlm/rag-prompt")


def result_generation(state):
    # LLM
    llm = ChatMistralAI(model="mistral-large-latest", api_key=api_key)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    
    docs=state['doc']
    
    docs="\n\n".join(docs)

    question=state["question"]
    # Run
    generation = rag_chain.invoke({"context": docs, "question": question})
    print(generation)
    return {"documents": state['doc'], "question": question, "generation": generation}