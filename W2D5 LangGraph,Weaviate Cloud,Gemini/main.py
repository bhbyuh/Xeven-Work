from fastapi import FastAPI,UploadFile
import uvicorn
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai 
from weaviate.auth import AuthApiKey
import weaviate
from route_question import route_quest
from retreivel_grader import grade_documents
from generation import result_generation
from halluconation_grader import halluconation_node
from question_rewriter import transform_query
from typing import List
from typing_extensions import TypedDict
from Answer_grader import ans_grader
from langgraph.graph import END, StateGraph, START
from pprint import pprint
from llm_call import gemini_generation
from generation import result_generation
import os
from route_question import route_quest
from langchain_core.runnables.graph import MermaidDrawMethod

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    vector: str
    documents: List[str]

app=FastAPI()

#Gemini API
api_key = os.getenv("Gemini_API")
genai.configure(api_key=api_key)

#Hugging Face key
huggingface_key = os.getenv("HF_Key")
headers = {
    "X-HuggingFace-Api-Key": huggingface_key,
}

#Weaviate Credentials
wcd_url = os.getenv("Wcd_Url") 
wcd_api_key = os.getenv("Wcd_Api_Key")

auth_config = AuthApiKey(api_key=wcd_api_key)

client = weaviate.Client(
    url=wcd_url,
    auth_client_secret=auth_config,
    additional_headers=headers
)

def uninfo_retrieve(state):
    print("---uninfo_retrieve Docs---")
    state["documents"]=[]
    question=state["question"]
    query_result = client.query.get('Uniinfo', ["title", "description"]).with_near_text({
            "concepts": [question]
        }).with_limit(2).do()
    count=0
    print(len(query_result))
    while (count<2):
        state["documents"].append(query_result['data']['Get']['Uniinfo'][count]["description"])
        count+=1

    return {"documents": state["documents"], "question": question}

def NoxVision_retrieve(state):
    print("---GAN_retrieve Docs---")
    state["documents"]=[]
    question=state["question"]
    query_result = client.query.get('NoxVision', ["title", "description"]).with_near_text({
            "concepts": [question]
        }).with_limit(2).do()
    count=0
    print(len(query_result))
    while (count<len(query_result)):
        state["documents"].append(query_result['data']['Get']['NoxVision'][count]["description"])
        count+=1

    return {"documents": state["documents"], "question": question}

def English_retrieve(state):
    print("---English_retrieve Retreive Docs---")
    state["documents"]=[]
    question=state["question"]
    
    query_result = client.query.get('English', ["title", "description"]).with_near_text({
            "concepts": [question]
        }).with_limit(2).do()
    count=0
    
    while (count<len(query_result)):
        state["documents"].append(query_result['data']['Get']['English'][count]["description"])
        count+=1

    return {"documents": state["documents"], "question": question}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """

    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    
    score = halluconation_node(state)
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = ans_grader(state)
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def grade_generation_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---GRADE GENERATION vs QUESTION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = ans_grader(state)
    grade = score.binary_score
    if grade == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("uninfo_retrieve", uninfo_retrieve)
workflow.add_node("NoxVision_retrieve", NoxVision_retrieve)
workflow.add_node("gemini_generation", gemini_generation)  # web search
workflow.add_node("English_retrieve", English_retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("result_generation", result_generation)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_conditional_edges(
    START,
    route_quest,
    {
        "Gemini": "gemini_generation",
        "Uniinfo": "uninfo_retrieve",
        "NoxVision":"NoxVision_retrieve",
        "English":"English_retrieve",
    },)
workflow.add_conditional_edges(
    "gemini_generation",
    grade_generation_and_question,
    {
        "useful": END,
        "not useful": "transform_query",
    },)

workflow.add_edge("uninfo_retrieve", "grade_documents")
workflow.add_edge("NoxVision_retrieve", "grade_documents")
workflow.add_edge("English_retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "result_generation",
    },)
workflow.add_conditional_edges(
    "result_generation",
    grade_generation_v_documents_and_question,
    {
        "not supported": "result_generation",
        "useful": END,
        "not useful": "transform_query",
    },)
workflow.add_conditional_edges(
    "transform_query",
    route_quest,
    {
        "Gemini": "gemini_generation",
        "Uniinfo": "uninfo_retrieve",
        "NoxVision":"NoxVision_retrieve",
        "English":"English_retrieve",
    },)

# Compile
graph_app = workflow.compile()

graph_png_bytes = graph_app.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API
)
with open("graph_output.png", "wb") as f:
    f.write(graph_png_bytes)

#API to upload file
@app.post("/uploadfile")
def get_file(file: UploadFile):

    temp_file_path=f"tempfile_{file.filename}"
    
    with open(temp_file_path,"wb") as temp_file:
        shutil.copyfileobj(file.file,temp_file)
    
    loader=PyMuPDFLoader(temp_file_path)
    docs=loader.load()

    text_splitter=CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    name=file.filename
    name=name.split(".")[0]
    # Create the collection if it doesn't exist
    schema = {
        "class": name,
        "properties": [
            {
                "name": "title",
                "dataType": ["text"]
            },
            {
                "name": "description",
                "dataType": ["text"]
            }
        ],
        "vectorizer": "text2vec-huggingface"  # Let Weaviate handle vectorization
    }

    # Create schema if it doesn't exist
    if not client.schema.exists("File"):
        client.schema.create_class(schema)

    count=0
    with client.batch as batch:
        batch.batch_size=20
        for page in docs:
            split_text=text_splitter.split_text(page.page_content)
            for split in split_text:
                object={
                    "title":file.filename,
                    "description":split
                }
                print(split)
                batch.add_data_object(object,class_name=name)
        
    pprint("Data objects created and embeddings stored in Weaviate!")
    return {"message": "Upload and embedding successful"}

#API to upload file
@app.post("/query")
def query(question: str):
    
    # # Run
    inputs = {
        "question": question,
        "generation":"In progress",
        "documents":[],
        "vector":None
    }
    for output in graph_app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    return value["generation"]

if __name__=="__main__":
    uvicorn.run(app=app,host="127.0.0.1",port=8000)