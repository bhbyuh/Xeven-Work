from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import os
from dotenv import load_dotenv
from fastapi import FastAPI,File,UploadFile
import shutil
from langgraph.graph import END, StateGraph, START
from langchain_core.runnables.graph import MermaidDrawMethod
import uvicorn
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone,ServerlessSpec
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

openai_key=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini",api_key=openai_key,temperature=0)
embeddings=OpenAIEmbeddings(model="text-embedding-ada-002",api_key=openai_key)

history = """
User: Where was Einstein born?
Assistant: He was born in Ulm, Germany.
User: What was his most famous theory?
Assistant: The theory of relativity.
"""

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

def standalone(question):
    if(history==None):
        return question
    else:
        # Template for transforming questions into standalone form
        template = """
        Given the conversation history and a follow-up question, your task is to rewrite the follow-up question to be standalone (self-contained). Ensure it contains all the necessary context for understanding.

        Conversation:
        {conversation_history}

        Follow-up Question:
        {follow_up_question}

        Standalone Question:
        """
        ouput_parser=StrOutputParser()
        template=ChatPromptTemplate.from_template(template)

        chain= template | llm | ouput_parser

        response=chain.invoke({"conversation_history":history,"follow_up_question":question})

        return response

class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource: Literal["vectorstore", "greeting"] = Field(
        ...,
        description="Based on the user question, choose to route it to 'vectorstore', return 'greeting' for greetings, or 'None' if it doesn't fit any of these."
    )

def routequestion(state):

    structured_llm_router = llm.with_structured_output(RouteQuery)

    question=state["question"]

    question=standalone(question)

    template = """
        You are an expert at routing questions to the appropriate response. Based on the question, decide whether to:
        - Return 'vector store' if the query is not greeting.
        - Return 'greeting' if the query is a greeting.

        Question: {question}
        """

    route_prompt = ChatPromptTemplate.from_template(template)
    
    question_router = route_prompt | structured_llm_router

    source=question_router.invoke({"question":state["question"]})
    if source.datasource == "vectorstore":
        return "vectorstore"
    elif source.datasource == "greeting":
        return "greeting"
    
def retreive_vectorstore(state):
    question=state["question"]
    
    indexes=["test",state["collection_name"]]
    question=standalone(question)
    
    # Initialize Pinecone and the Index
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    docs=[]
    for a in indexes:
        index_name=a

        index=pc.Index(index_name)

        vector_store = PineconeVectorStore(index,embedding=embeddings)

        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=llm)

        MultiQuery = retriever_from_llm.llm_chain.invoke({"question":question})
        
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vector_store.as_retriever()
        )
        docss=[]
        for a in MultiQuery:
            response = index.query(
            vector=embeddings.embed_query(a),
            top_k=2,
            include_values=True,
            include_metadata=True,
            )
            docss.append(response['matches'][0]["metadata"]["text"])
        for a in docss:
            compressed_docs = compression_retriever.invoke(a)
            docs.extend(compressed_docs)
    for i in range(len(docs)):
        docs[i]=docs[i].page_content
    print(docs)
    return {"documents":docs}

def genration(state):
    if (len(state["documents"])!=0):
        Prompt = '''Based only on the provided context, answer the following question concisely and accurately.

        Question:
        {question}

        Context:
        {context}

        If the answer is not found within the context, respond with "Information not available in the provided context."'''
        prompt=ChatPromptTemplate.from_template(Prompt)
        output_parser = StrOutputParser()
        chain=prompt | llm | output_parser
        docs=state['documents']
        docs=" ".join(docs)
        response=chain.invoke({"question":state['question'],"context":docs})

        return {"generation":response}
    else:
        return {"generation":"No response available"}

def greetings(state):
    Prompt = '''Respond appropriately to the following greeting:

    Greeting:
    {question}

    Provide a warm and concise response.'''
    prompt=ChatPromptTemplate.from_template(Prompt)
    output_parser = StrOutputParser()
    chain=prompt | llm | output_parser
    print(state['question'])
    response=chain.invoke({"question":state['question']})
    print(response)
    return {"generation":response}

def noresponse(state):
    return {"generation":"No answer available"}

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    collection_name: str

workflow = StateGraph(GraphState)

workflow.add_node("retreive_vectorstore", retreive_vectorstore)  # grade documents
workflow.add_node("genration", genration)  # generatae
workflow.add_node("greetings", greetings)  # generatae
workflow.add_node("noresponse", noresponse)  # generatae

workflow.add_conditional_edges(
    START,
    routequestion,
    {
        "vectorstore": "retreive_vectorstore",
        "greeting": "greetings",
        "None": "noresponse",
    },)

workflow.add_edge("retreive_vectorstore", "genration")
workflow.add_edge("genration",END)
workflow.add_edge("greetings", END)
workflow.add_edge("noresponse", END)

# Compile
graph_app = workflow.compile()

app=FastAPI()

@app.post("/uploadfile")
def uploadfile(file: UploadFile):

    temp_file_path=f"tempfile_{file.filename}"
    
    with open(temp_file_path,"wb") as temp_file:
        shutil.copyfileobj(file.file,temp_file)
    
    loader=PyMuPDFLoader(temp_file_path)
    docs=loader.load()

    name=file.filename
    index_name=name.split(".")[0]
    index_name=index_name.lower()

    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    dimension=1536
    pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    index = pc.Index(index_name)

    vectors=[]
    count=0
    for page in docs:
        split_text=text_splitter.split_text(page.page_content)
        for split in split_text:
            print(split)
            vector=embeddings.embed_query(split)
            vectors.append({"id": str(count), "values":vector,"metadata":{"text": split}})
            count+=1
            print(count)

    index.upsert(vectors)

    return {"Message":"Data SUccessfully uploaded"}

@app.get("/question")
def query(question:str,collection_name:str):
    # # Run
    inputs = {
        "question": question,
        "generation":"In progress",
        "documents":[],
        "collection_name":collection_name
    }
    for output in graph_app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        print("\n---\n")
    
    # Final generation
    return {"Result":value["generation"]}

if __name__=="__main__":
    uvicorn.run(app=app,host="127.0.0.1",port=8000)
