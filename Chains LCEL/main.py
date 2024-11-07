from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from fastapi import FastAPI,UploadFile
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import weaviate
from weaviate.auth import AuthApiKey
import uvicorn
from operator import itemgetter

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

# Initialize model
chat_llm = ChatOpenAI(
    model='gpt-4o-mini',
    openai_api_key=api_key,
    temperature=0
)

app=FastAPI()

#Hugging Face key
huggingface_key = ""
headers = {
    "X-HuggingFace-Api-Key": huggingface_key,
}

#Weaviate Credentials
wcd_url = ""
wcd_api_key =""

auth_config = AuthApiKey(api_key=wcd_api_key)

client = weaviate.Client(
    url=wcd_url,
    auth_client_secret=auth_config,
    additional_headers=headers
)

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
        "vectorizer": "text2vec-huggingface",  # Let Weaviate handle vectorization
        "model":"BAAI/bge-small-en-v1.5"
    }

    # Create schema if it doesn't exist
    if not client.schema.exists("File"):
        client.schema.create_class(schema)

    with client.batch as batch:
        print("File Loading")
        batch.batch_size=20
        for page in docs:
            split_text=text_splitter.split_text(page.page_content)
            for split in split_text:
                object={
                    "title":file.filename,
                    "description":split
                }
                batch.add_data_object(object,class_name=name)

    return {"Message":"Data SUccessfully uploaded"}

@app.get("/QA")
def questans(question:str):
    # Create a ChatPromptTemplate
    prompt_str = '''Give answer of question
        \n
        question:{question}'''
    
    query_fetcher= itemgetter("question")

    prompt = ChatPromptTemplate.from_template(prompt_str)

    setup={"question":query_fetcher}
    chain = (setup | prompt | chat_llm)

    response = chain.invoke({"question":question}).content
    return {"Response":response}

@app.get("/QA_retreival")
def questansretreiv(question:str,collection_name:str):
    # Create a ChatPromptTemplate
    documents=[]
    count=0
    query_result = client.query.get(collection_name, ["title", "description"]).with_near_text({
            "concepts": [question]
        }).with_limit(2).do()
    
    while count < len(query_result['data']['Get'][collection_name]):
        documents.append(query_result['data']['Get'][collection_name][count]["description"])
        count += 1

    Context=" ".join(documents)

    prompt_str = '''Based on the provided context, answer the following question:
        
    Question: {question}

    Context: {Context}

    Please provide a clear and concise response.'''

    
    prompt = ChatPromptTemplate.from_template(prompt_str)

    query_fetcher= itemgetter("question")
    context_fetcher=itemgetter("Context")

    prompt = ChatPromptTemplate.from_template(prompt_str)

    setup={"question":query_fetcher,"Context":context_fetcher}
    chain = (setup | prompt | chat_llm)

    response = chain.invoke({"question":question,"Context":Context}).content
    return {"Response":response}

@app.get("/conversation")
def conversation(question:str):
    # Create a ChatPromptTemplate
    history = [
    "User: Can you explain the difference between these two millionaire and billionaire?",
    "Assistant: Certainly! The primary difference between a millionaire and a billionaire lies in their net worth, which is a measure of their total assets minus liabilities.\n\n1. **Millionaire**: A millionaire is an individual whose net worth is at least one million units of currency (e.g., dollars, euros, etc.). This can include cash, investments, real estate, and other assets. Millionaires are often seen as wealthy, but their financial resources are limited compared to billionaires.\n\n2. **Billionaire**: A billionaire, on the other hand, has a net worth of at least one billion units of currency. This means that billionaires have at least 1,000 times more wealth than millionaires. Billionaires typically have significant investments, ownership stakes in large companies, and other high-value assets that contribute to their vast wealth.",
    ]
    history=" ".join(history)
    
    prompt_str = '''Answer the following question based on conversation history:

    Question: {question}

    History: {History}

    Please provide a detailed and relevant response.'''


    prompt = ChatPromptTemplate.from_template(prompt_str)

    query_fetcher= itemgetter("question")
    history_fetcher=itemgetter("History")

    prompt = ChatPromptTemplate.from_template(prompt_str)

    setup={"question":query_fetcher,"History":history_fetcher}
    chain = (setup | prompt | chat_llm)

    response = chain.invoke({"question":question,"History":history}).content
    return {"Response":response}

@app.get("/conversation_retreival")
def conversationretreiv(question:str,collection_name:str):
    # Create a ChatPromptTemplate
    documents=[]
    count=0
    query_result = client.query.get(collection_name, ["title", "description"]).with_near_text({
            "concepts": [question]
        }).with_limit(2).do()
    
    while count < len(query_result['data']['Get'][collection_name]):
        documents.append(query_result['data']['Get'][collection_name][count]["description"])
        count += 1

    Context=" ".join(documents)

    history = [
    "User: Can you explain the difference between these two options?",
    "Assistant: Sure, let me walk you through it step by step.",
    "User: That makes sense, but can it also handle this use case?",
    "Assistant: Yes, it's designed to be flexible enough for that.",
]
    history=" ".join(history)

    prompt_str = '''Answer the following question based on the provided context and conversation history:

    Question: {question}

    Context: {Context}

    History: {History}

    Please provide a detailed and relevant response.'''

    prompt = ChatPromptTemplate.from_template(prompt_str)

    query_fetcher= itemgetter("question")
    context_fetcher=itemgetter("Context")
    history_fetcher=itemgetter("History")

    prompt = ChatPromptTemplate.from_template(prompt_str)

    setup={"question":query_fetcher,"Context":context_fetcher,"History":history_fetcher}
    chain = (setup | prompt | chat_llm)

    response = chain.invoke({"question":question,"Context":Context,"History":history}).content
    return {"Response":response}

if __name__=="__main__":
    uvicorn.run(host="127.0.0.1",port=8000,app=app)
