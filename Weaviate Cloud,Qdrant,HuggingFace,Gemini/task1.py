from fastapi import FastAPI, File, UploadFile
import google.generativeai as genai 
import uvicorn
import os
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.docstore.document import Document
from typing import List
from qdrant_client import QdrantClient

# Initialize Hugging Face embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

app = FastAPI()

load_dotenv()

api_key = ""
url = ""
qdrant_api_key = ""

genai.configure(api_key=api_key)

qdrant_client = QdrantClient(
    url=url,
    api_key=qdrant_api_key,
    prefer_grpc=True 
)

@app.post("/question")
def take_ques(question: str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(question)
    return {"Answer": response.text}

@app.post("/uploadfile")
async def create_upload_file(files: List[UploadFile] = File(...)):
    all_documents = [] 
    for file in files:
        
        temp_file_path = f"temp_{file.filename}"
        
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()

        
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        
        for page in docs:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                document = Document(
                    page_content=text,
                    metadata={"file_name": page.metadata['source']}
                )
                all_documents.append(document)

    
    qdrant = Qdrant.from_documents(
        all_documents, 
        embeddings_model,
        url=url,
        prefer_grpc=True,
        collection_name="files",
        api_key="",
    )

    return {"message": f"{len(files)} files uploaded successfully."}

@app.get("/Query")
def query(question: str):
   
    num_chunks = 2 
    retriever = Qdrant(
        client=qdrant_client,
        collection_name="files",
        embeddings=embeddings_model
    ).as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    
    search_results = retriever.get_relevant_documents(question)
    
    relevant_texts = " ".join([doc.page_content for doc in search_results])
    
    print(relevant_texts)

    combined_text = f"Question: {question}\nContext: {relevant_texts}"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(combined_text)
    
    return {
        "Question": question,
        "Answer": response.text,
        "Meta Data":search_results[0].metadata['file_name']
    }

if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=8000)
