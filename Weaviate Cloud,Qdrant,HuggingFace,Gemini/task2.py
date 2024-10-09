from fastapi import FastAPI, File, UploadFile
import weaviate
from weaviate.auth import AuthApiKey
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
import uvicorn
import google.generativeai as genai 

app = FastAPI()

api_key = "AIzaSyCH0BB7Kv925qUJFX22WH14ghjB0d1cHDs"
genai.configure(api_key=api_key)

wcd_url = "https://dmudugysreujpgwgemi3g.c0.us-west3.gcp.weaviate.cloud"  # Update your Weaviate instance URL
wcd_api_key = "0TvCJRXIZUEeIBQObKBC9Q9YnaLf0MhDdYDz"           # Weaviate API Key

huggingface_key = "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"

auth_config = AuthApiKey(api_key=wcd_api_key)
headers = {
    "X-HuggingFace-Api-Key": huggingface_key,
}

client = weaviate.Client(
    url=wcd_url,
    auth_client_secret=auth_config,
    additional_headers=headers
)



@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"

    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)

    # Load PDF content using PyMuPDFLoader
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()

    chunk_size = 500
    chunk_overlap = 50

    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # Create the collection if it doesn't exist
    schema = {
        "class": "File",
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

    # Process each document and batch the data into Weaviate
    with client.batch as batch:
        batch.batch_size = 20  # Set a smaller batch size to avoid issues
        for page in docs:
            page_splits = splitter.split_text(page.page_content)
            for split in page_splits:
                data_obj = {
                    "title": temp_file_path,  # Use file name as title
                    "description": split,    # The split text chunk
                }
                # Add object to Weaviate; Weaviate will automatically handle the vectorization
                batch.add_data_object(
                    data_obj,
                    class_name="File"
                )
    
    print("Data objects created and embeddings stored in Weaviate!")
    return {"message": "Upload and embedding successful"}

@app.get("/query")
def query(question: str):
    # Use Weaviate's query method for retrieving data
    query_result = client.query.get("File", ["title", "description"]).with_near_text({
        "concepts": [question]
    }).with_limit(2).do()

    
    first=query_result['data']['Get']['File'][0]["description"]
    second=query_result['data']['Get']['File'][1]["description"]
    meta_data=query_result['data']['Get']['File'][0]["title"]
    relevant_texts = " ".join([first,second])

    print(second)

    combined_text = f"Question: {question}\nContext: {relevant_texts}"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(combined_text)
    
    return {
        "Question": question,
        "Answer": response.text,
        "Meta Data":meta_data
    }

if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=8000)
