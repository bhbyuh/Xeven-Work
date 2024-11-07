from fastapi import FastAPI
from fastapi import File, UploadFile
import google.generativeai as genai 
import uvicorn
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import weaviate
from weaviate.classes.init import Auth
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from weaviate.classes.query import MetadataQuery

app = FastAPI()

wcd_url = ""
wcd_api_key =""

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=Auth.api_key(wcd_api_key), 
)

class_schema = {
    "class": "File",
    "properties": [
        {
            "name": "title",
            "dataType": ["string"],
        },
        {
            "name": "content",
            "dataType": ["text"],
        },
        {
            "name": "embedding",
            "dataType": ["number[]"],  # Store embedding as an array of numbers (vector)
        }
    ],
    "vectorizer": "none"  # Disable automatic vectorization as you're providing the embeddings manually
}

# Create the class in Weaviate if it does not already exist
try:
    client.collections.create_from_dict(class_schema)
    print("Class 'File' created successfully.")
except weaviate.exceptions.WeaviateBaseError as e:
    print(f"Error creating class: {e}")

embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

load_dotenv()

api_key=os.getenv("API_KEY")

genai.configure(api_key=api_key)
app=FastAPI()

@app.post("/question")
def take_ques(question:str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(question)
    return {"Answer":response.text}

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    print(docs[0])
    chunk_size=500
    chunk_overlap=50

    splitter=CharacterTextSplitter(
    separator='\n',
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
    )
       
    # Process each document
    for page in docs:
        page_splits = splitter.split_text(page.page_content)
        for split in page_splits:
            # Generate embeddings for each chunk
            embedding = embeddings_model.embed_documents([split])[0]

            # Data object to send to Weaviate
            data_obj = {
                "title": temp_file_path, 
                "content": split,
                "embedding": embedding  
            }
            # Add the data object and embedding to Weaviate
            collection = client.collections.get("File")
            collection.data.insert(data_obj)
    
    print("Data objects created and embeddings stored in Weaviate!")
    return {"message": "Upload and embedding successful"}

@app.get("/Query")
def query(question:str):

    collection = client.collections.get("File")
    question_embedding = embeddings_model.embed_documents([question])[0]

    # collection = client.collections.get("File")

    # for item in collection.iterator(
    #     include_vector=True  # If using named vectors, you can specify ones to include e.g. ['title', 'body'], or True to include all
    # ):
    #     print(item.properties)
    #     print(item.vector)
    response = collection.query.near_vector(
        near_vector=question_embedding,
        limit=2,
        return_metadata=MetadataQuery(distance=True)
    )
    
    for o in response.objects:
        print(o.properties)
        print(o.metadata.distance)
        print(o.vector)

if __name__=='__main__':
    uvicorn.run(app=app,host='127.0.0.1',port=8000)
