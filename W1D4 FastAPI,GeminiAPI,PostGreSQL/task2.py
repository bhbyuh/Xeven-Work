from fastapi import FastAPI
import google.generativeai as genai 
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()

api_key=os.getenv("API_KEY")

genai.configure(api_key=api_key)
app=FastAPI()

@app.post("/question")
def take_ques(question:str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(question)
    return {"Answer":response.text}

if __name__=='__main__':
    uvicorn.run(app=app,host='127.0.0.1',port=8000)