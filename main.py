from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

sentiment_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",device=-1,torch_dtype="auto")


app = FastAPI()

class TextInput(BaseModel):
    text:str

@app.post("/analyze")
def analyze(input:TextInput):
    result = sentiment_model(input.text)
    return {"message":result}




