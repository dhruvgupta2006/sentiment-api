from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests

HF_TOKEN= os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"


class TextInput(BaseModel):
    text:str

@app.post("/analyze")
def analyze(input:TextInput):
    headers = {"authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": input.text})
    return {"message": response.json()}




