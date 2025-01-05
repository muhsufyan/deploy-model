from click import option
from fastapi import FastAPI
from fastapi.params import Body
from pydantic import BaseModel

# load model
from transformers import pipeline
model_name = "practice-ac/results"
classifier = pipeline("text-classification", model=model_name, tokenizer="bert-base-uncased")

def convert(pred):
  if pred == "LABEL_0":
    return "negative"
  else:
    return "positive"

app = FastAPI(title="Dokumentasi untuk api")

class Text(BaseModel):
    text: str

@app.post("/predict", tags=["inference model"], summary=["predict"])
async def prediction(capData: Text):
    pred = classifier(capData.text)
    return {
        "predict": convert(pred[0].get("label")),
        "data": [pred]
  }