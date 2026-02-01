import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing_extensions import Annotated

app = FastAPI()

# Load pipeline ONCE
with open("sentiment_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

class SentimentRequest(BaseModel):
    text: Annotated[str, Field(..., description="The text to analyze for sentiment")]

class SentimentResponse(BaseModel):
    sentiment: str

@app.post("/predict-sentiment", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    review_text = request.text  

    prediction = model.predict([review_text])[0]

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    return {"sentiment": label_map[prediction]}
