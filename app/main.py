from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import predict_emotion


# Initialize FastAPI
app = FastAPI(
    title="Emotion Classification API",
    description="An API that predicts emotions from text using a Bi-GRU model.",
    version="1.0.0"
)


# Request structure
class EmotionRequest(BaseModel):
    text: str


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Emotion Classification API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: EmotionRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    result = predict_emotion(text)

    return result