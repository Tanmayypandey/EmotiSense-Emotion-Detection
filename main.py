
import os
import re
import pickle
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ----------------------------------
# 1. Application Setup
# ----------------------------------

app = FastAPI(
    title="EmotiSense - Emotion Detection API",
    description="Detect emotions from text using a trained BiGRU model.",
    version="1.0.0"
)


# ----------------------------------
# 2. Model and Tokenizer Paths
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "Artifacts", "BiGRU_model.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR, "Artifacts", "tokenizer.pkl"
)

MAX_SEQUENCE_LENGTH = 50

# Must match the label order used while training the model.
EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

model = None
tokenizer = None


# ----------------------------------
# 3. Load Model and Tokenizer
# ----------------------------------

@app.on_event("startup")
def load_artifacts():
    global model, tokenizer

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.isfile(TOKENIZER_PATH):
        raise RuntimeError(
            f"Tokenizer file not found: {TOKENIZER_PATH}"
        )

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    print("BiGRU model loaded successfully.")
    print("Tokenizer loaded successfully.")


# ----------------------------------
# 4. Text Preprocessing
# ----------------------------------

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------------------
# 5. Request Schema
# ----------------------------------

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text whose emotion you want to detect."
    )


# ----------------------------------
# 6. Prediction API
# ----------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


@app.post("/predict")
def predict_emotion(request: PredictionRequest):

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model or tokenizer is not loaded."
        )

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter some text."
        )

    try:
        # Preprocess input
        cleaned_text = preprocess_text(request.text)

        if not cleaned_text:
            raise HTTPException(
                status_code=400,
                detail="Please enter meaningful text."
            )

        # Convert text to sequences
        sequence = tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        # Pad sequence to the training length
        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post"
        )

        # Predict probabilities
        predictions = model.predict(
            padded_sequence,
            verbose=0
        )

        probabilities = np.asarray(predictions[0]).flatten()

        if len(probabilities) != len(EMOTION_LABELS):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Model output count does not match "
                    "the configured emotion labels."
                )
            )

        predicted_index = int(np.argmax(probabilities))
        predicted_emotion = EMOTION_LABELS[predicted_index]
        confidence = float(probabilities[predicted_index])

        all_probabilities = {
            label: round(float(prob), 4)
            for label, prob in zip(
                EMOTION_LABELS, probabilities
            )
        }

        return {
            "input_text": request.text,
            "cleaned_text": cleaned_text,
            "predicted_emotion": predicted_emotion,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": all_probabilities
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ----------------------------------
# 7. Frontend
# ----------------------------------

@app.get("/")
def home():
    html_path = os.path.join(
        BASE_DIR, "static", "index.html"
    )

    if os.path.isfile(html_path):
        return FileResponse(html_path)

    return {
        "message": "EmotiSense API is running.",
        "docs": "/docs"
    }


static_dir = os.path.join(BASE_DIR, "static")

if os.path.isdir(static_dir):
    app.mount(
        "/static",
        StaticFiles(directory=static_dir),
        name="static"
    )
