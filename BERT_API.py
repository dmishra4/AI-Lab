import torch

from fastapi import FastAPI

from pydantic import BaseModel

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

# =========================
# LOAD SAVED MODEL
# =========================

print("\n===== LOADING SAVED MODEL =====")

model = BertForSequenceClassification.from_pretrained(
    "LLMScripts/saved_bert_model"
)

tokenizer = BertTokenizer.from_pretrained(
    "LLMScripts/saved_bert_model"
)

print("\nMODEL LOADED SUCCESSFULLY!")

# =========================
# FASTAPI APP
# =========================

app = FastAPI()

# =========================
# LABEL MAPPING
# =========================

label_map = {

    0: "Error",

    1: "Info",

    2: "Warning"
}

# =========================
# REQUEST MODEL
# =========================

class LogRequest(BaseModel):

    log_message: str

# =========================
# HOME API
# =========================

@app.get("/")

def home():

    return {

        "message": "BERT Log Classification API Running"
    }

# =========================
# PREDICTION API
# =========================

@app.post("/predict")

def predict_log(request: LogRequest):

    text = request.log_message

    # Encode text
    inputs = tokenizer(

        text,

        return_tensors="pt",

        padding=True,

        truncation=True,

        max_length=20
    )

    # Prediction
    with torch.no_grad():

        outputs = model(**inputs)

    predicted_label = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    predicted_category = label_map[predicted_label]

    return {

        "log_message": text,

        "predicted_category": predicted_category
    }


##  Command to run this file ::   uvicorn LLMScripts.BERT_API:app --reload

# Raw Dataset
#     ↓
# Cleaning
#     ↓
# Label Encoding
#     ↓
# Train/Test Split
#     ↓
# BERT Tokenization
#     ↓
# BERT Encoding
#     ↓
# PyTorch Dataset
#     ↓
# Load Pretrained BERT
#     ↓
# Train BERT Model
#     ↓
# Evaluate Accuracy
#     ↓
# Save Model
#     ↓
# Load Saved Model
#     ↓
# FastAPI Deployment
#     ↓
# Swagger UI Testing
#     ↓
# Real-time Prediction