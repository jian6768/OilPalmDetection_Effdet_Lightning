# app/main.py

from fastapi import FastAPI, UploadFile, File
from app.model_loader import load_model, make_prediction
from app.config.config import config

app = FastAPI(
    title=config.api_title,
    version=config.api_version,
    description=config.api_description
)

@app.on_event("startup")
def startup():
    app.model = load_model()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = make_prediction(model = app.model, image_bytes=image_bytes)
    return {
        "num_detections": len(result),
        "detections": result
    }
