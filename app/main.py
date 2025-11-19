# app/main.py

from fastapi import FastAPI, UploadFile, File
from app.model_loader import load_model, make_prediction, get_display_detection
from app.config.config import config
from PIL import Image
import io

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
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size # (width, height)
    
    # Run Inference (This part depends on your specific `make_prediction` wrapper)
    # Ensure 'raw_results' are the boxes directly from the model (0-512 scale)
    raw_results = make_prediction(model=app.model, image_bytes=image_bytes)
    
    # Resizes bbox back to original image size. 
    processed_results = get_display_detection(raw_results, original_size)
    
    return {
        "num_detections": len(processed_results),
        "detections": processed_results
    }