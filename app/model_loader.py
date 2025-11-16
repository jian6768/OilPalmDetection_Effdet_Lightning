# app/model_loader.py

import torch
from PIL import Image
import io
from torchvision import transforms


from app.models.effdet_model import EffDetLModel
from app.config.config import config
from app.config.classes import CLASS_NAMES

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# model = None


def load_model():
    #Remember to remove lr and batch size subsequently. Let model load itself. 
    model = EffDetLModel.load_from_checkpoint(config.checkpoint_path, bench_task = "predict")
    model.to(DEVICE).eval()
    print("Model loaded.")
    return model


# Preprocessing
def preprocess(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])

    return transform(img).unsqueeze(0).to(DEVICE)


# Postprocessing
def postprocess(pred):
    detections = []
    pred = pred[0].detach().cpu().tolist()

    for (x1, y1, x2, y2, score, cls) in pred:
        if score < config.score_threshold:
            continue

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
            "class_id": int(cls),
            "class_name": CLASS_NAMES.get(int(cls), "unknown")
        })

    return detections
