# app/model_loader.py

import torch
from PIL import Image
import io
from torchvision import transforms


from app.models.effdet_model import EffDetLModel
from app.config.config import config
from app.config.classes import CLASS_NAMES

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

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

def postprocess_images_outputs(preds):
    detections_list = []
    for i in range(len(preds)):
        detections_list.append(postprocess_image_output(preds[i]))
    return detections_list

def postprocess_image_output(pred):

    #outputs list of dictionaries. 
    detections = []
    pred = pred.detach().cpu().tolist()

    #Iterate through all detections within a particular prediction. 
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
