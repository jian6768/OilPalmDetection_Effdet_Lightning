# app/model_loader.py

import torch
from PIL import Image
import io
from torchvision import transforms


from app.models.effdet_model import EffDetLModel
from app.config.config import config
from app.config.classes import CLASS_NAMES
from effdet.data.transforms import RandomFlip, RandomResizePad, resolve_fill_color, ImageToNumpy, Compose, ResizePad

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def load_model():
    #Remember to remove lr and batch size subsequently. Let model load itself. 
    model = EffDetLModel.load_from_checkpoint(config.checkpoint_path, bench_task = "predict")
    model.to(DEVICE).eval()
    print("Model loaded.")
    return model


def make_prediction(model:EffDetLModel, image_bytes: bytes):
    img = preprocess_image(image_bytes=image_bytes)

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        pred = model(img).detach().cpu().squeeze(0)
        detections = postprocess_image_output(pred)

    return detections


# Preprocessing
def preprocess_image(image_bytes: bytes):

    #convert image to RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    fill_color = resolve_fill_color("mean", config.IMAGENET_DEFAULT_MEAN)

    image_tfl = [ResizePad(target_size=config.image_size, interpolation="bilinear", fill_color=fill_color), ImageToNumpy(),]

    transform = Compose(image_tfl)

    # Model expects BCHW. Preprocess image converts image into a format sutable for model and moves it to GPU. 
    return torch.from_numpy(transform(img, {})[0]).unsqueeze(0).float().to(DEVICE)



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

