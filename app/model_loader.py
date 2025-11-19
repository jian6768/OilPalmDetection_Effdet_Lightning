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


def get_display_detection(detections, original_size, model_img_size=config.image_size[0]):
    """
    Rescales boxes back to original image coordinates. Model requires 512x512 as input. 
    
    """
    orig_w, orig_h = original_size
    
    # 1. Calculate the scale factor used during preprocessing
    # (Ensure this logic matches your training preprocessing!)
    scale = model_img_size / max(orig_w, orig_h)
    
    # 2. Calculate Padding
    # This assumes your preprocessing aligns to TOP-LEFT.
    pad_x = 0 
    pad_y = 0 
    
    final_detections = []
    
    for det in detections:
        
        if isinstance(det, dict):
            box = det['bbox'] # Access by key instead of slice [0:4]
            score = det['score']
            class_id = det['class_id']
            # Preserve existing class name if available
            class_name = det.get('class_name', 'Overripe') 
        else:
            # Fallback if 'det' is a raw list/tensor [x1, y1, x2, y2, score, class]
            box = det[0:4]
            score = det[4]
            class_id = int(det[5])
            class_name = "Error"
        # --- FIX END ---

        # 4. Reverse the padding first
        x1 = box[0] - pad_x
        y1 = box[1] - pad_y
        x2 = box[2] - pad_x
        y2 = box[3] - pad_y
        
        # 5. Reverse the scaling
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale
        
        # 6. Clip to original image and format
        final_detections.append({
            "bbox": [
                max(0, float(x1)), 
                max(0, float(y1)), 
                min(orig_w, float(x2)), 
                min(orig_h, float(y2))
            ],
            "score": float(score),
            "class_id": class_id,
            "class_name": class_name
        })
        
    return final_detections




