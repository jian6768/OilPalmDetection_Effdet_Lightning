#import necessary libraries
from effdet.data import DetectionDatset, create_loader
from effdet.data.dataset_config import CocoCfg
from effdet.data.parsers import CocoParserCfg, create_parser
from pathlib import Path
import torch
import lightning as L
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from effdet import create_model
from effdet.data.transforms import RandomFlip, RandomResizePad, resolve_fill_color, ImageToNumpy, Compose, ResizePad

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

image_size = (512,512)
batch_size = 8

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def load_model() -> torch.nn.Module:

    #create and load model. 
    loaded_model = EffDetLModel.load_from_checkpoint("lightning_logs/version_67/checkpoints/epoch=14-step=5460.ckpt", model_architecture = model_architecture, num_classes = num_classes, bench_task = "predict", lr = 0.0002, batch_size = 8)
    loaded_model.to("cuda")
    loaded_model.eval()                            
