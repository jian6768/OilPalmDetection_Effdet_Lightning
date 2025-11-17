# app/config/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Tuple


#Using pydantic config allows modification of settings without touching code. It also avoids manual type conversion. 
class AppConfig(BaseSettings):
    # API metadata
    api_title: str = "Oil Palm Object Detection API"
    api_version: str = "1.0.0"
    api_description: str = "EffDet model to detect oil palm fruit maturity."

    # Model settings
    model_architecture: str = "tf_efficientdet_d0"
    num_classes: int = 5
    checkpoint_path: str = "weights/epoch=14-step=5460.ckpt"
    bench_task: str = "predict"
    score_threshold: float = 0.3

    # Preprocessing settings
    image_size: Tuple[int, int] = (512, 512)
    IMAGENET_DEFAULT_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD: List[float] = [0.229, 0.224, 0.225]

    #API key. 
    rf_key: str | None = None

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8"
    )


# Global instance of config
config = AppConfig()
