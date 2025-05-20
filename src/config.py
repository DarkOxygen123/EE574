from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DataConfig(BaseModel):
    data_dir: Path = Path("data")
    train_dir: Path = Path("data/Br35H-Mask-RCNN/TRAIN")
    val_dir: Path = Path("data/Br35H-Mask-RCNN/VAL")
    test_dir: Path = Path("data/Br35H-Mask-RCNN/TEST")
    annotations_file: Path = Path("data/Br35H-Mask-RCNN/annotations_all.json")
    image_size: tuple = (256, 256)
    batch_size: int = 32
    num_workers: int = 4

class PreprocessingConfig(BaseModel):
    normalization: bool = True
    histogram_equalization: bool = True
    gaussian_blur: bool = True
    median_blur: bool = True
    bilateral_filter: bool = True
    adaptive_threshold: bool = True
    otsu_threshold: bool = True
    canny_edge: bool = True
    sobel_edge: bool = True
    laplacian_edge: bool = True

class AugmentationConfig(BaseModel):
    rotation_range: int = 30
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = True
    fill_mode: str = "nearest"
    brightness_range: tuple = (0.8, 1.2)
    contrast_range: tuple = (0.8, 1.2)

class ModelConfig(BaseModel):
    classification_models: List[str] = ["resnet50", "efficientnet_b0", "densenet121"]
    segmentation_models: List[str] = ["unet", "fpn", "deeplabv3"]
    pretrained: bool = True
    num_classes: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3

class Config(BaseModel):
    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    model: ModelConfig = ModelConfig()
    seed: int = 42
    device: str = "cuda"
    num_epochs: int = 100
    early_stopping_patience: int = 10
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints") 