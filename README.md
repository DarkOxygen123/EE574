# Brain Tumor Image Analysis

A comprehensive framework for analyzing brain tumor images using various preprocessing techniques and deep learning models.

## Project Structure

```
.
├── data/                      # Data directory
│   └── Br35H-Mask-RCNN/      # Dataset
├── src/                      # Source code
│   ├── config.py            # Configuration
│   ├── data/               # Data handling
│   ├── models/             # Model definitions
│   ├── preprocessing/      # Preprocessing techniques
│   └── visualization/      # Visualization tools
├── logs/                    # Training logs
├── checkpoints/            # Model checkpoints
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Setup

1. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```python
from src.preprocessing.transforms import get_preprocessing_transforms
from src.visualization.visualize import visualize_preprocessing_effects

# Get preprocessing transforms
transforms = get_preprocessing_transforms(config)

# Visualize effects
visualize_preprocessing_effects(image, transforms, save_path='preprocessing_effects.png')
```

2. Data Augmentation:
```python
from src.preprocessing.transforms import get_augmentation_transforms
from src.visualization.visualize import visualize_augmentation_effects

# Get augmentation transforms
transforms = get_augmentation_transforms(config)

# Visualize effects
visualize_augmentation_effects(image, mask, transforms, save_path='augmentation_effects.png')
```

3. Model Training:
```python
from src.models.base import ClassificationModel, SegmentationModel

# Initialize model
model = ClassificationModel(config)  # or SegmentationModel

# Training loop
for epoch in range(config.num_epochs):
    # Training code here
    pass
```

## Features

- Comprehensive preprocessing techniques
- Advanced data augmentation
- Visualization tools for preprocessing and augmentation effects
- Base classes for classification and segmentation models
- Metrics tracking and visualization
- Configurable pipeline

## License

MIT 