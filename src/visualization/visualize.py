import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from albumentations import Compose

def visualize_preprocessing_effects(
    image: np.ndarray,
    transforms: Compose,
    save_path: Optional[Path] = None
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Apply each transform individually
    for i, transform in enumerate(transforms.transforms, 1):
        if i >= len(axes):
            break
            
        augmented = transform(image=image)
        axes[i].imshow(augmented['image'])
        axes[i].set_title(transform.__class__.__name__)
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_augmentation_effects(
    image: np.ndarray,
    mask: np.ndarray,
    transforms: Compose,
    n_samples: int = 4,
    save_path: Optional[Path] = None
) -> None:
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 5 * n_samples))
    
    for i in range(n_samples):
        augmented = transforms(image=image, mask=mask)
        
        axes[i, 0].imshow(augmented['image'])
        axes[i, 0].set_title(f'Augmented Image {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(augmented['mask'], cmap='gray')
        axes[i, 1].set_title(f'Augmented Mask {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_batch(
    images: np.ndarray,
    masks: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    n_samples: int = 4,
    save_path: Optional[Path] = None
) -> None:
    fig, axes = plt.subplots(n_samples, 3 if predictions is not None else 2, figsize=(15, 5 * n_samples))
    
    for i in range(n_samples):
        if i >= len(images):
            break
            
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        if predictions is not None:
            axes[i, 2].imshow(predictions[i], cmap='gray')
            axes[i, 2].set_title(f'Prediction {i+1}')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None
) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_title(metric_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close() 