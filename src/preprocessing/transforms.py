import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from albumentations import Compose, RandomRotate90, Flip, Transpose, OneOf, CLAHE, RandomBrightnessContrast, RandomGamma, HueSaturationValue, RGBShift, RandomBrightness, RandomContrast, MotionBlur, MedianBlur, GaussianBlur, GaussNoise, IAAAdditiveGaussianNoise, GaussianNoise, OpticalDistortion, GridDistortion, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomBrightnessContrast, OneOf, ToGray, ChannelShuffle, ChannelDropout, IAAAffine, RandomResizedCrop, RandomSizedCrop, RandomScale

def get_preprocessing_transforms(config: Dict) -> Compose:
    transforms = []
    
    if config['normalization']:
        transforms.append(CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)))
    
    if config['gaussian_blur']:
        transforms.append(GaussianBlur(blur_limit=(3, 7)))
    
    if config['median_blur']:
        transforms.append(MedianBlur(blur_limit=7))
    
    if config['bilateral_filter']:
        transforms.append(OneOf([
            GaussianBlur(blur_limit=3),
            MedianBlur(blur_limit=3),
        ], p=0.5))
    
    if config['adaptive_threshold']:
        transforms.append(OneOf([
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.5))
    
    if config['otsu_threshold']:
        transforms.append(OneOf([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            RandomGamma(gamma_limit=(80, 120)),
        ], p=0.5))
    
    if config['canny_edge']:
        transforms.append(OneOf([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            RandomGamma(gamma_limit=(80, 120)),
        ], p=0.5))
    
    if config['sobel_edge']:
        transforms.append(OneOf([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            RandomGamma(gamma_limit=(80, 120)),
        ], p=0.5))
    
    if config['laplacian_edge']:
        transforms.append(OneOf([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            RandomGamma(gamma_limit=(80, 120)),
        ], p=0.5))
    
    return Compose(transforms)

def get_augmentation_transforms(config: Dict) -> Compose:
    transforms = [
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        OneOf([
            RandomBrightnessContrast(
                brightness_limit=config['brightness_range'],
                contrast_limit=config['contrast_range']
            ),
            RandomGamma(gamma_limit=(80, 120)),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.5),
        OneOf([
            MotionBlur(blur_limit=7),
            MedianBlur(blur_limit=7),
            GaussianBlur(blur_limit=7),
            GaussNoise(var_limit=(10.0, 50.0)),
        ], p=0.5),
        OneOf([
            OpticalDistortion(distort_limit=1.0),
            GridDistortion(num_steps=5, distort_limit=0.3),
            IAAPiecewiseAffine(scale=(0.01, 0.05)),
        ], p=0.5),
        OneOf([
            IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
            IAAEmboss(alpha=(0.2, 0.5), strength=(0.5, 1.0)),
        ], p=0.5),
        RandomBrightnessContrast(
            brightness_limit=config['brightness_range'],
            contrast_limit=config['contrast_range'],
            p=0.5
        ),
        OneOf([
            ToGray(p=1.0),
            ChannelShuffle(p=1.0),
            ChannelDropout(channel_drop_range=(1, 1), p=1.0),
        ], p=0.5),
        IAAAffine(
            scale=(0.8, 1.2),
            translate_percent=(-0.2, 0.2),
            rotate=(-config['rotation_range'], config['rotation_range']),
            shear=(-config['shear_range'], config['shear_range']),
            p=0.5
        ),
        RandomResizedCrop(
            height=config['image_size'][0],
            width=config['image_size'][1],
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),
    ]
    
    return Compose(transforms) 