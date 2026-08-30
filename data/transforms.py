"""
data/transforms.py
-------------------
Albumentations pipelines for train / validation / inference.

Augmentations are deliberately conservative for radiography: aggressive
color jitter or hue shifts are meaningless on grayscale X-rays and can
distort clinically relevant intensity patterns, so we stick to geometric
and mild intensity perturbations that plausibly occur from acquisition
variance (patient positioning, exposure, sensor noise).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.03, 0.06),
                hole_width_range=(0.03, 0.06),
                fill=0,
                p=0.2,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_inference_transforms() -> A.Compose:
    """Same as validation — no test-time augmentation by default."""
    return get_val_transforms()
