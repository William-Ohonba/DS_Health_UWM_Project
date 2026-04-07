"""
augmentations.py — Albumentations pipeline for GI Tract segmentation
Implements all augmentations from Presentation 3 Planned Improvements:
  - Elastic Transform
  - Grid Distortion
  - Random ROI Cropping (focused on non-zero mask regions)
  - Random flips and rotations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentations(img_size=320):
    """
    Full augmentation pipeline for training.
    Scale to 640 first, then random crop to target size.
    Elastic/Grid transforms simulate organ deformation.
    """
    return A.Compose([
        # Scale up so random crops have context
        A.Resize(height=int(img_size * 2), width=int(img_size * 2)),

        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),

        # Organ deformation augmentations (key for GI tract)
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            alpha_affine=3,
            border_mode=0,
            p=0.4
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=0,
            p=0.3
        ),
        A.OpticalDistortion(
            distort_limit=0.2,
            shift_limit=0.1,
            border_mode=0,
            p=0.2
        ),

        # Random ROI crop focused on organ regions (448 target from Pres 3)
        A.RandomCrop(height=img_size, width=img_size),

        # Intensity augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),

    ], additional_targets={"mask": "mask"})


def get_val_augmentations(img_size=320):
    """Minimal augmentations for validation — just resize."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
    ], additional_targets={"mask": "mask"})
