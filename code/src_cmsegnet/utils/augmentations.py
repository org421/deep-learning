# -*- coding: utf-8 -*-
"""
Data Augmentations pour CMSeg-Net.

Inclut:
- Augmentations originales (Flip H/V, RandomNoise)
- Augmentations étendues optionnelles (rotation, JPEG, blur, etc.)
"""

import random
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# =============================================================================
# Original CMSeg-Net Augmentations (PyTorch style)
# =============================================================================

class RandomNoise:
    """
    Random Noise Augmentation.
    
    Fidèle au code original de CMSeg-Net.
    
    Args:
        noise_level: Niveau de bruit (défaut: 0.05)
        p: Probabilité d'application (défaut: 0.1)
    """
    def __init__(self, noise_level: float = 0.05, p: float = 0.1):
        self.noise_level = noise_level
        self.p = p

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            with torch.no_grad():
                noise = torch.randn_like(img_tensor) * self.noise_level
                noisy_img = img_tensor + noise
                noisy_img = torch.clamp(noisy_img, 0, 1)
            return noisy_img
        else:
            return img_tensor


class RandomHorizontalFlip:
    """Random Horizontal Flip."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        return image, mask


class RandomVerticalFlip:
    """Random Vertical Flip."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        return image, mask


class OriginalCMSegNetTransform:
    """
    Transformations originales de CMSeg-Net.
    
    Applique:
    - Random Horizontal Flip
    - Random Vertical Flip
    - Random Noise (sur l'image seulement)
    """
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        noise_level: float = 0.1,
        noise_prob: float = 0.05
    ):
        self.h_flip = RandomHorizontalFlip(p=0.5) if horizontal_flip else None
        self.v_flip = RandomVerticalFlip(p=0.5) if vertical_flip else None
        self.noise = RandomNoise(noise_level=noise_level, p=noise_prob)
    
    def __call__(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [C, H, W] tensor in [0, 1]
            mask: [1, H, W] or [H, W] tensor
            
        Returns:
            Transformed image and mask
        """
        if self.h_flip:
            image, mask = self.h_flip(image, mask)
        if self.v_flip:
            image, mask = self.v_flip(image, mask)
        
        # Noise only on image
        image = self.noise(image)
        
        return image, mask


# =============================================================================
# Extended Augmentations (Albumentations)
# =============================================================================

def get_training_augmentations(
    image_size: int = 512,
    use_extended: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> A.Compose:
    """
    Construit le pipeline d'augmentation pour l'entraînement.
    
    Args:
        image_size: Taille cible des images
        use_extended: Utiliser les augmentations étendues
        config: Configuration d'augmentation
        
    Returns:
        Pipeline Albumentations
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for extended augmentations")
    
    transforms = []
    
    # === Augmentations géométriques de base (comme l'original) ===
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    
    if use_extended:
        # === Augmentations géométriques étendues ===
        transforms.extend([
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=0,  # cv2.BORDER_CONSTANT
                p=0.3
            ),
        ])
        
        # === Qualité image (simule post-traitement) ===
        transforms.append(
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2)
        )
        
        # === Bruit (comme l'original mais plus varié) ===
        transforms.append(
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
            ], p=0.2)
        )
        
        # === Compression JPEG (important pour copy-move) ===
        transforms.append(
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3)
        )
        
        # === Colorimétrique ===
        transforms.append(
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.3)
        )
    
    # === Resize final ===
    transforms.append(A.Resize(image_size, image_size))
    
    return A.Compose(transforms)


def get_validation_augmentations(image_size: int = 512) -> A.Compose:
    """
    Augmentations pour validation (resize uniquement).
    
    Args:
        image_size: Taille cible
        
    Returns:
        Pipeline Albumentations
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required")
    
    return A.Compose([
        A.Resize(image_size, image_size),
    ])


# =============================================================================
# Normalization Utilities
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_image(
    image: np.ndarray,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> np.ndarray:
    """
    Normalise une image avec mean/std.
    
    Args:
        image: Image [H, W, C] in [0, 255] ou [0, 1]
        mean: Moyenne par canal
        std: Écart-type par canal
        
    Returns:
        Image normalisée
    """
    if image.max() > 1.0:
        image = image / 255.0
    
    image = (image - mean) / std
    return image.astype(np.float32)


def denormalize_image(
    image: np.ndarray,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> np.ndarray:
    """
    Dénormalise une image.
    
    Args:
        image: Image normalisée [H, W, C]
        mean: Moyenne utilisée
        std: Écart-type utilisé
        
    Returns:
        Image en [0, 255]
    """
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image
