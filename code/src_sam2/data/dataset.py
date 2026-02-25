"""
Dataset for Copy-Move Forgery Detection with DinoV2-UNet.

Simplified version without point prompts (UNet is a global segmenter).
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['ALBUMENTATIONS_SKIP_VERSION_CHECK'] = '1'

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A


class ForgeryDataset(Dataset):
    """
    Dataset for copy-move forgery detection.
    
    Returns:
        - image: (C, H, W) float32 in range [0, 255] (for DinoV2 normalization)
        - mask: (1, H, W) int64 binary mask (0 = authentic, 1 = forged)
        - img_path: str path to image
        - original_mask: np.ndarray original resolution mask for oF1 calculation
        - original_shape: (H, W) original image dimensions
    """
    
    def __init__(
        self,
        data_root: str,
        csv_file: str,
        images_dir: str,
        masks_dir: str,
        mode: str = "train",
        image_size: int = 518,  # DinoV2 native: 518 = 37 * 14
        augment_type: int = 0,
        num_pairs: int = 1,  # Kept for config compatibility, but ignored
        crop_prob: float = 0.0,
        resize_mode: str = None,
        include_authentic: bool = True,
        max_images: Optional[int] = None,
        split_name: Optional[str] = None,  # For display purposes (e.g., "test" instead of "val")
    ):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / images_dir
        self.masks_dir = self.data_root / masks_dir
        self.mode = mode
        self.split_name = split_name or mode  # Use split_name for display, fallback to mode
        self.image_size = image_size
        self.augment_type = augment_type
        self.crop_prob = crop_prob
        self.resize_mode = resize_mode
        self.include_authentic = include_authentic
        
        # Load data by scanning directories
        self.samples = self._load_samples(max_images)
        
        # Build augmentation pipeline
        self.noise_transform = self._build_noise_transforms()
        self.geometric_transform = self._build_geometric_transforms()
        
    def _load_samples(self, max_images: Optional[int]) -> List[Dict[str, Any]]:
        """Load samples by scanning directories."""
        samples = self._scan_directories()
        
        if not self.include_authentic:
            samples = [s for s in samples if s['label'] == 1]
        
        if max_images is not None and len(samples) > max_images:
            random.shuffle(samples)
            samples = samples[:max_images]
        
        print(f"Loaded {len(samples)} samples for {self.split_name}")
        
        num_forged = sum(1 for s in samples if s['label'] == 1)
        num_authentic = sum(1 for s in samples if s['label'] == 0)
        print(f"  -> {num_forged} forged, {num_authentic} authentic")
        
        masks_found = sum(1 for s in samples if s['mask_path'] is not None and s['mask_path'].exists())
        print(f"  -> {masks_found} masks found on disk")
        
        return samples
    
    def _scan_directories(self) -> List[Dict[str, Any]]:
        """Scan directories to find images and masks."""
        samples = []
        
        authentic_dir = self.images_dir / "authentic"
        forged_dir = self.images_dir / "forged"
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        
        if forged_dir.exists():
            for img_path in forged_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    mask_path = self._find_mask(img_path)
                    samples.append({
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'label': 1,
                    })
            
            if authentic_dir.exists() and self.include_authentic:
                for img_path in authentic_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        samples.append({
                            'image_path': img_path,
                            'mask_path': None,
                            'label': 0,
                        })
        
        elif self.images_dir.exists():
            for img_path in self.images_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    mask_path = self._find_mask(img_path)
                    label = 1 if mask_path and mask_path.exists() else 0
                    if label == 1 or self.include_authentic:
                        samples.append({
                            'image_path': img_path,
                            'mask_path': mask_path if label == 1 else None,
                            'label': label,
                        })
        
        return samples
    
    def _find_mask(self, image_path: Path) -> Optional[Path]:
        """Find corresponding mask for an image."""
        stem = image_path.stem
        possible_masks = [
            self.masks_dir / f"{stem}.npy",
            self.masks_dir / f"{stem}_mask.npy",
            self.masks_dir / f"{stem}.png",
            self.masks_dir / f"{stem}_mask.png",
            self.masks_dir / f"{stem}.jpg",
            self.masks_dir / f"{stem}.tif",
        ]
        
        for mask_path in possible_masks:
            if mask_path.exists():
                return mask_path
        
        return None
    
    def _build_noise_transforms(self) -> Optional[A.Compose]:
        """Build noise augmentation transforms."""
        if self.mode != "train" or self.augment_type < 1:
            return None
        
        return A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MotionBlur(blur_limit=(3, 7), p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
                A.CLAHE(clip_limit=4.0, p=1),
            ], p=0.3),
            A.ImageCompression(quality_range=(70, 100), p=0.3),
        ])
    
    def _build_geometric_transforms(self) -> Optional[A.Compose]:
        """Build geometric augmentation transforms."""
        if self.mode != "train" or self.augment_type < 2:
            return None
        
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as RGB numpy array."""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _load_mask(self, path: Optional[Path], img_shape: Tuple[int, int], resize: bool = True) -> np.ndarray:
        """
        Load mask or create empty mask for authentic images.
        
        Returns the COMPLETE forgery mask (all forged regions).
        Handles multi-instance masks by merging all channels.
        """
        if path is None or not path.exists():
            return np.zeros(img_shape[:2], dtype=np.uint8)
        
        if path.suffix.lower() == '.npy':
            mask = np.load(path)
        else:
            mask = Image.open(path)
            mask = np.array(mask)
        
        # Handle 3D masks
        if len(mask.shape) == 3:
            if mask.shape[0] == 1:  # (1, H, W)
                mask = mask[0, :, :]
            elif mask.shape[2] <= 4:  # (H, W, C) with C <= 4 (RGB or RGBA)
                mask = mask[:, :, 0]
            else:  # (C, H, W) multi-instance mask - merge all channels
                # Merge all instance channels into one binary mask
                mask = (mask.sum(axis=0) > 0).astype(np.uint8)
        
        # Resize if needed
        if resize and (mask.shape[0] != img_shape[0] or mask.shape[1] != img_shape[1]):
            mask = cv2.resize(mask.astype(np.float32), (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize
        mask = (mask > 0.5).astype(np.uint8)
        
        return mask
    
    def _resize_image_mask(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and mask to target size."""
        H, W = img.shape[:2]
        
        if self.resize_mode == 'crop_prob' and self.crop_prob > 0 and random.random() < self.crop_prob:
            # Random crop
            if H < self.image_size or W < self.image_size:
                # Pad if too small
                pad_h = max(0, self.image_size - H)
                pad_w = max(0, self.image_size - W)
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            # Random crop
            crop_transform = A.RandomCrop(width=self.image_size, height=self.image_size)
            transformed = crop_transform(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']
        else:
            # Simple resize
            img = cv2.resize(img, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        return img, mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.
        
        Returns:
            img_tensor: (C, H, W) float32 in [0, 255]
            mask_tensor: (1, H, W) int64 binary mask
            img_path: str
            original_mask: np.ndarray at original resolution
            original_shape: tuple (H, W)
        """
        sample = self.samples[idx]
        
        # Load image and mask
        img = self._load_image(sample['image_path'])
        original_shape = (img.shape[0], img.shape[1])
        
        # Load original mask (for oF1 at original resolution)
        original_mask = self._load_mask(sample['mask_path'], original_shape, resize=False)
        
        # Load mask at image size
        mask = self._load_mask(sample['mask_path'], img.shape)
        
        # Apply augmentations
        if self.mode == "train" and self.augment_type >= 1:
            if self.noise_transform is not None:
                transformed = self.noise_transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            
            if self.augment_type >= 2 and self.geometric_transform is not None:
                transformed = self.geometric_transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
        
        # Resize to target size
        img, mask = self._resize_image_mask(img, mask)
        
        # Convert to tensors
        # IMPORTANT: Keep image in [0, 255] range for DinoV2 normalization
        img_tensor = torch.tensor(img.transpose(2, 0, 1).astype('float32'), dtype=torch.float32)
        
        # Mask: (1, H, W) int64
        mask_tensor = torch.tensor(mask[None, ...], dtype=torch.int64)
        
        return img_tensor, mask_tensor, str(sample['image_path']), original_mask, original_shape
    
    def shuffle_samples(self, seed: Optional[int] = None):
        """Shuffle samples."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.samples)
    
    def get_filename(self, idx: int) -> str:
        """Get filename for a sample."""
        return self.samples[idx]['image_path'].stem