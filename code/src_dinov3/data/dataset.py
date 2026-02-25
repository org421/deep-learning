"""Dataset pour la detection de falsification d'images."""

import os
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Suppression des warnings
warnings.filterwarnings('ignore')
os.environ['ALBUMENTATIONS_SKIP_VERSION_CHECK'] = '1'

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A


class ForgeryDataset(Dataset):
    """Dataset pour charger les images et masques de falsification."""
    
    def __init__(
        self,
        data_root: str,
        csv_file: str,
        images_dir: str,
        masks_dir: str,
        mode: str = "train",
        image_size: int = 518,  # Taille native DinoV2 : 518 = 37 * 14
        augment_type: int = 0,
        num_pairs: int = 1,  # Garde pour compatibilite config, non utilise
        crop_prob: float = 0.0,
        resize_mode: str = None,
        include_authentic: bool = True,
        max_images: Optional[int] = None,
        split_name: Optional[str] = None,  # Nom pour affichage (ex: "test" au lieu de "val")
    ):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / images_dir
        self.masks_dir = self.data_root / masks_dir
        self.mode = mode
        self.split_name = split_name or mode  # Nom d'affichage, par defaut mode
        self.image_size = image_size
        self.augment_type = augment_type
        self.crop_prob = crop_prob
        self.resize_mode = resize_mode
        self.include_authentic = include_authentic
        
        # Chargement des donnees en scannant les dossiers
        self.samples = self._load_samples(max_images)
        
        # Construction du pipeline d'augmentation
        self.noise_transform = self._build_noise_transforms()
        self.geometric_transform = self._build_geometric_transforms()
        
    def _load_samples(self, max_images: Optional[int]) -> List[Dict[str, Any]]:
        """Charge les echantillons en scannant les dossiers."""
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
        """Scanne les dossiers pour trouver images et masques."""
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
        """Trouve le masque correspondant a une image."""
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
        """Construit les augmentations de bruit."""
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
        """Construit les augmentations geometriques."""
        if self.mode != "train" or self.augment_type < 2:
            return None
        
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Charge une image en tableau numpy RGB."""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _load_mask(self, path: Optional[Path], img_shape: Tuple[int, int], resize: bool = True) -> np.ndarray:
        """Charge le masque ou cree un masque vide pour les images authentiques."""
        if path is None or not path.exists():
            return np.zeros(img_shape[:2], dtype=np.uint8)
        
        if path.suffix.lower() == '.npy':
            mask = np.load(path)
        else:
            mask = Image.open(path)
            mask = np.array(mask)
        
        # Gestion des masques 3D
        if len(mask.shape) == 3:
            if mask.shape[0] == 1:  # (1, H, W)
                mask = mask[0, :, :]
            elif mask.shape[2] <= 4:  # (H, W, C) avec C <= 4 (RGB ou RGBA)
                mask = mask[:, :, 0]
            else:  # (C, H, W) masque multi-instance, on fusionne les canaux
                mask = (mask.sum(axis=0) > 0).astype(np.uint8)
        
        # Redimensionnement si necessaire
        if resize and (mask.shape[0] != img_shape[0] or mask.shape[1] != img_shape[1]):
            mask = cv2.resize(mask.astype(np.float32), (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarisation
        mask = (mask > 0.5).astype(np.uint8)
        
        return mask
    
    def _resize_image_mask(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Redimensionne l'image et le masque a la taille cible."""
        H, W = img.shape[:2]
        
        if self.resize_mode == 'crop_prob' and self.crop_prob > 0 and random.random() < self.crop_prob:
            # Crop aleatoire
            if H < self.image_size or W < self.image_size:
                # Padding si trop petit
                pad_h = max(0, self.image_size - H)
                pad_w = max(0, self.image_size - W)
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
            # Crop aleatoire
            crop_transform = A.RandomCrop(width=self.image_size, height=self.image_size)
            transformed = crop_transform(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']
        else:
            # Redimensionnement simple
            img = cv2.resize(img, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        return img, mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """Charge l'image et le masque pour un echantillon."""
        sample = self.samples[idx]
        
        # Chargement de l'image et du masque
        img = self._load_image(sample['image_path'])
        original_shape = (img.shape[0], img.shape[1])
        
        # Masque original (pour oF1 a la resolution d'origine)
        original_mask = self._load_mask(sample['mask_path'], original_shape, resize=False)
        
        # Masque a la taille de l'image
        mask = self._load_mask(sample['mask_path'], img.shape)
        
        # Application des augmentations
        if self.mode == "train" and self.augment_type >= 1:
            if self.noise_transform is not None:
                transformed = self.noise_transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            
            if self.augment_type >= 2 and self.geometric_transform is not None:
                transformed = self.geometric_transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
        
        # Redimensionnement a la taille cible
        img, mask = self._resize_image_mask(img, mask)
        
        # Conversion en tenseurs
        # On garde l'image dans [0, 255] pour la normalisation DinoV2
        img_tensor = torch.tensor(img.transpose(2, 0, 1).astype('float32'), dtype=torch.float32)
        
        # Masque : (1, H, W) int64
        mask_tensor = torch.tensor(mask[None, ...], dtype=torch.int64)
        
        return img_tensor, mask_tensor, str(sample['image_path']), original_mask, original_shape
    
    def shuffle_samples(self, seed: Optional[int] = None):
        """Melange les echantillons."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.samples)
    
    def get_filename(self, idx: int) -> str:
        """Retourne le nom de fichier d'un echantillon."""
        return self.samples[idx]['image_path'].stem