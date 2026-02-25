"""
Dataset pour CMSeg-Net (Copy-Move Forgery Detection).

Ce fichier gère le chargement des images et des masques
pour l'entraînement et la validation du modèle.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

# Tentative d'import d'Albumentations (optionnel)
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class CMSegNetDataset(Dataset):
    """
    Dataset principal pour la détection de falsification copy-move.

    Chaque échantillon contient :
    - une image RGB
    - un masque binaire (zone copiée ou non)
    - un label (0 = authentique, 1 = falsifiée)
    """
    
    def __init__(
        self,
        data_root: str,
        images_dir: str,
        masks_dir: str,
        mode: str = "train",
        image_size: int = 512,
        transform: Optional[Callable] = None,
        augment: Optional[Callable] = None,
        include_authentic: bool = True,
        max_samples: Optional[int] = None,
    ):
        # Chemins principaux
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / images_dir
        self.masks_dir = self.data_root / masks_dir
        
        # Paramètres du dataset
        self.mode = mode
        self.image_size = image_size
        self.transform = transform
        self.augment = augment
        self.include_authentic = include_authentic
        
        # Scan des fichiers images / masques
        self.samples = self._scan_files(max_samples)
        
        # Affichage d'infos utiles
        print(f"[DATASET] Mode: {mode}")
        print(f"[DATASET] Loaded {len(self.samples)} samples")
        num_forged = sum(1 for s in self.samples if s['label'] == 1)
        num_authentic = len(self.samples) - num_forged
        print(f"[DATASET] {num_forged} forged, {num_authentic} authentic")
    
    def _scan_files(self, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """
        Parcourt les dossiers pour récupérer les images
        et associer les masques quand ils existent.
        """
        samples = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        
        # Cas où il y a des sous-dossiers forged / authentic
        forged_dir = self.images_dir / "forged"
        authentic_dir = self.images_dir / "authentic"
        
        if forged_dir.exists():
            # Images falsifiées
            for img_path in forged_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    mask_path = self._find_mask(img_path)
                    samples.append({
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'label': 1,
                    })
            
            # Images authentiques (sans masque)
            if self.include_authentic and authentic_dir.exists():
                for img_path in authentic_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        samples.append({
                            'image_path': img_path,
                            'mask_path': None,
                            'label': 0,
                        })
        
        elif self.images_dir.exists():
            # Cas d'une structure plate (tout dans un seul dossier)
            for img_path in self.images_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    mask_path = self._find_mask(img_path)
                    
                    # Image avec masque -> falsifiée
                    if mask_path and mask_path.exists():
                        samples.append({
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'label': 1,
                        })
                    # Image sans masque -> authentique
                    elif self.include_authentic:
                        samples.append({
                            'image_path': img_path,
                            'mask_path': None,
                            'label': 0,
                        })
        
        # Limitation du nombre d'échantillons si demandé
        if max_samples and len(samples) > max_samples:
            random.shuffle(samples)
            samples = samples[:max_samples]
        
        return samples
    
    def _find_mask(self, image_path: Path) -> Optional[Path]:
        """
        Cherche le masque correspondant à une image
        selon différentes conventions de nommage.
        """
        stem = image_path.stem
        
        possible_masks = [
            self.masks_dir / f"{stem}.png",
            self.masks_dir / f"{stem}_mask.png",
            self.masks_dir / f"{stem}.npy",
            self.masks_dir / f"{stem}_mask.npy",
            self.masks_dir / f"{stem}.jpg",
            self.masks_dir / f"{stem}.tif",
        ]
        
        for mask_path in possible_masks:
            if mask_path.exists():
                return mask_path
        
        return None
    
    def _load_image(self, path: Path) -> np.ndarray:
        """
        Charge une image et la convertit en RGB.
        """
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _load_mask(self, path: Optional[Path], shape: Tuple[int, int]) -> np.ndarray:
        """
        Charge un masque ou retourne un masque vide
        si l'image est authentique.
        """
        if path is None or not path.exists():
            return np.zeros(shape, dtype=np.float32)
        
        # Chargement selon le format
        if path.suffix.lower() == '.npy':
            mask = np.load(path)
        else:
            mask = np.array(Image.open(path))
        
        # Si le masque a plusieurs canaux, on en garde un seul
        if len(mask.shape) == 3:
            if mask.shape[2] <= 4:
                mask = mask[:, :, 0]
            else:
                mask = (mask.sum(axis=2) > 0).astype(np.float32)
        
        # Redimensionnement si nécessaire
        if mask.shape[:2] != shape:
            mask = cv2.resize(
                mask.astype(np.float32),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Binarisation du masque
        mask = (mask > 0.5).astype(np.float32)
        
        return mask
    
    def __len__(self) -> int:
        """Retourne la taille du dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retourne un échantillon du dataset.
        """
        sample = self.samples[idx]
        
        # Chargement de l'image
        image = self._load_image(sample['image_path'])
        original_shape = image.shape[:2]
        
        # Chargement du masque
        mask = self._load_mask(sample['mask_path'], original_shape)
        original_mask = mask.copy()
        
        # Redimensionnement image et masque
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Transformations géométriques (image + masque)
        if self.mode == 'train' and self.transform is not None:
            if HAS_ALBUMENTATIONS and isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                # Cas des transforms PyTorch
                image_tensor = torch.from_numpy(
                    image.transpose(2, 0, 1)
                ).float() / 255.0
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
                
                image_tensor, mask_tensor = self.transform(
                    image_tensor,
                    mask_tensor
                )
                
                image = (
                    image_tensor.numpy().transpose(1, 2, 0) * 255
                ).astype(np.uint8)
                mask = mask_tensor.squeeze(0).numpy()
        
        # Conversion finale en tenseurs PyTorch
        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Augmentations de bruit (image seulement)
        if self.mode == 'train' and self.augment is not None:
            image = self.augment(image)
        
        return {
            'image': image,
            'mask': mask,
            'label': sample['label'],
            'image_path': str(sample['image_path']),
            'original_mask': original_mask,
            'original_shape': original_shape,
        }


class FakeParaEggDataset(CMSegNetDataset):
    """
    Dataset spécifique au jeu de données FakeParaEgg.
    """
    
    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        image_size: int = 512,
        transform: Optional[Callable] = None,
        augment: Optional[Callable] = None,
        **kwargs
    ):
        # Séparation train / test
        if mode == 'train':
            images_dir = "train/images"
            masks_dir = "train/masks"
        else:
            images_dir = "test/images"
            masks_dir = "test/masks"
        
        super().__init__(
            data_root=data_root,
            images_dir=images_dir,
            masks_dir=masks_dir,
            mode=mode,
            image_size=image_size,
            transform=transform,
            augment=augment,
            **kwargs
        )


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Fonction de collate pour le DataLoader.
    Permet de regrouper les données dans un batch.
    """
    return {
        'image': torch.stack([s['image'] for s in batch]),
        'mask': torch.stack([s['mask'] for s in batch]),
        'label': torch.tensor([s['label'] for s in batch]),
        'image_path': [s['image_path'] for s in batch],
        'original_mask': [s['original_mask'] for s in batch],
        'original_shape': [s['original_shape'] for s in batch],
    }
