"""
Script d'inférence automatisé pour SegFormer.

Pipeline:
1. Grid search sur validation pour trouver les meilleurs (threshold, min_area)
2. Évaluation finale sur test avec les meilleurs hyperparamètres
3. Calcul des métriques par catégorie d'instances (1-2, 3+)
4. Visualisation des prédictions
5. Génération d'un rapport récapitulatif

Usage:
    python main_inference.py --work_dir ./work_dir --data_root ../data
    python main_inference.py --work_dir ./work_dir_type2_otherparam --data_root ../data --visualize_image 43955.png
"""

import os
import sys
import argparse
import json
import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from scipy import ndimage

warnings.filterwarnings('ignore')
os.environ['ALBUMENTATIONS_SKIP_VERSION_CHECK'] = '1'

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import Config
from utils.kaggle_metric import KaggleMetricCalculator, mask_to_instances
from utils.metrics import dice_score, iou_score, compute_confusion_matrix


# =============================================================================
# Configuration
# =============================================================================

GRID_SEARCH_PARAMS = {
    'threshold': [0.25, 0.5, 0.75],
    'min_area': [50, 100],
}

MODEL_CHECKPOINTS = [
    'model_best_loss.pth',
    'model_best_loss_ema.pth', 
    'model_best_oF1.pth',
    'model_best_oF1_ema.pth',
]

# Catégories d'instances pour les métriques par difficulté
INSTANCE_CATEGORIES = {
    '1-2': lambda x: 1 <= x <= 2,
    '3+': lambda x: x > 2,
}


# =============================================================================
# Dataset adaptatif
# =============================================================================

class AdaptiveInferenceDataset(Dataset):
    """Dataset qui s'adapte automatiquement à la structure disponible."""
    
    def __init__(
        self,
        data_root: str,
        split: str,
        image_size: int = 512,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # Essayer de trouver les images
        self.images_dir = self._find_images_dir()
        self.masks_dir = self._find_masks_dir()
        
        # Charger les échantillons
        self.samples = self._load_samples()
        
        print(f"✓ [{split.upper()}] {len(self.samples)} échantillons")
        if len(self.samples) > 0:
            forged_count = sum(1 for s in self.samples if s['label'] == 1)
            authentic_count = len(self.samples) - forged_count
            print(f"  - Forgés: {forged_count}, Authentiques: {authentic_count}")
            print(f"  - Images: {self.images_dir}")
            print(f"  - Masques: {self.masks_dir}")
    
    def _find_images_dir(self) -> Path:
        """Trouve le dossier images en essayant plusieurs emplacements."""
        candidates = [
            self.data_root / self.split / "images",
            self.data_root / self.split,
            self.data_root / "images" / self.split,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                if (candidate / "forged").exists() or (candidate / "authentic").exists():
                    return candidate
        
        return candidates[0]
    
    def _find_masks_dir(self) -> Path:
        """Trouve le dossier masks en essayant plusieurs emplacements."""
        candidates = [
            self.data_root / self.split / "masks",
            self.data_root / "train" / "masks",
            self.data_root / "masks" / self.split,
            self.data_root / "masks",
        ]
        
        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("*.npy")):
                return candidate
        
        return candidates[0]
    
    def _count_instances(self, mask: np.ndarray) -> int:
        """Compte le nombre d'instances dans un masque."""
        if mask.ndim == 3 and mask.shape[0] > 1:
            return mask.shape[0]
        
        binary_mask = (mask > 0.5).astype(np.uint8)
        labeled, num_instances = ndimage.label(binary_mask)
        return num_instances
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Charge tous les échantillons disponibles."""
        samples = []
        
        if not self.images_dir.exists():
            print(f"⚠️  Dossier images introuvable: {self.images_dir}")
            return samples
        
        for label_name in ['forged', 'authentic']:
            label_dir = self.images_dir / label_name
            if not label_dir.exists():
                continue
            
            image_files = list(label_dir.glob('*.png')) + list(label_dir.glob('*.jpg')) + list(label_dir.glob('*.jpeg'))
            
            for img_path in image_files:
                mask_path = None
                num_instances = 0
                
                if label_name == 'forged':
                    mask_name = f"{img_path.stem}.npy"
                    mask_path = self.masks_dir / mask_name
                    
                    if mask_path.exists():
                        try:
                            mask = np.load(mask_path)
                            num_instances = self._count_instances(mask)
                        except:
                            pass
                
                samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'label': 1 if label_name == 'forged' else 0,
                    'image_name': img_path.name,
                    'num_instances': num_instances,
                })
        
        return samples
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Charge une image RGB."""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _load_mask(self, path: Optional[Path], img_shape: Tuple[int, int]) -> np.ndarray:
        """Charge un masque ou crée un masque vide."""
        if path is None or not path.exists():
            return np.zeros(img_shape[:2], dtype=np.uint8)
        
        try:
            if path.suffix == '.npy':
                mask = np.load(path)
            else:
                mask = np.array(Image.open(path).convert('L'))
            
            if mask.ndim == 3:
                if mask.shape[0] == 1:
                    mask = mask[0]
                elif mask.shape[2] <= 4:
                    mask = mask[:, :, 0]
                else:
                    mask = (mask.sum(axis=0) > 0).astype(np.uint8)
            
            return (mask > 0.5).astype(np.uint8)
        except Exception as e:
            return np.zeros(img_shape[:2], dtype=np.uint8)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        img = self._load_image(sample['image_path'])
        original_shape = img.shape[:2]
        
        original_mask = self._load_mask(sample['mask_path'], original_shape)
        
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        mask_resized = cv2.resize(original_mask, (self.image_size, self.image_size), 
                                  interpolation=cv2.INTER_NEAREST)
        
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1).astype('float32'))
        mask_tensor = torch.from_numpy(mask_resized[None, ...].astype('int64'))
        
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'original_mask': original_mask,
            'original_shape': original_shape,
            'image_path': str(sample['image_path']),
            'label': sample['label'],
            'image_name': sample['image_name'],
            'num_instances': sample['num_instances'],
        }


def collate_fn(batch):
    """Collate function pour le DataLoader."""
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'original_mask': [b['original_mask'] for b in batch],
        'original_shape': [b['original_shape'] for b in batch],
        'image_path': [b['image_path'] for b in batch],
        'label': [b['label'] for b in batch],
        'image_name': [b['image_name'] for b in batch],
        'num_instances': [b['num_instances'] for b in batch],
    }


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def load_model(checkpoint_path: str, config: Config, device: str) -> nn.Module:
    """Charge un modèle SegFormer depuis un checkpoint."""
    from networks.segformer_model import SegFormer
    
    variant = getattr(config.model, 'variant', 'segformer_b2')
    
    model = SegFormer(
        variant=variant,
        pretrained=False,
        freeze_backbone=False,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.5,
    min_area: int = 100,
    store_predictions: bool = False,
    visualize_image: str = None,
) -> Dict[str, Any]:
    """Exécute l'inférence et calcule les métriques (pixel + Kaggle oF1).
    
    Args:
        store_predictions: Si True, stocke toutes les prédictions (consomme beaucoup de RAM).
        visualize_image: Si défini, ne stocke que la prédiction de cette image.
    """
    
    model.eval()
    
    # Calculateur de métriques Kaggle
    kaggle_calc = KaggleMetricCalculator(
        threshold=threshold,
        min_area=min_area,
    )
    
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    predictions = []
    
    # Métriques par catégorie d'instances
    metrics_by_category = {cat: {
        'dice': [], 'iou': [], 'precision': [], 'recall': [], 'oF1': [], 'count': 0
    } for cat in INSTANCE_CATEGORIES.keys()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inférence (th={threshold}, area={min_area})", leave=False):
            images = batch['image'].to(device)
            original_masks = batch['original_mask']
            original_shapes = batch['original_shape']
            labels = batch['label']
            image_names = batch['image_name']
            image_paths = batch['image_path']
            num_instances_list = batch['num_instances']
            
            # Forward pass
            try:
                outputs, _ = model(images)
            except:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            # Appliquer sigmoid
            if outputs.min() < 0:
                probs = torch.sigmoid(outputs)
            else:
                probs = outputs
            
            probs = probs.cpu().numpy()
            
            for i in range(len(images)):
                pred = probs[i, 0]
                gt_mask = original_masks[i]
                orig_shape = original_shapes[i]
                label = labels[i]
                num_inst = num_instances_list[i]
                
                # Redimensionner
                pred_resized = cv2.resize(pred, (orig_shape[1], orig_shape[0]))
                
                # Mettre à jour le calculateur Kaggle
                kaggle_calc.update(pred_resized, gt_mask.astype(np.float32), orig_shape)
                
                # Calculer métriques pixel
                dice = dice_score(pred_resized, gt_mask, threshold=threshold)
                iou = iou_score(pred_resized, gt_mask, threshold=threshold)
                cm = compute_confusion_matrix(pred_resized, gt_mask, threshold=threshold)
                
                all_dice.append(dice)
                all_iou.append(iou)
                all_precision.append(cm['precision'])
                all_recall.append(cm['recall'])
                
                # Métriques par catégorie (seulement pour les forgés)
                if label == 1 and num_inst > 0:
                    for cat_name, cat_func in INSTANCE_CATEGORIES.items():
                        if cat_func(num_inst):
                            metrics_by_category[cat_name]['dice'].append(dice)
                            metrics_by_category[cat_name]['iou'].append(iou)
                            metrics_by_category[cat_name]['precision'].append(cm['precision'])
                            metrics_by_category[cat_name]['recall'].append(cm['recall'])
                            metrics_by_category[cat_name]['count'] += 1
                
                # Stocker pour visualisation (seulement si demandé)
                if store_predictions or (visualize_image and visualize_image in image_names[i]):
                    predictions.append({
                        'image_name': image_names[i],
                        'image_path': image_paths[i],
                        'pred': pred_resized,
                        'gt': gt_mask,
                        'num_instances': num_inst,
                    })
                
                # Libérer la mémoire de pred_resized si pas stocké
                del pred_resized
    
    gc.collect()
    
    # Calculer les métriques Kaggle
    kaggle_metrics = kaggle_calc.compute()
    
    # Calculer les moyennes par catégorie
    for cat_name in metrics_by_category:
        cat = metrics_by_category[cat_name]
        if cat['count'] > 0:
            cat['dice_mean'] = np.mean(cat['dice'])
            cat['iou_mean'] = np.mean(cat['iou'])
            cat['precision_mean'] = np.mean(cat['precision'])
            cat['recall_mean'] = np.mean(cat['recall'])
            p, r = cat['precision_mean'], cat['recall_mean']
            cat['f1_mean'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        else:
            cat['dice_mean'] = 0
            cat['iou_mean'] = 0
            cat['precision_mean'] = 0
            cat['recall_mean'] = 0
            cat['f1_mean'] = 0
    
    return {
        'oF1': kaggle_metrics['oF1'],
        'forged_oF1': kaggle_metrics['forged_oF1'],
        'dice': np.mean(all_dice) if all_dice else 0.0,
        'iou': np.mean(all_iou) if all_iou else 0.0,
        'precision': np.mean(all_precision) if all_precision else 0.0,
        'recall': np.mean(all_recall) if all_recall else 0.0,
        'predictions': predictions,
        'n_samples': len(all_dice),
        'by_category': metrics_by_category,
        'kaggle_details': kaggle_metrics,
    }


def grid_search_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> Tuple[float, int, Dict[str, Any], List[Dict]]:
    """
    Grid search sur validation pour trouver les meilleurs hyperparamètres.
    Optimise sur oF1.
    
    Returns:
        best_threshold, best_min_area, best_metrics, all_results
    """
    
    print("\n  🔍 Grid Search sur Validation...")
    
    best_oF1 = -1
    best_threshold = 0.5
    best_min_area = 100
    best_metrics = None
    all_results = []
    
    for threshold in GRID_SEARCH_PARAMS['threshold']:
        for min_area in GRID_SEARCH_PARAMS['min_area']:
            metrics = run_inference(model, val_loader, device, threshold=threshold, min_area=min_area,
                                    store_predictions=False)
            
            result = {
                'threshold': threshold,
                'min_area': min_area,
                'oF1': metrics['oF1'],
                'forged_oF1': metrics['forged_oF1'],
            }
            all_results.append(result)
            
            print(f"    th={threshold}, area={min_area} -> "
                  f"oF1={metrics['oF1']:.4f}, forged_oF1={metrics['forged_oF1']:.4f}")
            
            if metrics['oF1'] > best_oF1:
                best_oF1 = metrics['oF1']
                best_threshold = threshold
                best_min_area = min_area
                best_metrics = metrics
    
    print(f"  ✓ Meilleur: threshold={best_threshold}, min_area={best_min_area}, oF1={best_oF1:.4f}")
    
    return best_threshold, best_min_area, best_metrics, all_results


def visualize_prediction(
    image_path: str,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float,
    save_path: str,
    title: str = "",
):
    """Visualise une prédiction avec le masque GT et prédit."""
    
    img = np.array(Image.open(image_path).convert('RGB'))
    
    # Binariser la prédiction
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    
    # Redimensionner si nécessaire
    if pred_binary.shape != img.shape[:2]:
        pred_binary = cv2.resize(pred_binary.astype(np.uint8), (img.shape[1], img.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape != img.shape[:2]:
        gt_mask = cv2.resize(gt_mask.astype(np.float32), (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    # Créer la figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Image originale
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Image Originale', fontsize=12)
    axes[0, 0].axis('off')
    
    # Ground Truth
    axes[0, 1].imshow(img)
    gt_overlay = np.zeros((*img.shape[:2], 4))
    gt_overlay[gt_mask > 0] = [0, 1, 0, 0.5]
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title('Ground Truth (Vert)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Prédiction
    axes[1, 0].imshow(img)
    pred_overlay = np.zeros((*img.shape[:2], 4))
    pred_overlay[pred_binary > 0] = [1, 0, 0, 0.5]
    axes[1, 0].imshow(pred_overlay)
    axes[1, 0].set_title(f'Prédiction (Rouge) - th={threshold}', fontsize=12)
    axes[1, 0].axis('off')
    
    # Comparaison
    axes[1, 1].imshow(img)
    comparison = np.zeros((*img.shape[:2], 4))
    
    tp = (pred_binary > 0) & (gt_mask > 0)
    fp = (pred_binary > 0) & (gt_mask == 0)
    fn = (pred_binary == 0) & (gt_mask > 0)
    
    comparison[tp] = [0, 1, 0, 0.6]
    comparison[fp] = [1, 0, 0, 0.6]
    comparison[fn] = [1, 1, 0, 0.6]
    
    axes[1, 1].imshow(comparison)
    axes[1, 1].set_title('Comparaison (Vert=TP, Rouge=FP, Jaune=FN)', fontsize=12)
    axes[1, 1].axis('off')
    
    # Légende
    handles = [
        mpatches.Patch(color='green', alpha=0.6, label='TP (Vrai Positif)'),
        mpatches.Patch(color='red', alpha=0.6, label='FP (Faux Positif)'),
        mpatches.Patch(color='yellow', alpha=0.6, label='FN (Faux Négatif)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualisation sauvegardée: {save_path}")


def detect_available_splits(data_root: Path) -> List[str]:
    """Détecte automatiquement les splits disponibles."""
    splits = []
    for split_name in ['train', 'val', 'test']:
        candidates = [
            data_root / split_name / "images",
            data_root / split_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                forged = candidate / "forged"
                authentic = candidate / "authentic"
                if forged.exists() or authentic.exists():
                    splits.append(split_name)
                    break
    
    return splits


# =============================================================================
# Fonction principale
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inférence SegFormer avec Pipeline Complète")
    parser.add_argument('--work_dir', type=str, required=True,
                        help='Répertoire contenant les expérimentations')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Répertoire racine des données')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Répertoire de sortie')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Taille du batch')
    parser.add_argument('--visualize_image', type=str, default='43955.png',
                        help='Nom de l\'image à visualiser')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda ou cpu)')
    args = parser.parse_args()
    
    # Vérifier le device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA non disponible, utilisation du CPU")
        args.device = 'cpu'
    
    # Créer le répertoire de sortie
    work_dir_name = Path(args.work_dir).name
    output_dir = Path(args.output_dir) / work_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("🚀 INFÉRENCE SegFormer - Pipeline Complète")
    print("="*80)
    print("Pipeline:")
    print("  1. Grid search sur validation → meilleurs hyperparamètres")
    print("  2. Évaluation finale sur test")
    print("  3. Métriques par catégorie d'instances (1-2, 3+)")
    print("  4. Visualisation des prédictions")
    print("  5. Rapport récapitulatif")
    print("="*80)
    print(f"📁 Work dir: {args.work_dir}")
    print(f"📁 Data root: {args.data_root}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {args.device}")
    if args.visualize_image:
        print(f"🎨 Visualiser: {args.visualize_image}")
    print("="*80)
    
    # Vérifier work_dir
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        print(f"❌ Work dir introuvable: {work_dir}")
        return
    
    # Détecter les splits disponibles
    data_root = Path(args.data_root)
    available_splits = detect_available_splits(data_root)
    
    if not available_splits:
        print(f"❌ Aucun split (train/val/test) trouvé dans {data_root}")
        return
    
    print(f"\n📊 Splits disponibles: {', '.join(available_splits)}")
    
    # Déterminer les splits pour validation et test
    if 'val' in available_splits and 'test' in available_splits:
        val_split = 'val'
        test_split = 'test'
        print(f"✓ Mode complet: val={val_split}, test={test_split}")
    elif 'train' in available_splits and 'test' in available_splits:
        val_split = 'train'
        test_split = 'test'
        print(f"⚠️  Mode partiel: val={val_split}, test={test_split}")
    else:
        val_split = available_splits[0]
        test_split = available_splits[0]
        print(f"⚠️  Mode minimal: val=test={val_split}")
    
    # Trouver les expérimentations
    experiments = sorted([d for d in work_dir.iterdir() if d.is_dir()])
    
    if len(experiments) == 0:
        print(f"❌ Aucune expérimentation trouvée dans {work_dir}")
        return
    
    print(f"\n📂 Expérimentations trouvées: {len(experiments)}")
    for exp in experiments:
        checkpoints = [cp for cp in MODEL_CHECKPOINTS if (exp / cp).exists()]
        print(f"  - {exp.name}: {len(checkpoints)} checkpoints")
    
    # Rapport
    report_lines = []
    report_lines.append("="*100)
    report_lines.append("RAPPORT D'INFÉRENCE - SegFormer - Pipeline Complète")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Validation split: {val_split}")
    report_lines.append(f"Test split: {test_split}")
    report_lines.append("="*100)
    report_lines.append("")
    report_lines.append("Pipeline:")
    report_lines.append("  1. Grid search sur validation → meilleurs hyperparamètres")
    report_lines.append("  2. Évaluation finale sur test")
    report_lines.append("  3. Métriques par catégorie d'instances (1-2, 3+)")
    report_lines.append("  4. Visualisation des prédictions")
    report_lines.append("  5. Rapport récapitulatif")
    report_lines.append("")
    
    all_results = []
    
    for exp_idx, exp_dir in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"[{exp_idx+1}/{len(experiments)}] 📂 {exp_dir.name}")
        print("="*60)
        
        # Charger la config
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            print(f"  ⚠️  SKIP: config.yaml non trouvé")
            continue
        
        config = Config.from_yaml(str(config_path))
        config.data.data_root = args.data_root
        
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"EXPÉRIMENTATION: {exp_dir.name}")
        report_lines.append(f"Variant: {getattr(config.model, 'variant', 'segformer_b2')}")
        report_lines.append("="*80)
        
        # Créer les datasets
        try:
            val_dataset = AdaptiveInferenceDataset(
                data_root=config.data.data_root,
                split=val_split,
                image_size=config.data.image_size,
            )
            
            test_dataset = AdaptiveInferenceDataset(
                data_root=config.data.data_root,
                split=test_split,
                image_size=config.data.image_size,
            )
            
            if len(val_dataset) == 0 or len(test_dataset) == 0:
                print(f"  ⚠️  SKIP: datasets vides")
                continue
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn,
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn,
            )
        except Exception as e:
            print(f"  ❌ ERREUR chargement données: {e}")
            continue
        
        # Traiter chaque checkpoint
        for checkpoint_name in MODEL_CHECKPOINTS:
            checkpoint_path = exp_dir / checkpoint_name
            if not checkpoint_path.exists():
                continue
            
            print(f"\n  🔄 [{checkpoint_name}]")
            
            try:
                # Charger le modèle
                model = load_model(str(checkpoint_path), config, args.device)
                
                # ÉTAPE 1: Grid search sur validation
                best_th, best_area, val_metrics, grid_results = grid_search_validation(
                    model, val_loader, args.device
                )
                
                # ÉTAPE 2: Évaluation sur test
                print(f"\n  📊 Évaluation sur Test (th={best_th}, area={best_area})...")
                test_metrics = run_inference(model, test_loader, args.device,
                                             threshold=best_th, min_area=best_area,
                                             store_predictions=False,
                                             visualize_image=args.visualize_image)
                
                # Résultats
                result = {
                    'experiment': exp_dir.name,
                    'checkpoint': checkpoint_name,
                    'best_threshold': best_th,
                    'best_min_area': best_area,
                    'val_oF1': val_metrics['oF1'],
                    'val_forged_oF1': val_metrics['forged_oF1'],
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'test_oF1': test_metrics['oF1'],
                    'test_forged_oF1': test_metrics['forged_oF1'],
                    'test_dice': test_metrics['dice'],
                    'test_iou': test_metrics['iou'],
                    'test_precision': test_metrics['precision'],
                    'test_recall': test_metrics['recall'],
                    'test_n_samples': test_metrics['n_samples'],
                    'test_by_category': test_metrics['by_category'],
                    'grid_search_results': grid_results,
                }
                all_results.append(result)
                
                # Afficher
                print(f"\n  📊 Résultats sur TEST:")
                print(f"    oF1:        {test_metrics['oF1']:.4f}")
                print(f"    forged_oF1: {test_metrics['forged_oF1']:.4f}")
                print(f"    Dice:       {test_metrics['dice']:.4f}")
                print(f"    IoU:        {test_metrics['iou']:.4f}")
                print(f"    Precision:  {test_metrics['precision']:.4f}")
                print(f"    Recall:     {test_metrics['recall']:.4f}")
                
                # ÉTAPE 3: Métriques par catégorie
                print(f"\n  📊 Métriques par catégorie d'instances:")
                for cat_name, cat_metrics in test_metrics['by_category'].items():
                    if cat_metrics['count'] > 0:
                        print(f"    [{cat_name}] (n={cat_metrics['count']})")
                        print(f"      F1={cat_metrics['f1_mean']:.4f}, "
                              f"P={cat_metrics['precision_mean']:.4f}, "
                              f"R={cat_metrics['recall_mean']:.4f}, "
                              f"Dice={cat_metrics['dice_mean']:.4f}, "
                              f"IoU={cat_metrics['iou_mean']:.4f}")
                
                # Ajouter au rapport
                report_lines.append(f"\n--- {checkpoint_name} ---")
                report_lines.append(f"Best hyperparameters: threshold={best_th}, min_area={best_area}")
                report_lines.append("")
                report_lines.append("Grid Search Results (Validation):")
                report_lines.append(f"{'Threshold':<12} {'Min Area':<12} {'oF1':<12} {'forged_oF1':<12}")
                report_lines.append("-"*48)
                for gr in grid_results:
                    report_lines.append(f"{gr['threshold']:<12} {gr['min_area']:<12} {gr['oF1']:<12.4f} {gr['forged_oF1']:<12.4f}")
                
                report_lines.append("")
                report_lines.append("Validation Metrics (best params):")
                report_lines.append(f"  oF1:        {val_metrics['oF1']:.4f}")
                report_lines.append(f"  forged_oF1: {val_metrics['forged_oF1']:.4f}")
                report_lines.append(f"  Dice:       {val_metrics['dice']:.4f}")
                report_lines.append(f"  IoU:        {val_metrics['iou']:.4f}")
                report_lines.append(f"  Precision:  {val_metrics['precision']:.4f}")
                report_lines.append(f"  Recall:     {val_metrics['recall']:.4f}")
                
                report_lines.append("")
                report_lines.append("Test Metrics:")
                report_lines.append(f"  oF1:        {test_metrics['oF1']:.4f}")
                report_lines.append(f"  forged_oF1: {test_metrics['forged_oF1']:.4f}")
                report_lines.append(f"  Dice:       {test_metrics['dice']:.4f}")
                report_lines.append(f"  IoU:        {test_metrics['iou']:.4f}")
                report_lines.append(f"  Precision:  {test_metrics['precision']:.4f}")
                report_lines.append(f"  Recall:     {test_metrics['recall']:.4f}")
                
                report_lines.append("")
                report_lines.append("Test Metrics by Instance Category:")
                report_lines.append(f"{'Category':<12} {'Count':<8} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Dice':<10} {'IoU':<10}")
                report_lines.append("-"*72)
                for cat_name, cat_metrics in test_metrics['by_category'].items():
                    if cat_metrics['count'] > 0:
                        report_lines.append(
                            f"{cat_name:<12} {cat_metrics['count']:<8} "
                            f"{cat_metrics['f1_mean']:<10.4f} "
                            f"{cat_metrics['precision_mean']:<12.4f} "
                            f"{cat_metrics['recall_mean']:<10.4f} "
                            f"{cat_metrics['dice_mean']:<10.4f} "
                            f"{cat_metrics['iou_mean']:<10.4f}"
                        )
                
                # ÉTAPE 4: Visualisation
                if args.visualize_image:
                    for pred in test_metrics['predictions']:
                        if args.visualize_image in pred['image_name']:
                            vis_path = output_dir / f"visualization_{exp_dir.name}_{checkpoint_name.replace('.pth', '')}_{args.visualize_image}"
                            visualize_prediction(
                                pred['image_path'],
                                pred['pred'],
                                pred['gt'],
                                best_th,
                                str(vis_path),
                                title=f"{exp_dir.name} / {checkpoint_name}\nth={best_th}, area={best_area}, oF1={test_metrics['oF1']:.4f}"
                            )
                            break
                
                # Libérer la mémoire
                del model
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ❌ ERREUR: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Tableau récapitulatif
    report_lines.append("\n")
    report_lines.append("="*155)
    report_lines.append("TABLEAU RÉCAPITULATIF")
    report_lines.append("="*155)
    
    if all_results:
        header = (f"{'Experiment':<50} {'Checkpoint':<25} {'Th':<6} {'Area':<6} "
                  f"{'Val oF1':<10} {'Test oF1':<10} {'Test fgd_oF1':<12} "
                  f"{'Dice':<8} {'IoU':<8} {'P':<8} {'R':<8}")
        report_lines.append(header)
        report_lines.append("-"*155)
        
        # Trier par oF1 test décroissant
        all_results_sorted = sorted(all_results, key=lambda x: x['test_oF1'], reverse=True)
        
        for r in all_results_sorted:
            line = (f"{r['experiment'][:49]:<50} {r['checkpoint']:<25} "
                    f"{r['best_threshold']:<6} {r['best_min_area']:<6} "
                    f"{r['val_oF1']:<10.4f} {r['test_oF1']:<10.4f} {r['test_forged_oF1']:<12.4f} "
                    f"{r['test_dice']:<8.4f} {r['test_iou']:<8.4f} "
                    f"{r['test_precision']:<8.4f} {r['test_recall']:<8.4f}")
            report_lines.append(line)
        
        # Meilleur modèle
        report_lines.append("\n")
        report_lines.append("🏆 MEILLEUR MODÈLE (Test oF1):")
        best = all_results_sorted[0]
        report_lines.append(f"  Experiment:  {best['experiment']}")
        report_lines.append(f"  Checkpoint:  {best['checkpoint']}")
        report_lines.append(f"  Threshold:   {best['best_threshold']}")
        report_lines.append(f"  Min Area:    {best['best_min_area']}")
        report_lines.append(f"  oF1:         {best['test_oF1']:.4f}")
        report_lines.append(f"  forged_oF1:  {best['test_forged_oF1']:.4f}")
        report_lines.append(f"  Dice:        {best['test_dice']:.4f}")
        report_lines.append(f"  IoU:         {best['test_iou']:.4f}")
        report_lines.append(f"  Precision:   {best['test_precision']:.4f}")
        report_lines.append(f"  Recall:      {best['test_recall']:.4f}")
        report_lines.append("")
        report_lines.append("  Métriques par catégorie:")
        for cat_name, cat_metrics in best['test_by_category'].items():
            if cat_metrics['count'] > 0:
                report_lines.append(f"    [{cat_name}] (n={cat_metrics['count']}): "
                                  f"F1={cat_metrics['f1_mean']:.4f}, "
                                  f"Dice={cat_metrics['dice_mean']:.4f}")
    
    # Sauvegarder le rapport
    report_path = output_dir / f"inference_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Sauvegarder en JSON
    json_path = output_dir / f"inference_results_{timestamp}.json"
    json_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != 'test_by_category'}
        r_copy['test_by_category'] = {}
        for cat, metrics in r.get('test_by_category', {}).items():
            r_copy['test_by_category'][cat] = {
                k: v for k, v in metrics.items() 
                if k not in ['dice', 'iou', 'precision', 'recall', 'oF1']
            }
        json_results.append(r_copy)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"✅ PIPELINE TERMINÉE !")
    print(f"📄 Rapport: {report_path}")
    print(f"📊 JSON: {json_path}")
    if args.visualize_image:
        vis_files = list(output_dir.glob(f"visualization_*{args.visualize_image}"))
        if vis_files:
            print(f"🎨 Visualisations: {len(vis_files)} images générées")
    print("="*80)


if __name__ == '__main__':
    main()