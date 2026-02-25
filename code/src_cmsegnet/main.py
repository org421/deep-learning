# -*- coding: utf-8 -*-
"""
Main Training Script for CMSeg-Net Copy-Move Forgery Detection.

Based on the original CMSeg-Net architecture:
- MobileNetV2 encoder
- CoSA modules (Corr + ASPP + SAM)
- U-Net decoder
- MixedLoss (BCE + Dice)

Features:
- Gradient Accumulation
- Mixed Precision (AMP)
- EMA (optional) with EMA validation
- Early Stopping
- oF1 metric tracking on original resolution
- Saves best_loss, best_oF1, best_loss_ema, best_oF1_ema
- Grid search for best threshold/min_area combination (inference mode)

Usage:
    python main.py --mode train
    python main.py --mode inference --checkpoint path/to/model.pth
"""

import os
import sys
import gc
import argparse
import json
import time
import random
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2  # For fast connected components (replaces scipy.ndimage)

# Detect if running in SSH/non-interactive terminal
IS_INTERACTIVE = sys.stdout.isatty()


def get_tqdm_kwargs(desc: str, disable: bool = False, config=None) -> dict:
    """Get tqdm kwargs adapted for SSH/non-interactive terminals."""
    if config is not None:
        disable_tqdm = getattr(config.logging, 'disable_tqdm', False)
        if disable_tqdm:
            disable = True
    
    base_kwargs = {
        'desc': desc,
        'disable': disable,
    }
    if not IS_INTERACTIVE:
        base_kwargs.update({
            'ncols': 100,
            'ascii': True,
            'dynamic_ncols': False,
            'mininterval': 30.0,
            'maxinterval': 60.0,
        })
    return base_kwargs


# Local imports
from utils.config_loader import Config, get_device_info
from utils.losses import MixedLoss, get_loss_function
from utils.metrics import AverageMeter, compute_batch_metrics
from utils.kaggle_metric import KaggleMetricCalculator, evaluate_single_image
from utils.ema import EMAModel
from utils.augmentations import (
    RandomNoise, 
    OriginalCMSegNetTransform,
    get_training_augmentations,
    get_validation_augmentations,
)
from data.dataset import CMSegNetDataset, collate_fn
from networks.model import UnetMobilenetV2, build_cmsegnet


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self, 
        patience: int = 20, 
        min_delta: float = 0.0001, 
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# =============================================================================
# METRICS LOGGER
# =============================================================================

class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'train_bce': [],
            'gradient_norm': [],
            'gradient_norm_clipped': [],
            'val_loss': [],
            'val_oF1': [],
            'val_forged_oF1': [],
            'val_precision': [],
            'val_recall': [],
            'val_dice': [],
            'val_iou': [],
            'learning_rate': [],
        }
    
    def log(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self):
        """Save metrics to JSON file."""
        with open(self.save_path / 'metrics_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot(self):
        """Plot training curves (improved version aligned with SAFIRE)."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Train vs Val Loss (on same plot for comparison)
        if self.history['train_loss'] or self.history['val_loss']:
            if self.history['train_loss']:
                axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
            if self.history['val_loss']:
                axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='orange')
            axes[0, 0].set_title('Train vs Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # oF1 metrics (oF1 and forged_oF1 together)
        if self.history['val_oF1']:
            axes[0, 1].plot(self.history['val_oF1'], label='Val oF1', color='green')
            if self.history['val_forged_oF1']:
                axes[0, 1].plot(self.history['val_forged_oF1'], label='Val forged_oF1', color='darkgreen', linestyle='--')
            axes[0, 1].set_title('Validation oF1')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Precision / Recall
        if self.history['val_precision'] and self.history['val_recall']:
            axes[0, 2].plot(self.history['val_precision'], label='Precision', color='blue')
            axes[0, 2].plot(self.history['val_recall'], label='Recall', color='red')
            axes[0, 2].set_title('Precision / Recall')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Dice / IoU
        if self.history['val_dice'] or self.history['val_iou']:
            if self.history['val_dice']:
                axes[1, 0].plot(self.history['val_dice'], label='Val Dice', color='purple')
            if self.history['val_iou']:
                axes[1, 0].plot(self.history['val_iou'], label='Val IoU', color='magenta', linestyle='--')
            axes[1, 0].set_title('Validation Dice / IoU')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient Norm (before and after clipping)
        if self.history['gradient_norm'] or self.history['gradient_norm_clipped']:
            if self.history['gradient_norm']:
                axes[1, 1].plot(self.history['gradient_norm'], label='Before Clipping', color='red', alpha=0.7)
            if self.history['gradient_norm_clipped']:
                axes[1, 1].plot(self.history['gradient_norm_clipped'], label='After Clipping', color='blue', alpha=0.7)
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        if self.history['learning_rate']:
            axes[1, 2].plot(self.history['learning_rate'], label='Learning Rate', color='teal')
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_path / 'training_curves.png', dpi=150)
        plt.close()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_precision_recall_dice(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute precision, recall, dice score, and IoU."""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > 0.5).astype(np.float32)
    
    pred_flat = pred_binary.ravel()
    gt_flat = gt_binary.ravel()
    
    valid_mask = gt_flat >= 0
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    intersection = 2 * tp
    union = 2 * tp + fp + fn
    dice = intersection / union if union > 0 else 0
    
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'dice': dice, 'iou': iou}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    save_path: Path,
    name: str = "latest",
    ema_model: Optional[EMAModel] = None,
    scaler: Optional[GradScaler] = None,
):
    """Save checkpoint."""
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.state_dict()
    
    torch.save(checkpoint, save_path / f"model_{name}.pth")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    config: Config,
    epoch: int,
    ema_model: Optional[EMAModel] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    epoch_loss = AverageMeter()
    epoch_dice = AverageMeter()
    epoch_bce = AverageMeter()
    gradient_norms = []
    gradient_norms_clipped = []
    
    accumulation_steps = config.training.gradient_accumulation_steps
    use_amp = config.training.use_amp
    grad_clip = config.training.grad_clip_norm
    
    optimizer.zero_grad()
    
    desc = f"[Epoch {epoch}] Training"
    if accumulation_steps > 1:
        desc += f" (accum={accumulation_steps})"
    
    pbar = tqdm(loader, **get_tqdm_kwargs(desc, config=config))
    
    for step, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                
                # Resize outputs to match mask size if needed
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs, size=masks.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                
                total_loss, dice_loss, bce_loss = criterion(outputs, masks)
            
            # Scale loss for accumulation
            loss = total_loss / accumulation_steps
            scaler.scale(loss).backward()
            
            # Update weights
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                
                # Compute gradient norm before clipping
                grad_norm = compute_gradient_norm(model)
                gradient_norms.append(grad_norm)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                
                # Compute gradient norm after clipping
                grad_norm_clipped = compute_gradient_norm(model)
                gradient_norms_clipped.append(grad_norm_clipped)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if ema_model is not None:
                    ema_model.update(model)
        
        else:
            outputs = model(images)
            
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            
            total_loss, dice_loss, bce_loss = criterion(outputs, masks)
            
            loss = total_loss / accumulation_steps
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                # Compute gradient norm before clipping
                grad_norm = compute_gradient_norm(model)
                gradient_norms.append(grad_norm)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                
                # Compute gradient norm after clipping
                grad_norm_clipped = compute_gradient_norm(model)
                gradient_norms_clipped.append(grad_norm_clipped)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if ema_model is not None:
                    ema_model.update(model)
        
        batch_size = images.shape[0]
        epoch_loss.update(total_loss.item(), batch_size)
        epoch_dice.update(dice_loss.item(), batch_size)
        epoch_bce.update(bce_loss.item(), batch_size)
        
        # Progress bar update
        if step % config.logging.log_every == 0:
            current_grad = gradient_norms[-1] if gradient_norms else 0
            current_grad_clipped = gradient_norms_clipped[-1] if gradient_norms_clipped else 0
            pbar.set_postfix({
                'loss': f"{epoch_loss.avg:.4f}",
                'grad': f"{current_grad:.2f}",
                'grad_clip': f"{current_grad_clipped:.2f}"
            })
    
    # Handle last batch if not aligned with accumulation
    if (step + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    valid_grads = [g for g in gradient_norms if np.isfinite(g)]
    avg_grad_norm = np.mean(valid_grads) if valid_grads else float('nan')
    
    valid_grads_clipped = [g for g in gradient_norms_clipped if np.isfinite(g)]
    avg_grad_norm_clipped = np.mean(valid_grads_clipped) if valid_grads_clipped else float('nan')
    
    return {
        'train_loss': epoch_loss.avg,
        'train_dice': epoch_dice.avg,
        'train_bce': epoch_bce.avg,
        'gradient_norm': avg_grad_norm,
        'gradient_norm_clipped': avg_grad_norm_clipped,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    epoch: int = 0,
) -> Dict[str, float]:
    """Validation with oF1 metric calculation on original resolution.
    
    MEMORY-SAFE version:
    - Computes oF1 incrementally DURING inference (no storage)
    - Sequential processing (no ThreadPoolExecutor = no memory accumulation)
    - No OOM possible - only 1 image processed at a time
    """
    model.eval()
    
    # Loss accumulator
    total_loss_sum = 0.0
    total_samples = 0
    
    # Incremental pixel metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # oF1 scores (just floats, tiny memory)
    all_oF1_scores = []
    forged_oF1_scores = []
    
    # Debug counters
    debug_gt_authentic = 0
    debug_gt_forged = 0
    debug_pred_authentic = 0
    debug_pred_forged = 0
    
    threshold = config.inference.threshold
    min_area = config.inference.min_area
    
    # Check if oF1 computation is enabled
    compute_oF1 = getattr(config.evaluation, 'compute_oF1', True)
    
    pbar = tqdm(loader, **get_tqdm_kwargs(f"[Epoch {epoch}] Validating", config=config))
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        original_masks = batch['original_mask']
        original_shapes = batch['original_shape']
        labels = batch['label']
        
        outputs = model(images)
        
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(
                outputs, size=masks.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        total_loss, _, _ = criterion(outputs, masks)
        
        batch_size = images.shape[0]
        total_loss_sum += total_loss.item() * batch_size
        total_samples += batch_size
        
        pred_sigmoid = torch.sigmoid(outputs)
        
        for i in range(batch_size):
            is_forged = (labels[i] == 1)
            orig_h, orig_w = original_shapes[i]
            
            pred_single = pred_sigmoid[i:i+1]
            pred_resized = F.interpolate(
                pred_single,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )[0, 0]
            
            pred_binary = (pred_resized > threshold).cpu().numpy().astype(np.uint8)
            
            if min_area > 1:
                pred_binary = remove_small_regions(pred_binary, min_area)
            
            orig_gt = original_masks[i]
            if orig_gt.shape[0] > orig_h or orig_gt.shape[1] > orig_w:
                orig_gt = orig_gt[:orig_h, :orig_w]
            gt_binary = (orig_gt > 0.5).astype(np.uint8)
            
            # Debug counters
            pred_has_forgery = pred_binary.sum() > 0
            gt_has_forgery = gt_binary.sum() > 0
            
            if gt_has_forgery:
                debug_gt_forged += 1
            else:
                debug_gt_authentic += 1
            
            if pred_has_forgery:
                debug_pred_forged += 1
            else:
                debug_pred_authentic += 1
            
            # Pixel metrics for forged images
            if is_forged:
                tp = np.sum((pred_binary == 1) & (gt_binary == 1))
                fp = np.sum((pred_binary == 1) & (gt_binary == 0))
                fn = np.sum((pred_binary == 0) & (gt_binary == 1))
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            # Compute oF1 SEQUENTIALLY (no .copy(), no memory accumulation)
            if compute_oF1:
                score = compute_oF1_direct(pred_binary, gt_binary)
                all_oF1_scores.append(score)
                if is_forged:
                    forged_oF1_scores.append(score)
            
            # Libérer immédiatement les arrays numpy
            del pred_binary, gt_binary
        
        # Cleanup GPU memory
        del images, masks, outputs, pred_sigmoid
        
        # Periodic cleanup
        if batch_idx % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # ==================== COMPUTE FINAL METRICS ====================
    avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0
    oF1 = np.mean(all_oF1_scores) if all_oF1_scores else 0.0
    forged_oF1 = np.mean(forged_oF1_scores) if forged_oF1_scores else 0.0
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    
    return {
        'val_loss': avg_loss,
        'val_oF1': oF1,
        'val_forged_oF1': forged_oF1,
        'val_precision': precision,
        'val_recall': recall,
        'val_dice': dice,
        'val_iou': iou,
        'debug_gt_authentic': debug_gt_authentic,
        'debug_gt_forged': debug_gt_forged,
        'debug_pred_authentic': debug_pred_authentic,
        'debug_pred_forged': debug_pred_forged,
    }


@torch.no_grad()
def validate_with_params(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    threshold: float,
    min_area: int,
    epoch: int = 0,
    show_progress: bool = False,
) -> Dict[str, float]:
    """Validation with specific threshold and min_area parameters.
    
    Used for grid search - allows overriding config values.
    """
    model.eval()
    
    # Loss accumulator
    total_loss_sum = 0.0
    total_samples = 0
    
    # Incremental pixel metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # oF1 scores
    all_oF1_scores = []
    forged_oF1_scores = []
    
    # Debug counters
    debug_gt_authentic = 0
    debug_gt_forged = 0
    debug_pred_authentic = 0
    debug_pred_forged = 0
    
    compute_oF1 = getattr(config.evaluation, 'compute_oF1', True)
    
    if show_progress:
        pbar = tqdm(loader, **get_tqdm_kwargs(f"thresh={threshold:.2f}, min_area={min_area}", config=config))
    else:
        pbar = loader
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        original_masks = batch['original_mask']
        original_shapes = batch['original_shape']
        labels = batch['label']
        
        outputs = model(images)
        
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(
                outputs, size=masks.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        total_loss, _, _ = criterion(outputs, masks)
        
        batch_size = images.shape[0]
        total_loss_sum += total_loss.item() * batch_size
        total_samples += batch_size
        
        pred_sigmoid = torch.sigmoid(outputs)
        
        for i in range(batch_size):
            is_forged = (labels[i] == 1)
            orig_h, orig_w = original_shapes[i]
            
            pred_single = pred_sigmoid[i:i+1]
            pred_resized = F.interpolate(
                pred_single,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )[0, 0]
            
            pred_binary = (pred_resized > threshold).cpu().numpy().astype(np.uint8)
            
            if min_area > 1:
                pred_binary = remove_small_regions(pred_binary, min_area)
            
            orig_gt = original_masks[i]
            if orig_gt.shape[0] > orig_h or orig_gt.shape[1] > orig_w:
                orig_gt = orig_gt[:orig_h, :orig_w]
            gt_binary = (orig_gt > 0.5).astype(np.uint8)
            
            pred_has_forgery = pred_binary.sum() > 0
            gt_has_forgery = gt_binary.sum() > 0
            
            if gt_has_forgery:
                debug_gt_forged += 1
            else:
                debug_gt_authentic += 1
            
            if pred_has_forgery:
                debug_pred_forged += 1
            else:
                debug_pred_authentic += 1
            
            if is_forged:
                tp = np.sum((pred_binary == 1) & (gt_binary == 1))
                fp = np.sum((pred_binary == 1) & (gt_binary == 0))
                fn = np.sum((pred_binary == 0) & (gt_binary == 1))
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            if compute_oF1:
                score = compute_oF1_direct(pred_binary, gt_binary)
                all_oF1_scores.append(score)
                if is_forged:
                    forged_oF1_scores.append(score)
            
            del pred_binary, gt_binary
        
        del images, masks, outputs, pred_sigmoid
        
        if batch_idx % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0
    oF1 = np.mean(all_oF1_scores) if all_oF1_scores else 0.0
    forged_oF1 = np.mean(forged_oF1_scores) if forged_oF1_scores else 0.0
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    
    return {
        'val_loss': avg_loss,
        'val_oF1': oF1,
        'val_forged_oF1': forged_oF1,
        'val_precision': precision,
        'val_recall': recall,
        'val_dice': dice,
        'val_iou': iou,
        'debug_gt_authentic': debug_gt_authentic,
        'debug_gt_forged': debug_gt_forged,
        'debug_pred_authentic': debug_pred_authentic,
        'debug_pred_forged': debug_pred_forged,
    }


def compute_oF1_direct(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """Compute oF1 score directly from binary masks. Thread-safe."""
    pred_has_forgery = pred_binary.sum() > 0
    gt_has_forgery = gt_binary.sum() > 0
    
    # Fast path for authentic cases
    if not pred_has_forgery and not gt_has_forgery:
        return 1.0
    if pred_has_forgery != gt_has_forgery:
        return 0.0
    
    # Both have forgeries - extract instances
    pred_instances = extract_instances_fast(pred_binary)
    gt_instances = extract_instances_fast(gt_binary)
    
    if len(pred_instances) == 0 or len(gt_instances) == 0:
        return 0.0
    
    return compute_oF1_from_instances(pred_instances, gt_instances)


def extract_instances_fast(binary_mask: np.ndarray) -> List[np.ndarray]:
    """Extract connected component instances from binary mask using cv2."""
    if binary_mask.sum() == 0:
        return []
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    
    instances = []
    for i in range(1, num_labels):
        instance = (labels == i).astype(np.uint8)
        instances.append(instance)
    
    return instances


def compute_oF1_from_instances(pred_instances: List[np.ndarray], 
                                gt_instances: List[np.ndarray],
                                max_instances: int = 50) -> float:
    """Compute oF1 score using Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment
    
    # Garder seulement les max_instances plus grandes instances prédites
    if len(pred_instances) > max_instances:
        pred_instances = sorted(pred_instances, key=lambda x: x.sum(), reverse=True)[:max_instances]
    
    num_pred = len(pred_instances)
    num_gt = len(gt_instances)
    
    if num_pred == 0 or num_gt == 0:
        return 0.0
    
    # Compute F1 matrix
    size = max(num_pred, num_gt)
    f1_matrix = np.zeros((size, size))
    
    for i in range(num_pred):
        pred_bool = pred_instances[i].astype(bool)
        for j in range(num_gt):
            gt_bool = gt_instances[j].astype(bool)
            
            tp = np.sum(pred_bool & gt_bool)
            fp = np.sum(pred_bool & ~gt_bool)
            fn = np.sum(~pred_bool & gt_bool)
            
            if (tp + fp) == 0 or (tp + fn) == 0:
                f1 = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_matrix[i, j] = f1
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-f1_matrix)
    
    valid_matches = [(r, c) for r, c in zip(row_ind, col_ind) if r < num_pred and c < num_gt]
    
    if not valid_matches:
        return 0.0
    
    matched_f1_sum = sum(f1_matrix[r, c] for r, c in valid_matches)
    mean_f1 = matched_f1_sum / num_gt
    
    excess_penalty = num_gt / max(num_pred, num_gt)
    
    return mean_f1 * excess_penalty


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area - OPTIMIZED with cv2."""
    if min_area <= 1:
        return mask
    
    # Use cv2 instead of scipy - much faster
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create output mask
    output = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 1
    
    return output


# =============================================================================
# DEPRECATED FUNCTIONS (kept for reference, no longer used)
# The new validate() function computes oF1 incrementally without storing RLEs
# =============================================================================

# def mask_to_rle(mask: np.ndarray) -> str:
#     """DEPRECATED - Convert binary mask to RLE string."""
#     pass

# def compute_global_oF1(pred_rles, gt_rles, shapes) -> dict:
#     """DEPRECATED - Was causing OOM when many forgeries detected."""
#     pass


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function."""
    
    # Load config
    config = Config.from_yaml("configs/config.yaml")
    
    # Set seed
    seed = config.experiment.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Device
    device_info = get_device_info(config)
    device = device_info['device']
    
    print("\n" + "=" * 70)
    print("CMSeg-Net Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(config.experiment.work_dir) / f"run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Output: {save_path}")
    
    # Save config and script
    config.save(save_path / "config.yaml")
    shutil.copyfile(__file__, save_path / "main.py")
    
    # ==========================================================================
    # DATA
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("Loading data...")
    print("-" * 70)
    
    # Augmentations
    if config.augmentation.extended.enabled:
        train_transform = get_training_augmentations(
            image_size=config.data.image_size,
            use_extended=True
        )
    else:
        train_transform = OriginalCMSegNetTransform(
            horizontal_flip=config.augmentation.horizontal_flip,
            vertical_flip=config.augmentation.vertical_flip,
            noise_level=config.augmentation.random_noise.noise_level,
            noise_prob=config.augmentation.random_noise.probability,
        )
    
    # Noise augmentation
    if not config.augmentation.extended.enabled:
        train_augment = RandomNoise(
            noise_level=config.augmentation.random_noise.noise_level,
            p=config.augmentation.random_noise.probability
        )
    else:
        train_augment = None
    
    # Datasets
    train_dataset = CMSegNetDataset(
        data_root=config.data.data_root,
        images_dir=config.data.train_images_dir,
        masks_dir=config.data.train_masks_dir,
        mode='train',
        image_size=config.data.image_size,
        transform=train_transform if config.augmentation.extended.enabled else None,
        augment=train_augment,
        include_authentic=config.data.include_authentic,
        max_samples=config.data.max_samples_train,
    )
    
    val_dataset = CMSegNetDataset(
        data_root=config.data.data_root,
        images_dir=config.data.val_images_dir,
        masks_dir=config.data.val_masks_dir,
        mode='val',
        image_size=config.data.image_size,
        transform=None,
        augment=None,
        include_authentic=config.data.include_authentic,
        max_samples=config.data.max_samples_val,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # ==========================================================================
    # MODEL
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("Building model...")
    print("-" * 70)
    
    model = build_cmsegnet(device=str(device))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==========================================================================
    # LOSS, OPTIMIZER, SCHEDULER
    # ==========================================================================
    
    criterion = MixedLoss(
        bce_weight=getattr(config.loss, 'bce_weight', 1.0),
        dice_weight=getattr(config.loss, 'dice_weight', 0.5),
        tversky_weight=getattr(config.loss, 'tversky_weight', 0.0),
        pos_weight=getattr(config.loss, 'pos_weight', 1.0),
        tversky_alpha=getattr(config.loss, 'tversky_alpha', 0.3),
        tversky_beta=getattr(config.loss, 'tversky_beta', 0.7),
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Scheduler
    scheduler = None
    scheduler_type = getattr(config.training.scheduler, 'type', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.scheduler.min_lr,
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.training.scheduler.mode,
            factor=config.training.scheduler.factor,
            patience=config.training.scheduler.patience,
            min_lr=config.training.scheduler.min_lr,
        )
    
    # EMA
    ema_model = None
    if config.training.ema.enabled:
        ema_model = EMAModel(model, decay=config.training.ema.decay)
        print(f"EMA enabled with decay={config.training.ema.decay}")
    
    # AMP
    scaler = GradScaler() if config.training.use_amp else None
    if config.training.use_amp:
        print("Mixed precision (AMP) enabled")
    
    # Early stopping
    early_stopping = None
    if config.training.early_stopping.enabled:
        es_mode = 'min' if config.training.early_stopping.metric == 'val_loss' else 'max'
        early_stopping = EarlyStopping(
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
            mode=es_mode,
        )
        print(f"Early stopping enabled: patience={config.training.early_stopping.patience}, metric={config.training.early_stopping.metric}")
    
    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    metrics_logger = MetricsLogger(save_path)
    best_val_loss = float('inf')
    best_oF1 = 0.0
    best_ema_val_loss = float('inf')
    best_ema_oF1 = 0.0
    
    for epoch in range(1, config.training.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, config, epoch, ema_model
        )
        
        # Scheduler step for cosine (after epoch)
        if scheduler is not None and scheduler_type == 'cosine':
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config, epoch)
        
        # Validate EMA model
        ema_val_metrics = None
        if ema_model is not None:
            ema_val_metrics = validate(
                ema_model.get_model(), val_loader, criterion, device, config, epoch
            )
        
        # Log metrics
        all_metrics = {
            **train_metrics,
            **val_metrics,
            'learning_rate': current_lr,
        }
        metrics_logger.log(epoch, all_metrics)
        metrics_logger.save()
        metrics_logger.plot()
        
        # Print epoch summary
        print(f"\n[Epoch {epoch}/{config.training.epochs}]")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        print(f"  Val oF1:    {val_metrics['val_oF1']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        if ema_val_metrics is not None:
            print(f"  EMA Val Loss: {ema_val_metrics['val_loss']:.4f}")
            print(f"  EMA Val oF1:  {ema_val_metrics['val_oF1']:.4f}")
        
        # Save best models
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, 
                           save_path, "best_loss", ema_model, scaler)
            print(f"  -> New best val_loss: {best_val_loss:.4f}")
        
        # Save best oF1 model
        if val_metrics['val_oF1'] > best_oF1:
            best_oF1 = val_metrics['val_oF1']
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, 
                           save_path, "best_oF1", ema_model, scaler)
            print(f"  -> New best oF1: {best_oF1:.4f}")
        
        # Save best EMA models
        if ema_val_metrics is not None:
            if ema_val_metrics['val_loss'] < best_ema_val_loss:
                best_ema_val_loss = ema_val_metrics['val_loss']
                ema_checkpoint = {
                    "model": ema_model.get_model().state_dict(),
                    "epoch": epoch,
                    "metrics": ema_val_metrics,
                }
                torch.save(ema_checkpoint, save_path / "model_best_loss_ema.pth")
                print(f"  -> New best EMA val_loss: {best_ema_val_loss:.4f}")
            
            if ema_val_metrics['val_oF1'] > best_ema_oF1:
                best_ema_oF1 = ema_val_metrics['val_oF1']
                ema_checkpoint = {
                    "model": ema_model.get_model().state_dict(),
                    "epoch": epoch,
                    "metrics": ema_val_metrics,
                }
                torch.save(ema_checkpoint, save_path / "model_best_oF1_ema.pth")
                print(f"  -> New best EMA oF1: {best_ema_oF1:.4f}")
        
        # Early stopping check
        if early_stopping is not None:
            metric_value = val_metrics['val_loss'] if config.training.early_stopping.metric == 'val_loss' else val_metrics['val_oF1']
            if early_stopping(metric_value):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"  Best val_loss: {best_val_loss:.4f}")
                print(f"  Best oF1: {best_oF1:.4f}")
                break
        
        # Scheduler step for plateau (after validation)
        if scheduler is not None and scheduler_type == 'plateau':
            plateau_metric = val_metrics['val_oF1'] if config.training.scheduler.mode == 'max' else val_metrics['val_loss']
            scheduler.step(plateau_metric)
    
    # Save periodic checkpoint (skip if save_every is 0 or None)
    if config.training.save_every and config.training.save_every > 0:
        if epoch % config.training.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, 
                           {'epoch': epoch}, save_path, f"epoch_{epoch}", ema_model, scaler)

    # Final summary
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best oF1 score: {best_oF1:.4f}")
    if ema_model is not None:
        print(f"  Best EMA validation loss: {best_ema_val_loss:.4f}")
        print(f"  Best EMA oF1 score: {best_ema_oF1:.4f}")
    print(f"  Models saved to: {save_path}")
    print("=" * 70)


# =============================================================================
# INFERENCE WITH GRID SEARCH
# =============================================================================

def evaluate_single_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    thresholds: List[float],
    min_areas: List[int],
) -> Dict[str, Any]:
    """Evaluate a single checkpoint with grid search for best threshold/min_area.
    
    Returns dict with best params and metrics for this checkpoint.
    """
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    checkpoint_name = os.path.basename(checkpoint_path)
    epoch = checkpoint.get('epoch', 'N/A')
    train_oF1 = checkpoint.get('metrics', {}).get('val_oF1', 'N/A')
    
    print(f"\n  Checkpoint: {checkpoint_name}")
    print(f"  Epoch: {epoch}, Training oF1: {train_oF1}")
    
    # Grid search
    best_oF1 = -1
    best_params = None
    best_metrics = None
    all_results = []
    
    for thresh in thresholds:
        for min_area in min_areas:
            metrics = validate_with_params(
                model, loader, criterion, device, config,
                threshold=thresh,
                min_area=min_area,
                epoch=0,
                show_progress=False
            )
            
            result = {
                'threshold': thresh,
                'min_area': min_area,
                'oF1': metrics['val_oF1'],
                'forged_oF1': metrics.get('val_forged_oF1', 0),
                'loss': metrics['val_loss'],
                'precision': metrics['val_precision'],
                'recall': metrics['val_recall'],
                'dice': metrics['val_dice'],
                'iou': metrics['val_iou'],
            }
            all_results.append(result)
            
            if metrics['val_oF1'] > best_oF1:
                best_oF1 = metrics['val_oF1']
                best_params = (thresh, min_area)
                best_metrics = metrics
    
    print(f"  -> Best: oF1={best_oF1:.4f} (thresh={best_params[0]:.2f}, min_area={best_params[1]})")
    
    return {
        'checkpoint': checkpoint_name,
        'checkpoint_path': checkpoint_path,
        'epoch': epoch,
        'training_oF1': train_oF1 if isinstance(train_oF1, (int, float)) else None,
        'best_threshold': best_params[0],
        'best_min_area': best_params[1],
        'best_oF1': best_oF1,
        'best_forged_oF1': best_metrics.get('val_forged_oF1', 0),
        'best_precision': best_metrics['val_precision'],
        'best_recall': best_metrics['val_recall'],
        'best_dice': best_metrics['val_dice'],
        'best_iou': best_metrics['val_iou'],
        'best_loss': best_metrics['val_loss'],
        'all_results': sorted(all_results, key=lambda x: x['oF1'], reverse=True),
    }


def run_inference(args):
    """Run inference on validation or test set with grid search for best threshold/min_area.
    
    Supports both single checkpoint file and directory of checkpoints.
    """
    config = Config.from_yaml("configs/config.yaml")
    
    device_info = get_device_info(config)
    device = device_info['device']
    
    split = args.split if args.split else 'val'
    
    print("\n" + "=" * 70)
    print(f"CMSeg-Net Inference - {split} set")
    print("=" * 70)
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        raise ValueError("--checkpoint is required for inference mode")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint/directory not found: {checkpoint_path}")
    
    # Determine if it's a directory or single file
    if os.path.isdir(checkpoint_path):
        # Find all .pth files in directory
        checkpoint_files = sorted([
            os.path.join(checkpoint_path, f) 
            for f in os.listdir(checkpoint_path) 
            if f.endswith('.pth')
        ])
        if not checkpoint_files:
            raise FileNotFoundError(f"No .pth files found in: {checkpoint_path}")
        print(f"Found {len(checkpoint_files)} checkpoints in: {checkpoint_path}")
        for f in checkpoint_files:
            print(f"  - {os.path.basename(f)}")
    else:
        # Single checkpoint file
        checkpoint_files = [checkpoint_path]
        print(f"Single checkpoint: {checkpoint_path}")
    
    # Dataset
    if split == 'val':
        images_dir = config.data.val_images_dir
        masks_dir = config.data.val_masks_dir
        max_samples = config.data.max_samples_val
    elif split == 'test':
        images_dir = config.data.test_images_dir
        masks_dir = config.data.test_masks_dir
        max_samples = getattr(config.data, 'max_samples_test', None)
    else:
        images_dir = config.data.train_images_dir
        masks_dir = config.data.train_masks_dir
        max_samples = config.data.max_samples_train
    
    dataset = CMSegNetDataset(
        data_root=config.data.data_root,
        images_dir=images_dir,
        masks_dir=masks_dir,
        mode='val',
        image_size=config.data.image_size,
        include_authentic=config.data.include_authentic,
        max_samples=max_samples,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )
    
    print(f"\nDataset: {len(dataset)} samples")
    
    # Get threshold and min_area lists from config
    thresholds = getattr(config.inference, 'thresholds', None)
    min_areas = getattr(config.inference, 'min_areas', None)
    
    # Fallback to single values if lists not provided
    if thresholds is None:
        threshold_single = getattr(config.inference, 'threshold', 0.5)
        thresholds = [threshold_single] if isinstance(threshold_single, (int, float)) else threshold_single
    
    if min_areas is None:
        min_area_single = getattr(config.inference, 'min_area', 1)
        min_areas = [min_area_single] if isinstance(min_area_single, (int, float)) else min_area_single
    
    # Ensure they are lists
    if not isinstance(thresholds, (list, tuple)):
        thresholds = [thresholds]
    if not isinstance(min_areas, (list, tuple)):
        min_areas = [min_areas]
    
    # Convert to proper types
    thresholds = [float(t) for t in thresholds]
    min_areas = [int(m) for m in min_areas]
    
    print(f"\nThresholds to test: {thresholds}")
    print(f"Min areas to test: {min_areas}")
    
    total_combinations = len(thresholds) * len(min_areas)
    print(f"Combinations per model: {total_combinations}")
    
    # Setup criterion
    criterion = MixedLoss(
        bce_weight=getattr(config.loss, 'bce_weight', 1.0),
        dice_weight=getattr(config.loss, 'dice_weight', 0.5),
        tversky_weight=getattr(config.loss, 'tversky_weight', 0.0),
        pos_weight=getattr(config.loss, 'pos_weight', 1.0),
        tversky_alpha=getattr(config.loss, 'tversky_alpha', 0.3),
        tversky_beta=getattr(config.loss, 'tversky_beta', 0.7),
    )
    
    # Build model once (weights will be reloaded for each checkpoint)
    model = build_cmsegnet(device=str(device))
    
    # Evaluate all checkpoints
    print("\n" + "-" * 70)
    print(f"Evaluating {len(checkpoint_files)} checkpoint(s)...")
    print("-" * 70)
    
    all_checkpoint_results = []
    
    for i, ckpt_path in enumerate(checkpoint_files, 1):
        print(f"\n[{i}/{len(checkpoint_files)}] Processing...")
        
        result = evaluate_single_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            config=config,
            thresholds=thresholds,
            min_areas=min_areas,
        )
        all_checkpoint_results.append(result)
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    # Sort by best oF1
    all_checkpoint_results_sorted = sorted(
        all_checkpoint_results, 
        key=lambda x: x['best_oF1'], 
        reverse=True
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - ALL CHECKPOINTS")
    print("=" * 70)
    print(f"\n{'Rank':<6}{'Checkpoint':<30}{'oF1':<10}{'Thresh':<10}{'MinArea':<10}{'Precision':<12}{'Recall':<10}")
    print("-" * 98)
    
    for i, r in enumerate(all_checkpoint_results_sorted, 1):
        ckpt_name = r['checkpoint'][:28] + '..' if len(r['checkpoint']) > 30 else r['checkpoint']
        print(f"{i:<6}{ckpt_name:<30}{r['best_oF1']:<10.4f}{r['best_threshold']:<10.2f}{r['best_min_area']:<10}{r['best_precision']:<12.4f}{r['best_recall']:<10.4f}")
    
    # Best overall
    best_overall = all_checkpoint_results_sorted[0]
    
    print("\n" + "=" * 70)
    print("BEST OVERALL MODEL")
    print("=" * 70)
    print(f"  Checkpoint: {best_overall['checkpoint']}")
    print(f"  Path:       {best_overall['checkpoint_path']}")
    print(f"  Epoch:      {best_overall['epoch']}")
    print(f"  Threshold:  {best_overall['best_threshold']:.2f}")
    print(f"  Min Area:   {best_overall['best_min_area']}")
    print(f"  oF1:        {best_overall['best_oF1']:.4f}")
    print(f"  Forged oF1: {best_overall['best_forged_oF1']:.4f}")
    print(f"  Precision:  {best_overall['best_precision']:.4f}")
    print(f"  Recall:     {best_overall['best_recall']:.4f}")
    print(f"  Dice:       {best_overall['best_dice']:.4f}")
    print(f"  IoU:        {best_overall['best_iou']:.4f}")
    print(f"  Loss:       {best_overall['best_loss']:.4f}")
    print("=" * 70)
    
    # Save results to JSON if output specified
    if args.output:
        results = {
            'directory': checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path),
            'split': split,
            'num_checkpoints': len(checkpoint_files),
            'grid_search': {
                'thresholds': thresholds,
                'min_areas': min_areas,
                'combinations_per_model': total_combinations,
            },
            'best_overall': {
                'checkpoint': best_overall['checkpoint'],
                'checkpoint_path': best_overall['checkpoint_path'],
                'epoch': best_overall['epoch'],
                'threshold': best_overall['best_threshold'],
                'min_area': best_overall['best_min_area'],
                'oF1': best_overall['best_oF1'],
                'forged_oF1': best_overall['best_forged_oF1'],
                'precision': best_overall['best_precision'],
                'recall': best_overall['best_recall'],
                'dice': best_overall['best_dice'],
                'iou': best_overall['best_iou'],
                'loss': best_overall['best_loss'],
            },
            'all_checkpoints': [
                {
                    'checkpoint': r['checkpoint'],
                    'epoch': r['epoch'],
                    'best_threshold': r['best_threshold'],
                    'best_min_area': r['best_min_area'],
                    'best_oF1': r['best_oF1'],
                    'best_forged_oF1': r['best_forged_oF1'],
                    'best_precision': r['best_precision'],
                    'best_recall': r['best_recall'],
                    'best_dice': r['best_dice'],
                    'best_iou': r['best_iou'],
                    'best_loss': r['best_loss'],
                    'all_grid_results': r['all_results'][:5],  # Top 5 per checkpoint
                }
                for r in all_checkpoint_results_sorted
            ],
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return best_overall


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='CMSeg-Net Training and Inference')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'inference'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for inference')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for inference results')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'train':
        main()
    else:
        run_inference(args)
