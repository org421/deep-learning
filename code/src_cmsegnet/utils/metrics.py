# -*- coding: utf-8 -*-
"""
Metrics for CMSeg-Net evaluation.

Includes pixel-level metrics (Dice, IoU, F1) and detection metrics.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_pixel_metrics(
    pred: np.ndarray, 
    target: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute pixel-level metrics.
    
    Args:
        pred: Predicted mask (probabilities) [H, W] or [B, H, W]
        target: Ground truth mask (binary) [H, W] or [B, H, W]
        threshold: Threshold for binarization
        
    Returns:
        Dict with dice, iou, precision, recall, f1, accuracy, specificity
    """
    # Binarize prediction
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0.5).astype(np.uint8)
    
    # Flatten
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # Compute TP, FP, FN, TN
    tp = np.sum(pred_flat & target_flat)
    fp = np.sum(pred_flat & ~target_flat)
    fn = np.sum(~pred_flat & target_flat)
    tn = np.sum(~pred_flat & ~target_flat)
    
    # Metrics
    eps = 1e-7
    
    # Dice / F1
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    
    # IoU / Jaccard
    iou = (tp + eps) / (tp + fp + fn + eps)
    
    # Precision
    precision = (tp + eps) / (tp + fp + eps)
    
    # Recall / Sensitivity
    recall = (tp + eps) / (tp + fn + eps)
    
    # F1 (same as Dice for binary)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    
    # Accuracy
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    
    # Specificity
    specificity = (tn + eps) / (tn + fp + eps)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity,
    }


def compute_batch_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics for a batch of predictions.
    
    Args:
        preds: Predictions [B, 1, H, W] or [B, H, W]
        targets: Targets [B, 1, H, W] or [B, H, W]
        threshold: Threshold for binarization
        
    Returns:
        Dict with averaged metrics
    """
    # Convert to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Remove channel dim if present
    if len(preds.shape) == 4:
        preds = preds[:, 0]
    if len(targets.shape) == 4:
        targets = targets[:, 0]
    
    # Compute metrics for each sample
    metrics_list = []
    for i in range(len(preds)):
        metrics = compute_pixel_metrics(preds[i], targets[i], threshold)
        metrics_list.append(metrics)
    
    # Average
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    return avg_metrics


def compute_detection_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 100
) -> Dict[str, float]:
    """
    Compute detection-level metrics (for oF1 calculation).
    
    A detection is correct if it overlaps with ground truth.
    
    Args:
        preds: Predictions [B, H, W]
        targets: Targets [B, H, W]
        threshold: Threshold for binarization
        min_area: Minimum area for a valid region
        
    Returns:
        Dict with detection metrics
    """
    from scipy import ndimage
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, target in zip(preds, targets):
        pred_binary = (pred > threshold).astype(np.uint8)
        target_binary = (target > 0.5).astype(np.uint8)
        
        # Label connected components
        pred_labels, num_pred = ndimage.label(pred_binary)
        target_labels, num_target = ndimage.label(target_binary)
        
        # Filter by min area
        for i in range(1, num_pred + 1):
            if np.sum(pred_labels == i) < min_area:
                pred_labels[pred_labels == i] = 0
        
        for i in range(1, num_target + 1):
            if np.sum(target_labels == i) < min_area:
                target_labels[target_labels == i] = 0
        
        # Re-label
        pred_labels, num_pred = ndimage.label(pred_labels > 0)
        target_labels, num_target = ndimage.label(target_labels > 0)
        
        # Match predictions to targets
        matched_targets = set()
        matched_preds = set()
        
        for i in range(1, num_pred + 1):
            pred_mask = pred_labels == i
            for j in range(1, num_target + 1):
                target_mask = target_labels == j
                if np.any(pred_mask & target_mask):
                    matched_preds.add(i)
                    matched_targets.add(j)
        
        tp = len(matched_preds)
        fp = num_pred - len(matched_preds)
        fn = num_target - len(matched_targets)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    eps = 1e-7
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    
    return {
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
    }
