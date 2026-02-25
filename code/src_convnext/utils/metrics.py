"""
Metrics for forgery detection evaluation.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef


def pixel_auc(pred: np.ndarray, gt: np.ndarray, ignore_value: int = -1) -> float:
    """
    Compute pixel-level AUC.
    
    Args:
        pred: Prediction array (H, W) with values in [0, 1]
        gt: Ground truth array (H, W) with values {0, 1} or {0, 1, -1}
        ignore_value: Value to ignore in computation
        
    Returns:
        AUC score
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    
    # Filter ignored values
    mask = gt_flat != ignore_value
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    if len(np.unique(gt_flat)) < 2:
        return 0.5
    
    try:
        return roc_auc_score(gt_flat, pred_flat)
    except:
        return 0.5


def pixel_AP(pred: np.ndarray, gt: np.ndarray, ignore_value: int = -1) -> float:
    """
    Compute pixel-level Average Precision.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    
    mask = gt_flat != ignore_value
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    if len(np.unique(gt_flat)) < 2:
        return 0.0
    
    try:
        return average_precision_score(gt_flat, pred_flat)
    except:
        return 0.0


def f1_fixed_tamp(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, ignore_value: int = -1) -> float:
    """
    Compute F1 score at fixed threshold.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    
    mask = gt_flat != ignore_value
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    pred_binary = (pred_flat >= threshold).astype(int)
    
    try:
        return f1_score(gt_flat, pred_binary)
    except:
        return 0.0


def f1_best_tamp(pred: np.ndarray, gt: np.ndarray, ignore_value: int = -1) -> float:
    """
    Compute best F1 score over all thresholds.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    
    mask = gt_flat != ignore_value
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    if len(np.unique(gt_flat)) < 2:
        return 0.0
    
    # Try multiple thresholds
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0.0
    
    for t in thresholds:
        pred_binary = (pred_flat >= t).astype(int)
        try:
            f1 = f1_score(gt_flat, pred_binary)
            best_f1 = max(best_f1, f1)
        except:
            continue
    
    return best_f1


def mcc_tamp(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, ignore_value: int = -1) -> float:
    """
    Compute Matthews Correlation Coefficient.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    
    mask = gt_flat != ignore_value
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    pred_binary = (pred_flat >= threshold).astype(int)
    
    try:
        return matthews_corrcoef(gt_flat, pred_binary)
    except:
        return 0.0


def iou_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, ignore_value: int = -1) -> float:
    """
    Compute Intersection over Union.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    
    mask = gt_flat != ignore_value
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    pred_binary = (pred_flat >= threshold).astype(int)
    
    intersection = np.logical_and(pred_binary, gt_flat).sum()
    union = np.logical_or(pred_binary, gt_flat).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def dice_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, ignore_value: int = -1) -> float:
    """
    Compute Dice coefficient.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    
    mask = gt_flat != ignore_value
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    pred_binary = (pred_flat >= threshold).astype(int)
    
    intersection = np.logical_and(pred_binary, gt_flat).sum()
    total = pred_binary.sum() + gt_flat.sum()
    
    if total == 0:
        return 1.0
    
    return 2 * intersection / total


def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, ignore_value: int = -1) -> dict:
    """
    Compute confusion matrix components.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    
    mask = gt_flat != ignore_value
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    pred_binary = (pred_flat >= threshold).astype(int)
    
    TP = np.logical_and(pred_binary == 1, gt_flat == 1).sum()
    TN = np.logical_and(pred_binary == 0, gt_flat == 0).sum()
    FP = np.logical_and(pred_binary == 1, gt_flat == 0).sum()
    FN = np.logical_and(pred_binary == 0, gt_flat == 1).sum()
    
    return {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'accuracy': (TP + TN) / max(1, TP + TN + FP + FN),
        'precision': TP / max(1, TP + FP),
        'recall': TP / max(1, TP + FN),
        'specificity': TN / max(1, TN + FP),
    }


def adjusted_rand_index(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index for multi-class segmentation.
    """
    try:
        from sklearn.metrics import adjusted_rand_score
        pred_flat = pred.ravel()
        gt_flat = gt.ravel()
        return adjusted_rand_score(gt_flat, pred_flat)
    except:
        return 0.0


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
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def average(self):
        return self.avg


class MetricsTracker:
    """Track multiple metrics during evaluation."""
    
    def __init__(self, metric_names: list = None):
        if metric_names is None:
            metric_names = ['AUC', 'AP', 'F1_fixed', 'F1_best', 'MCC', 'IoU', 'Dice']
        
        self.metric_names = metric_names
        self.meters = {name: AverageMeter() for name in metric_names}
        
        self.metric_fns = {
            'AUC': pixel_auc,
            'AP': pixel_AP,
            'F1_fixed': f1_fixed_tamp,
            'F1_best': f1_best_tamp,
            'MCC': mcc_tamp,
            'IoU': iou_score,
            'Dice': dice_score,
        }
    
    def update(self, pred: np.ndarray, gt: np.ndarray):
        """Update all metrics with a new prediction-ground truth pair."""
        for name in self.metric_names:
            if name in self.metric_fns:
                value = self.metric_fns[name](pred, gt)
                self.meters[name].update(value)
    
    def get_averages(self) -> dict:
        """Get average values for all metrics."""
        return {name: meter.average() for name, meter in self.meters.items()}
    
    def reset(self):
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()
    
    def __str__(self) -> str:
        """String representation of current averages."""
        lines = []
        for name, meter in self.meters.items():
            lines.append(f"{name}: {meter.average():.4f}")
        return " | ".join(lines)
