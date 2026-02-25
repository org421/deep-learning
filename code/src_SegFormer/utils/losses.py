"""
Loss functions for Scientific Image Forgery Detection.

Features:
- Configurable losses: BCE, Dice, Focal, Combined
- Inverse Label Weighting (ILW) for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict


# =============================================================================
# Inverse Label Weighting BCE Loss
# =============================================================================

class ILW_BCEWithLogitsLoss(nn.Module):
    """
    Inverse Label Weighting BCE Loss.
    
    Automatically balances loss based on class distribution in each sample.
    This helps handle severe class imbalance in forgery detection where
    forged regions are typically much smaller than authentic regions.
    """
    
    def __init__(
        self, 
        reduction: str = 'mean', 
        ignore_label: int = -1,
        max_weight: float = 10.0,
        min_weight: float = 0.1,
    ) -> None:
        super().__init__()
        assert reduction == 'mean', "Only 'mean' reduction is supported"
        self.reduction = reduction
        self.ignore_label = ignore_label
        self.max_weight = max_weight
        self.min_weight = min_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Count zeros and ones per sample
        num_of_zeros = (target == 0).view((target.shape[0], -1)).sum(dim=-1).clamp(min=1)
        num_of_ones = (target == 1).view((target.shape[0], -1)).sum(dim=-1).clamp(min=1)
        
        # Calculate inverse label weights
        total = num_of_ones + num_of_zeros
        weight_zeros = torch.clamp(0.5 * total / num_of_zeros, min=self.min_weight, max=self.max_weight)
        weight_zero_elements = weight_zeros[:, None, None, None].expand_as(target)
        
        weight_ones = torch.clamp(0.5 * total / num_of_ones, min=self.min_weight, max=self.max_weight)
        weight_one_elements = weight_ones[:, None, None, None].expand_as(target)
        
        # Apply weights based on target value
        weights_elements = torch.where(target == 0, weight_zero_elements, weight_one_elements)
        weights_elements = torch.where(
            target == self.ignore_label, 
            torch.zeros_like(weights_elements), 
            weights_elements
        )
        
        # Compute weighted BCE
        target_clean = target.clone().float()
        target_clean[target == self.ignore_label] = 0
        
        bce = F.binary_cross_entropy_with_logits(input, target_clean, reduction='none')
        ilw_bce = bce * weights_elements
        
        # Compute mean over valid pixels
        valid_count = (target != self.ignore_label).sum()
        if valid_count > 0:
            return ilw_bce.sum() / valid_count
        else:
            return torch.tensor(0.0, device=input.device, requires_grad=True)


class BCEWithLogitsLossIgnore(nn.Module):
    """Standard BCE Loss with ignore label support."""
    
    def __init__(self, ignore_label: int = -1) -> None:
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        valid_mask = (target != self.ignore_label).float()
        target_clean = target.clone().float()
        target_clean[target == self.ignore_label] = 0
        
        bce = F.binary_cross_entropy_with_logits(input, target_clean, reduction='none')
        bce = bce * valid_mask
        
        valid_count = valid_mask.sum()
        if valid_count > 0:
            return bce.sum() / valid_count
        else:
            return torch.tensor(0.0, device=input.device, requires_grad=True)


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces loss for well-classified examples, focusing on hard examples.
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean', 
        ignore_label: int = -1
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_label = ignore_label
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        valid_mask = (target != self.ignore_label).float()
        target_clean = target.clone().float()
        target_clean[target == self.ignore_label] = 0
        
        p = torch.sigmoid(input)
        p_t = p * target_clean + (1 - p) * (1 - target_clean)
        
        alpha_t = self.alpha * target_clean + (1 - self.alpha) * (1 - target_clean)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        bce = F.binary_cross_entropy_with_logits(input, target_clean, reduction='none')
        focal_loss = focal_weight * bce * valid_mask
        
        if self.reduction == 'mean':
            return focal_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Dice Loss
# =============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0, ignore_label: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_label = ignore_label
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        valid_mask = (target != self.ignore_label).float()
        
        input_sigmoid = torch.sigmoid(input) * valid_mask
        target_clean = target.clone().float()
        target_clean[target == self.ignore_label] = 0
        target_clean = target_clean * valid_mask
        
        intersection = (input_sigmoid * target_clean).sum(dim=(2, 3))
        union = input_sigmoid.sum(dim=(2, 3)) + target_clean.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


# =============================================================================
# Focal Tversky Loss (optimized for sparse forgery masks)
# =============================================================================

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for forgery detection.

    Tversky index generalizes Dice by allowing asymmetric FP/FN weighting.
    Focal exponent focuses training on hard examples.

    Args:
        alpha: Weight for false positives (lower = tolerate more FP)
        beta: Weight for false negatives (higher = penalize missed forgeries more)
        gamma: Focal exponent (lower = focus more on hard examples)
        smooth: Smoothing factor to avoid division by zero
        ignore_label: Label to ignore in loss computation
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
        ignore_label: int = -1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_label = ignore_label

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        valid_mask = (target != self.ignore_label).float()

        pred = torch.sigmoid(input) * valid_mask
        target_clean = target.clone().float()
        target_clean[target == self.ignore_label] = 0
        target_clean = target_clean * valid_mask

        TP = (pred * target_clean).sum(dim=(2, 3))
        FP = ((1 - target_clean) * pred).sum(dim=(2, 3))
        FN = (target_clean * (1 - pred)).sum(dim=(2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky.mean()


# =============================================================================
# Edge Loss (Dice-based for boundary supervision)
# =============================================================================

class EdgeLoss(nn.Module):
    """
    Edge Loss for forgery boundary supervision.

    Supervises the edge head with morphological gradient of the GT mask.
    Uses Dice loss on the edge predictions.
    """

    def __init__(self, smooth: float = 1.0, ignore_label: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_label = ignore_label
        # Morphological kernel for edge extraction (dilation - erosion)
        kernel = torch.ones(1, 1, 5, 5)
        self.register_buffer('morph_kernel', kernel)

    def _extract_edges(self, mask: Tensor) -> Tensor:
        """Extract edges from binary mask using morphological gradient."""
        mask_float = mask.float()
        if mask_float.dim() == 3:
            mask_float = mask_float.unsqueeze(1)
        # Dilation
        dilated = F.max_pool2d(mask_float, kernel_size=5, stride=1, padding=2)
        # Erosion (= 1 - dilation of (1 - mask))
        eroded = 1.0 - F.max_pool2d(1.0 - mask_float, kernel_size=5, stride=1, padding=2)
        # Edge = dilation - erosion
        edge = dilated - eroded
        return edge.clamp(0, 1)

    def forward(self, edge_logits: Tensor, target_mask: Tensor) -> Tensor:
        """
        Args:
            edge_logits: (B, 1, H, W) predicted edge logits
            target_mask: (B, 1, H, W) GT segmentation mask
        """
        valid_mask = (target_mask != self.ignore_label).float()

        # Extract edge GT from mask
        target_clean = target_mask.clone().float()
        target_clean[target_mask == self.ignore_label] = 0
        edge_gt = self._extract_edges(target_clean) * valid_mask

        # Dice on edges
        pred = torch.sigmoid(edge_logits) * valid_mask
        intersection = (pred * edge_gt).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + edge_gt.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# =============================================================================
# Combined Loss (with Focal Tversky + Edge + Classification)
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss with forensic-specific components.

    Components:
    - ILW-BCE: class-balanced binary cross-entropy (main)
    - Focal Tversky: asymmetric FP/FN weighting for sparse masks (replaces Dice)
    - Edge Dice: boundary supervision for forgery edges (auxiliary)
    - Classification BCE: image-level forged vs authentic (auxiliary)
    - Focal: optional focal loss
    """

    def __init__(
        self,
        use_dice: bool = True,
        use_bce: bool = True,
        use_focal: bool = False,
        use_ilw: bool = True,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        focal_weight: float = 1.0,
        edge_weight: float = 0.5,
        cls_weight: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 0.75,
        dice_smooth: float = 1.0,
        ignore_label: int = -1,
        ilw_max_weight: float = 10.0,
        ilw_min_weight: float = 0.1,
    ):
        super().__init__()
        self.use_dice = use_dice
        self.use_bce = use_bce
        self.use_focal = use_focal

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.edge_weight = edge_weight
        self.cls_weight = cls_weight

        # Focal Tversky replaces standard Dice
        if use_dice:
            self.tversky_loss = FocalTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=tversky_gamma,
                smooth=dice_smooth,
                ignore_label=ignore_label,
            )

        if use_bce:
            if use_ilw:
                self.bce_loss = ILW_BCEWithLogitsLoss(
                    ignore_label=ignore_label,
                    max_weight=ilw_max_weight,
                    min_weight=ilw_min_weight,
                )
            else:
                self.bce_loss = BCEWithLogitsLossIgnore(ignore_label=ignore_label)

        if use_focal:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                ignore_label=ignore_label
            )

        # Edge loss (always active)
        self.edge_loss = EdgeLoss(smooth=dice_smooth, ignore_label=ignore_label)

        # Classification loss (always active)
        self.cls_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        edge_logits: Tensor = None,
        cls_logits: Tensor = None,
        cls_target: Tensor = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            input: (B, 1, H, W) segmentation logits
            target: (B, 1, H, W) GT mask
            edge_logits: (B, 1, H, W) edge prediction logits
            cls_logits: (B, 1) image-level classification logits
            cls_target: (B,) image-level GT (1=forged, 0=authentic)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=input.device, requires_grad=True)

        # Focal Tversky (replaces Dice — better for sparse forgery masks)
        if self.use_dice:
            tversky = self.tversky_loss(input, target)
            losses['tversky'] = tversky
            total_loss = total_loss + self.dice_weight * tversky

        # ILW-BCE
        if self.use_bce:
            bce = self.bce_loss(input, target)
            losses['bce'] = bce
            total_loss = total_loss + self.bce_weight * bce

        # Focal
        if self.use_focal:
            focal = self.focal_loss(input, target)
            losses['focal'] = focal
            total_loss = total_loss + self.focal_weight * focal

        # Edge supervision
        if edge_logits is not None:
            edge = self.edge_loss(edge_logits, target)
            losses['edge'] = edge
            total_loss = total_loss + self.edge_weight * edge

        # Image-level classification
        if cls_logits is not None and cls_target is not None:
            cls = self.cls_loss(cls_logits.squeeze(-1), cls_target.float())
            losses['cls'] = cls
            total_loss = total_loss + self.cls_weight * cls

        losses['total'] = total_loss
        return losses


# =============================================================================
# Pixel Accuracy Metric
# =============================================================================

class PixelAccWithIgnoreLabel(nn.Module):
    """Compute pixel accuracy while ignoring specified label."""
    
    def __init__(self, ignore_label: int = -1):
        super().__init__()
        self.ignore_label = ignore_label
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        pred = (input >= 0).long()
        
        correct = (pred == target)
        valid_mask = (target != self.ignore_label)
        correct_valid = correct & valid_mask
        
        correct_count = correct_valid.view(correct_valid.shape[0], -1).sum(dim=1)
        valid_count = valid_mask.view(valid_mask.shape[0], -1).sum(dim=1)
        
        acc = correct_count.float() / valid_count.float().clamp(min=1)
        return acc


# =============================================================================
# Loss Builder from Config
# =============================================================================

def build_loss_from_config(config) -> nn.Module:
    """Build loss function from configuration."""
    loss_config = config.loss
    ignore_label = loss_config.ignore_label
    
    ilw_max_weight = getattr(loss_config.bce, 'ilw_max_weight', 10.0)
    ilw_min_weight = getattr(loss_config.bce, 'ilw_min_weight', 0.1)
    
    if loss_config.type == "bce":
        if loss_config.bce.use_ilw:
            return ILW_BCEWithLogitsLoss(
                ignore_label=ignore_label,
                max_weight=ilw_max_weight,
                min_weight=ilw_min_weight,
            )
        else:
            return BCEWithLogitsLossIgnore(ignore_label=ignore_label)
    
    elif loss_config.type == "dice":
        return DiceLoss(
            smooth=loss_config.dice.smooth,
            ignore_label=ignore_label
        )
    
    elif loss_config.type == "focal":
        return FocalLoss(
            alpha=loss_config.focal.alpha,
            gamma=loss_config.focal.gamma,
            ignore_label=ignore_label
        )
    
    elif loss_config.type == "combined":
        return CombinedLoss(
            use_dice=loss_config.dice.enabled,
            use_bce=loss_config.bce.enabled,
            use_focal=loss_config.focal.enabled,
            use_ilw=loss_config.bce.use_ilw,
            dice_weight=loss_config.dice.weight,
            bce_weight=loss_config.bce.weight,
            focal_weight=loss_config.focal.weight,
            edge_weight=getattr(loss_config, 'edge_weight', 0.5),
            cls_weight=getattr(loss_config, 'cls_weight', 0.3),
            focal_alpha=loss_config.focal.alpha,
            focal_gamma=loss_config.focal.gamma,
            tversky_alpha=getattr(loss_config, 'tversky_alpha', 0.3),
            tversky_beta=getattr(loss_config, 'tversky_beta', 0.7),
            tversky_gamma=getattr(loss_config, 'tversky_gamma', 0.75),
            dice_smooth=loss_config.dice.smooth,
            ignore_label=ignore_label,
            ilw_max_weight=ilw_max_weight,
            ilw_min_weight=ilw_min_weight,
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_config.type}")

# =============================================================================
# Region to Region Contrastive Loss
# =============================================================================

class R2R_ContrastiveLoss_TwoSource(nn.Module):
    """
    Region to Region Contrastive Loss for two sources used in SAFIRE pre-training.
    """
    def __init__(self, temperature: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    @staticmethod
    @torch.no_grad()
    def _validate_inputs(pos: torch.Tensor, neg: torch.Tensor) -> None:
        if pos.dim() != 2 or neg.dim() != 2:
            raise ValueError("Both <positive_keys> and <negative_keys> must be 2-D tensors (N, D).")
        if pos.size(1) != neg.size(1):
            raise ValueError("Positive and negative keys must share the same feature dimension.")

    def forward(self, positive_keys: torch.Tensor, negative_keys: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(positive_keys, negative_keys)
        positive_keys = F.normalize(positive_keys, dim=-1)
        negative_keys = F.normalize(negative_keys, dim=-1)
        
        pos_sim = positive_keys @ positive_keys.T
        neg_sim = negative_keys @ negative_keys.T
        
        pos_mask = (torch.ones_like(pos_sim) - torch.eye(pos_sim.size(0), device=pos_sim.device)).bool()
        neg_mask = (torch.ones_like(neg_sim) - torch.eye(neg_sim.size(0), device=neg_sim.device)).bool()
        
        pos_sim = pos_sim.masked_select(pos_mask).view(pos_sim.size(0), -1)
        neg_sim = neg_sim.masked_select(neg_mask).view(neg_sim.size(0), -1)
        
        pos_anchor_logit = pos_sim.mean(dim=1, keepdim=True)
        neg_anchor_logit = neg_sim.mean(dim=1, keepdim=True)
        
        cross_sim = positive_keys @ negative_keys.T
        
        pos_logits = torch.cat([pos_anchor_logit, cross_sim], dim=1)
        neg_logits = torch.cat([neg_anchor_logit, cross_sim.T], dim=1)
        
        labels_pos = torch.zeros(pos_logits.size(0), dtype=torch.long, device=pos_logits.device)
        labels_neg = torch.zeros(neg_logits.size(0), dtype=torch.long, device=neg_logits.device)
        
        loss_pos = F.cross_entropy(pos_logits / self.temperature, labels_pos, reduction=self.reduction)
        loss_neg = F.cross_entropy(neg_logits / self.temperature, labels_neg, reduction=self.reduction)
        
        return loss_pos + loss_neg