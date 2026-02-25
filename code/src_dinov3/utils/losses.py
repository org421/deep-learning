"""Fonctions de loss pour la detection de falsification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict


# =============================================================================
# BCE avec ponderation inverse
# =============================================================================

class ILW_BCEWithLogitsLoss(nn.Module):
    """BCE avec ponderation inverse des labels."""
    
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
        # comptage des 0 et 1 par sample
        num_of_zeros = (target == 0).view((target.shape[0], -1)).sum(dim=-1).clamp(min=1)
        num_of_ones = (target == 1).view((target.shape[0], -1)).sum(dim=-1).clamp(min=1)
        
        # poids inverses
        total = num_of_ones + num_of_zeros
        weight_zeros = torch.clamp(0.5 * total / num_of_zeros, min=self.min_weight, max=self.max_weight)
        weight_zero_elements = weight_zeros[:, None, None, None].expand_as(target)
        
        weight_ones = torch.clamp(0.5 * total / num_of_ones, min=self.min_weight, max=self.max_weight)
        weight_one_elements = weight_ones[:, None, None, None].expand_as(target)
        
        # application des poids
        weights_elements = torch.where(target == 0, weight_zero_elements, weight_one_elements)
        weights_elements = torch.where(
            target == self.ignore_label, 
            torch.zeros_like(weights_elements), 
            weights_elements
        )
        
        # BCE ponderee
        target_clean = target.clone().float()
        target_clean[target == self.ignore_label] = 0
        
        bce = F.binary_cross_entropy_with_logits(input, target_clean, reduction='none')
        ilw_bce = bce * weights_elements
        
        # moyenne sur les pixels valides
        valid_count = (target != self.ignore_label).sum()
        if valid_count > 0:
            return ilw_bce.sum() / valid_count
        else:
            return torch.tensor(0.0, device=input.device, requires_grad=True)


class BCEWithLogitsLossIgnore(nn.Module):
    """BCE classique avec ignore label."""
    
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
    """Focal loss pour gerer le desequilibre de classes."""
    
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
    """Dice loss pour la segmentation."""
    
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
# Loss combinee
# =============================================================================

class CombinedLoss(nn.Module):
    """Loss combinee (BCE + Dice + Focal)."""
    
    def __init__(
        self,
        use_dice: bool = True,
        use_bce: bool = True,
        use_focal: bool = False,
        use_ilw: bool = True,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
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
        
        if use_dice:
            self.dice_loss = DiceLoss(smooth=dice_smooth, ignore_label=ignore_label)
        
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
    
    def forward(self, input: Tensor, target: Tensor) -> Dict[str, Tensor]:
        losses = {}
        total_loss = torch.tensor(0.0, device=input.device, requires_grad=True)
        
        if self.use_dice:
            dice = self.dice_loss(input, target)
            losses['dice'] = dice
            total_loss = total_loss + self.dice_weight * dice
        
        if self.use_bce:
            bce = self.bce_loss(input, target)
            losses['bce'] = bce
            total_loss = total_loss + self.bce_weight * bce
        
        if self.use_focal:
            focal = self.focal_loss(input, target)
            losses['focal'] = focal
            total_loss = total_loss + self.focal_weight * focal
        
        losses['total'] = total_loss
        return losses


# =============================================================================
# Pixel accuracy
# =============================================================================

class PixelAccWithIgnoreLabel(nn.Module):
    """Pixel accuracy en ignorant certains labels."""
    
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
# Construction de la loss depuis la config
# =============================================================================

def build_loss_from_config(config) -> nn.Module:
    """Construit la loss a partir de la config."""
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
            focal_alpha=loss_config.focal.alpha,
            focal_gamma=loss_config.focal.gamma,
            dice_smooth=loss_config.dice.smooth,
            ignore_label=ignore_label,
            ilw_max_weight=ilw_max_weight,
            ilw_min_weight=ilw_min_weight,
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_config.type}")

# =============================================================================
# Loss contrastive region-to-region
# =============================================================================

class R2R_ContrastiveLoss_TwoSource(nn.Module):
    """Loss contrastive region-to-region."""
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