# -*- coding: utf-8 -*-
"""
Loss Functions pour CMSeg-Net.

Inclut:
- MixedLoss (BCE + Dice) - original
- TverskyLoss - pour améliorer le recall
- pos_weight support - pour le déséquilibre de classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss pour segmentation.
    Fidèle au code original de CMSeg-Net.
    """
    def __init__(self, smooth: float = 1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Prédictions (avant sigmoid) [B, 1, H, W]
            targets: Masques cibles [B, 1, H, W]
            
        Returns:
            Dice loss scalaire
        """
        bs = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(bs, -1)
        m2 = targets.view(bs, -1)

        intersection = (m1 * m2).sum(dim=1)
        union = m1.sum(dim=1) + m2.sum(dim=1) + self.smooth

        score = (2. * intersection + self.smooth) / union
        loss = 1 - score.mean()

        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Généralisation du Dice Loss.
    
    Permet de contrôler séparément la pénalité pour FP et FN.
    Utile quand le recall est faible (comme dans ton cas: 34%).
    
    Formule: TL = 1 - (TP + smooth) / (TP + α*FP + β*FN + smooth)
    
    Args:
        alpha: Poids des faux positifs (faible = tolérant aux FP)
        beta: Poids des faux négatifs (élevé = sévère sur FN, boost recall)
        smooth: Terme de lissage pour éviter division par zéro
        
    Exemples:
        alpha=0.5, beta=0.5 → équivalent au Dice Loss
        alpha=0.3, beta=0.7 → favorise le recall (recommandé pour toi)
        alpha=0.2, beta=0.8 → favorise encore plus le recall
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Prédictions (avant sigmoid) [B, 1, H, W]
            targets: Masques cibles [B, 1, H, W]
            
        Returns:
            Tversky loss scalaire
        """
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (probs_flat * targets_flat).sum()
        FP = (probs_flat * (1 - targets_flat)).sum()
        FN = ((1 - probs_flat) * targets_flat).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Combine Tversky avec mécanisme Focal.
    
    Ajoute un exposant gamma pour se concentrer sur les exemples difficiles.
    
    Args:
        alpha: Poids des FP
        beta: Poids des FN
        gamma: Exposant focal (1.0 = Tversky normal, >1 = focus sur exemples difficiles)
        smooth: Terme de lissage
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, 
                 gamma: float = 1.0, smooth: float = 1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        TP = (probs_flat * targets_flat).sum()
        FP = (probs_flat * (1 - targets_flat)).sum()
        FN = ((1 - probs_flat) * targets_flat).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Focal component
        focal_tversky = torch.pow((1 - tversky), self.gamma)
        
        return focal_tversky


class MixedLoss(nn.Module):
    """
    Mixed Loss = BCE + Dice + Tversky (optionnel).
    
    Version améliorée avec:
    - pos_weight pour BCE (gère le déséquilibre de classes)
    - Tversky Loss optionnel (améliore le recall)
    
    Args:
        bce_weight: Poids de la BCE loss
        dice_weight: Poids de la Dice loss  
        tversky_weight: Poids de la Tversky loss (0 = désactivé)
        pos_weight: Poids des positifs dans BCE (>1 pénalise plus les FN)
        tversky_alpha: Alpha pour Tversky (poids FP)
        tversky_beta: Beta pour Tversky (poids FN)
    """
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 0.5,
        tversky_weight: float = 0.0,
        pos_weight: float = 1.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super(MixedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.pos_weight_value = pos_weight  # Stocker la valeur float
        
        self.soft_dice = SoftDiceLoss()
        
        # Tversky (optionnel)
        if tversky_weight > 0:
            self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        else:
            self.tversky = None

    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> tuple:
        """
        Args:
            y_pred: Prédictions (logits) [B, 1, H, W]
            y_true: Masques cibles [B, 1, H, W]
            
        Returns:
            Tuple (total_loss, dice_loss, bce_loss)
        """
        # BCE loss avec pos_weight sur le même device que y_pred
        if self.pos_weight_value != 1.0:
            pos_weight = torch.tensor([self.pos_weight_value], device=y_pred.device, dtype=y_pred.dtype)
            bce_loss = F.binary_cross_entropy_with_logits(
                y_pred, y_true, pos_weight=pos_weight
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        
        dice_loss = self.soft_dice(y_pred, y_true)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        # Ajouter Tversky si activé
        if self.tversky is not None and self.tversky_weight > 0:
            tversky_loss = self.tversky(y_pred, y_true)
            total_loss = total_loss + self.tversky_weight * tversky_loss
        
        return total_loss, dice_loss, bce_loss


class DiceLoss(nn.Module):
    """Dice Loss standard."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss pour gérer le déséquilibre de classes."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-7, max=1-1e-7)
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        loss = focal_weight * bce
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Loss avec BCE, Dice, Tversky et optionnellement Focal.
    
    Plus flexible que MixedLoss pour expérimentation.
    """
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 0.5,
        tversky_weight: float = 0.0,
        focal_weight: float = 0.0,
        pos_weight: float = 1.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.pos_weight_value = pos_weight  # Stocker la valeur float
        
        self.dice = SoftDiceLoss(smooth=smooth)
        
        if tversky_weight > 0:
            self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        else:
            self.tversky = None
        
        if focal_weight > 0:
            self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.focal = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        if self.bce_weight > 0:
            if self.pos_weight_value != 1.0:
                pos_weight = torch.tensor([self.pos_weight_value], device=pred.device, dtype=pred.dtype)
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred, target, pos_weight=pos_weight
                )
            else:
                bce_loss = F.binary_cross_entropy_with_logits(pred, target)
            loss += self.bce_weight * bce_loss
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(pred, target)
        
        if self.tversky_weight > 0 and self.tversky is not None:
            loss += self.tversky_weight * self.tversky(pred, target)
        
        if self.focal_weight > 0 and self.focal is not None:
            loss += self.focal_weight * self.focal(pred, target)
        
        return loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory pour créer la fonction de loss.
    
    Args:
        loss_type: Type de loss ('mixed', 'dice', 'bce', 'focal', 'tversky', 'combined')
        **kwargs: Arguments supplémentaires
        
    Returns:
        Module de loss
    """
    if loss_type == 'mixed':
        return MixedLoss(
            bce_weight=kwargs.get('bce_weight', 1.0),
            dice_weight=kwargs.get('dice_weight', 0.5),
            tversky_weight=kwargs.get('tversky_weight', 0.0),
            pos_weight=kwargs.get('pos_weight', 1.0),
            tversky_alpha=kwargs.get('tversky_alpha', 0.3),
            tversky_beta=kwargs.get('tversky_beta', 0.7),
        )
    elif loss_type == 'dice':
        return DiceLoss(smooth=kwargs.get('smooth', 1.0))
    elif loss_type == 'bce':
        pos_weight = kwargs.get('pos_weight', 1.0)
        if pos_weight != 1.0:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0)
        )
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=kwargs.get('tversky_alpha', 0.3),
            beta=kwargs.get('tversky_beta', 0.7)
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            bce_weight=kwargs.get('bce_weight', 1.0),
            dice_weight=kwargs.get('dice_weight', 0.5),
            tversky_weight=kwargs.get('tversky_weight', 0.0),
            focal_weight=kwargs.get('focal_weight', 0.0),
            pos_weight=kwargs.get('pos_weight', 1.0),
            tversky_alpha=kwargs.get('tversky_alpha', 0.3),
            tversky_beta=kwargs.get('tversky_beta', 0.7),
            focal_alpha=kwargs.get('focal_alpha', 0.25),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            smooth=kwargs.get('smooth', 1.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
