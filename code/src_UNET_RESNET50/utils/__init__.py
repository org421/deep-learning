"""
SAFIRE Utils module.
"""

from .losses import (
    ILW_BCEWithLogitsLoss,
    BCEWithLogitsLossIgnore,
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    PixelAccWithIgnoreLabel,
    R2R_ContrastiveLoss_TwoSource,
    build_loss_from_config,
)

from .kaggle_metric import (
    KaggleMetricCalculator,
    mask_to_instances,
    rle_encode,
    rle_decode,
    evaluate_single_image,
    calculate_competition_score,
)

from .metrics import AverageMeter

from .config_loader import Config, get_device_info

from .ema import EMAModel

__all__ = [
    # Losses
    "ILW_BCEWithLogitsLoss",
    "BCEWithLogitsLossIgnore", 
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "PixelAccWithIgnoreLabel",
    "R2R_ContrastiveLoss_TwoSource",
    "build_loss_from_config",
    # Kaggle metric
    "KaggleMetricCalculator",
    "mask_to_instances",
    "rle_encode",
    "rle_decode",
    "evaluate_single_image",
    "calculate_competition_score",
    # Other
    "AverageMeter",
    "Config",
    "get_device_info",
    "EMAModel",
]
