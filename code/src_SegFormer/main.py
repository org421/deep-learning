"""
Main Training Script for Image Forgery Detection.
Supports: UCM-NetV2 and SegFormer models.
Optimisé pour 12 GB VRAM.

Usage:
    python main.py --mode train
    python main.py --mode inference --checkpoint path/to/model.pth
    python main.py --mode inference --checkpoint path/to/model.pth --split test
"""

import os
import argparse
import gc
import shutil
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

warnings.filterwarnings('ignore')
os.environ['ALBUMENTATIONS_SKIP_VERSION_CHECK'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# ── imports corrigés (non dépréciés) ─────────────────────────────────────
from torch.amp import autocast          # ← was torch.cuda.amp.autocast
from torch.cuda.amp import GradScaler   # ← GradScaler reste ici pour l'instant
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.config_loader import Config, get_device_info
from utils.losses import (
    ILW_BCEWithLogitsLoss,
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    PixelAccWithIgnoreLabel,
)
from utils.metrics import AverageMeter
from utils.kaggle_metric import KaggleMetricCalculator, evaluate_single_image
from utils.ema import EMAModel
from Data_.dataset import ForgeryDataset

# ── Model imports ─────────────────────────────────────────────────────────
from networks.ucm_netv2_model import UCMNetV2, build_ucm_netv2
from networks.ucm_netv2_model import load_checkpoint as ucm_load_checkpoint
from networks.segformer_model import SegFormer, build_segformer
from networks.segformer_model import load_checkpoint as segformer_load_checkpoint


def _is_segformer_variant(variant: str) -> bool:
    """Check if the model variant is a SegFormer variant."""
    return variant.startswith("segformer_")


# =============================================================================
# Déterminer le dtype AMP optimal selon la GPU
# =============================================================================
def _get_amp_dtype(device: str) -> torch.dtype:
    """bf16 si supporté (Ampere+), sinon fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

AMP_DTYPE: torch.dtype = torch.float16   # sera mis à jour après init GPU


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0001, mode: str = 'min'):
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


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = {
            'train_loss': [],
            'train_bce': [],
            'train_dice': [],
            'gradient_norm': [],
            'val_loss': [],
            'val_oF1': [],
            'val_precision': [],
            'val_recall': [],
            'val_dice': [],
            'val_iou': [],
            'val_bce': [],
            'learning_rate': [],
            'backbone_frozen': [],
        }

    def log(self, epoch: int, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def save(self):
        with open(self.save_path / 'metrics_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        if self.history['train_loss']:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
            axes[0, 0].set_title('Train Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].legend()

        if self.history['val_loss']:
            axes[0, 1].plot(self.history['val_loss'], label='Val Loss', color='orange')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].legend()

        if self.history['val_oF1']:
            axes[0, 2].plot(self.history['val_oF1'], label='Val oF1', color='green')
            axes[0, 2].set_title('Validation oF1')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].legend()

        if self.history['val_precision'] and self.history['val_recall']:
            axes[1, 0].plot(self.history['val_precision'], label='Precision')
            axes[1, 0].plot(self.history['val_recall'], label='Recall')
            axes[1, 0].set_title('Precision / Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()

        if self.history['val_dice']:
            axes[1, 1].plot(self.history['val_dice'], label='Val Dice', color='purple')
            axes[1, 1].set_title('Validation Dice')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()

        if self.history['gradient_norm']:
            axes[1, 2].plot(self.history['gradient_norm'], label='Gradient Norm', color='red')
            axes[1, 2].set_title('Gradient Norm')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].legend()

        plt.tight_layout()
        plt.savefig(self.save_path / 'training_curves.png', dpi=150)
        plt.close()


def setup_distributed(config: Config) -> dict:
    """Setup distributed training if enabled, or single GPU training."""
    device_info = get_device_info(config)

    if device_info['is_distributed'] and config.distributed.enabled:
        torch.cuda.set_device(device_info['local_rank'])
        dist.init_process_group(
            backend=config.distributed.backend,
            init_method="env://",
            rank=device_info['global_rank'],
            world_size=device_info['world_size'],
        )
        if device_info['is_main']:
            print(f"Distributed training enabled with {device_info['world_size']} GPUs")
    elif device_info['device'].startswith('cuda'):
        gpu_id = getattr(config.experiment, 'gpu_id', 0)
        device_info['device'] = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        if device_info['is_main']:
            print(f"Single GPU training on cuda:{gpu_id}")
    else:
        if device_info['is_main']:
            print("Training on CPU")

    return device_info


def setup_experiment(config: Config, device_info: dict) -> Path:
    """Setup experiment directory and save config."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = Path(config.experiment.work_dir) / f"{config.experiment.name}-{run_id}"

    if device_info['is_main']:
        save_path.mkdir(parents=True, exist_ok=True)
        config.save(str(save_path / "config.yaml"))
        shutil.copyfile(__file__, save_path / "main.py")
        print(f"Experiment directory: {save_path}")

    return save_path


def collate_fn(batch):
    """Collate function for variable-size original masks."""
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    img_paths = [item[2] for item in batch]
    original_masks = [item[3] for item in batch]
    original_shapes = torch.tensor([item[4] for item in batch])
    return images, masks, img_paths, original_masks, original_shapes


def create_dataloaders(config: Config, device_info: dict):
    """Create train and validation dataloaders."""
    max_samples_train = getattr(config.data, 'max_samples_train', None)
    max_samples_val   = getattr(config.data, 'max_samples_val', None)

    train_dataset = ForgeryDataset(
        data_root=config.data.data_root,
        csv_file=config.data.train_csv,
        images_dir=config.data.train_images_dir,
        masks_dir=config.data.train_masks_dir,
        mode="train",
        image_size=config.data.image_size,
        augment_type=config.augmentation.type,
        crop_prob=config.augmentation.crop_prob,
        resize_mode=getattr(config.augmentation, 'resize_mode', 'crop_prob'),
        include_authentic=config.data.include_authentic,
        max_images=max_samples_train,
    )

    val_dataset = ForgeryDataset(
        data_root=config.data.data_root,
        csv_file=config.data.val_csv,
        images_dir=config.data.val_images_dir,
        masks_dir=config.data.val_masks_dir,
        mode="val",
        image_size=config.data.image_size,
        augment_type=0,
        include_authentic=config.data.include_authentic,
        max_images=max_samples_val,
    )

    train_sampler = None
    val_sampler   = None

    if device_info['is_distributed']:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset, shuffle=False)

    pin_memory  = getattr(config.data, 'pin_memory', False)
    num_workers = config.data.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),   # ← évite de recréer les workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),   # ←
    )

    if device_info['is_main']:
        print(f"Training samples: {len(train_dataset)}" + (f" (limited to {max_samples_train})" if max_samples_train else ""))
        print(f"Validation samples: {len(val_dataset)}" + (f" (limited to {max_samples_val})" if max_samples_val else ""))

    return train_loader, val_loader, train_sampler, val_sampler, train_dataset, val_dataset


def create_model(config: Config, device_info: dict) -> nn.Module:
    """Create and setup model (UCM-NetV2 or SegFormer)."""
    variant = config.model.variant

    if _is_segformer_variant(variant):
        model = build_segformer(
            variant=variant,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone,
            device=device_info['device'],
        )
        _load_fn = segformer_load_checkpoint
    else:
        model = build_ucm_netv2(
            variant=variant,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone,
            device=device_info['device'],
            use_grad_checkpoint=True,
        )
        _load_fn = ucm_load_checkpoint

    # Load pretrained checkpoint if specified
    if config.model.pretrained_checkpoint and os.path.exists(config.model.pretrained_checkpoint):
        if device_info['is_main']:
            print(f"Loading pretrained weights from: {config.model.pretrained_checkpoint}")
        _load_fn(model, config.model.pretrained_checkpoint, device_info['device'])

    # Wrap in DDP if distributed
    if device_info['is_distributed']:
        model = DDP(
            model,
            device_ids=[device_info['local_rank']],
            output_device=device_info['local_rank'],
            find_unused_parameters=True,
            bucket_cap_mb=config.distributed.bucket_cap_mb,
        )

    if device_info['is_main']:
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    return model


def create_loss_functions(config: Config, device_info: dict) -> Dict[str, nn.Module]:
    """Create loss functions from config."""
    ignore_label = config.loss.ignore_label

    losses = {}

    edge_weight     = config.loss.edge_weight
    cls_weight      = config.loss.cls_weight
    tversky_alpha   = config.loss.tversky_alpha
    tversky_beta    = config.loss.tversky_beta
    tversky_gamma   = config.loss.tversky_gamma

    ilw_max_weight = getattr(config.loss.bce, 'ilw_max_weight', 10.0)
    ilw_min_weight = getattr(config.loss.bce, 'ilw_min_weight', 0.1)

    if config.loss.type == "combined":
        losses['main'] = CombinedLoss(
            use_dice=config.loss.dice.enabled,
            use_bce=config.loss.bce.enabled,
            use_focal=config.loss.focal.enabled,
            use_ilw=config.loss.bce.use_ilw,
            dice_weight=config.loss.dice.weight,
            bce_weight=config.loss.bce.weight,
            focal_weight=config.loss.focal.weight,
            edge_weight=edge_weight,
            cls_weight=cls_weight,
            focal_alpha=config.loss.focal.alpha,
            focal_gamma=config.loss.focal.gamma,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma,
            dice_smooth=config.loss.dice.smooth,
            ignore_label=ignore_label,
            ilw_max_weight=ilw_max_weight,
            ilw_min_weight=ilw_min_weight,
        )
    elif config.loss.type == "bce":
        if config.loss.bce.use_ilw:
            losses['main'] = ILW_BCEWithLogitsLoss(
                ignore_label=ignore_label,
                max_weight=ilw_max_weight,
                min_weight=ilw_min_weight,
            )
        else:
            losses['main'] = nn.BCEWithLogitsLoss()
    elif config.loss.type == "dice":
        losses['main'] = DiceLoss(smooth=config.loss.dice.smooth, ignore_label=ignore_label)
    elif config.loss.type == "focal":
        losses['main'] = FocalLoss(
            alpha=config.loss.focal.alpha,
            gamma=config.loss.focal.gamma,
            ignore_label=ignore_label,
        )

    losses['mse'] = nn.MSELoss(reduction='mean')
    losses['acc'] = PixelAccWithIgnoreLabel(ignore_label=ignore_label)

    if device_info['is_main']:
        print(f"Loss type: {config.loss.type}")
        if config.loss.type == "combined":
            enabled = []
            if config.loss.bce.enabled:
                enabled.append(f"ILW-BCE(w={config.loss.bce.weight})")
            if config.loss.dice.enabled:
                enabled.append(f"FocalTversky(w={config.loss.dice.weight})")
            if config.loss.focal.enabled:
                enabled.append(f"Focal(w={config.loss.focal.weight})")
            enabled.append(f"Edge(w={edge_weight})")
            enabled.append(f"Cls(w={cls_weight})")
            print(f"  Components: {', '.join(enabled)}")

    return losses


def _get_encoder_prefixes(model: nn.Module) -> tuple:
    """Return encoder parameter prefixes based on model type."""
    if isinstance(model, SegFormer):
        return ('image_encoder.',)
    else:
        # UCM-NetV2 prefixes
        return (
            'encoder1', 'ebn1',
            'block_0_1', 'block0', 'block1', 'block2', 'block3',
            'norm1', 'norm2', 'norm3', 'norm4', 'norm5',
            'patch_embed1', 'patch_embed2', 'patch_embed3',
            'patch_embed4', 'patch_embed5',
        )


def create_optimizer_scheduler(model: nn.Module, config: Config, backbone_frozen: bool = True):
    """Create optimizer and learning rate scheduler."""
    base_model = model.module if hasattr(model, 'module') else model

    backbone_params = []
    decoder_params  = []

    encoder_prefixes = _get_encoder_prefixes(base_model)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if any(name.startswith(prefix) for prefix in encoder_prefixes):
                backbone_params.append(param)
            else:
                decoder_params.append(param)

    backbone_lr_mult = getattr(config.training, 'backbone_lr_multiplier', 0.1)

    param_groups = [
        {'params': decoder_params, 'lr': config.training.lr},
    ]

    if not backbone_frozen and backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': config.training.lr * backbone_lr_mult,
            'name': 'backbone'
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    sched_type = getattr(config.training.scheduler, 'type', 'cosine_warm_restarts')

    if sched_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=getattr(config.training.scheduler, 'factor', 0.5),
            patience=getattr(config.training.scheduler, 'patience', 5),
            min_lr=getattr(config.training.scheduler, 'min_lr', 1e-6),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.scheduler.T_0,
            T_mult=config.training.scheduler.T_mult,
            eta_min=config.training.scheduler.eta_min,
        )

    return optimizer, scheduler


def update_optimizer_for_unfrozen_backbone(model: nn.Module, optimizer: torch.optim.Optimizer, config: Config):
    """Add backbone parameters to optimizer when unfreezing."""
    base_model = model.module if hasattr(model, 'module') else model

    encoder_prefixes = _get_encoder_prefixes(base_model)
    backbone_params = []
    for name, param in base_model.named_parameters():
        if any(name.startswith(prefix) for prefix in encoder_prefixes) and param.requires_grad:
            backbone_params.append(param)

    if backbone_params:
        backbone_lr_mult = getattr(config.training, 'backbone_lr_multiplier', 0.1)
        current_lr = optimizer.param_groups[0]['lr']

        optimizer.add_param_group({
            'params': backbone_params,
            'lr': current_lr * backbone_lr_mult,
            'name': 'backbone'
        })

        print(f"Added {len(backbone_params)} backbone parameters to optimizer with LR={current_lr * backbone_lr_mult:.6f}")


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
    gt_binary   = (gt > 0.5).astype(np.float32)

    pred_flat = pred_binary.ravel()
    gt_flat   = gt_binary.ravel()

    valid_mask = gt_flat >= 0
    pred_flat  = pred_flat[valid_mask]
    gt_flat    = gt_flat[valid_mask]

    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    intersection = 2 * tp
    union        = 2 * tp + fp + fn
    dice = intersection / union if union > 0 else 0

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return {'precision': precision, 'recall': recall, 'dice': dice, 'iou': iou}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    losses: Dict[str, nn.Module],
    config: Config,
    device_info: dict,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    ema_model: Optional[EMAModel] = None,
) -> dict:
    """Train for one epoch — optimized for speed and low VRAM."""
    global AMP_DTYPE
    model.train()

    epoch_loss     = AverageMeter()
    epoch_bce_loss = AverageMeter()
    epoch_dice_loss= AverageMeter()
    last_grad_norm = 0.0

    use_amp = config.training.use_amp and scaler is not None
    main_loss_fn = losses['main']
    mse_loss     = losses['mse']
    acc_fn       = losses['acc']

    accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    accumulation_steps = max(1, accumulation_steps)
    log_every = config.logging.log_every

    desc = f"[Epoch {epoch}] Training"
    if accumulation_steps > 1:
        desc += f" (accum={accumulation_steps})"

    pbar = tqdm(dataloader, desc=desc, disable=not device_info['is_main'])

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        image, gt_mask, img_paths, _, _ = batch
        image   = image.to(device_info['device'], non_blocking=True)
        gt_mask = gt_mask.to(device_info['device'], non_blocking=True)

        # Classification target image-level
        cls_target = (gt_mask.view(gt_mask.shape[0], -1).sum(dim=1) > 0).float()

        with autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=use_amp):
            pred, aux_outputs = model(image, forward_type=0, upscale_output=True)

            edge_logits = aux_outputs['edge_logits']
            cls_logits  = aux_outputs['cls_logits']
            iou_pred    = aux_outputs['iou_pred']

            # Resize GT to match prediction size (safety net)
            if gt_mask.shape[-2:] != pred.shape[-2:]:
                gt_mask_resized = F.interpolate(
                    gt_mask.float(), size=pred.shape[-2:], mode="nearest",
                ).long()
            else:
                gt_mask_resized = gt_mask

            # Resize edge_logits to match GT if needed
            if edge_logits.shape[-2:] != gt_mask_resized.shape[-2:]:
                edge_logits = F.interpolate(
                    edge_logits, size=gt_mask_resized.shape[-2:],
                    mode='bilinear', align_corners=False
                )

            # Compute loss
            if isinstance(main_loss_fn, CombinedLoss):
                loss_dict = main_loss_fn(
                    pred, gt_mask_resized.float(),
                    edge_logits=edge_logits,
                    cls_logits=cls_logits,
                    cls_target=cls_target,
                )
                loss     = loss_dict['total']
                bce_val  = loss_dict.get('bce',    torch.tensor(0.0)).item()
                dice_val = loss_dict.get('tversky', torch.tensor(0.0)).item()
            else:
                loss     = main_loss_fn(pred, gt_mask_resized.float())
                bce_val  = loss.item()
                dice_val = 0.0

            # IoU prediction loss
            acc      = acc_fn(pred, gt_mask_resized.float())
            iou_loss = mse_loss(iou_pred.squeeze(), acc)

            total_loss  = loss + config.loss.lambda_pred_score * iou_loss
            scaled_loss = total_loss / accumulation_steps

        # Backward
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Update metrics
        batch_size = image.shape[0]
        epoch_loss.update(total_loss.item(), batch_size)
        epoch_bce_loss.update(bce_val, batch_size)
        epoch_dice_loss.update(dice_val, batch_size)

        if device_info['is_distributed']:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        # Optimizer step after accumulation
        is_update_step = (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader)
        if is_update_step:
            if use_amp:
                scaler.unscale_(optimizer)
                if device_info['is_main']:
                    last_grad_norm = compute_gradient_norm(model)
                if config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if device_info['is_main']:
                    last_grad_norm = compute_gradient_norm(model)
                if config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if ema_model is not None:
                ema_model.update(model)

        # Log periodically
        if device_info['is_main'] and step % log_every == 0:
            pbar.set_postfix({'loss': f"{epoch_loss.avg:.4f}", 'grad': f"{last_grad_norm:.2f}"})

    return {
        'train_loss': epoch_loss.avg,
        'train_bce': epoch_bce_loss.avg,
        'train_dice': epoch_dice_loss.avg,
        'gradient_norm': last_grad_norm,
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    losses: Dict[str, nn.Module],
    config: Config,
    device_info: dict,
    epoch: int,
    use_amp: bool = False,
) -> dict:
    """Validate for one epoch with oF1 metric."""
    global AMP_DTYPE
    model.eval()

    epoch_loss      = AverageMeter()
    epoch_bce_loss  = AverageMeter()
    epoch_dice_loss = AverageMeter()

    precision_meter = AverageMeter()
    recall_meter    = AverageMeter()
    dice_meter      = AverageMeter()
    iou_meter       = AverageMeter()

    oF1_calculator = KaggleMetricCalculator(
        threshold=config.inference.threshold,
        min_area=config.inference.min_area,
    )

    main_loss_fn = losses['main']

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] Validating", disable=not device_info['is_main'])

    with torch.no_grad():
        for batch in pbar:
            image, gt_mask, img_paths, original_masks, original_shapes = batch
            image   = image.to(device_info['device'])
            gt_mask = gt_mask.to(device_info['device'])

            with autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=use_amp):
                pred, aux_outputs = model(image, forward_type=0, upscale_output=True)

                if gt_mask.shape[-2:] != pred.shape[-2:]:
                    gt_mask_resized = F.interpolate(
                        gt_mask.float(),
                        size=pred.shape[-2:],
                        mode="nearest"
                    ).long()
                else:
                    gt_mask_resized = gt_mask

                if isinstance(main_loss_fn, CombinedLoss):
                    loss_dict      = main_loss_fn(pred, gt_mask_resized.float())
                    loss           = loss_dict['total']
                    bce_loss_val   = loss_dict.get('bce',    torch.tensor(0.0))
                    dice_loss_val  = loss_dict.get('tversky', torch.tensor(0.0))
                else:
                    loss          = main_loss_fn(pred, gt_mask_resized.float())
                    bce_loss_val  = loss
                    dice_loss_val = torch.tensor(0.0)

            if device_info['is_distributed']:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            batch_size = image.shape[0]
            epoch_loss.update(loss.item(), batch_size)
            epoch_bce_loss.update(bce_loss_val.item() if torch.is_tensor(bce_loss_val) else bce_loss_val, batch_size)
            epoch_dice_loss.update(dice_loss_val.item() if torch.is_tensor(dice_loss_val) else dice_loss_val, batch_size)

            if device_info['is_main']:
                # .float() obligatoire : pred peut être bf16/fp16 après autocast
                # et numpy ne supporte ni bf16 ni fp16.
                # On caste une seule fois ici ; ça couvre aussi F.interpolate ci-dessous.
                pred_sigmoid = torch.sigmoid(pred).float()

                pred_np = pred_sigmoid.cpu().numpy()
                gt_np   = gt_mask_resized.cpu().numpy()

                for i in range(batch_size):
                    metrics = compute_precision_recall_dice(pred_np[i, 0], gt_np[i, 0])
                    precision_meter.update(metrics['precision'])
                    recall_meter.update(metrics['recall'])
                    dice_meter.update(metrics['dice'])
                    iou_meter.update(metrics['iou'])

                    orig_h = original_shapes[i, 0].item()
                    orig_w = original_shapes[i, 1].item()
                    orig_shape = (orig_h, orig_w)
                    orig_gt    = original_masks[i]

                    pred_single = pred_sigmoid[i:i+1]
                    pred_resized_tensor = F.interpolate(
                        pred_single,
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    pred_resized = pred_resized_tensor[0, 0].cpu().numpy()

                    oF1_calculator.update(pred_resized, orig_gt, original_shape=orig_shape)

    oF1_metrics = oF1_calculator.compute()

    return {
        'val_loss':       epoch_loss.avg,
        'val_bce':        epoch_bce_loss.avg,
        'val_dice':       dice_meter.avg,
        'val_iou':        iou_meter.avg,
        'val_oF1':        oF1_metrics['oF1'],
        'val_forged_oF1': oF1_metrics.get('forged_oF1', 0.0),
        'val_precision':  precision_meter.avg,
        'val_recall':     recall_meter.avg,
        'debug_gt_authentic':            oF1_metrics.get('debug_gt_authentic', 0),
        'debug_gt_forged':               oF1_metrics.get('debug_gt_forged', 0),
        'debug_pred_authentic':          oF1_metrics.get('debug_pred_authentic', 0),
        'debug_pred_forged':             oF1_metrics.get('debug_pred_forged', 0),
        'debug_gt_forged_pred_authentic':oF1_metrics.get('debug_gt_forged_pred_authentic', 0),
    }


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
    backbone_frozen: bool = True,
):
    """Save checkpoint."""
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "backbone_frozen": backbone_frozen,
    }

    if ema_model is not None:
        checkpoint["ema"] = ema_model.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, save_path / f"model_{name}.pth")


def main():
    """Main training function."""
    global AMP_DTYPE

    config = Config.from_yaml()

    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)

    # GPU performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Déterminer le dtype AMP optimal APRÈS init CUDA
        AMP_DTYPE = _get_amp_dtype('cuda')

    # Setup distributed training
    device_info = setup_distributed(config)

    if device_info['is_main']:
        print(f"AMP dtype: {AMP_DTYPE}")   # bf16 ou fp16

    # Setup experiment directory
    save_path = setup_experiment(config, device_info)

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler, train_dataset, val_dataset = \
        create_dataloaders(config, device_info)

    # Create model
    model = create_model(config, device_info)

    # Create loss functions
    losses = create_loss_functions(config, device_info)

    # Track backbone freeze state
    backbone_frozen         = config.model.freeze_backbone
    freeze_backbone_epochs  = config.model.freeze_backbone_epochs

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(model, config, backbone_frozen)

    # Initialize AMP scaler
    scaler = GradScaler() if config.training.use_amp else None
    if device_info['is_main'] and config.training.use_amp:
        print("AMP (Automatic Mixed Precision) enabled")

    # Initialize EMA — shadow sur CPU pour économiser la VRAM
    ema_model = None
    if config.training.use_ema:
        base_model = model.module if hasattr(model, 'module') else model
        ema_model = EMAModel(base_model, decay=config.training.ema_decay)
        if device_info['is_main']:
            print(f"EMA enabled (CPU shadow) with decay={config.training.ema_decay}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if config.training.resume_checkpoint and os.path.exists(config.training.resume_checkpoint):
        checkpoint = torch.load(config.training.resume_checkpoint, map_location=device_info['device'], weights_only=False)

        state_dict = checkpoint.get("model", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch     = checkpoint['epoch'] + 1
        backbone_frozen = checkpoint.get('backbone_frozen', True)

        if 'ema' in checkpoint and ema_model is not None:
            ema_model.load_state_dict(checkpoint['ema'])

        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])

        if device_info['is_main']:
            print(f"Resuming from epoch {start_epoch}")

    # Initialize tracking
    metrics_logger = MetricsLogger(save_path) if device_info['is_main'] else None

    early_stopping_mode = 'min' if config.training.early_stopping.metric == 'val_loss' else 'max'
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping.patience,
        min_delta=config.training.early_stopping.min_delta,
        mode=early_stopping_mode
    ) if config.training.early_stopping.enabled else None

    best_val_loss      = float('inf')
    best_oF1           = 0.0
    best_forged_oF1    = 0.0
    best_ema_val_loss  = float('inf')
    best_ema_oF1       = 0.0

    # Print training config summary
    if device_info['is_main']:
        print("\n" + "="*60)
        print("Training Configuration Summary")
        print("="*60)
        print(f"  Device: {device_info['device']}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  AMP dtype: {AMP_DTYPE}")
        model_family = "SegFormer" if _is_segformer_variant(config.model.variant) else "UCM-NetV2"
        print(f"  Model: {model_family} ({config.model.variant})")
        print(f"  Grad checkpoint: ON")
        print(f"  Epochs: {config.training.num_epochs}")
        print(f"  Batch size: {config.training.batch_size}")
        accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
        if accumulation_steps > 1:
            print(f"  Gradient accumulation: {accumulation_steps} steps")
            print(f"  Effective batch size: {config.training.batch_size * accumulation_steps}")
        print(f"  Learning rate: {config.training.lr}")
        print(f"  Backbone LR multiplier: {getattr(config.training, 'backbone_lr_multiplier', 0.1)}")
        print(f"  Loss type: {config.loss.type}")
        print(f"  AMP: {config.training.use_amp}")
        print(f"  EMA: {config.training.use_ema} (decay={config.training.ema_decay}, CPU shadow)")
        print(f"  Gradient clipping: {config.training.grad_clip_norm}")
        print(f"  Early stopping: {config.training.early_stopping.enabled} (patience={config.training.early_stopping.patience})")
        print(f"  Backbone frozen: {backbone_frozen}")
        print(f"  Freeze backbone epochs: {freeze_backbone_epochs}")
        print("="*60 + "\n")

    # Warmup config
    warmup_epochs = getattr(config.training, 'warmup_epochs', 0)
    target_lr     = config.training.lr

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.training.num_epochs):
        gc.collect()

        # Linear warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            warmup_lr = target_lr * max(0.1, warmup_factor)
            for pg in optimizer.param_groups:
                if pg.get('name') == 'backbone':
                    backbone_lr_mult = getattr(config.training, 'backbone_lr_multiplier', 1.0)
                    pg['lr'] = warmup_lr * backbone_lr_mult
                else:
                    pg['lr'] = warmup_lr
            if device_info['is_main']:
                print(f"[Warmup] Epoch {epoch}: LR = {warmup_lr:.6f}")

        # Unfreeze backbone si nécessaire
        if backbone_frozen and epoch >= freeze_backbone_epochs:
            if device_info['is_main']:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}: UNFREEZING BACKBONE")
                print(f"{'='*60}")

            base_model = model.module if hasattr(model, 'module') else model
            base_model.set_backbone_freeze(False)
            backbone_frozen = False

            update_optimizer_for_unfrozen_backbone(model, optimizer, config)

            if device_info['is_main']:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Trainable parameters after unfreeze: {trainable_params:,}")
                print(f"{'='*60}\n")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_dataset.shuffle_samples(epoch)

        # ── Train ────────────────────────────────────────────────────────
        train_metrics = train_epoch(
            model, train_loader, optimizer, losses,
            config, device_info, epoch, scaler, ema_model
        )

        current_lr = optimizer.param_groups[0]['lr']

        if device_info['is_main']:
            frozen_str = " [backbone frozen]" if backbone_frozen else " [backbone training]"
            print(f"Epoch {epoch} | Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Grad Norm: {train_metrics['gradient_norm']:.2f} | LR: {current_lr:.6f}{frozen_str}")

        # Sync processes
        if device_info['is_distributed']:
            dist.barrier()

        # ── Validate ─────────────────────────────────────────────────────
        if (epoch + 1) % config.training.val_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            val_metrics = validate_epoch(
                model, val_loader, losses, config, device_info, epoch,
                use_amp=config.training.use_amp
            )

            # ── Validate EMA : déplacer sur GPU puis rapatrier sur CPU ───
            ema_val_metrics = None
            if ema_model is not None:
                # Libérer un peu de VRAM avant de charger l'EMA
                torch.cuda.empty_cache()

                ema_gpu_model = ema_model.get_model(device=device_info['device'])
                ema_val_metrics = validate_epoch(
                    ema_gpu_model, val_loader, losses,
                    config, device_info, epoch, use_amp=config.training.use_amp
                )
                # Rapatrier sur CPU immédiatement
                ema_model.offload_to_cpu()
                torch.cuda.empty_cache()

            if device_info['is_main']:
                print(f"Epoch {epoch} | Val Loss: {val_metrics['val_loss']:.4f} | "
                      f"oF1: {val_metrics['val_oF1']:.4f} | "
                      f"forged_oF1: {val_metrics.get('val_forged_oF1', 0):.4f} | "
                      f"Precision: {val_metrics['val_precision']:.4f} | "
                      f"Recall: {val_metrics['val_recall']:.4f} | "
                      f"Dice: {val_metrics['val_dice']:.4f} | "
                      f"IoU: {val_metrics['val_iou']:.4f}")

                if ema_val_metrics is not None:
                    print(f"       EMA | Val Loss: {ema_val_metrics['val_loss']:.4f} | "
                          f"oF1: {ema_val_metrics['val_oF1']:.4f} | "
                          f"IoU: {ema_val_metrics['val_iou']:.4f}")

                # Log metrics
                all_metrics = {
                    **train_metrics,
                    **val_metrics,
                    'learning_rate': current_lr,
                    'backbone_frozen': int(backbone_frozen),
                }
                metrics_logger.log(epoch, all_metrics)
                metrics_logger.save()
                metrics_logger.plot()

                # Save best val_loss model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                                   save_path, "best_loss", ema_model, scaler, backbone_frozen)
                    print(f"  -> New best val_loss: {best_val_loss:.4f}")

                # Save best oF1 model
                if val_metrics['val_oF1'] > best_oF1:
                    best_oF1 = val_metrics['val_oF1']
                    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                                   save_path, "best_oF1", ema_model, scaler, backbone_frozen)
                    print(f"  -> New best oF1: {best_oF1:.4f}")

                # Save best forged_oF1 model
                current_forged_oF1 = val_metrics.get('val_forged_oF1', 0.0)
                if current_forged_oF1 > best_forged_oF1:
                    best_forged_oF1 = current_forged_oF1
                    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                                   save_path, "best_forged_oF1", ema_model, scaler, backbone_frozen)
                    print(f"  -> New best forged_oF1: {best_forged_oF1:.4f}")

                # Save best EMA models if better
                if ema_val_metrics is not None:
                    if ema_val_metrics['val_loss'] < best_ema_val_loss:
                        best_ema_val_loss = ema_val_metrics['val_loss']
                        ema_checkpoint = {
                            "model": ema_model.shadow.state_dict(),
                            "epoch": epoch,
                            "metrics": ema_val_metrics,
                        }
                        torch.save(ema_checkpoint, save_path / "model_best_loss_ema.pth")
                        print(f"  -> New best EMA val_loss: {best_ema_val_loss:.4f}")

                    if ema_val_metrics['val_oF1'] > best_ema_oF1:
                        best_ema_oF1 = ema_val_metrics['val_oF1']
                        ema_checkpoint = {
                            "model": ema_model.shadow.state_dict(),
                            "epoch": epoch,
                            "metrics": ema_val_metrics,
                        }
                        torch.save(ema_checkpoint, save_path / "model_best_oF1_ema.pth")
                        print(f"  -> New best EMA oF1: {best_ema_oF1:.4f}")

                # Early stopping check
                es_metric = config.training.early_stopping.metric
                if es_metric == 'val_loss':
                    metric_value = val_metrics['val_loss']
                elif es_metric == 'val_forged_oF1':
                    # ── FIX : fallback sur val_oF1 si forged_oF1 est 0 ──
                    metric_value = val_metrics.get('val_forged_oF1', 0.0)
                    if metric_value == 0.0:
                        metric_value = val_metrics.get('val_oF1', 0.0)
                else:
                    metric_value = val_metrics['val_oF1']

                if early_stopping is not None and epoch >= warmup_epochs:
                    if early_stopping(metric_value):
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        print(f"  Best val_loss: {best_val_loss:.4f}")
                        print(f"  Best oF1: {best_oF1:.4f}")
                        print(f"  Best forged_oF1: {best_forged_oF1:.4f}")
                        break

            # Step scheduler (after validation) — skip during warmup
            # ── FIX : même fallback pour le scheduler ──
            if epoch >= warmup_epochs:
                sched_type = getattr(config.training.scheduler, 'type', 'cosine_warm_restarts')
                if sched_type == "reduce_on_plateau":
                    sched_metric = val_metrics.get('val_forged_oF1', 0.0)
                    if sched_metric == 0.0:
                        sched_metric = val_metrics.get('val_oF1', 0.0)
                    scheduler.step(sched_metric)
                else:
                    scheduler.step()

        # Sync processes
        if device_info['is_distributed']:
            dist.barrier()

    if device_info['is_main']:
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best oF1 score: {best_oF1:.4f}")
        print(f"  Best forged_oF1: {best_forged_oF1:.4f}")
        print(f"  Models saved to: {save_path}")
        print("="*60)


def run_inference(args):
    """Run inference/evaluation on a dataset."""
    global AMP_DTYPE
    config = Config.from_yaml()

    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)

    if torch.cuda.is_available():
        AMP_DTYPE = _get_amp_dtype('cuda')

    device_info = setup_distributed(config)

    split = args.split if args.split else 'val'

    print("\n" + "="*60)
    print(f"INFERENCE MODE - Evaluating on {split} set")
    print("="*60)

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        raise ValueError("--checkpoint is required for inference mode")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    max_samples = getattr(config.data, f'max_samples_{split}', None)

    if split == 'val':
        dataset = ForgeryDataset(
            data_root=config.data.data_root,
            csv_file=config.data.val_csv,
            images_dir=config.data.val_images_dir,
            masks_dir=config.data.val_masks_dir,
            mode="val",
            image_size=config.data.image_size,
            augment_type=0,
            include_authentic=config.data.include_authentic,
            max_images=max_samples,
            split_name="val",
        )
    elif split == 'test':
        dataset = ForgeryDataset(
            data_root=config.data.data_root,
            csv_file=config.data.test_csv,
            images_dir=config.data.test_images_dir,
            masks_dir=config.data.test_masks_dir,
            mode="val",
            image_size=config.data.image_size,
            augment_type=0,
            include_authentic=config.data.include_authentic,
            max_images=max_samples,
            split_name="test",
        )
    elif split == 'train':
        dataset = ForgeryDataset(
            data_root=config.data.data_root,
            csv_file=config.data.train_csv,
            images_dir=config.data.train_images_dir,
            masks_dir=config.data.train_masks_dir,
            mode="val",
            image_size=config.data.image_size,
            augment_type=0,
            include_authentic=config.data.include_authentic,
            max_images=max_samples,
            split_name="train",
        )
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

    pin_memory  = getattr(config.data, 'pin_memory', False)
    num_workers = config.data.num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )

    num_forged    = sum(1 for s in dataset.samples if s['label'] == 1)
    num_authentic = sum(1 for s in dataset.samples if s['label'] == 0)

    print(f"\nDataset: {len(dataset)} samples")
    print(f"  -> {num_forged} forged, {num_authentic} authentic")
    print(f"Using threshold: {config.inference.threshold}")

    model = create_model(config, device_info)
    losses = create_loss_functions(config, device_info)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_info['device'], weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    base_model = model.module if hasattr(model, 'module') else model
    base_model.load_state_dict(state_dict)

    if 'epoch' in checkpoint:
        print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Checkpoint metrics: oF1={checkpoint['metrics'].get('val_oF1', 'N/A')}")

    print(f"\nEvaluating on {split} set...")
    metrics = validate_epoch(
        model, dataloader, losses, config, device_info, epoch=0,
        use_amp=config.training.use_amp
    )

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({split} set)")
    print("="*60)
    print(f"  Threshold: {config.inference.threshold:.2f}")
    print(f"  Loss:      {metrics['val_loss']:.4f}")
    print(f"  oF1:       {metrics['val_oF1']:.4f}")
    print(f"  Forged oF1:{metrics.get('val_forged_oF1', 0):.4f}")
    print(f"  Precision: {metrics['val_precision']:.4f}")
    print(f"  Recall:    {metrics['val_recall']:.4f}")
    print(f"  Dice:      {metrics['val_dice']:.4f}")
    print(f"  IoU:       {metrics['val_iou']:.4f}")
    print("="*60)

    if args.output:
        results = {
            'checkpoint': checkpoint_path,
            'split': split,
            'threshold': config.inference.threshold,
            'metrics': {
                'loss': metrics['val_loss'],
                'oF1': metrics['val_oF1'],
                'forged_oF1': metrics.get('val_forged_oF1', 0),
                'precision': metrics['val_precision'],
                'recall': metrics['val_recall'],
                'dice': metrics['val_dice'],
                'iou': metrics['val_iou'],
            }
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Forgery Detection Training and Inference (UCM-NetV2 / SegFormer)')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Mode: train or inference (default: train)')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for inference')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split for inference (default: val)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for inference results')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'inference':
        run_inference(args)
