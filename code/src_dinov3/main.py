"""
Script principal - entrainement et inference DinoV3 pour la detection de falsification.
Projet M2 DSC - Kaggle competition.
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
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
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
from utils.kaggle_metric import KaggleMetricCalculator
from utils.ema import EMAModel
from data.dataset import ForgeryDataset
from networks.dinov3_upernet_model import DinoV3UperNet, build_dinov3_upernet, load_checkpoint


class EarlyStopping:
    """Arret automatique si pas d'amelioration."""
    
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
    """Sauvegarde les metriques d'entrainement."""
    
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
        """Enregistre les metriques d'une epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self):
        """Sauvegarde en JSON."""
        with open(self.save_path / 'metrics_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot(self):
        """Trace les courbes d'entrainement."""
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
    """Config du multi-GPU ou single GPU."""
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
    """Cree le dossier de l'experience et sauvegarde la config."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = Path(config.experiment.work_dir) / f"{config.experiment.name}-{run_id}"
    
    if device_info['is_main']:
        save_path.mkdir(parents=True, exist_ok=True)
        config.save(str(save_path / "config.yaml"))
        shutil.copyfile(__file__, save_path / "main.py")
        print(f"Experiment directory: {save_path}")
    
    return save_path


def collate_fn(batch):
    """Collate custom pour gerer les masques de taille variable."""
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    img_paths = [item[2] for item in batch]
    original_masks = [item[3] for item in batch]
    original_shapes = torch.tensor([item[4] for item in batch])
    return images, masks, img_paths, original_masks, original_shapes


def create_dataloaders(config: Config, device_info: dict):
    """Cree les dataloaders train et val."""
    max_samples_train = getattr(config.data, 'max_samples_train', None)
    max_samples_val = getattr(config.data, 'max_samples_val', None)
    
    train_dataset = ForgeryDataset(
        data_root=config.data.data_root,
        csv_file=config.data.train_csv,
        images_dir=config.data.train_images_dir,
        masks_dir=config.data.train_masks_dir,
        mode="train",
        image_size=config.data.image_size,
        augment_type=config.augmentation.type,
        crop_prob=config.augmentation.crop_prob,
        resize_mode=config.augmentation.resize_mode,
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
    val_sampler = None
    
    if device_info['is_distributed']:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    pin_memory = getattr(config.data, 'pin_memory', False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )
    
    if device_info['is_main']:
        print(f"Training samples: {len(train_dataset)}" + (f" (limited to {max_samples_train})" if max_samples_train else ""))
        print(f"Validation samples: {len(val_dataset)}" + (f" (limited to {max_samples_val})" if max_samples_val else ""))
    
    return train_loader, val_loader, train_sampler, val_sampler, train_dataset, val_dataset


def create_model(config: Config, device_info: dict) -> nn.Module:
    """Instancie le modele DinoV3."""
    weights_path = getattr(config.model, 'weights_path', None)

    # params du decoder
    fpn_channels = getattr(config.model, 'fpn_channels', 256)
    ppm_channels = getattr(config.model, 'ppm_channels', 512)
    dropout_ratio = getattr(config.model, 'dropout_ratio', 0.1)
    
    model = build_dinov3_upernet(
        backbone=config.model.backbone,
        weights_path=weights_path,  # chemin vers le .pth local
        freeze_backbone=config.model.freeze_backbone,
        fpn_channels=fpn_channels,
        ppm_channels=ppm_channels,
        num_groups=config.model.num_groups,
        dropout_ratio=dropout_ratio,
        img_size=config.data.image_size,
        device=device_info['device'],
    )
    
    # charge un checkpoint pretraine si specifie
    if config.model.pretrained_checkpoint and os.path.exists(config.model.pretrained_checkpoint):
        if device_info['is_main']:
            print(f"Loading pretrained weights from: {config.model.pretrained_checkpoint}")
        load_checkpoint(model, config.model.pretrained_checkpoint, device_info['device'])
    
    # DDP pour le multi-GPU
    if device_info['is_distributed']:
        model = DDP(
            model,
            device_ids=[device_info['local_rank']],
            output_device=device_info['local_rank'],
            find_unused_parameters=True,
            bucket_cap_mb=config.distributed.bucket_cap_mb,
        )
    
    if device_info['is_main']:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_loss_functions(config: Config, device_info: dict) -> Dict[str, nn.Module]:
    """Cree les fonctions de loss selon la config."""
    ignore_label = config.loss.ignore_label
    
    losses = {}
    
    if config.loss.type == "combined":
        losses['main'] = CombinedLoss(
            use_dice=config.loss.dice.enabled,
            use_bce=config.loss.bce.enabled,
            use_focal=config.loss.focal.enabled,
            use_ilw=config.loss.bce.use_ilw,
            dice_weight=config.loss.dice.weight,
            bce_weight=config.loss.bce.weight,
            focal_weight=config.loss.focal.weight,
            focal_alpha=config.loss.focal.alpha,
            focal_gamma=config.loss.focal.gamma,
            dice_smooth=config.loss.dice.smooth,
            ignore_label=ignore_label,
        )
    elif config.loss.type == "bce":
        if config.loss.bce.use_ilw:
            losses['main'] = ILW_BCEWithLogitsLoss(ignore_label=ignore_label)
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
                enabled.append(f"BCE(w={config.loss.bce.weight})")
            if config.loss.dice.enabled:
                enabled.append(f"Dice(w={config.loss.dice.weight})")
            if config.loss.focal.enabled:
                enabled.append(f"Focal(w={config.loss.focal.weight})")
            print(f"  Components: {', '.join(enabled)}")
    
    return losses


def create_optimizer_scheduler(model: nn.Module, config: Config, backbone_frozen: bool = True):
    """Cree l'optimizer AdamW et le scheduler cosine."""
    base_model = model.module if hasattr(model, 'module') else model

    # on separe les params backbone / decoder
    backbone_params = []
    decoder_params = []
    
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if 'image_encoder' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
    
    # lr differents pour backbone et decoder
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
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.scheduler.T_0,
        T_mult=config.training.scheduler.T_mult,
        eta_min=config.training.scheduler.eta_min,
    )
    
    return optimizer, scheduler


def update_optimizer_for_unfrozen_backbone(model: nn.Module, optimizer: torch.optim.Optimizer, config: Config):
    """Ajoute les params du backbone a l'optimizer apres le degel."""
    base_model = model.module if hasattr(model, 'module') else model
    
    backbone_params = []
    for name, param in base_model.named_parameters():
        if 'image_encoder' in name and param.requires_grad:
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
    """Calcule la norme des gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_precision_recall_dice(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calcule precision, recall, dice et IoU."""
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
    """Entraine un epoch."""
    model.train()
    
    epoch_loss = AverageMeter()
    epoch_bce_loss = AverageMeter()
    epoch_dice_loss = AverageMeter()
    gradient_norms = []
    
    use_amp = config.training.use_amp and scaler is not None
    main_loss_fn = losses['main']
    mse_loss = losses['mse']
    acc_fn = losses['acc']
    
    accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    accumulation_steps = max(1, accumulation_steps)
    
    desc = f"[Epoch {epoch}] Training"
    if accumulation_steps > 1:
        desc += f" (accum={accumulation_steps})"
    
    pbar = tqdm(dataloader, desc=desc, disable=not device_info['is_main'])
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        # on depack le batch
        image, gt_mask, img_paths, _, _ = batch
        image = image.to(device_info['device'])
        gt_mask = gt_mask.to(device_info['device'])
        
        with autocast(enabled=use_amp):
            # forward pass
            pred, iou_pred = model(image, forward_type=0, upscale_output=False)
            
            # resize du GT si taille differente
            if gt_mask.shape[-2:] != pred.shape[-2:]:
                gt_mask_resized = F.interpolate(
                    gt_mask.float(),
                    size=pred.shape[-2:],
                    mode="nearest",
                ).long()
            else:
                gt_mask_resized = gt_mask
            
            # calcul de la loss
            if isinstance(main_loss_fn, CombinedLoss):
                loss_dict = main_loss_fn(pred, gt_mask_resized.float())
                loss = loss_dict['total']
                bce_val = loss_dict.get('bce', torch.tensor(0.0)).item()
                dice_val = loss_dict.get('dice', torch.tensor(0.0)).item()
            else:
                loss = main_loss_fn(pred, gt_mask_resized.float())
                bce_val = loss.item()
                dice_val = 0.0
            
            # loss auxiliaire IoU
            acc = acc_fn(pred, gt_mask_resized.float())
            iou_loss = mse_loss(iou_pred.squeeze(), acc)
            
            total_loss = loss + config.loss.lambda_pred_score * iou_loss
            
            # scaling pour l'accumulation de gradients
            scaled_loss = total_loss / accumulation_steps
        
        # backward
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # maj des metriques
        batch_size = image.shape[0]
        epoch_loss.update(total_loss.item(), batch_size)
        epoch_bce_loss.update(bce_val, batch_size)
        epoch_dice_loss.update(dice_val, batch_size)
        
        if device_info['is_distributed']:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        
        # step optimizer apres accumulation
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            if use_amp:
                if config.training.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                
                grad_norm = compute_gradient_norm(model)
                gradient_norms.append(grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                if config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                
                grad_norm = compute_gradient_norm(model)
                gradient_norms.append(grad_norm)
                
                optimizer.step()
            
            optimizer.zero_grad()
            
            if ema_model is not None:
                ema_model.update(model)
        
        if device_info['is_main'] and step % config.logging.log_every == 0:
            current_grad = gradient_norms[-1] if gradient_norms else 0
            pbar.set_postfix({'loss': f"{epoch_loss.avg:.4f}", 'grad': f"{current_grad:.2f}"})
    
    valid_grads = [g for g in gradient_norms if np.isfinite(g)]
    avg_grad_norm = np.mean(valid_grads) if valid_grads else float('nan')
    
    return {
        'train_loss': epoch_loss.avg,
        'train_bce': epoch_bce_loss.avg,
        'train_dice': epoch_dice_loss.avg,
        'gradient_norm': avg_grad_norm,
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
    """Validation sur un epoch avec calcul du oF1."""
    model.eval()
    
    epoch_loss = AverageMeter()
    epoch_bce_loss = AverageMeter()
    epoch_dice_loss = AverageMeter()
    
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    oF1_calculator = KaggleMetricCalculator(
        threshold=config.inference.threshold,
        min_area=config.inference.min_area,
    )
    
    main_loss_fn = losses['main']
    
    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] Validating", disable=not device_info['is_main'])
    
    with torch.no_grad():
        for batch in pbar:
            image, gt_mask, img_paths, original_masks, original_shapes = batch
            image = image.to(device_info['device'])
            gt_mask = gt_mask.to(device_info['device'])
            
            with autocast(enabled=use_amp):
                pred, _ = model(image, forward_type=0)
                
                # resize GT si besoin
                if gt_mask.shape[-2:] != pred.shape[-2:]:
                    gt_mask_resized = F.interpolate(
                        gt_mask.float(),
                        size=pred.shape[-2:],
                        mode="nearest"
                    ).long()
                else:
                    gt_mask_resized = gt_mask
                
                # loss
                if isinstance(main_loss_fn, CombinedLoss):
                    loss_dict = main_loss_fn(pred, gt_mask_resized.float())
                    loss = loss_dict['total']
                    bce_loss_val = loss_dict.get('bce', torch.tensor(0.0))
                    dice_loss_val = loss_dict.get('dice', torch.tensor(0.0))
                else:
                    loss = main_loss_fn(pred, gt_mask_resized.float())
                    bce_loss_val = loss
                    dice_loss_val = torch.tensor(0.0)
            
            if device_info['is_distributed']:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            
            batch_size = image.shape[0]
            epoch_loss.update(loss.item(), batch_size)
            epoch_bce_loss.update(bce_loss_val.item() if torch.is_tensor(bce_loss_val) else bce_loss_val, batch_size)
            epoch_dice_loss.update(dice_loss_val.item() if torch.is_tensor(dice_loss_val) else dice_loss_val, batch_size)
            
            if device_info['is_main']:
                # sigmoid sur GPU
                pred_sigmoid = torch.sigmoid(pred)
                
                # metriques sur les masques resizes
                pred_np = pred_sigmoid.cpu().numpy()
                gt_np = gt_mask_resized.cpu().numpy()
                
                for i in range(batch_size):
                    # precision/recall/dice
                    metrics = compute_precision_recall_dice(pred_np[i, 0], gt_np[i, 0])
                    precision_meter.update(metrics['precision'])
                    recall_meter.update(metrics['recall'])
                    dice_meter.update(metrics['dice'])
                    iou_meter.update(metrics['iou'])
                    
                    # pour le oF1 on resize a la resolution originale
                    orig_h = original_shapes[i, 0].item()
                    orig_w = original_shapes[i, 1].item()
                    orig_shape = (orig_h, orig_w)
                    orig_gt = original_masks[i]
                    
                    pred_single = pred_sigmoid[i:i+1]
                    pred_resized_tensor = F.interpolate(
                        pred_single,
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    pred_resized = pred_resized_tensor[0, 0].cpu().numpy()
                    
                    # maj oF1
                    oF1_calculator.update(pred_resized, orig_gt, original_shape=orig_shape)
    
    oF1_metrics = oF1_calculator.compute()
    
    return {
        'val_loss': epoch_loss.avg,
        'val_bce': epoch_bce_loss.avg,
        'val_dice': dice_meter.avg,
        'val_iou': iou_meter.avg,
        'val_oF1': oF1_metrics['oF1'],
        'val_forged_oF1': oF1_metrics.get('forged_oF1', 0.0),
        'val_precision': precision_meter.avg,
        'val_recall': recall_meter.avg,
        'debug_gt_authentic': oF1_metrics.get('debug_gt_authentic', 0),
        'debug_gt_forged': oF1_metrics.get('debug_gt_forged', 0),
        'debug_pred_authentic': oF1_metrics.get('debug_pred_authentic', 0),
        'debug_pred_forged': oF1_metrics.get('debug_pred_forged', 0),
        'debug_gt_forged_pred_authentic': oF1_metrics.get('debug_gt_forged_pred_authentic', 0),
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
    """Sauvegarde un checkpoint."""
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
    """Boucle d'entrainement principale."""
    # chargement config
    config = Config.from_yaml()
    
    # seed
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    
    # config multi-GPU
    device_info = setup_distributed(config)

    # dossier experience
    save_path = setup_experiment(config, device_info)

    # dataloaders
    train_loader, val_loader, train_sampler, val_sampler, train_dataset, val_dataset = \
        create_dataloaders(config, device_info)

    # modele
    model = create_model(config, device_info)

    # fonctions de loss
    losses = create_loss_functions(config, device_info)

    # etat du gel du backbone
    backbone_frozen = config.model.freeze_backbone
    freeze_backbone_epochs = config.model.freeze_backbone_epochs

    # optimizer et scheduler
    optimizer, scheduler = create_optimizer_scheduler(model, config, backbone_frozen)

    # scaler AMP
    scaler = GradScaler() if config.training.use_amp else None
    if device_info['is_main'] and config.training.use_amp:
        print("AMP (Automatic Mixed Precision) enabled")
    
    # EMA
    ema_model = None
    if config.training.use_ema:
        base_model = model.module if hasattr(model, 'module') else model
        ema_model = EMAModel(base_model, decay=config.training.ema_decay, device=device_info['device'])
        if device_info['is_main']:
            print(f"EMA enabled with decay={config.training.ema_decay}")
    
    # reprise depuis un checkpoint si specifie
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
        start_epoch = checkpoint['epoch'] + 1
        backbone_frozen = checkpoint.get('backbone_frozen', True)
        
        if 'ema' in checkpoint and ema_model is not None:
            ema_model.load_state_dict(checkpoint['ema'])
        
        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        
        if device_info['is_main']:
            print(f"Resuming from epoch {start_epoch}")
    
    # suivi des metriques
    metrics_logger = MetricsLogger(save_path) if device_info['is_main'] else None
    
    early_stopping_mode = 'min' if config.training.early_stopping.metric == 'val_loss' else 'max'
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping.patience,
        min_delta=config.training.early_stopping.min_delta,
        mode=early_stopping_mode
    ) if config.training.early_stopping.enabled else None
    
    best_val_loss = float('inf')
    best_oF1 = 0.0
    best_ema_val_loss = float('inf')
    best_ema_oF1 = 0.0
    
    # resume config
    if device_info['is_main']:
        print("\n" + "="*60)
        print("Training Configuration Summary")
        print("="*60)
        print(f"  Device: {device_info['device']}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Model: DinoV2-UNET ({config.model.backbone})")
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
        print(f"  EMA: {config.training.use_ema} (decay={config.training.ema_decay})")
        print(f"  Gradient clipping: {config.training.grad_clip_norm}")
        print(f"  Early stopping: {config.training.early_stopping.enabled} (patience={config.training.early_stopping.patience})")
        print(f"  Backbone frozen: {backbone_frozen}")
        print(f"  Freeze backbone epochs: {freeze_backbone_epochs}")
        print("="*60 + "\n")
    
    # boucle d'entrainement
    for epoch in range(start_epoch, config.training.num_epochs):
        gc.collect()

        # degel du backbone si epoch atteint
        if backbone_frozen and epoch >= freeze_backbone_epochs:
            if device_info['is_main']:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}: UNFREEZING BACKBONE")
                print(f"{'='*60}")
            
            # degel backbone
            base_model = model.module if hasattr(model, 'module') else model
            base_model.set_backbone_freeze(False)
            backbone_frozen = False

            # ajout params backbone a l'optimizer
            update_optimizer_for_unfrozen_backbone(model, optimizer, config)
            
            if device_info['is_main']:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Trainable parameters after unfreeze: {trainable_params:,}")
                print(f"{'='*60}\n")
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        train_dataset.shuffle_samples(epoch)
        
        # entrainement
        train_metrics = train_epoch(
            model, train_loader, optimizer, losses,
            config, device_info, epoch, scaler, ema_model
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if device_info['is_main']:
            frozen_str = " [backbone frozen]" if backbone_frozen else " [backbone training]"
            print(f"Epoch {epoch} | Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Grad Norm: {train_metrics['gradient_norm']:.2f} | LR: {current_lr:.6f}{frozen_str}")
        
        # maj scheduler
        scheduler.step()

        # synchro multi-GPU
        if device_info['is_distributed']:
            dist.barrier()

        # validation
        if (epoch + 1) % config.training.val_every == 0:
            val_metrics = validate_epoch(
                model, val_loader, losses, config, device_info, epoch,
                use_amp=config.training.use_amp
            )
            
            # validation avec modele EMA si dispo
            ema_val_metrics = None
            if ema_model is not None:
                ema_val_metrics = validate_epoch(
                    ema_model.get_model(), val_loader, losses, 
                    config, device_info, epoch, use_amp=config.training.use_amp
                )
            
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
                
                # sauvegarde metriques
                all_metrics = {
                    **train_metrics,
                    **val_metrics,
                    'learning_rate': current_lr,
                    'backbone_frozen': int(backbone_frozen),
                }
                metrics_logger.log(epoch, all_metrics)
                metrics_logger.save()
                metrics_logger.plot()
                
                # sauvegarde meilleur val_loss
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, 
                                   save_path, "best_loss", ema_model, scaler, backbone_frozen)
                    print(f"  -> New best val_loss: {best_val_loss:.4f}")
                
                # sauvegarde meilleur oF1
                if val_metrics['val_oF1'] > best_oF1:
                    best_oF1 = val_metrics['val_oF1']
                    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, 
                                   save_path, "best_oF1", ema_model, scaler, backbone_frozen)
                    print(f"  -> New best oF1: {best_oF1:.4f}")
                
                # sauvegarde meilleurs modeles EMA
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
                
                # arret premature si pas d'amelioration
                if early_stopping is not None:
                    metric_value = val_metrics['val_loss'] if config.training.early_stopping.metric == 'val_loss' else val_metrics['val_oF1']
                    if early_stopping(metric_value):
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        print(f"  Best val_loss: {best_val_loss:.4f}")
                        print(f"  Best oF1: {best_oF1:.4f}")
                        break
        
        # synchro multi-GPU
        if device_info['is_distributed']:
            dist.barrier()

    if device_info['is_main']:
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best oF1 score: {best_oF1:.4f}")
        print(f"  Models saved to: {save_path}")
        print("="*60)


def run_inference(args):
    """Inference / evaluation sur un dataset."""
    # chargement config
    config = Config.from_yaml()

    # seed
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)

    # device
    device_info = setup_distributed(config)

    # choix du split
    split = args.split if args.split else 'val'
    
    print("\n" + "="*60)
    print(f"INFERENCE MODE - Evaluating on {split} set")
    print("="*60)
    
    # verification du checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        raise ValueError("--checkpoint is required for inference mode")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # dataset selon le split
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
            mode="val",  # pas d'augmentation
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
            mode="val",  # pas d'augmentation pour l'eval
            image_size=config.data.image_size,
            augment_type=0,
            include_authentic=config.data.include_authentic,
            max_images=max_samples,
            split_name="train",
        )
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
    
    pin_memory = getattr(config.data, 'pin_memory', False)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    # comptage forged/authentic
    num_forged = sum(1 for s in dataset.samples if s['label'] == 1)
    num_authentic = sum(1 for s in dataset.samples if s['label'] == 0)
    
    print(f"\nDataset: {len(dataset)} samples")
    print(f"  -> {num_forged} forged, {num_authentic} authentic")
    print(f"Using threshold: {config.inference.threshold}")
    
    # modele
    model = create_model(config, device_info)

    # fonctions de loss (pour val_loss)
    losses = create_loss_functions(config, device_info)

    # chargement checkpoint
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
    
    # evaluation
    print(f"\nEvaluating on {split} set...")
    metrics = validate_epoch(
        model, dataloader, losses, config, device_info, epoch=0,
        use_amp=config.training.use_amp
    )
    
    # affichage resultats
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
    
    # sauvegarde JSON
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
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description='DinoV3-UNET Training and Inference')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Mode: train or inference (default: train)')
    
    # arguments inference
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