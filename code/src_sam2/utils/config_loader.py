"""
Configuration loader and manager for DinoV2-UNET training.
"""

import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Priority:
    1. DINOV2_ROOT environment variable (if set)
    2. Current working directory (CWD)
    """
    if 'DINOV2_ROOT' in os.environ:
        return Path(os.environ['DINOV2_ROOT'])
    return Path.cwd()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, searches in:
                    1. {CWD}/configs/config.yaml
                    2. {script_dir}/../configs/config.yaml (fallback)
    """
    if config_path is None:
        # Try CWD first
        cwd_config = get_project_root() / "configs" / "config.yaml"
        if cwd_config.exists():
            config_path = cwd_config
        else:
            # Fallback to script-relative path
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def resolve_path(path: str, base_dir: Path = None) -> Path:
    """
    Resolve a path relative to the project root.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    
    if base_dir is None:
        base_dir = get_project_root()
    
    return base_dir / p


@dataclass
class ExperimentConfig:
    name: str = "dinov2_unet_forgery_detection"
    seed: int = 42
    work_dir: str = "./work_dir"
    device: str = "cuda"
    gpu_id: int = 0


@dataclass
class DataConfig:
    data_root: str = "./data"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    train_images_dir: str = "train/images"
    train_masks_dir: str = "train/masks"
    val_images_dir: str = "val/images"
    val_masks_dir: str = "val/masks"
    test_images_dir: str = "test/images"
    test_masks_dir: str = "test/masks"
    image_size: int = 518  # DinoV2 native: 518 = 37 * 14
    num_workers: int = 4
    include_authentic: bool = True
    max_images_per_class: Optional[int] = None
    pin_memory: bool = True
    max_samples_train: Optional[int] = None
    max_samples_val: Optional[int] = None
    max_samples_test: Optional[int] = None


@dataclass
class ModelConfig:
    # DinoV2 backbone
    backbone: str = "dinov2_vitb14"
    pretrained: bool = True
    
    # Chemin vers les poids locaux (pour mode offline)
    weights_path: Optional[str] = None
    
    # UNET decoder
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    num_groups: int = 8  # GroupNorm groups (gradient accumulation compatible)
    
    # Backbone freezing
    freeze_backbone: bool = True
    freeze_backbone_epochs: int = 5  # Number of epochs to keep backbone frozen
    
    # For compatibility with SAFIRE
    num_pairs: int = 3
    
    # Legacy fields (for loading old configs)
    type: str = "dinov2_unet"
    sam_checkpoint: str = ""
    pretrained_checkpoint: str = ""


@dataclass
class SchedulerConfig:
    type: str = "cosine_warm_restarts"
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 20
    min_delta: float = 0.0001
    metric: str = "val_loss"
    mode: str = "min"


@dataclass
class TrainingConfig:
    num_epochs: int = 150
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    lr: float = 0.0001
    weight_decay: float = 0.01
    backbone_lr_multiplier: float = 0.1  # LR for backbone after unfreeze
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    lambda_pred_score: float = 0.1
    resume_checkpoint: str = ""
    save_every: int = 10
    val_every: int = 1
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    grad_clip_norm: float = 1.0


@dataclass
class AugmentationConfig:
    type: int = 1
    crop_prob: float = 0.3
    resize_mode: str = "crop_prob"


@dataclass
class DistributedConfig:
    enabled: bool = True
    backend: str = "nccl"
    bucket_cap_mb: int = 25


@dataclass
class InferenceConfig:
    points_per_side: int = 16
    points_per_batch: int = 256
    threshold: float = 0.5
    min_area: int = 100
    cluster_type: str = "kmeans"
    kmeans_cluster_num: int = 2
    dbscan_eps: float = 0.2
    dbscan_min_samples: int = 1
    output_dir_binary: str = "./outputs/binary"
    output_dir_multi: str = "./outputs/multi"


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    log_every: int = 10
    verbose: bool = True


@dataclass
class BCEConfig:
    enabled: bool = True
    weight: float = 1.0
    use_ilw: bool = True
    ilw_max_weight: float = 10.0
    ilw_min_weight: float = 0.1


@dataclass
class DiceConfig:
    enabled: bool = True
    weight: float = 1.0
    smooth: float = 1.0


@dataclass
class FocalConfig:
    enabled: bool = False
    weight: float = 1.0
    alpha: float = 0.25
    gamma: float = 2.0


@dataclass
class LossConfig:
    type: str = "combined"
    bce: BCEConfig = field(default_factory=BCEConfig)
    dice: DiceConfig = field(default_factory=DiceConfig)
    focal: FocalConfig = field(default_factory=FocalConfig)
    lambda_pred_score: float = 0.1
    ignore_label: int = -1


@dataclass
class EvaluationConfig:
    compute_oF1: bool = True
    threshold: float = 0.5
    min_area: int = 100
    merge_distance: int = 0


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str = None) -> 'Config':
        """Load config from YAML file and create Config object."""
        raw_config = load_config(config_path)
        
        # Parse nested configs
        scheduler_cfg = SchedulerConfig(**raw_config.get('training', {}).get('scheduler', {}))
        early_stopping_cfg = EarlyStoppingConfig(**raw_config.get('training', {}).get('early_stopping', {}))
        
        training_dict = raw_config.get('training', {}).copy()
        training_dict.pop('scheduler', None)
        training_dict.pop('early_stopping', None)
        training_cfg = TrainingConfig(**training_dict, scheduler=scheduler_cfg, early_stopping=early_stopping_cfg)
        
        aug_cfg = AugmentationConfig(**raw_config.get('augmentation', {}))
        
        # Parse model config with defaults for missing fields
        model_raw = raw_config.get('model', {}).copy()
        # Handle decoder_channels as list
        if 'decoder_channels' in model_raw and isinstance(model_raw['decoder_channels'], list):
            pass  # Keep as is
        model_cfg = ModelConfig(**model_raw)
        
        # Parse loss config
        loss_raw = raw_config.get('loss', {})
        bce_cfg = BCEConfig(**loss_raw.get('bce', {}))
        dice_cfg = DiceConfig(**loss_raw.get('dice', {}))
        focal_cfg = FocalConfig(**loss_raw.get('focal', {}))
        loss_dict = {k: v for k, v in loss_raw.items() if k not in ['bce', 'dice', 'focal']}
        loss_cfg = LossConfig(**loss_dict, bce=bce_cfg, dice=dice_cfg, focal=focal_cfg)
        
        # Parse evaluation config
        eval_raw = raw_config.get('evaluation', {}).copy()
        oF1_raw = eval_raw.pop('oF1', {})
        eval_cfg = EvaluationConfig(**{**eval_raw, **oF1_raw})
        
        return cls(
            experiment=ExperimentConfig(**raw_config.get('experiment', {})),
            data=DataConfig(**raw_config.get('data', {})),
            model=model_cfg,
            training=training_cfg,
            augmentation=aug_cfg,
            distributed=DistributedConfig(**raw_config.get('distributed', {})),
            inference=InferenceConfig(**raw_config.get('inference', {})),
            logging=LoggingConfig(**raw_config.get('logging', {})),
            loss=loss_cfg,
            evaluation=eval_cfg,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        training_dict = {k: v for k, v in self.training.__dict__.items() 
                        if k not in ['scheduler', 'early_stopping']}
        training_dict['scheduler'] = self.training.scheduler.__dict__
        training_dict['early_stopping'] = self.training.early_stopping.__dict__
        
        loss_dict = {k: v for k, v in self.loss.__dict__.items()
                    if k not in ['bce', 'dice', 'focal']}
        loss_dict['bce'] = self.loss.bce.__dict__
        loss_dict['dice'] = self.loss.dice.__dict__
        loss_dict['focal'] = self.loss.focal.__dict__
        
        return {
            'experiment': self.experiment.__dict__,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': training_dict,
            'augmentation': self.augmentation.__dict__,
            'distributed': self.distributed.__dict__,
            'inference': self.inference.__dict__,
            'logging': self.logging.__dict__,
            'loss': loss_dict,
            'evaluation': self.evaluation.__dict__,
        }
    
    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_device_info(config=None):
    """
    Get device information for training.
    """
    if config is not None:
        preferred_device = getattr(config.experiment, 'device', 'cuda')
        gpu_id = getattr(config.experiment, 'gpu_id', 0)
    else:
        preferred_device = 'cuda'
        gpu_id = 0
    
    if 'RANK' in os.environ:
        global_rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        is_distributed = world_size > 1
    else:
        global_rank = 0
        local_rank = gpu_id
        world_size = 1
        is_distributed = False
    
    if preferred_device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
    else:
        device = 'cpu'
        is_distributed = False
    
    return {
        'global_rank': global_rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'is_main': global_rank == 0,
        'device': device,
        'is_distributed': is_distributed,
    }
