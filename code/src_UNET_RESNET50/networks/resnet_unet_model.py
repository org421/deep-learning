"""
ResNet-UNET Model for Copy-Move Forgery Detection.

Architecture:
- Encoder: ResNet (18, 34, 50, 101) with ImageNet pretrained weights
- Decoder: UNET-style decoder from segmentation_models_pytorch
- Output: Binary segmentation mask

Uses segmentation_models_pytorch for clean implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
import segmentation_models_pytorch as smp


class ResNetUNet(nn.Module):
    """
    ResNet + UNET for Image Forgery Detection.
    
    Uses segmentation_models_pytorch with same interface as DinoV2UNet.
    """
    
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        pixel_mean: List[float] = [123.675, 116.28, 103.53],  # ImageNet mean (0-255)
        pixel_std: List[float] = [58.395, 57.12, 57.375],      # ImageNet std (0-255)
    ):
        """
        Initialize ResNet-UNET model.
        
        Args:
            backbone: ResNet backbone name (resnet18, resnet34, resnet50, resnet101)
            pretrained: Load ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights
            decoder_channels: Channels for decoder blocks
            pixel_mean: Mean for image normalization (0-255 range)
            pixel_std: Std for image normalization (0-255 range)
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        
        # Create UNet with ResNet encoder using segmentation_models_pytorch
        encoder_weights = "imagenet" if pretrained else None
        
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,  # Binary segmentation
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=True,
        )
        
        # IoU prediction head (for compatibility with training loop)
        # Takes the segmentation output (1 channel after sigmoid)
        self.iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Image normalization buffers (0-255 range, same as DINOv2)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder/backbone weights."""
        for param in self.unet.encoder.parameters():
            param.requires_grad = False
    
    def _unfreeze_encoder(self):
        """Unfreeze encoder/backbone weights."""
        for param in self.unet.encoder.parameters():
            param.requires_grad = True
    
    def set_backbone_freeze(self, freeze: bool):
        """Set backbone freeze state."""
        self.freeze_backbone = freeze
        if freeze:
            self._freeze_encoder()
        else:
            self._unfreeze_encoder()
    
    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values from [0, 255] to normalized range."""
        x = (x - self.pixel_mean) / self.pixel_std
        return x.float()
    
    def forward(
        self, 
        image: torch.Tensor, 
        point_prompt: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        forward_type: int = 0,
        upscale_output: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Forward pass compatible with training loop.
        
        Args:
            image: Input images (B, C, H, W) in [0, 255] range
            point_prompt: Ignored (for compatibility with DINOv2 interface)
            forward_type: 0 = single output, 1 = multiple outputs (for compatibility)
            upscale_output: Whether to upscale output to input size
            
        Returns:
            For forward_type=0: (segmentation_logits, iou_predictions)
            For forward_type=1: (list of segmentation_logits, list of iou_predictions)
        """
        B, C, H, W = image.shape
        
        # Preprocess (normalize)
        x = self.preprocess(image)
        
        # Forward through entire UNet (encoder + decoder + segmentation_head)
        seg_logits = self.unet(x)
        
        # Handle output size
        if upscale_output and seg_logits.shape[-2:] != (H, W):
            seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)
        
        # IoU prediction from segmentation logits (use sigmoid output)
        iou_pred = self.iou_head(torch.sigmoid(seg_logits))
        
        if forward_type == 0:
            return seg_logits, iou_pred
        
        elif forward_type == 1:
            # Multiple predictions (for compatibility)
            if point_prompt is not None:
                points, labels = point_prompt
                num_prompts = points.shape[1]
            else:
                num_prompts = 1
            
            group_of_masks = [seg_logits for _ in range(num_prompts)]
            group_of_ious = [iou_pred for _ in range(num_prompts)]
            
            return group_of_masks, group_of_ious
        
        else:
            raise NotImplementedError(f"forward_type {forward_type} not implemented")
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def build_resnet_unet(
    backbone: str = "resnet50",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    decoder_channels: List[int] = [256, 128, 64, 32, 16],
    device: str = "cuda",
) -> ResNetUNet:
    """
    Build ResNet-UNET model.
    
    Args:
        backbone: ResNet backbone name (resnet18, resnet34, resnet50, resnet101)
        pretrained: Load ImageNet pretrained weights
        freeze_backbone: Freeze backbone
        decoder_channels: Decoder channel configuration
        device: Device to put model on
        
    Returns:
        ResNetUNet model
    """
    model = ResNetUNet(
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        decoder_channels=decoder_channels,
    )
    
    return model.to(device)


def load_checkpoint(model: ResNetUNet, checkpoint_path: str, device: str = "cuda") -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: ResNet-UNET model
        checkpoint_path: Path to checkpoint
        device: Device to load on
        
    Returns:
        Checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Remove module. prefix if present (from DDP)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        for k in missing_keys[:5]:
            print(f"  - {k}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        for k in unexpected_keys[:5]:
            print(f"  - {k}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys) - 5} more")
    
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return checkpoint