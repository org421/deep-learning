"""
SAM2-UNET Model for Copy-Move Forgery Detection - Version OFFLINE.

Architecture:
- Encoder: SAM2 Hiera backbone (sans torch.hub)
- Decoder: UNET-style decoder avec skip connections multi-échelles
- Normalisation: GroupNorm

SAM2's Hiera encoder produces hierarchical features at 4 scales,
which is ideal for U-Net style decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from pathlib import Path

from .hiera_local import build_sam2_encoder, SAM2_CONFIGS


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        padding: int = 1,
        num_groups: int = 8,
    ):
        super().__init__()
        
        num_groups = min(num_groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Upsample + Conv block for decoder."""
    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        
        num_groups = min(num_groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class UNetDecoder(nn.Module):
    """
    UNET-style decoder with multi-scale skip connections from Hiera.
    
    Hiera produces 4 scales of features, which we use as skip connections.
    """
    
    def __init__(
        self,
        encoder_channels: List[int],  # e.g., [256, 256, 256, 256] from SAM2 neck
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 1,
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.decoder_channels = decoder_channels
        self.num_encoder_features = len(encoder_channels)
        
        # Initial convolution on deepest features
        self.initial_conv = ConvBlock(encoder_channels[-1], decoder_channels[0], num_groups=num_groups)
        
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        # Build decoder stages - go from deep to shallow
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            # Skip connection from encoder (going backwards through encoder features)
            encoder_idx = self.num_encoder_features - 2 - i  # -2 because we start after initial_conv
            if encoder_idx >= 0:
                skip_ch = encoder_channels[encoder_idx]
                self.skip_projections.append(
                    nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
                )
            else:
                self.skip_projections.append(None)
            
            self.upsample_blocks.append(UpsampleBlock(in_ch, out_ch, num_groups=num_groups))
            
            # Concatenation with skip
            concat_ch = out_ch * 2 if encoder_idx >= 0 else out_ch
            self.decoder_blocks.append(ConvBlock(concat_ch, out_ch, num_groups=num_groups))
        
        # Final upsampling to original resolution
        # SAM2 at 1024 input: stage 1 = 256x256, stage 2 = 128x128, etc.
        # We need to upsample more to get back to input resolution
        self.final_upsample = nn.Sequential(
            UpsampleBlock(decoder_channels[-1], decoder_channels[-1], num_groups=num_groups),
            UpsampleBlock(decoder_channels[-1], decoder_channels[-1], num_groups=num_groups),
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1),
        )
        
        # IoU prediction head
        self.iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, 
        encoder_features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_features: List of features from Hiera [stage1, stage2, stage3, stage4]
                             from shallow to deep (increasing depth)
            target_size: Optional output size (H, W)
            
        Returns:
            seg_logits: (B, 1, H, W) segmentation logits
            iou_pred: (B, 1) IoU prediction
        """
        # Reverse features so deepest is first
        encoder_features = encoder_features[::-1]
        
        # Start with deepest features
        x = self.initial_conv(encoder_features[0])
        
        # Decoder path with skip connections
        for i, (upsample, decoder_block) in enumerate(zip(self.upsample_blocks, self.decoder_blocks)):
            x = upsample(x)
            
            # Add skip connection if available
            skip_idx = i + 1
            if skip_idx < len(encoder_features) and self.skip_projections[i] is not None:
                skip = encoder_features[skip_idx]
                skip = self.skip_projections[i](skip)
                
                # Match spatial dimensions
                if x.shape[-2:] != skip.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
                x = torch.cat([x, skip], dim=1)
            
            x = decoder_block(x)
        
        # Final upsampling
        x = self.final_upsample(x)
        
        features_for_iou = x
        seg_logits = self.segmentation_head(x)
        
        # Resize to target if needed
        if target_size is not None and seg_logits.shape[-2:] != target_size:
            seg_logits = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)
        
        iou_pred = self.iou_head(features_for_iou)
        
        return seg_logits, iou_pred


class SAM2Encoder(nn.Module):
    """
    SAM2 Hiera Encoder - Version OFFLINE.
    
    Loads weights from a local file.
    """
    
    def __init__(
        self,
        model_name: str = "sam2_hiera_b",
        weights_path: Optional[str] = None,
        freeze: bool = True,
        img_size: int = 1024,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Get config
        if model_name not in SAM2_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(SAM2_CONFIGS.keys())}")
        
        config = SAM2_CONFIGS[model_name]
        self.embed_dim = config['embed_dim']
        self.out_chans = config['out_chans']
        
        # Build backbone LOCAL
        self.backbone = build_sam2_encoder(
            model_name=model_name,
            weights_path=weights_path,
            img_size=img_size,
            device='cpu'  # Load on CPU first
        )
        
        if freeze:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def set_freeze(self, freeze: bool):
        self.freeze = freeze
        if freeze:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from SAM2 encoder.
        
        Args:
            x: (B, 3, H, W) input image
            
        Returns:
            List of features at 4 scales, from shallow to deep
        """
        with torch.set_grad_enabled(not self.freeze):
            features = self.backbone(x)
        
        return features
    
    @property
    def output_channels(self) -> List[int]:
        """Return output channels for each stage."""
        # SAM2 neck outputs same channels for all stages
        return [self.out_chans] * 4


class SAM2UNet(nn.Module):
    """
    SAM2 + UNET for Image Forgery Detection - Version OFFLINE.
    
    Uses SAM2's Hiera backbone for hierarchical features and
    a U-Net style decoder for pixel-wise segmentation.
    """
    
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        backbone: str = "sam2_hiera_b",
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_groups: int = 8,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        img_size: int = 1024,
    ):
        super().__init__()
        
        self.image_encoder = SAM2Encoder(
            model_name=backbone,
            weights_path=weights_path,
            freeze=freeze_backbone,
            img_size=img_size,
        )
        
        encoder_channels = self.image_encoder.output_channels
        self.mask_decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=1,
            num_groups=num_groups,
        )
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        self.backbone_name = backbone
        self.img_size = img_size
    
    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image."""
        x = (x - self.pixel_mean) / self.pixel_std
        return x.float()
    
    def set_backbone_freeze(self, freeze: bool):
        """Set backbone freeze state."""
        self.image_encoder.set_freeze(freeze)
    
    def forward(
        self, 
        image: torch.Tensor, 
        point_prompt: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        forward_type: int = 0,
        upscale_output: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Forward pass.
        
        Args:
            image: (B, 3, H, W) input image in [0, 255]
            point_prompt: Ignored (kept for API compatibility)
            forward_type: 0 = normal, 1 = return lists (for compatibility)
            upscale_output: Whether to upscale output to input size
            
        Returns:
            seg_logits: (B, 1, H, W) or List
            iou_pred: (B, 1) or List
        """
        x = self.preprocess(image)
        
        B, C, H, W = x.shape
        
        # Hiera requires input size divisible by 32 (stride of the network)
        # Pad or resize if needed
        stride = 32
        target_h = ((H + stride - 1) // stride) * stride
        target_w = ((W + stride - 1) // stride) * stride
        
        if H != target_h or W != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Encode
        encoder_features = self.image_encoder(x)
        
        # Decode
        output_size = (H, W) if upscale_output else None
        seg_logits, iou_pred = self.mask_decoder(encoder_features, target_size=output_size)
        
        if forward_type == 0:
            return seg_logits, iou_pred
        
        elif forward_type == 1:
            # Return as lists (for compatibility with some training loops)
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


def build_sam2_unet(
    backbone: str = "sam2_hiera_b",
    weights_path: Optional[str] = None,
    freeze_backbone: bool = True,
    decoder_channels: List[int] = [256, 128, 64, 32],
    num_groups: int = 8,
    img_size: int = 1024,
    device: str = "cuda",
) -> SAM2UNet:
    """
    Build SAM2-UNET model with local weights.
    
    Args:
        backbone: "sam2_hiera_t", "sam2_hiera_s", "sam2_hiera_b", "sam2_hiera_b+", "sam2_hiera_l"
        weights_path: Path to .pt file (e.g., "model/sam2_hiera_base_plus.pt")
        freeze_backbone: Freeze the backbone
        decoder_channels: Decoder configuration
        num_groups: Groups for GroupNorm
        img_size: Image size (1024 recommended for SAM2)
        device: Device
        
    Returns:
        SAM2UNet model
    """
    model = SAM2UNet(
        backbone=backbone,
        weights_path=weights_path,
        freeze_backbone=freeze_backbone,
        decoder_channels=decoder_channels,
        num_groups=num_groups,
        img_size=img_size,
    )
    
    return model.to(device)


def load_checkpoint(model: SAM2UNet, checkpoint_path: str, device: str = "cuda") -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return checkpoint
