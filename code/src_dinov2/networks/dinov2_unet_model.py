"""
DinoV2-UNET Model for Copy-Move Forgery Detection.

Architecture:
- Encoder: DinoV2 ViT with intermediate feature extraction (blocks 3, 6, 9, 12)
- Decoder: UNET-style decoder with skip connections from intermediate layers
- Normalization: GroupNorm (compatible with gradient accumulation)
- Output: Binary segmentation mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm (gradient accumulation compatible)."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        padding: int = 1,
        num_groups: int = 8,  # GroupNorm groups
    ):
        super().__init__()
        
        # Ensure num_groups divides out_channels
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
    UNET-style decoder with TRUE multi-scale skip connections.
    
    Takes features from DinoV2 intermediate layers and progressively 
    upsamples with skip connections to produce full-resolution output.
    """
    
    def __init__(
        self,
        encoder_channels: List[int],  # [block12_ch, block9_ch, block6_ch, block3_ch]
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 1,
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.decoder_channels = decoder_channels
        
        # Initial projection from deepest encoder features
        self.initial_conv = ConvBlock(encoder_channels[0], decoder_channels[0], num_groups=num_groups)
        
        # Decoder stages with skip connections
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            # Skip connection projection (encoder features to decoder channels)
            if i + 1 < len(encoder_channels):
                skip_ch = encoder_channels[i + 1]
                self.skip_projections.append(
                    nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
                )
            else:
                self.skip_projections.append(None)
            
            # Upsample block
            self.upsample_blocks.append(UpsampleBlock(in_ch, out_ch, num_groups=num_groups))
            
            # Decoder conv block (concatenated features)
            concat_ch = out_ch * 2 if i + 1 < len(encoder_channels) else out_ch
            self.decoder_blocks.append(ConvBlock(concat_ch, out_ch, num_groups=num_groups))
        
        # Final upsample to full resolution (4x to match image size)
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
        
        # IoU prediction head (for compatibility with SAFIRE)
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
        Forward pass through decoder.
        
        Args:
            encoder_features: List of features from encoder [block12, block9, block6, block3]
            target_size: Target output size (H, W)
            
        Returns:
            Tuple of (segmentation_logits, iou_prediction)
        """
        # Start with deepest features
        x = self.initial_conv(encoder_features[0])
        
        # Decoder with skip connections
        for i, (upsample, decoder_block) in enumerate(zip(self.upsample_blocks, self.decoder_blocks)):
            # Upsample
            x = upsample(x)
            
            # Add skip connection if available
            if i + 1 < len(encoder_features) and self.skip_projections[i] is not None:
                skip = encoder_features[i + 1]
                skip = self.skip_projections[i](skip)
                
                # Ensure same spatial size
                if x.shape[-2:] != skip.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
                x = torch.cat([x, skip], dim=1)
            
            # Decoder conv
            x = decoder_block(x)
        
        # Final upsample to higher resolution
        x = self.final_upsample(x)
        
        # Features for IoU prediction
        features_for_iou = x
        
        # Segmentation output
        seg_logits = self.segmentation_head(x)
        
        # Upsample to target size if specified
        if target_size is not None and seg_logits.shape[-2:] != target_size:
            seg_logits = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)
        
        # IoU prediction
        iou_pred = self.iou_head(features_for_iou)
        
        return seg_logits, iou_pred


class DinoV2Encoder(nn.Module):
    """
    DinoV2 Vision Transformer as encoder with INTERMEDIATE FEATURE EXTRACTION.
    
    Extracts features from blocks 3, 6, 9, 12 for true multi-scale UNET.
    """
    
    # DinoV2 configurations
    DINOV2_CONFIGS = {
        'dinov2_vits14': {'embed_dim': 384, 'num_blocks': 12},
        'dinov2_vitb14': {'embed_dim': 768, 'num_blocks': 12},
        'dinov2_vitl14': {'embed_dim': 1024, 'num_blocks': 24},
        'dinov2_vitg14': {'embed_dim': 1536, 'num_blocks': 40},
    }
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        freeze: bool = True,
        intermediate_layers: List[int] = None,  # Which blocks to extract
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Get config
        if model_name not in self.DINOV2_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.DINOV2_CONFIGS.keys())}")
        
        config = self.DINOV2_CONFIGS[model_name]
        self.embed_dim = config['embed_dim']
        self.num_blocks = config['num_blocks']
        
        # Default intermediate layers based on model depth
        if intermediate_layers is None:
            if self.num_blocks == 12:
                # ViT-S/B: blocks 3, 6, 9, 12 (1-indexed → 2, 5, 8, 11 in 0-indexed)
                self.intermediate_layers = [2, 5, 8, 11]
            elif self.num_blocks == 24:
                # ViT-L: blocks 6, 12, 18, 24
                self.intermediate_layers = [5, 11, 17, 23]
            else:
                # ViT-G: blocks 10, 20, 30, 40
                self.intermediate_layers = [9, 19, 29, 39]
        else:
            self.intermediate_layers = intermediate_layers
        
        # Load DinoV2 model
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
        self.patch_size = self.backbone.patch_size
        
        if freeze:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def set_freeze(self, freeze: bool):
        """Set backbone freeze state."""
        self.freeze = freeze
        if freeze:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from DinoV2 intermediate layers.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            List of features [block12_feat, block9_feat, block6_feat, block3_feat]
            Each feature has shape (B, embed_dim, H/14, W/14)
        """
        B, C, H, W = x.shape
        
        # Ensure input size is compatible with patch size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size must be divisible by patch_size {self.patch_size}"
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Get intermediate features using DinoV2's built-in method
        with torch.set_grad_enabled(not self.freeze):
            # get_intermediate_layers returns list of features from specified blocks
            # Each feature is (B, num_patches + 1, embed_dim) with CLS token
            intermediate_outputs = self.backbone.get_intermediate_layers(
                x, 
                n=self.intermediate_layers,
                reshape=False,  # Keep as sequence
                return_class_token=False,  # Don't need CLS token
            )
        
        # Convert to spatial format (B, C, H, W)
        features = []
        for feat in intermediate_outputs:
            # feat shape: (B, num_patches, embed_dim)
            feat = feat.reshape(B, num_patches_h, num_patches_w, -1)
            feat = feat.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)
            features.append(feat)
        
        # Reverse order: deepest first [block12, block9, block6, block3]
        features = features[::-1]
        
        return features
    
    @property
    def output_channels(self) -> List[int]:
        """Return output channels for each scale (all same for ViT)."""
        return [self.embed_dim] * len(self.intermediate_layers)


class DinoV2UNet(nn.Module):
    """
    DinoV2 + UNET for Image Forgery Detection.
    
    TRUE multi-scale architecture with intermediate feature extraction.
    Uses GroupNorm for gradient accumulation compatibility.
    """
    
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_groups: int = 8,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],  # ImageNet mean
        pixel_std: List[float] = [58.395, 57.12, 57.375],      # ImageNet std
    ):
        """
        Initialize DinoV2-UNET model.
        
        Args:
            backbone: DinoV2 backbone model name
            pretrained: Load pretrained weights
            freeze_backbone: Freeze backbone weights
            decoder_channels: Channels for decoder blocks
            num_groups: Number of groups for GroupNorm
            pixel_mean: Mean for image normalization
            pixel_std: Std for image normalization
        """
        super().__init__()
        
        # Encoder
        self.image_encoder = DinoV2Encoder(
            model_name=backbone,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        
        # Decoder with true multi-scale skip connections
        encoder_channels = self.image_encoder.output_channels  # [768, 768, 768, 768] for ViT-B
        self.mask_decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=1,
            num_groups=num_groups,
        )
        
        # Image normalization
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        # Store config
        self.backbone_name = backbone
        self.patch_size = self.image_encoder.patch_size
    
    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values."""
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
        Forward pass compatible with SAFIRE training loop.
        
        Args:
            image: Input images (B, C, H, W) in [0, 255] range
            point_prompt: Tuple of (points, labels) - for compatibility
            forward_type: 0 = single output, 1 = multiple outputs per point
            upscale_output: Whether to upscale output to input size
            
        Returns:
            For forward_type=0: (segmentation_logits, iou_predictions)
            For forward_type=1: (list of segmentation_logits, list of iou_predictions)
        """
        # Preprocess
        x = self.preprocess(image)
        
        # Ensure size is compatible with patch_size
        B, C, H, W = x.shape
        target_h = (H // self.patch_size) * self.patch_size
        target_w = (W // self.patch_size) * self.patch_size
        
        if H != target_h or W != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Extract multi-scale encoder features
        encoder_features = self.image_encoder(x)
        
        # Decode
        output_size = (H, W) if upscale_output else None
        seg_logits, iou_pred = self.mask_decoder(encoder_features, target_size=output_size)
        
        if forward_type == 0:
            # Single prediction
            return seg_logits, iou_pred
        
        elif forward_type == 1:
            # Multiple predictions (for training with multiple point prompts)
            if point_prompt is not None:
                points, labels = point_prompt
                num_prompts = points.shape[1]
            else:
                num_prompts = 1
            
            # Return same prediction for each prompt (UNET doesn't use points)
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


def build_dinov2_unet(
    backbone: str = "dinov2_vitb14",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    decoder_channels: List[int] = [256, 128, 64, 32],
    num_groups: int = 8,
    device: str = "cuda",
) -> DinoV2UNet:
    """
    Build DinoV2-UNET model.
    
    Args:
        backbone: DinoV2 backbone name
        pretrained: Load pretrained weights
        freeze_backbone: Freeze backbone
        decoder_channels: Decoder channel configuration
        num_groups: Number of groups for GroupNorm
        device: Device to put model on
        
    Returns:
        DinoV2UNet model
    """
    model = DinoV2UNet(
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        decoder_channels=decoder_channels,
        num_groups=num_groups,
    )
    
    return model.to(device)


def load_checkpoint(model: DinoV2UNet, checkpoint_path: str, device: str = "cuda") -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: DinoV2-UNET model
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
