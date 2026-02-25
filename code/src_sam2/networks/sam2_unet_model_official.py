"""
SAM2-UNET Model for Copy-Move Forgery Detection.

Utilise le repo officiel de Facebook:
    git clone https://github.com/facebookresearch/sam2.git

Architecture:
- Encoder: SAM2 Hiera backbone (depuis repo officiel)
- Decoder: UNET-style decoder avec skip connections multi-échelles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from pathlib import Path

# Import depuis le nouveau module qui utilise le repo officiel
from .sam2_encoder import build_sam2_encoder, SAM2_CONFIGS


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
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
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
        
        # Build decoder stages
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            encoder_idx = self.num_encoder_features - 2 - i
            if encoder_idx >= 0:
                skip_ch = encoder_channels[encoder_idx]
                self.skip_projections.append(
                    nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
                )
            else:
                self.skip_projections.append(None)
            
            self.upsample_blocks.append(UpsampleBlock(in_ch, out_ch, num_groups=num_groups))
            
            concat_ch = out_ch * 2 if encoder_idx >= 0 else out_ch
            self.decoder_blocks.append(ConvBlock(concat_ch, out_ch, num_groups=num_groups))
        
        # Final upsampling
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
        # Reverse features so deepest is first
        encoder_features = encoder_features[::-1]
        
        # Start with deepest features
        x = self.initial_conv(encoder_features[0])
        
        # Decoder path with skip connections
        for i, (upsample, decoder_block) in enumerate(zip(self.upsample_blocks, self.decoder_blocks)):
            x = upsample(x)
            
            skip_idx = i + 1
            if skip_idx < len(encoder_features) and self.skip_projections[i] is not None:
                skip = encoder_features[skip_idx]
                skip = self.skip_projections[i](skip)
                
                if x.shape[-2:] != skip.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
                x = torch.cat([x, skip], dim=1)
            
            x = decoder_block(x)
        
        # Final upsampling
        x = self.final_upsample(x)
        
        features_for_iou = x
        seg_logits = self.segmentation_head(x)
        
        if target_size is not None and seg_logits.shape[-2:] != target_size:
            seg_logits = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)
        
        iou_pred = self.iou_head(features_for_iou)
        
        return seg_logits, iou_pred


class SAM2Encoder(nn.Module):
    """
    SAM2 Hiera Encoder - Utilise le repo officiel Facebook.
    """
    
    def __init__(
        self,
        model_name: str = "sam2_hiera_b+",
        weights_path: Optional[str] = None,
        freeze: bool = True,
        img_size: int = 1024,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        if model_name not in SAM2_CONFIGS:
            raise ValueError(f"Modèle inconnu: {model_name}. Choix: {list(SAM2_CONFIGS.keys())}")
        
        config = SAM2_CONFIGS[model_name]
        self.embed_dim = config['embed_dim']
        self.out_chans = config['out_chans']
        
        # Construire l'encodeur avec le repo officiel
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
        with torch.set_grad_enabled(not self.freeze):
            features = self.backbone(x)
        return features
    
    @property
    def output_channels(self) -> List[int]:
        return [self.out_chans] * 4


class SAM2UNet(nn.Module):
    """
    SAM2 + UNET for Image Forgery Detection.
    
    Utilise le repo officiel de Facebook pour SAM2.
    """
    
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        backbone: str = "sam2_hiera_b+",
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
        x = (x - self.pixel_mean) / self.pixel_std
        return x.float()
    
    def set_backbone_freeze(self, freeze: bool):
        self.image_encoder.set_freeze(freeze)
    
    def forward(
        self, 
        image: torch.Tensor, 
        point_prompt: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        forward_type: int = 0,
        upscale_output: bool = True,
    ) -> Tuple[Any, Any]:
        x = self.preprocess(image)
        
        B, C, H, W = x.shape
        
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_sam2_unet(
    backbone: str = "sam2_hiera_b+",
    weights_path: Optional[str] = None,
    freeze_backbone: bool = True,
    decoder_channels: List[int] = [256, 128, 64, 32],
    num_groups: int = 8,
    img_size: int = 1024,
    device: str = "cuda",
) -> SAM2UNet:
    """
    Build SAM2-UNET model avec le repo officiel.
    
    Args:
        backbone: "sam2_hiera_t", "sam2_hiera_s", "sam2_hiera_b+", "sam2_hiera_l"
        weights_path: Chemin vers le .pt (ex: "model/sam2_hiera_base_plus.pt")
        freeze_backbone: Geler le backbone
        decoder_channels: Config du décodeur
        num_groups: Groupes pour GroupNorm
        img_size: Taille d'image
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
