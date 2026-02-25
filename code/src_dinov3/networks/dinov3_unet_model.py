"""Modele DinoV3-UNet pour la detection de falsification. Encoder ViT + decoder UNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from pathlib import Path

# Import du package dinov3
import sys
sys.path.insert(0, 'model')  # chemin vers dinov3

try:
    # on verifie si dinov3 est disponible
    import dinov3
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False
    print("WARNING: dinov3 package not found. Make sure the dinov3 repo is in 'model/' directory")


# ==================== Configs DINOv3 ====================

DINOV3_CONFIGS = {
    'dinov3_vits16': {
        'builder': 'vit_small',
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
    },
    'dinov3_vitb16': {
        'builder': 'vit_base',
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
    },
    'dinov3_vitl16': {
        'builder': 'vit_large',
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
    },
    'dinov3_vith16plus': {
        'builder': 'vit_huge2',
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 20,
    },
    'dinov3_vit7b16': {
        'builder': 'vit_7b',
        'embed_dim': 4096,
        'depth': 40,
        'num_heads': 32,
    },
}


class ConvBlock(nn.Module):
    """Bloc convolutif avec GroupNorm."""
    
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
    """Bloc upsample + conv pour le decoder."""
    
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
    """Decoder UNet avec skip connections."""
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 1,
        num_groups: int = 8,
    ):
        super().__init__()
        
        self.decoder_channels = decoder_channels
        
        self.initial_conv = ConvBlock(encoder_channels[0], decoder_channels[0], num_groups=num_groups)
        
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            if i + 1 < len(encoder_channels):
                skip_ch = encoder_channels[i + 1]
                self.skip_projections.append(
                    nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
                )
            else:
                self.skip_projections.append(None)
            
            self.upsample_blocks.append(UpsampleBlock(in_ch, out_ch, num_groups=num_groups))
            
            concat_ch = out_ch * 2 if i + 1 < len(encoder_channels) else out_ch
            self.decoder_blocks.append(ConvBlock(concat_ch, out_ch, num_groups=num_groups))
        
        self.final_upsample = nn.Sequential(
            UpsampleBlock(decoder_channels[-1], decoder_channels[-1], num_groups=num_groups),
            UpsampleBlock(decoder_channels[-1], decoder_channels[-1], num_groups=num_groups),
        )
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1),
        )
        
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
        
        x = self.initial_conv(encoder_features[0])
        
        for i, (upsample, decoder_block) in enumerate(zip(self.upsample_blocks, self.decoder_blocks)):
            x = upsample(x)
            
            if i + 1 < len(encoder_features) and self.skip_projections[i] is not None:
                skip = encoder_features[i + 1]
                skip = self.skip_projections[i](skip)
                
                if x.shape[-2:] != skip.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
                x = torch.cat([x, skip], dim=1)
            
            x = decoder_block(x)
        
        x = self.final_upsample(x)
        
        features_for_iou = x
        seg_logits = self.segmentation_head(x)
        
        if target_size is not None and seg_logits.shape[-2:] != target_size:
            seg_logits = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)
        
        iou_pred = self.iou_head(features_for_iou)
        
        return seg_logits, iou_pred


class DinoV3Encoder(nn.Module):
    """Encoder DinoV3 base sur le ViT officiel."""
    
    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        freeze: bool = True,
        intermediate_layers: List[int] = None,
        img_size: int = 512,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # recup config
        if model_name not in DINOV3_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(DINOV3_CONFIGS.keys())}")
        
        config = DINOV3_CONFIGS[model_name]
        self.embed_dim = config['embed_dim']
        self.num_blocks = config['depth']
        
        # couches intermediaires par defaut
        if intermediate_layers is None:
            if self.num_blocks == 12:
                self.intermediate_layers = [2, 5, 8, 11]
            elif self.num_blocks == 24:
                self.intermediate_layers = [5, 11, 17, 23]
            elif self.num_blocks == 32:  # vith16plus
                self.intermediate_layers = [7, 15, 23, 31]
            else:  # 40 blocks (7B)
                self.intermediate_layers = [9, 19, 29, 39]
        else:
            self.intermediate_layers = intermediate_layers
        
        # construction du backbone
        self.backbone = self._build_backbone(model_name, img_size)
        self.patch_size = self.backbone.patch_size  # normalement 16
        
        # charger les poids si fournis
        if weights_path is not None:
            self._load_weights(weights_path)
        
        if freeze:
            self._freeze_backbone()
    
    def _build_backbone(self, model_name: str, img_size: int) -> nn.Module:
        """Construit le backbone via les fonctions hub DINOv3."""
        # import des fonctions hub
        from dinov3.hub.backbones import (
            dinov3_vits16,
            dinov3_vitb16,
            dinov3_vitl16,
            dinov3_vith16plus,
            dinov3_vit7b16,
        )
        
        # correspondance nom -> fonction hub
        hub_functions = {
            'dinov3_vits16': dinov3_vits16,
            'dinov3_vitb16': dinov3_vitb16,
            'dinov3_vitl16': dinov3_vitl16,
            'dinov3_vith16plus': dinov3_vith16plus,
            'dinov3_vit7b16': dinov3_vit7b16,
        }
        
        if model_name not in hub_functions:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(hub_functions.keys())}")
        
        hub_fn = hub_functions[model_name]
        
        # creation du modele sans poids pre-entraines (on les charge apres)
        model = hub_fn(pretrained=False)
        
        return model
    
    def _load_weights(self, weights_path: str):
        """Charge les poids pre-entraines."""
        print(f"Loading DINOv3 weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        # gerer les differents formats de checkpoint
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # retirer le prefixe si present
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
        
        # chargement flexible
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
            for key in missing[:5]:
                print(f"  - {key}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
            for key in unexpected[:5]:
                print(f"  - {key}")
            if len(unexpected) > 5:
                print(f"  ... and {len(unexpected) - 5} more")
        
        print("DINOv3 weights loaded successfully!")
    
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
        B, C, H, W = x.shape
        
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size must be divisible by patch_size {self.patch_size}"
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        with torch.set_grad_enabled(not self.freeze):
            # extraction des features intermediaires
            intermediate_outputs = self.backbone.get_intermediate_layers(
                x, 
                n=self.intermediate_layers,
                reshape=True,  # format spatial (B, C, H, W)
                return_class_token=False,
            )
        
        # tuple de tenseurs (B, C, H, W)
        features = list(intermediate_outputs)
        
        # ordre inverse (le plus profond en premier pour le decoder)
        features = features[::-1]
        
        return features
    
    @property
    def output_channels(self) -> List[int]:
        return [self.embed_dim] * len(self.intermediate_layers)


class DinoV3UNet(nn.Module):
    """Modele DinoV3 + UNet pour la segmentation."""
    
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        backbone: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_groups: int = 8,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        img_size: int = 512,
    ):
        super().__init__()
        
        self.image_encoder = DinoV3Encoder(
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
        self.patch_size = self.image_encoder.patch_size
    
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
        target_h = (H // self.patch_size) * self.patch_size
        target_w = (W // self.patch_size) * self.patch_size
        
        if H != target_h or W != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        encoder_features = self.image_encoder(x)
        
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


def build_dinov3_unet(
    backbone: str = "dinov3_vitb16",
    weights_path: Optional[str] = None,
    freeze_backbone: bool = True,
    decoder_channels: List[int] = [256, 128, 64, 32],
    num_groups: int = 8,
    img_size: int = 512,
    device: str = "cuda",
) -> DinoV3UNet:
    """Construit le modele complet DinoV3-UNet."""
    model = DinoV3UNet(
        backbone=backbone,
        weights_path=weights_path,
        freeze_backbone=freeze_backbone,
        decoder_channels=decoder_channels,
        num_groups=num_groups,
        img_size=img_size,
    )
    
    return model.to(device)


def load_checkpoint(model: DinoV3UNet, checkpoint_path: str, device: str = "cuda") -> dict:
    """Charge un checkpoint."""
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
