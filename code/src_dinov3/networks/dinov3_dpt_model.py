"""Modele DinoV3-DPT pour la detection de falsification. Encoder ViT + decoder DPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from pathlib import Path

# import du package DINOv3 officiel
import sys
sys.path.insert(0, 'model')

try:
    import dinov3
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False
    print("WARNING: dinov3 package not found. Make sure the dinov3 repo is in 'model/' directory")


# ==================== configs DINOv3 ====================

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


# ==================== composants DPT ====================

class ResidualConvUnit(nn.Module):
    """Unite convolutionnelle residuelle du DPT."""
    
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        
        num_groups = min(num_groups, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class FeatureFusionBlock(nn.Module):
    """Bloc de fusion de features du DPT."""
    
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        
        self.resconv1 = ResidualConvUnit(channels, num_groups)
        self.resconv2 = ResidualConvUnit(channels, num_groups)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            # upsample x pour correspondre a la taille du skip
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        
        x = self.resconv1(x)
        x = self.resconv2(x)
        
        # upsample x2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        return x


class Reassemble(nn.Module):
    """Module reassemble du DPT."""
    
    def __init__(
        self, 
        embed_dim: int, 
        out_channels: int = 256,
        num_groups: int = 8,
    ):
        super().__init__()
        
        num_groups = min(num_groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        
        # projection embed_dim -> out_channels
        self.project = nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        
        # ajustements d'echelle pour chaque couche
        # les features ViT sont a resolution 1/16 (patch_size=16)
        self.upsample_4x = nn.ConvTranspose2d(
            out_channels, out_channels, 
            kernel_size=4, stride=4, padding=0, bias=False
        )
        self.upsample_2x = nn.ConvTranspose2d(
            out_channels, out_channels,
            kernel_size=2, stride=2, padding=0, bias=False
        )
        self.identity = nn.Identity()
        self.downsample_2x = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward: 4 features ViT -> 4 features multi-echelle."""
        assert len(features) == 4, f"Expected 4 features, got {len(features)}"
        
        reassembled = []
        
        for i, feat in enumerate(features):
            # projection vers dimension commune
            x = self.norm(self.project(feat))
            
            # transformation selon l'echelle
            if i == 0:  # couche la moins profonde -> res 1/4
                x = self.upsample_4x(x)
            elif i == 1:  # -> res 1/8
                x = self.upsample_2x(x)
            elif i == 2:  # -> res 1/16 (meme que sortie ViT)
                x = self.identity(x)
            else:  # couche la plus profonde -> res 1/32
                x = self.downsample_2x(x)
            
            reassembled.append(x)
        
        return reassembled


class DPTDecoder(nn.Module):
    """Decoder DPT complet."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        decoder_channels: int = 256,
        num_classes: int = 1,
        num_groups: int = 8,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels
        
        # reassemble: features ViT -> features multi-echelle
        self.reassemble = Reassemble(
            embed_dim=embed_dim,
            out_channels=decoder_channels,
            num_groups=num_groups,
        )
        
        # blocs de fusion (bottom-up, de 1/32 a 1/4)
        self.fusion4 = FeatureFusionBlock(decoder_channels, num_groups)  # 1/32 -> 1/16
        self.fusion3 = FeatureFusionBlock(decoder_channels, num_groups)  # 1/16 -> 1/8
        self.fusion2 = FeatureFusionBlock(decoder_channels, num_groups)  # 1/8 -> 1/4
        self.fusion1 = FeatureFusionBlock(decoder_channels, num_groups)  # 1/4 -> 1/2
        
        # tete de prediction finale
        self.output_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, decoder_channels // 2),
            nn.GELU(),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(decoder_channels // 2, num_classes, kernel_size=1),
        )
        
        # tete de prediction IoU (compatibilite avec le code d'entrainement)
        self.iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        encoder_features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward du decoder DPT."""
        # on inverse l'ordre (l'encoder donne deep first, on veut shallow first)
        features = encoder_features[::-1]
        
        # reassemble vers multi-echelle
        reassembled = self.reassemble(features)
        
        layer_1_4, layer_1_8, layer_1_16, layer_1_32 = reassembled
        
        # fusion bottom-up avec skip connections
        x = self.fusion4(layer_1_32, None)       # 1/32 -> 1/16
        x = self.fusion3(x, layer_1_16)          # 1/16 -> 1/8 (+ skip)
        x = self.fusion2(x, layer_1_8)           # 1/8 -> 1/4 (+ skip)
        x = self.fusion1(x, layer_1_4)           # 1/4 -> 1/2 (+ skip)
        
        # prediction IoU a partir des features fusionnees
        iou_pred = self.iou_head(x)
        
        # prediction finale
        seg_logits = self.output_conv(x)
        
        # upsample vers la taille cible
        seg_logits = F.interpolate(seg_logits, scale_factor=2, mode='bilinear', align_corners=False)
        
        if target_size is not None and seg_logits.shape[-2:] != target_size:
            seg_logits = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)
        
        return seg_logits, iou_pred


# ==================== encoder DinoV3 ====================

class DinoV3Encoder(nn.Module):
    """Encoder DinoV3 avec extraction de features intermediaires."""
    
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
        
        # couches intermediaires par defaut pour DINOv3
        if intermediate_layers is None:
            if self.num_blocks == 12:  # ViT-S, ViT-B
                self.intermediate_layers = [2, 5, 8, 11]
            elif self.num_blocks == 24:  # ViT-L
                self.intermediate_layers = [5, 11, 17, 23]
            elif self.num_blocks == 32:  # ViT-H
                self.intermediate_layers = [7, 15, 23, 31]
            else:  # 40 blocks (7B)
                self.intermediate_layers = [9, 19, 29, 39]
        else:
            self.intermediate_layers = intermediate_layers
        
        # construction du backbone
        self.backbone = self._build_backbone(model_name, img_size)
        self.patch_size = self.backbone.patch_size  # normalement 16
        
        # chargement des poids
        if weights_path is not None:
            self._load_weights(weights_path)
        
        if freeze:
            self._freeze_backbone()
    
    def _build_backbone(self, model_name: str, img_size: int) -> nn.Module:
        """Construit le backbone via les fonctions hub DINOv3."""
        from dinov3.hub.backbones import (
            dinov3_vits16,
            dinov3_vitb16,
            dinov3_vitl16,
            dinov3_vith16plus,
            dinov3_vit7b16,
        )
        
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
        model = hub_fn(pretrained=False)
        
        return model
    
    def _load_weights(self, weights_path: str):
        """Charge les poids pre-entraines."""
        print(f"Loading DINOv3 weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
        
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
        
        with torch.set_grad_enabled(not self.freeze):
            # utilise get_intermediate_layers officiel
            intermediate_outputs = self.backbone.get_intermediate_layers(
                x, 
                n=self.intermediate_layers,
                reshape=True,
                return_class_token=False,
            )
        
        features = list(intermediate_outputs)
        
        # inversion (deep first pour compatibilite, le decoder re-inverse)
        features = features[::-1]
        
        return features
    
    @property
    def output_channels(self) -> List[int]:
        return [self.embed_dim] * len(self.intermediate_layers)


# ==================== modele principal ====================

class DinoV3DPT(nn.Module):
    """Modele DinoV3 + DPT pour la segmentation."""
    
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        backbone: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        decoder_channels: int = 256,
        num_groups: int = 8,
        dropout_ratio: float = 0.1,
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
        
        embed_dim = self.image_encoder.embed_dim
        
        self.mask_decoder = DPTDecoder(
            embed_dim=embed_dim,
            decoder_channels=decoder_channels,
            num_classes=1,
            num_groups=num_groups,
            dropout_ratio=dropout_ratio,
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


# ==================== fonctions de construction ====================

def build_dinov3_dpt(
    backbone: str = "dinov3_vitb16",
    weights_path: Optional[str] = None,
    freeze_backbone: bool = True,
    decoder_channels: int = 256,
    num_groups: int = 8,
    dropout_ratio: float = 0.1,
    img_size: int = 512,
    device: str = "cuda",
) -> DinoV3DPT:
    """Construit le modele complet DinoV3-DPT."""
    model = DinoV3DPT(
        backbone=backbone,
        weights_path=weights_path,
        freeze_backbone=freeze_backbone,
        decoder_channels=decoder_channels,
        num_groups=num_groups,
        dropout_ratio=dropout_ratio,
        img_size=img_size,
    )
    
    return model.to(device)


def load_checkpoint(model: DinoV3DPT, checkpoint_path: str, device: str = "cuda") -> dict:
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


# ==================== test ====================

if __name__ == "__main__":
    # test de l'architecture du modele
    print("Testing DinoV3-DPT Model...")
    
    # modele factice (sans poids reels)
    if DINOV3_AVAILABLE:
        model = DinoV3DPT(
            backbone="dinov3_vitb16",
            weights_path=None,  # pas de poids pour le test
            freeze_backbone=True,
            decoder_channels=256,
            img_size=512,
        )
        
        # test du forward pass
        dummy_input = torch.randn(2, 3, 512, 512)
        
        with torch.no_grad():
            seg_logits, iou_pred = model(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output seg_logits shape: {seg_logits.shape}")
        print(f"Output iou_pred shape: {iou_pred.shape}")
        print(f"Total params: {model.get_total_params():,}")
        print(f"Trainable params: {model.get_trainable_params():,}")
    else:
        print("DINOv3 package not available. Cannot test model.")
