"""
Hiera Vision Transformer - Version Offline pour SAM2.

Architecture locale pour chargement sans internet.
Compatible avec les poids officiels de Meta SAM2.

Hiera est un ViT hierarchique avec pooling progressif qui produit
des features multi-échelles, idéal pour la segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Optional, Tuple, Dict, Callable
import math


# =============================================================================
# Hiera Building Blocks
# =============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    
    Args:
        x: (B, H, W, C)
        window_size: window size
        
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
        (Hp, Wp): padded height and width
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, 
    window_size: int, 
    pad_hw: Tuple[int, int], 
    hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        pad_hw: (Hp, Wp)
        hw: (H, W) before padding
        
    Returns:
        x: (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with kernel and stride."""
    
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (7, 7),
        stride: Tuple[int, int] = (4, 4),
        padding: Tuple[int, int] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head Attention with optional window attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HieraBlock(nn.Module):
    """Hiera Transformer Block with optional window attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        window_size: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Window partition if needed
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
            x = x.view(-1, self.window_size * self.window_size, C)
        else:
            pad_hw = (H, W)
            x = x.view(B, H * W, C)

        x = self.attn(x)

        # Window unpartition if needed
        if self.window_size > 0:
            x = x.view(-1, self.window_size, self.window_size, C)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = x.view(B, H, W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling."""
    
    def __init__(self, dim: int, out_dim: Optional[int] = None, norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            (B, H/2, W/2, 2*C)
        """
        B, H, W, C = x.shape
        
        # Padding if needed
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# =============================================================================
# Hiera Image Encoder (SAM2 Backbone)
# =============================================================================

class Hiera(nn.Module):
    """
    Hiera: A Hierarchical Vision Transformer.
    
    This is the image encoder used in SAM2. It produces multi-scale features
    through progressive downsampling.
    """
    
    def __init__(
        self,
        img_size: int = 1024,
        in_chans: int = 3,
        embed_dim: int = 96,
        num_heads: List[int] = [1, 2, 4, 8],
        depths: List[int] = [2, 3, 16, 3],
        window_sizes: List[int] = [8, 4, 14, 7],
        global_attn_indexes: List[int] = [5, 7, 11, 13, 15, 17, 21, 23],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depths = depths
        self.num_stages = len(depths)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        # Stochastic depth
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        dim = embed_dim
        block_idx = 0
        
        for stage_idx in range(self.num_stages):
            depth = depths[stage_idx]
            n_heads = num_heads[stage_idx]
            window_size = window_sizes[stage_idx]
            
            # Blocks for this stage
            blocks = []
            for i in range(depth):
                # Use global attention at specified indices
                use_window = block_idx not in global_attn_indexes
                block = HieraBlock(
                    dim=dim,
                    num_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    window_size=window_size if use_window else 0,
                )
                blocks.append(block)
                block_idx += 1
            
            self.stages.append(nn.ModuleList(blocks))
            
            # Downsample between stages (except last)
            if stage_idx < self.num_stages - 1:
                out_dim = dim * 2
                downsample = PatchMerging(dim, out_dim=out_dim)
                self.downsamples.append(downsample)
                dim = out_dim
            else:
                self.downsamples.append(nn.Identity())
        
        # Output dimensions for each stage
        self.stage_dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        
        # Final norm
        self.norm = nn.LayerNorm(self.stage_dims[-1])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning final features."""
        x = self.patch_embed(x)
        
        for stage, downsample in zip(self.stages, self.downsamples):
            for block in stage:
                x = block(x)
            x = downsample(x)
        
        x = self.norm(x)
        return x
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: List[int],
        reshape: bool = True,
    ) -> List[torch.Tensor]:
        """
        Get features from intermediate stages.
        
        Args:
            x: Input tensor (B, C, H, W)
            n: List of stage indices to extract (0-indexed)
            reshape: If True, return features in (B, C, H, W) format
            
        Returns:
            List of features from specified stages
        """
        x = self.patch_embed(x)
        
        outputs = []
        
        for stage_idx, (stage, downsample) in enumerate(zip(self.stages, self.downsamples)):
            for block in stage:
                x = block(x)
            
            if stage_idx in n:
                if reshape:
                    # (B, H, W, C) -> (B, C, H, W)
                    feat = x.permute(0, 3, 1, 2).contiguous()
                else:
                    feat = x.clone()
                outputs.append(feat)
            
            x = downsample(x)
        
        return outputs


# =============================================================================
# SAM2 Image Encoder Wrapper
# =============================================================================

class SAM2ImageEncoder(nn.Module):
    """
    SAM2 Image Encoder wrapper around Hiera.
    
    Adds position embeddings and neck for FPN-like multi-scale features.
    """
    
    def __init__(
        self,
        img_size: int = 1024,
        backbone_type: str = "hiera_b",
        out_chans: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        
        # Build backbone based on type
        backbone_configs = {
            "hiera_t": {
                "embed_dim": 96,
                "num_heads": [1, 2, 4, 8],
                "depths": [1, 2, 7, 2],
                "window_sizes": [8, 4, 14, 7],
                "global_attn_indexes": [5, 7, 9],
            },
            "hiera_s": {
                "embed_dim": 96,
                "num_heads": [1, 2, 4, 8],
                "depths": [1, 2, 11, 2],
                "window_sizes": [8, 4, 14, 7],
                "global_attn_indexes": [7, 9, 11, 13],
            },
            "hiera_b": {
                "embed_dim": 96,
                "num_heads": [1, 2, 4, 8],
                "depths": [2, 3, 16, 3],
                "window_sizes": [8, 4, 14, 7],
                "global_attn_indexes": [5, 7, 11, 13, 15, 17, 21, 23],
            },
            "hiera_b+": {
                "embed_dim": 112,
                "num_heads": [2, 4, 8, 16],
                "depths": [2, 3, 16, 3],
                "window_sizes": [8, 4, 14, 7],
                "global_attn_indexes": [5, 7, 11, 13, 15, 17, 21, 23],
            },
            "hiera_l": {
                "embed_dim": 144,
                "num_heads": [2, 4, 8, 16],
                "depths": [2, 6, 36, 4],
                "window_sizes": [8, 4, 14, 7],
                "global_attn_indexes": [12, 16, 20, 24, 28, 32, 36, 40],
            },
        }
        
        if backbone_type not in backbone_configs:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        config = backbone_configs[backbone_type]
        self.backbone = Hiera(img_size=img_size, **config)
        self.backbone_type = backbone_type
        
        # Neck: project multi-scale features to out_chans
        self.neck = nn.ModuleList()
        for dim in self.backbone.stage_dims:
            self.neck.append(
                nn.Sequential(
                    nn.Conv2d(dim, out_chans, kernel_size=1, bias=False),
                    nn.LayerNorm([out_chans]),
                    nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                    nn.LayerNorm([out_chans]),
                )
            )
        
        self.out_chans = out_chans
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: (B, 3, H, W) input image
            
        Returns:
            List of features at different scales
        """
        # Get features from all stages
        features = self.backbone.get_intermediate_layers(
            x, 
            n=list(range(self.backbone.num_stages)),
            reshape=True,
        )
        
        # Apply neck
        out_features = []
        for feat, neck in zip(features, self.neck):
            out_features.append(neck(feat))
        
        return out_features


# =============================================================================
# Configurations
# =============================================================================

SAM2_CONFIGS = {
    'sam2_hiera_t': {
        'backbone_type': 'hiera_t',
        'embed_dim': 96,
        'out_chans': 256,
    },
    'sam2_hiera_s': {
        'backbone_type': 'hiera_s',
        'embed_dim': 96,
        'out_chans': 256,
    },
    'sam2_hiera_b': {
        'backbone_type': 'hiera_b',
        'embed_dim': 96,
        'out_chans': 256,
    },
    'sam2_hiera_b+': {
        'backbone_type': 'hiera_b+',
        'embed_dim': 112,
        'out_chans': 256,
    },
    'sam2_hiera_l': {
        'backbone_type': 'hiera_l',
        'embed_dim': 144,
        'out_chans': 256,
    },
}


def build_sam2_encoder(
    model_name: str = 'sam2_hiera_b',
    weights_path: Optional[str] = None,
    img_size: int = 1024,
    device: str = 'cuda'
) -> SAM2ImageEncoder:
    """
    Build SAM2 image encoder and load local weights.
    
    Args:
        model_name: Name of model configuration
        weights_path: Path to .pt weights file
        img_size: Input image size
        device: Device to load model on
        
    Returns:
        SAM2ImageEncoder model
    """
    if model_name not in SAM2_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(SAM2_CONFIGS.keys())}")
    
    config = SAM2_CONFIGS[model_name]
    
    model = SAM2ImageEncoder(
        img_size=img_size,
        backbone_type=config['backbone_type'],
        out_chans=config['out_chans'],
    )
    
    # Load weights if provided
    if weights_path is not None:
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats from SAM2
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Extract only image encoder weights
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('image_encoder.'):
                new_key = k.replace('image_encoder.', '')
                encoder_state_dict[new_key] = v
            elif k.startswith('trunk.'):
                # SAM2 uses 'trunk' for backbone
                new_key = k.replace('trunk.', 'backbone.')
                encoder_state_dict[new_key] = v
            elif not any(k.startswith(prefix) for prefix in ['sam_mask_decoder', 'sam_prompt_encoder', 'memory_encoder', 'memory_attention']):
                encoder_state_dict[k] = v
        
        # Load with flexibility
        if encoder_state_dict:
            missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
            print("Weights loaded successfully!")
        else:
            print("Warning: No encoder weights found in checkpoint")
    
    return model.to(device)
