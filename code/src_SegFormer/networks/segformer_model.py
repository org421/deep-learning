"""
SegFormer Model for Copy-Move Forgery Detection.

Architecture:
- Encoder: SegFormer MiT (Mix Transformer) - variants B0 to B5
- Decoder: All-MLP decoder (standard SegFormer from Xie et al., NeurIPS 2021)
- Output: Binary segmentation mask + auxiliary heads

MLP decoder (from paper):
  Linear projection per stage -> upsample to 1/4 -> concat -> fuse -> segment
  Lightweight and memory-efficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
import math


class OverlapPatchEmbed(nn.Module):
    """Overlapping Patch Embedding for SegFormer."""

    def __init__(
        self,
        patch_size: int = 7,
        stride: int = 4,
        in_channels: int = 3,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Efficient Self-Attention with Spatial Reduction."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MixFFN(nn.Module):
    """Mix Feed-Forward Network with depthwise convolution."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size=3, padding=1, groups=hidden_features
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with Efficient Self-Attention and Mix-FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
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
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MixTransformerEncoder(nn.Module):
    """
    Mix Transformer Encoder (MiT) for SegFormer.

    B2 configuration:
    - embed_dims: [64, 128, 320, 512]
    - depths: [3, 4, 6, 3]
    - num_heads: [1, 2, 5, 8]
    - sr_ratios: [8, 4, 2, 1]
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [64, 128, 320, 512],
        depths: List[int] = [3, 4, 6, 3],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        sr_ratios: List[int] = [8, 4, 2, 1],
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_stages = len(depths)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Patch embeddings
        self.patch_embeds = nn.ModuleList()
        for i in range(self.num_stages):
            if i == 0:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7, stride=4,
                    in_channels=in_channels,
                    embed_dim=embed_dims[0]
                )
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3, stride=2,
                    in_channels=embed_dims[i - 1],
                    embed_dim=embed_dims[i]
                )
            self.patch_embeds.append(patch_embed)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        cur = 0
        for i in range(self.num_stages):
            blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    sr_ratio=sr_ratios[i],
                )
                for j in range(depths[i])
            ])
            self.blocks.append(blocks)
            cur += depths[i]

        # Layer norms at the end of each stage
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims[i]) for i in range(self.num_stages)
        ])

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
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Returns:
            List of features at 4 scales: [1/4, 1/8, 1/16, 1/32]
        """
        features = []
        B = x.shape[0]

        for i in range(self.num_stages):
            x, H, W = self.patch_embeds[i](x)

            for block in self.blocks[i]:
                x = block(x, H, W)

            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)

        return features

    @property
    def output_channels(self) -> List[int]:
        return self.embed_dims


class LayerNorm2d(nn.Module):
    """LayerNorm for 4D tensors (B, C, H, W). Gradient accumulation compatible."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x


class SegFormerMLPDecoder(nn.Module):
    """
    Standard SegFormer All-MLP Decoder from the paper (Xie et al., NeurIPS 2021).

    Architecture:
      1. MLP layer per encoder stage: project each stage to decoder_embed_dim
      2. Upsample all to 1/4 resolution
      3. Concatenate + fuse with linear layer
      4. Segmentation head

    Plus forensic auxiliary heads for forgery detection.
    Much lighter than FPN — fits in 3GB VRAM.
    """

    def __init__(
        self,
        encoder_channels: List[int] = [64, 128, 320, 512],
        decoder_embed_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim

        # --- MLP layers: one per encoder stage (Linear proj from paper) ---
        self.linear_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, decoder_embed_dim, kernel_size=1, bias=False),
                LayerNorm2d(decoder_embed_dim),
            )
            for ch in encoder_channels
        ])

        # --- Fuse concatenated features ---
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_embed_dim * len(encoder_channels), decoder_embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(decoder_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)

        # --- Segmentation head (main task) ---
        self.segmentation_head = nn.Conv2d(decoder_embed_dim, num_classes, kernel_size=1)

        # --- Edge head (auxiliary: forgery boundary detection) ---
        self.edge_head = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, decoder_embed_dim // 4, kernel_size=1, bias=False),
            LayerNorm2d(decoder_embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_embed_dim // 4, num_classes, kernel_size=1),
        )

        # --- Classification head (auxiliary: forged vs authentic image-level) ---
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_embed_dim, 1),
        )

        # --- IoU prediction head ---
        self.iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """
        Args:
            features: [stage1, stage2, stage3, stage4] from encoder
                      resolutions: [1/4, 1/8, 1/16, 1/32]
            target_size: (H, W) for final output

        Returns:
            dict with keys:
              'seg_logits': (B, 1, H, W) segmentation logits
              'edge_logits': (B, 1, H, W) edge logits
              'cls_logits': (B, 1) image-level classification logits
              'iou_pred': (B, 1) IoU prediction
        """
        # --- 1. Project each stage to decoder_embed_dim ---
        projected = []
        for i, (proj, feat) in enumerate(zip(self.linear_proj, features)):
            x = proj(feat)
            # Upsample all to stage1 resolution (1/4)
            if i > 0:
                x = F.interpolate(x, size=features[0].shape[2:], mode='bilinear', align_corners=False)
            projected.append(x)

        # --- 2. Concat + fuse ---
        x = torch.cat(projected, dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)

        # --- Heads ---
        iou_pred = self.iou_head(x)
        cls_logits = self.cls_head(x)

        seg_logits = self.segmentation_head(x)
        edge_logits = self.edge_head(x)

        # Upsample 4x to match input resolution
        if target_size is not None:
            seg_logits = F.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)
            edge_logits = F.interpolate(edge_logits, size=target_size, mode='bilinear', align_corners=False)
        else:
            seg_logits = F.interpolate(seg_logits, scale_factor=4, mode='bilinear', align_corners=False)
            edge_logits = F.interpolate(edge_logits, scale_factor=4, mode='bilinear', align_corners=False)

        return {
            'seg_logits': seg_logits,
            'edge_logits': edge_logits,
            'cls_logits': cls_logits,
            'iou_pred': iou_pred,
        }


class SegFormer(nn.Module):
    """
    SegFormer for Image Forgery Detection.

    MiT encoder + All-MLP decoder (paper default).
    Uses LayerNorm for gradient accumulation compatibility.
    """

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    # SegFormer configurations
    SEGFORMER_CONFIGS = {
        'segformer_b0': {
            'embed_dims': [32, 64, 160, 256],
            'depths': [2, 2, 2, 2],
            'num_heads': [1, 2, 5, 8],
            'decoder_embed_dim': 256,
        },
        'segformer_b1': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [2, 2, 2, 2],
            'num_heads': [1, 2, 5, 8],
            'decoder_embed_dim': 256,
        },
        'segformer_b2': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 4, 6, 3],
            'num_heads': [1, 2, 5, 8],
            'decoder_embed_dim': 768,
        },
        'segformer_b3': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 4, 18, 3],
            'num_heads': [1, 2, 5, 8],
            'decoder_embed_dim': 768,
        },
        'segformer_b4': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 8, 27, 3],
            'num_heads': [1, 2, 5, 8],
            'decoder_embed_dim': 768,
        },
        'segformer_b5': {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 6, 40, 3],
            'num_heads': [1, 2, 5, 8],
            'decoder_embed_dim': 768,
        },
    }

    def __init__(
        self,
        variant: str = "segformer_b2",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        num_classes: int = 1,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        skip_preprocess: bool = False,
    ):
        """
        Initialize SegFormer model.

        Args:
            variant: SegFormer variant (b0-b5)
            pretrained: Load pretrained weights
            freeze_backbone: Freeze backbone weights
            num_classes: Number of output classes
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
            pixel_mean: Mean for image normalization (ImageNet [0-255] range)
            pixel_std: Std for image normalization (ImageNet [0-255] range)
            skip_preprocess: If True, skip internal normalization. Default False
                because the dataset returns images in [0, 255] range.
        """
        super().__init__()

        if variant not in self.SEGFORMER_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(self.SEGFORMER_CONFIGS.keys())}")

        config = self.SEGFORMER_CONFIGS[variant]
        self.variant = variant
        self.freeze_backbone = freeze_backbone
        self.skip_preprocess = skip_preprocess

        # Encoder
        self.image_encoder = MixTransformerEncoder(
            in_channels=3,
            embed_dims=config['embed_dims'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Decoder (Standard SegFormer All-MLP decoder from paper)
        self.mask_decoder = SegFormerMLPDecoder(
            encoder_channels=config['embed_dims'],
            decoder_embed_dim=config['decoder_embed_dim'],
            num_classes=num_classes,
        )

        # Image normalization (only used if skip_preprocess=False)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights(variant)

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _load_pretrained_weights(self, variant: str):
        """Load pretrained weights from timm or custom source."""
        try:
            import timm
            # Try to load from timm
            timm_name = f"mit_{variant.split('_')[1]}"
            pretrained_model = timm.create_model(timm_name, pretrained=True)

            # Load encoder weights
            encoder_state = {}
            for k, v in pretrained_model.state_dict().items():
                if not k.startswith('head'):
                    encoder_state[k] = v

            # Try to load what we can
            missing, unexpected = self.image_encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded pretrained weights for {variant}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Initializing with random weights.")

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True

    def set_backbone_freeze(self, freeze: bool):
        """Set backbone freeze state."""
        self.freeze_backbone = freeze
        if freeze:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values (skipped if dataset already normalizes)."""
        if self.skip_preprocess:
            return x.float()
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
            point_prompt: Ignored (for API compatibility)
            forward_type: 0 = single output, 1 = multiple outputs
            upscale_output: Whether to upscale output to input size

        Returns:
            For forward_type=0: (seg_logits, decoder_outputs_dict)
                decoder_outputs_dict contains: iou_pred, edge_logits, cls_logits
            For forward_type=1: (list of seg_logits, list of decoder_outputs_dict)
        """
        B, C, H, W = image.shape

        # Preprocess
        x = self.preprocess(image)

        # Extract multi-scale encoder features
        encoder_features = self.image_encoder(x)

        # Decode
        output_size = (H, W) if upscale_output else None
        decoder_out = self.mask_decoder(encoder_features, target_size=output_size)

        seg_logits = decoder_out['seg_logits']
        aux_outputs = {
            'iou_pred': decoder_out['iou_pred'],
            'edge_logits': decoder_out['edge_logits'],
            'cls_logits': decoder_out['cls_logits'],
        }

        if forward_type == 0:
            return seg_logits, aux_outputs
        elif forward_type == 1:
            if point_prompt is not None:
                points, labels = point_prompt
                num_prompts = points.shape[1]
            else:
                num_prompts = 1

            group_of_masks = [seg_logits for _ in range(num_prompts)]
            group_of_aux = [aux_outputs for _ in range(num_prompts)]

            return group_of_masks, group_of_aux
        else:
            raise NotImplementedError(f"forward_type {forward_type} not implemented")

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def build_segformer(
    variant: str = "segformer_b2",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: str = "cuda",
) -> SegFormer:
    """
    Build SegFormer model.

    Args:
        variant: SegFormer variant (b0-b5)
        pretrained: Load pretrained weights
        freeze_backbone: Freeze backbone
        device: Device to put model on

    Returns:
        SegFormer model
    """
    model = SegFormer(
        variant=variant,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    return model.to(device)


def load_checkpoint(model: SegFormer, checkpoint_path: str, device: str = "cuda") -> dict:
    """
    Load model checkpoint.

    Args:
        model: SegFormer model
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

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
