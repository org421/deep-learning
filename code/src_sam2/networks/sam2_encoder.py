"""
SAM2 Encoder utilisant le repo officiel de Facebook.
Import direct sans hydra pour éviter les problèmes d'instantiation.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


def _add_sam2_to_path():
    """Ajoute le repo sam2 au PYTHONPATH."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    possible_paths = [
        project_root / "model" / "sam2",
        Path.cwd() / "model" / "sam2",
    ]
    
    for sam2_path in possible_paths:
        if (sam2_path / "sam2").exists():
            if str(sam2_path) not in sys.path:
                sys.path.insert(0, str(sam2_path))
            return sam2_path
    
    raise ImportError("Repo sam2 non trouvé dans model/sam2")


SAM2_REPO_PATH = _add_sam2_to_path()

# Import direct des classes SAM2 (sans hydra)
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck


# Configurations
SAM2_CONFIGS = {
    'sam2_hiera_t': {
        'embed_dim': 96,
        'num_heads': 1,
        'stages': [1, 2, 7, 2],
        'global_att_blocks': [5, 7, 9],
        'window_pos_embed_bkg_spatial_size': [7, 7],
        'out_chans': 256,
    },
    'sam2_hiera_s': {
        'embed_dim': 96,
        'num_heads': 1,
        'stages': [1, 2, 11, 2],
        'global_att_blocks': [7, 10, 13],
        'window_pos_embed_bkg_spatial_size': [7, 7],
        'out_chans': 256,
    },
    'sam2_hiera_b+': {
        'embed_dim': 112,
        'num_heads': 2,
        'stages': [2, 3, 16, 3],
        'global_att_blocks': [12, 16, 20],
        'window_pos_embed_bkg_spatial_size': [14, 14],
        'out_chans': 256,
    },
    'sam2_hiera_l': {
        'embed_dim': 144,
        'num_heads': 2,
        'stages': [2, 6, 36, 4],
        'global_att_blocks': [23, 33, 43],
        'window_pos_embed_bkg_spatial_size': [7, 7],
        'out_chans': 256,
    },
}
SAM2_CONFIGS['sam2_hiera_b'] = SAM2_CONFIGS['sam2_hiera_b+']


class LayerNorm2d(nn.Module):
    """LayerNorm pour format (B, C, H, W)."""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SAM2ImageEncoder(nn.Module):
    """
    Wrapper autour du backbone Hiera de SAM2.
    Construit manuellement sans utiliser hydra.
    """
    
    def __init__(
        self,
        model_name: str = 'sam2_hiera_b+',
        weights_path: Optional[str] = None,
        device: str = 'cpu',
    ):
        super().__init__()
        
        if model_name not in SAM2_CONFIGS:
            raise ValueError(f"Modèle inconnu: {model_name}")
        
        cfg = SAM2_CONFIGS[model_name]
        self.model_name = model_name
        self.out_chans = cfg['out_chans']
        
        print(f"Construction de SAM2 Hiera: {model_name}")
        
        # Construire le backbone Hiera manuellement
        self.trunk = Hiera(
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            stages=cfg['stages'],
            global_att_blocks=cfg['global_att_blocks'],
            window_pos_embed_bkg_spatial_size=cfg['window_pos_embed_bkg_spatial_size'],
        )
        
        # Récupérer les dimensions de sortie de chaque stage
        embed_dim = cfg['embed_dim']
        self.stage_dims = [embed_dim * (2 ** i) for i in range(4)]
        
        # Notre propre neck (évite le FpnNeck de SAM2 qui utilise un LayerNorm incompatible)
        self.neck = nn.ModuleList()
        for dim in self.stage_dims:
            self.neck.append(
                nn.Sequential(
                    nn.Conv2d(dim, self.out_chans, kernel_size=1, bias=False),
                    LayerNorm2d(self.out_chans),
                    nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, padding=1, bias=False),
                    LayerNorm2d(self.out_chans),
                )
            )
        
        # Charger les poids si fournis
        if weights_path:
            self._load_weights(weights_path, device)
    
    def _load_weights(self, weights_path: str, device: str):
        """Charge les poids du checkpoint SAM2."""
        print(f"Chargement des poids depuis: {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Extraire les poids du trunk (backbone)
        trunk_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('image_encoder.trunk.'):
                new_key = k.replace('image_encoder.trunk.', '')
                trunk_state_dict[new_key] = v
        
        if trunk_state_dict:
            missing, unexpected = self.trunk.load_state_dict(trunk_state_dict, strict=False)
            print(f"Trunk: {len(trunk_state_dict)} poids chargés")
            if missing:
                print(f"  Missing: {len(missing)}")
            if unexpected:
                print(f"  Unexpected: {len(unexpected)}")
        else:
            print("Attention: Aucun poids trouvé pour le trunk")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extrait les features multi-échelles.
        
        Args:
            x: (B, 3, H, W) image normalisée
            
        Returns:
            Liste de 4 features au format (B, 256, H_i, W_i)
        """
        # Extraire les features du trunk
        features = self._get_trunk_features(x)
        
        # Appliquer notre neck
        out_features = []
        for feat, neck_layer in zip(features, self.neck):
            # Hiera retourne toujours en (B, H, W, C) -> convertir en (B, C, H, W)
            if feat.dim() == 4 and feat.shape[-1] in self.stage_dims:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out_features.append(neck_layer(feat))
        
        return out_features
    
    def _get_trunk_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extrait les features intermédiaires du trunk Hiera."""
        
        # Méthode 1: utiliser _get_intermediate_layers si disponible
        if hasattr(self.trunk, '_get_intermediate_layers'):
            try:
                feats, pos = self.trunk._get_intermediate_layers(x)
                # Convertir en (B, C, H, W) - Hiera retourne toujours (B, H, W, C)
                processed = []
                for feat in feats:
                    if feat.dim() == 3:  # (B, N, C)
                        B, N, C = feat.shape
                        H = W = int(N ** 0.5)
                        feat = feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                    elif feat.dim() == 4 and feat.shape[-1] in self.stage_dims:  # (B, H, W, C)
                        feat = feat.permute(0, 3, 1, 2).contiguous()
                    processed.append(feat)
                return processed
            except Exception as e:
                print(f"_get_intermediate_layers failed: {e}")
        
        # Méthode 2: forward avec hooks
        features = []
        hooks = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Enregistrer des hooks sur les blocs de sortie de chaque stage
        # Structure Hiera: les stages sont définis par stage_ends
        if hasattr(self.trunk, 'stage_ends'):
            stage_ends = self.trunk.stage_ends
        else:
            # Calculer stage_ends depuis la config
            stages = SAM2_CONFIGS[self.model_name]['stages']
            stage_ends = []
            cumsum = 0
            for s in stages:
                cumsum += s
                stage_ends.append(cumsum - 1)
        
        # Parcourir les blocs
        block_idx = 0
        for name, module in self.trunk.named_modules():
            if name.startswith('blocks.') and name.count('.') == 1:
                if block_idx in stage_ends:
                    h = module.register_forward_hook(hook_fn)
                    hooks.append(h)
                block_idx += 1
        
        # Forward
        try:
            _ = self.trunk(x)
        except Exception as e:
            # Parfois le forward direct échoue, essayer autrement
            print(f"Forward direct échoué: {e}")
        
        # Retirer les hooks
        for h in hooks:
            h.remove()
        
        # Convertir les features - Hiera retourne (B, H, W, C)
        processed = []
        for feat in features:
            if isinstance(feat, tuple):
                feat = feat[0]
            if feat.dim() == 3:  # (B, N, C)
                B, N, C = feat.shape
                H = W = int(N ** 0.5)
                feat = feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            elif feat.dim() == 4 and feat.shape[-1] in self.stage_dims:  # (B, H, W, C)
                feat = feat.permute(0, 3, 1, 2).contiguous()
            processed.append(feat)
        
        # S'assurer qu'on a 4 features
        while len(processed) < 4:
            if processed:
                processed.append(processed[-1])
            else:
                raise RuntimeError("Impossible d'extraire les features du trunk")
        
        return processed[:4]
    
    @property
    def output_channels(self) -> List[int]:
        return [self.out_chans] * 4


def build_sam2_encoder(
    model_name: str = 'sam2_hiera_b+',
    weights_path: Optional[str] = None,
    img_size: int = 1024,
    device: str = 'cuda'
) -> SAM2ImageEncoder:
    """Construit l'encodeur SAM2."""
    encoder = SAM2ImageEncoder(
        model_name=model_name,
        weights_path=weights_path,
        device=device,
    )
    return encoder.to(device)
