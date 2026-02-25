# -*- coding: utf-8 -*-
"""
CMSeg-Net Model - Copy-Move Forgery Detection Network

Fidèle au code original de CMSeg-Net.
Architecture: MobileNetV2 Encoder + CoSA Modules + U-Net Decoder

Basé sur: "Copy-Move Detection in Optical Microscopy: A Segmentation Network and A Dataset"
IEEE Signal Processing Letters, 2025
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .cor import Corr


# =============================================================================
# MobileNetV2 Building Blocks
# =============================================================================

def conv_bn(inp, oup, stride):
    """Conv + BatchNorm + ReLU6"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    """1x1 Conv + BatchNorm + ReLU6"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    """
    Bloc Inverted Residual de MobileNetV2.
    Fidèle à l'architecture originale.
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# =============================================================================
# MobileNetV2 Encoder
# =============================================================================

class MobileNetV2(nn.Module):
    """
    MobileNetV2 Encoder pour CMSeg-Net.
    Fidèle au code original.
    """
    def __init__(self, n_class=1000, input_size=512, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        # Configuration MobileNetV2 standard
        interverted_residual_setting = [
            # t, c, n, s (expand_ratio, channels, num_blocks, stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        
        # Building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        
        # Building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        
        # Make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # Building classifier (non utilisé pour la segmentation)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# =============================================================================
# Spatial Attention Module
# =============================================================================

class SpatialAttention(nn.Module):
    """
    Module d'Attention Spatiale (SAM).
    Fidèle au code original de CMSeg-Net.
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        input_tensor = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = torch.sigmoid(x)
        return input_tensor + input_tensor * y


# =============================================================================
# CMSeg-Net Main Model
# =============================================================================

class UnetMobilenetV2(nn.Module):
    """
    CMSeg-Net: Copy-Move Segmentation Network.
    
    Fidèle au code original.
    
    Architecture:
    1. MobileNetV2 Encoder → Features multi-échelles
    2. Modules CoSA (Corr + ASPP + SAM) → Détection de corrélation
    3. U-Net Decoder → Masque de segmentation
    
    Args:
        num_classes: Nombre de classes de sortie (défaut: 1 pour binaire)
        num_filters: Non utilisé (pour compatibilité)
        pretrained: Charger les poids MobileNetV2 pré-entraînés
        Dropout: Taux de dropout (non utilisé)
        path: Chemin vers les poids MobileNetV2
    """
    def __init__(
        self, 
        num_classes: int = 1, 
        num_filters: int = 32, 
        pretrained: bool = True,
        Dropout: float = 0.2, 
        path: str = 'mobilenet_v2.pth.tar'
    ):
        super(UnetMobilenetV2, self).__init__()
        
        # Encoder MobileNetV2
        self.encoder = MobileNetV2(n_class=1000)
        self.num_classes = num_classes

        # =====================================================================
        # Decoder (U-Net style avec skip connections)
        # =====================================================================
        
        # Niveau 1: 1280 → 96
        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        # Niveau 2: 96 → 32
        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        # Niveau 3: 32 → 24
        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        # Niveau 4: 24 → 16
        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, stride=4, padding=0)
        self.invres4 = InvertedResidual(32, 16, 1, 6)
        
        # Niveau 5: 16 → 3 (non utilisé dans forward original)
        self.dconv5 = nn.ConvTranspose2d(16, 3, 4, padding=1, stride=2)
        self.invres5 = InvertedResidual(6, 3, 1, 6)
        
        # Transformation pour x1
        self.trans = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        
        # Couches de sortie
        self.conv_last = nn.Conv2d(16, 3, 1)
        self.conv_score = nn.Conv2d(3, 1, 1)

        # Pour compatibilité
        self.dconv_final = nn.ConvTranspose2d(1, 1, 4, padding=1, stride=2)

        # =====================================================================
        # Charger les poids pré-entraînés MobileNetV2
        # =====================================================================
        if pretrained and os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            self.encoder.load_state_dict(state_dict, strict=False)
            print(f"[MODEL] Loaded MobileNetV2 weights from {path}")
        elif pretrained:
            print(f"[MODEL] Warning: Pretrained weights not found at {path}")
        else:
            self._init_weights()

        # =====================================================================
        # Modules CoSA (Correlation-assisted Spatial Attention)
        # =====================================================================
        
        # Modules de corrélation avec différents topk
        self.corr16 = Corr(topk=16)
        self.corr24 = Corr(topk=24)
        self.corr32 = Corr(topk=32)
        self.corr96 = Corr(topk=96)
        
        # Modules ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp1 = models.segmentation.deeplabv3.ASPP(
            in_channels=96, out_channels=96, atrous_rates=[4, 8, 12, 16]
        )
        self.aspp2 = models.segmentation.deeplabv3.ASPP(
            in_channels=32, out_channels=32, atrous_rates=[4, 8, 12, 16]
        )
        self.aspp3 = models.segmentation.deeplabv3.ASPP(
            in_channels=24, out_channels=24, atrous_rates=[4, 8, 12, 16]
        )
        self.aspp4 = models.segmentation.deeplabv3.ASPP(
            in_channels=16, out_channels=16, atrous_rates=[4, 8, 12, 16]
        )

        # Modules d'attention spatiale
        self.sam1 = SpatialAttention()
        self.sam2 = SpatialAttention()
        self.sam3 = SpatialAttention()
        self.sam4 = SpatialAttention()

    def _init_weights(self):
        """Initialise les poids si pas de pré-entraînement."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass de CMSeg-Net.
        
        Args:
            x: Image d'entrée [B, 3, H, W]
            
        Returns:
            Logits de segmentation [B, 1, H/4, W/4]
        """
        # =====================================================================
        # Encoder avec extraction de features multi-échelles
        # =====================================================================
        
        # Niveau 1: features[0-1] → x1 (16 canaux)
        for n in range(0, 2):
            x = self.encoder.features[n](x)
        x1 = x
        x1 = self.aspp4(x1)
        x1 = self.sam1(x1)

        # Niveau 2: features[2-3] → x2 (24 canaux)
        for n in range(2, 4):
            x = self.encoder.features[n](x)
        x2 = x
        x2 = self.corr24(x2)
        x2 = self.aspp3(x2)
        x2 = self.sam2(x2)

        # Niveau 3: features[4-6] → x3 (32 canaux)
        for n in range(4, 7):
            x = self.encoder.features[n](x)
        x3 = x
        x3 = self.corr32(x3)
        x3 = self.aspp2(x3)
        x3 = self.sam3(x3)

        # Niveau 4: features[7-13] → x4 (96 canaux)
        for n in range(7, 14):
            x = self.encoder.features[n](x)
        x4 = x
        x4 = self.corr96(x4)
        x4 = self.aspp1(x4)
        x4 = self.sam4(x4)

        # Niveau 5: features[14-18] → x5 (1280 canaux)
        for n in range(14, 19):
            x = self.encoder.features[n](x)
        x5 = x

        # =====================================================================
        # Decoder avec skip connections
        # =====================================================================
        
        # Up1: x5 + x4
        up1 = torch.cat([x4, self.dconv1(x5)], dim=1)
        up1 = self.invres1(up1)

        # Up2: up1 + x3
        up2 = torch.cat([x3, self.dconv2(up1)], dim=1)
        up2 = self.invres2(up2)

        # Up3: up2 + x2
        up3 = torch.cat([x2, self.dconv3(up2)], dim=1)
        up3 = self.invres3(up3)

        # Up4: up3 + x1
        up4 = torch.cat([self.trans(x1), self.dconv4(up3)], dim=1)
        up4 = self.invres4(up4)
        
        # Sortie
        x = self.conv_last(up4)
        x = self.conv_score(x)

        return x

    def freeze_encoder(self):
        """Gèle l'encoder MobileNetV2."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[MODEL] Encoder frozen")

    def unfreeze_encoder(self):
        """Dégèle l'encoder MobileNetV2."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("[MODEL] Encoder unfrozen")

    def get_encoder_params(self):
        """Retourne les paramètres de l'encoder."""
        return self.encoder.parameters()

    def get_decoder_params(self):
        """Retourne les paramètres du decoder (tout sauf encoder)."""
        encoder_params = set(self.encoder.parameters())
        return [p for p in self.parameters() if p not in encoder_params]


# =============================================================================
# Model Factory
# =============================================================================

def build_cmsegnet(
    pretrained_mobilenet: Optional[str] = None,
    pretrained_cmsegnet: Optional[str] = None,
    device: str = 'cuda',
    strict_weights: bool = True
) -> UnetMobilenetV2:
    """
    Construit le modèle CMSeg-Net.
    
    Args:
        pretrained_mobilenet: Chemin vers les poids MobileNetV2
        pretrained_cmsegnet: Chemin vers les poids CMSeg-Net complet
        device: Device cible
        strict_weights: Si True, lève une erreur si les poids n'existent pas
        
    Returns:
        Modèle CMSeg-Net
        
    Raises:
        FileNotFoundError: Si strict_weights=True et les fichiers de poids n'existent pas
    """
    print("\n" + "=" * 70)
    print("[MODEL] Building CMSeg-Net...")
    print("=" * 70)
    
    # Vérifier les poids MobileNetV2
    if pretrained_mobilenet:
        if os.path.exists(pretrained_mobilenet):
            print(f"[MODEL] ✓ MobileNetV2 weights found: {pretrained_mobilenet}")
            model = UnetMobilenetV2(pretrained=True, path=pretrained_mobilenet)
        else:
            msg = f"[MODEL] ✗ MobileNetV2 weights NOT FOUND: {pretrained_mobilenet}"
            print(msg)
            if strict_weights:
                raise FileNotFoundError(msg)
            print("[MODEL] → Proceeding WITHOUT MobileNetV2 pretrained weights")
            model = UnetMobilenetV2(pretrained=False)
    else:
        print("[MODEL] → No MobileNetV2 weights specified (training from scratch)")
        model = UnetMobilenetV2(pretrained=False)
    
    # Vérifier les poids CMSeg-Net
    if pretrained_cmsegnet:
        if os.path.exists(pretrained_cmsegnet):
            print(f"[MODEL] ✓ CMSeg-Net weights found: {pretrained_cmsegnet}")
            state_dict = torch.load(pretrained_cmsegnet, map_location='cpu', weights_only=False)
            
            # Gérer les checkpoints qui contiennent plus que juste le state_dict
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Retirer le préfixe "module." si présent (DataParallel)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # Charger les poids
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[MODEL] ✓ Loaded CMSeg-Net weights from {pretrained_cmsegnet}")
            if missing:
                print(f"[MODEL]   Missing keys ({len(missing)}): {missing[:5]}..." if len(missing) > 5 else f"[MODEL]   Missing keys: {missing}")
            if unexpected:
                print(f"[MODEL]   Unexpected keys ({len(unexpected)}): {unexpected[:5]}..." if len(unexpected) > 5 else f"[MODEL]   Unexpected keys: {unexpected}")
        else:
            msg = f"[MODEL] ✗ CMSeg-Net weights NOT FOUND: {pretrained_cmsegnet}"
            print(msg)
            print(f"[MODEL]   Searched in: {os.path.abspath(pretrained_cmsegnet)}")
            if strict_weights:
                raise FileNotFoundError(
                    f"\n{'='*70}\n"
                    f"ERREUR CRITIQUE: Fichier de poids introuvable!\n"
                    f"  Chemin spécifié: {pretrained_cmsegnet}\n"
                    f"  Chemin absolu: {os.path.abspath(pretrained_cmsegnet)}\n"
                    f"\n"
                    f"Solutions:\n"
                    f"  1. Vérifiez que le fichier existe\n"
                    f"  2. Placez le fichier dans le bon répertoire\n"
                    f"  3. Mettez pretrained_cmsegnet: null dans config.yaml pour entraîner sans poids\n"
                    f"{'='*70}"
                )
            print("[MODEL] → Proceeding WITHOUT CMSeg-Net pretrained weights")
    else:
        print("[MODEL] → No CMSeg-Net weights specified (fine-tuning from MobileNetV2 only)")
    
    print("=" * 70 + "\n")
    
    return model.to(device)
