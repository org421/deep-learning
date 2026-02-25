"""
Exponential Moving Average (EMA) for model weights.

=== OPTIMISATION VRAM ===
L'EMA garde désormais ses poids sur le CPU par défaut.
La méthode ``get_model(device)`` déplace temporairement le modèle sur le GPU
uniquement pour la phase de validation, puis le remet sur CPU.
Économie : ~720 K params × 4 bytes = ~3 MB pour UCM-NetV2 large.
Sur des modèles plus grands (SegFormer, DINOv2) l'économie est considérable.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMAModel:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters that is updated
    with exponential moving average at each training step.

    Le modèle shadow vit sur le CPU pour économiser la VRAM GPU.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[str] = None,   # ignoré — shadow toujours sur CPU
    ):
        self.decay = decay

        # Copie sur CPU — pas besoin de garder sur GPU
        self.shadow = deepcopy(model).cpu()
        self.shadow.eval()

        # Désactiver les gradients sur la copie
        for param in self.shadow.parameters():
            param.requires_grad = False

        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA weights.
        Les poids du modèle actif sont détachés et copiés sur CPU pour la mise à jour.
        """
        self.num_updates += 1

        # Decay adaptatif (warmup au début)
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        model_params  = dict(model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name in shadow_params:
            if name in model_params:
                # Détacher + cpu pour éviter de garder le graphe de calcul GPU
                new_val = model_params[name].detach().cpu()
                shadow_params[name].data.mul_(decay).add_(new_val, alpha=1 - decay)

        # Copier les buffers (running stats BN, etc.)
        model_buffers  = dict(model.named_buffers())
        shadow_buffers = dict(self.shadow.named_buffers())
        for name in shadow_buffers:
            if name in model_buffers:
                shadow_buffers[name].data.copy_(model_buffers[name].detach().cpu())

    def get_model(self, device: Optional[str] = None) -> nn.Module:
        """
        Retourne le modèle EMA, éventuellement déplacé sur un device.

        Usage typique :
            ema_model = ema.get_model(device='cuda:0')   # pour la val
            # ... validation ...
            # Le shadow reste sur CPU, pas besoin de le revoir.
        """
        if device is not None:
            return self.shadow.to(device)
        return self.shadow

    def offload_to_cpu(self):
        """Rapatrier le shadow sur CPU après une validation sur GPU."""
        self.shadow = self.shadow.cpu()

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            'shadow': self.shadow.state_dict(),
            'decay': self.decay,
            'num_updates': self.num_updates,
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        self.shadow.load_state_dict(state_dict['shadow'])
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        # S'assurer que le shadow est sur CPU
        self.shadow = self.shadow.cpu()

    def to(self, device: str):
        """Déplacer le shadow (utilise avec précaution — préférez get_model(device))."""
        self.shadow.to(device)
        return self
