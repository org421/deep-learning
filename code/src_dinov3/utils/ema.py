"""
EMA - moyenne mobile exponentielle des poids du modele.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMAModel:
    """EMA des poids du modele."""
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[str] = None,
    ):
        # copie shadow du modele
        self.decay = decay
        self.device = device
        
        # copie du modele
        self.shadow = deepcopy(model)
        self.shadow.eval()
        
        # pas de gradients pour le shadow
        for param in self.shadow.parameters():
            param.requires_grad = False
        
        if device is not None:
            self.shadow.to(device)
        
        self.num_updates = 0
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Met a jour les poids EMA."""
        self.num_updates += 1
        
        # decay adaptatif (warmup au debut)
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        # maj des params shadow
        model_params = dict(model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())
        
        for name in shadow_params:
            if name in model_params:
                # formule EMA
                shadow_params[name].data.mul_(decay).add_(
                    model_params[name].data, alpha=1 - decay
                )
        
        # copie des buffers (stats BatchNorm etc.)
        model_buffers = dict(model.named_buffers())
        shadow_buffers = dict(self.shadow.named_buffers())
        
        for name in shadow_buffers:
            if name in model_buffers:
                shadow_buffers[name].data.copy_(model_buffers[name].data)
    
    def get_model(self) -> nn.Module:
        """Renvoie le modele EMA."""
        return self.shadow
    
    def state_dict(self) -> dict:
        """Pour la sauvegarde."""
        return {
            'shadow': self.shadow.state_dict(),
            'decay': self.decay,
            'num_updates': self.num_updates,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Chargement depuis checkpoint."""
        self.shadow.load_state_dict(state_dict['shadow'])
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
    
    def to(self, device: str):
        """Deplace sur le device."""
        self.shadow.to(device)
        self.device = device
        return self
