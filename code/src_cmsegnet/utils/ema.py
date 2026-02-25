# -*- coding: utf-8 -*-
"""
Exponential Moving Average (EMA) for model weights.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from copy import deepcopy


class EMAModel:
    """
    Exponential Moving Average of model weights.
    
    Maintains a shadow copy of model weights that is updated
    with exponential moving average at each training step.
    
    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.999)
        device: Device for shadow weights
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: str = None
    ):
        self.decay = decay
        self.device = device
        
        # Create shadow model
        self.shadow = deepcopy(model)
        self.shadow.eval()
        
        # Disable gradients for shadow
        for param in self.shadow.parameters():
            param.requires_grad = False
        
        if device:
            self.shadow = self.shadow.to(device)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update shadow weights with EMA.
        
        Args:
            model: Current model with updated weights
        """
        for shadow_param, model_param in zip(
            self.shadow.parameters(), 
            model.parameters()
        ):
            shadow_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1 - self.decay
            )
        
        # Also update buffers (e.g., BatchNorm running stats)
        for shadow_buffer, model_buffer in zip(
            self.shadow.buffers(),
            model.buffers()
        ):
            shadow_buffer.data.copy_(model_buffer.data)
    
    def get_model(self) -> nn.Module:
        """Get the shadow model."""
        return self.shadow
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict of shadow model."""
        return self.shadow.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict into shadow model."""
        self.shadow.load_state_dict(state_dict)


class EMA:
    """
    Simple EMA implementation (alternative to EMAModel).
    
    Stores shadow weights as a dict rather than a full model.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights after each training step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] + 
                        (1 - self.decay) * param.data
                    )
                else:
                    self.shadow[name] = param.data.clone()
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict[str, Any]:
        return {'decay': self.decay, 'shadow': self.shadow}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
