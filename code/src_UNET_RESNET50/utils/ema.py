"""
Exponential Moving Average (EMA) for model weights.
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
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[str] = None,
    ):
        """
        Initialize EMA model.
        
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update)
            device: Device for EMA weights
        """
        self.decay = decay
        self.device = device
        
        # Create shadow copy of model
        self.shadow = deepcopy(model)
        self.shadow.eval()
        
        # Disable gradients for shadow model
        for param in self.shadow.parameters():
            param.requires_grad = False
        
        if device is not None:
            self.shadow.to(device)
        
        self.num_updates = 0
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA weights.
        
        Args:
            model: Current model with updated weights
        """
        self.num_updates += 1
        
        # Compute adaptive decay based on number of updates
        # This provides warmup at the beginning
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        # Update shadow parameters
        model_params = dict(model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())
        
        for name in shadow_params:
            if name in model_params:
                # EMA update: shadow = decay * shadow + (1 - decay) * model
                shadow_params[name].data.mul_(decay).add_(
                    model_params[name].data, alpha=1 - decay
                )
        
        # Update buffers (BatchNorm running stats, etc.)
        model_buffers = dict(model.named_buffers())
        shadow_buffers = dict(self.shadow.named_buffers())
        
        for name in shadow_buffers:
            if name in model_buffers:
                shadow_buffers[name].data.copy_(model_buffers[name].data)
    
    def get_model(self) -> nn.Module:
        """Get the EMA model for evaluation."""
        return self.shadow
    
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
    
    def to(self, device: str):
        """Move EMA model to device."""
        self.shadow.to(device)
        self.device = device
        return self
