# -*- coding: utf-8 -*-
"""
Configuration Loader for CMSeg-Net.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class ConfigDict:
    """
    Wrapper to access dict with attribute syntax.
    """
    def __init__(self, d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class Config:
    """
    Configuration manager for CMSeg-Net training.
    """
    
    def __init__(self, config_dict: dict):
        self._raw = config_dict
        
        # Convert to attribute access
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, config_path: str = "configs/config.yaml") -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Config object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    def to_dict(self) -> dict:
        """Convert config back to dict."""
        return self._raw.copy()
    
    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self._raw, f, default_flow_style=False)
    
    def __repr__(self):
        return f"Config({self._raw})"


def get_device_info(config: Config) -> Dict[str, Any]:
    """
    Get device information from config.
    
    Args:
        config: Config object
        
    Returns:
        Dict with device info
    """
    import torch
    
    device_str = getattr(config.experiment, 'device', 'cuda')
    gpu_id = getattr(config.experiment, 'gpu_id', 0)
    
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        is_cuda = True
    else:
        device = torch.device('cpu')
        is_cuda = False
    
    return {
        'device': device,
        'is_cuda': is_cuda,
        'is_distributed': False,  # Single GPU for now
        'is_main': True,
        'local_rank': 0,
        'world_size': 1,
    }


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """
    Convenience function to load config.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config object
    """
    return Config.from_yaml(config_path)
