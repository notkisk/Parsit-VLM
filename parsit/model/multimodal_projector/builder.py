import torch
import torch.nn as nn
import re

from .pooler_projector import PoolerProjector


def init_projector_weights(module):
    """Initialize projector weights using He/Kaiming initialization for GELU activations."""
    if isinstance(module, nn.Linear):
        # He initialization works better with GELU than default Xavier
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    """
    Build vision projector with proper initialization and optional dropout.
    
    Args:
        config: Model configuration object
        delay_load: Whether to delay loading (compatibility parameter)
        **kwargs: Additional arguments
    
    Returns:
        nn.Module: The constructed projector
    """
    projector_type = getattr(config, "mm_projector_type", "linear")
    dropout_rate = getattr(config, "mm_projector_dropout", 0.0)  # Default no dropout

    if projector_type == "linear":
        linear = nn.Linear(config.mm_hidden_size, config.hidden_size)
        linear.apply(init_projector_weights)
        return linear

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        
        for _ in range(1, mlp_depth):
            modules.extend([
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(config.hidden_size, config.hidden_size)
            ])
        
        projector = nn.Sequential(*modules)
        projector.apply(init_projector_weights)
        return projector

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        
        # Add MLP layers with consistent architecture
        for _ in range(1, mlp_depth):
            modules.extend([
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(config.hidden_size, config.hidden_size)
            ])
        
        # Add residual blocks
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        
        projector = nn.Sequential(*modules)
        projector.apply(init_projector_weights)
        return projector

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")