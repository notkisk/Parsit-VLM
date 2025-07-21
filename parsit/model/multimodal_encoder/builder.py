import os
from .siglip_encoder import SigLipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    if "siglip" in vision_tower or is_absolute_path_exists:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}. Parsit only supports SigLIP encoders.")