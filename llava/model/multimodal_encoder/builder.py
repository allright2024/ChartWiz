import os
from .deplot_encoder import DeplotVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    return DeplotVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

