"""Neural depth shared by still pixel art, video, and camera animation."""
from .model import depth_for_file, image_and_depth, predict_near, scale_depth

__all__ = ['depth_for_file', 'image_and_depth', 'predict_near', 'scale_depth']
