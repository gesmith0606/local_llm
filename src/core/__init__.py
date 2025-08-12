"""
Core package for AI image generation.
"""

__version__ = "1.0.0"
__author__ = "AI Image Generation Team"

from .config import Config
from .generator import ImageGenerator
from .models import ModelManager

__all__ = ["Config", "ImageGenerator", "ModelManager"]
