"""Image processing utilities for the AI image generation project."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import hashlib

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PIL.ExifTags import TAGS
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for generated images."""
    filename: str
    prompt: str
    enhanced_prompt: Optional[str] = None
    model_name: str = ""
    width: int = 0
    height: int = 0
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    generation_time: float = 0.0
    timestamp: datetime = None
    file_size: int = 0
    format: str = "PNG"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ImageProcessor:
    """Utility class for image processing operations."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
    
    def resize_image(
        self,
        image: Image.Image,
        width: int,
        height: int,
        maintain_aspect: bool = True,
        resample: int = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """Resize an image to specified dimensions.
        
        Args:
            image: PIL Image to resize
            width: Target width
            height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            resample: Resampling algorithm
            
        Returns:
            Resized PIL Image
        """
        if maintain_aspect:
            image.thumbnail((width, height), resample)
            return image
        else:
            return image.resize((width, height), resample)
    
    def crop_image(
        self,
        image: Image.Image,
        crop_box: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Crop an image to specified box.
        
        Args:
            image: PIL Image to crop
            crop_box: (left, top, right, bottom) coordinates
            
        Returns:
            Cropped PIL Image
        """
        return image.crop(crop_box)
    
    def enhance_image(
        self,
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0
    ) -> Image.Image:
        """Apply enhancements to an image.
        
        Args:
            image: PIL Image to enhance
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)
            
        Returns:
            Enhanced PIL Image
        """
        enhanced = image
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)
        
        return enhanced
    
    def apply_filter(
        self,
        image: Image.Image,
        filter_type: str,
        **kwargs
    ) -> Image.Image:
        """Apply a filter to an image.
        
        Args:
            image: PIL Image to filter
            filter_type: Type of filter to apply
            **kwargs: Filter-specific parameters
            
        Returns:
            Filtered PIL Image
        """
        filters = {
            'blur': ImageFilter.BLUR,
            'contour': ImageFilter.CONTOUR,
            'detail': ImageFilter.DETAIL,
            'edge_enhance': ImageFilter.EDGE_ENHANCE,
            'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
            'emboss': ImageFilter.EMBOSS,
            'find_edges': ImageFilter.FIND_EDGES,
            'sharpen': ImageFilter.SHARPEN,
            'smooth': ImageFilter.SMOOTH,
            'smooth_more': ImageFilter.SMOOTH_MORE,
        }
        
        filter_obj = filters.get(filter_type.lower())
        if filter_obj:
            return image.filter(filter_obj)
        
        # Handle parametric filters
        if filter_type.lower() == 'gaussian_blur':
            radius = kwargs.get('radius', 2)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif filter_type.lower() == 'unsharp_mask':
            radius = kwargs.get('radius', 2)
            percent = kwargs.get('percent', 150)
            threshold = kwargs.get('threshold', 3)
            return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        
        logger.warning(f"Unknown filter type: {filter_type}")
        return image
    
    def create_collage(
        self,
        images: List[Image.Image],
        grid_size: Optional[Tuple[int, int]] = None,
        spacing: int = 10,
        background_color: Union[str, Tuple[int, int, int]] = "white"
    ) -> Image.Image:
        """Create a collage from multiple images.
        
        Args:
            images: List of PIL Images
            grid_size: (rows, cols) for grid layout. Auto-calculated if None
            spacing: Spacing between images in pixels
            background_color: Background color
            
        Returns:
            Collage as PIL Image
        """
        if not images:
            raise ValueError("No images provided for collage")
        
        # Calculate grid size if not provided
        if grid_size is None:
            n_images = len(images)
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        # Get maximum dimensions
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Calculate collage dimensions
        collage_width = cols * max_width + (cols + 1) * spacing
        collage_height = rows * max_height + (rows + 1) * spacing
        
        # Create collage canvas
        collage = Image.new('RGB', (collage_width, collage_height), background_color)
        
        # Place images
        for i, image in enumerate(images):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            
            # Calculate position
            x = spacing + col * (max_width + spacing)
            y = spacing + row * (max_height + spacing)
            
            # Center image in cell
            x_offset = (max_width - image.width) // 2
            y_offset = (max_height - image.height) // 2
            
            collage.paste(image, (x + x_offset, y + y_offset))
        
        return collage
    
    def add_watermark(
        self,
        image: Image.Image,
        watermark_text: str,
        position: str = "bottom_right",
        opacity: float = 0.5,
        font_size: int = 24
    ) -> Image.Image:
        """Add a text watermark to an image.
        
        Args:
            image: PIL Image to watermark
            watermark_text: Text to add as watermark
            position: Position of watermark
            opacity: Watermark opacity (0.0 to 1.0)
            font_size: Font size for watermark
            
        Returns:
            Watermarked PIL Image
        """
        # Create a copy to avoid modifying original
        watermarked = image.copy().convert("RGBA")
        
        # Create watermark overlay
        overlay = Image.new('RGBA', watermarked.size, (255, 255, 255, 0))
        
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(overlay)
            
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Get text size
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            margin = 20
            positions = {
                "top_left": (margin, margin),
                "top_right": (watermarked.width - text_width - margin, margin),
                "bottom_left": (margin, watermarked.height - text_height - margin),
                "bottom_right": (watermarked.width - text_width - margin, watermarked.height - text_height - margin),
                "center": ((watermarked.width - text_width) // 2, (watermarked.height - text_height) // 2)
            }
            
            pos = positions.get(position, positions["bottom_right"])
            
            # Draw watermark
            alpha = int(255 * opacity)
            draw.text(pos, watermark_text, font=font, fill=(255, 255, 255, alpha))
            
            # Composite with original
            watermarked = Image.alpha_composite(watermarked, overlay)
            
        except ImportError:
            logger.warning("PIL ImageDraw/ImageFont not available for watermarking")
        
        return watermarked.convert("RGB")
    
    def get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive information about an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with image information
        """
        info = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
        }
        
        # Add EXIF data if available
        if hasattr(image, '_getexif') and image._getexif():
            exif_data = {}
            for tag_id, value in image._getexif().items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
            info["exif"] = exif_data
        
        # Calculate file size estimate
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        info["estimated_size"] = len(buffer.getvalue())
        
        return info
    
    def calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate a hash of the image for deduplication.
        
        Args:
            image: PIL Image to hash
            
        Returns:
            MD5 hash string
        """
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        # Calculate hash
        return hashlib.md5(image_bytes).hexdigest()


# Convenience functions
async def save_image(
    image: Image.Image,
    filepath: Union[str, Path],
    format: str = "PNG",
    quality: int = 95,
    metadata: Optional[ImageMetadata] = None
) -> Path:
    """Save an image to disk asynchronously.
    
    Args:
        image: PIL Image to save
        filepath: Path to save the image
        format: Image format (PNG, JPEG, etc.)
        quality: Quality for lossy formats (1-100)
        metadata: Optional metadata to embed
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in executor to avoid blocking
    def _save():
        save_kwargs = {"format": format}
        
        if format.upper() in ("JPEG", "JPG"):
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        
        # Add metadata if provided
        if metadata:
            pnginfo = None
            if format.upper() == "PNG":
                from PIL.PngImagePlugin import PngInfo
                pnginfo = PngInfo()
                pnginfo.add_text("prompt", metadata.prompt)
                if metadata.enhanced_prompt:
                    pnginfo.add_text("enhanced_prompt", metadata.enhanced_prompt)
                pnginfo.add_text("model", metadata.model_name)
                pnginfo.add_text("seed", str(metadata.seed) if metadata.seed else "")
                pnginfo.add_text("guidance_scale", str(metadata.guidance_scale))
                pnginfo.add_text("steps", str(metadata.num_inference_steps))
                save_kwargs["pnginfo"] = pnginfo
        
        image.save(filepath, **save_kwargs)
        return filepath
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _save)


async def load_image(filepath: Union[str, Path]) -> Image.Image:
    """Load an image from disk asynchronously.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Loaded PIL Image
    """
    filepath = Path(filepath)
    
    def _load():
        return Image.open(filepath)
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load)


def create_thumbnail(
    image: Image.Image,
    size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    """Create a thumbnail of an image.
    
    Args:
        image: Source PIL Image
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail PIL Image
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail
