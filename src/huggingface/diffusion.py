"""Diffusion model implementations using Hugging Face transformers and diffusers."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import asynccontextmanager
import torch
from diffusers import (
    StableDiffusionPipeline as HFStableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
)
from PIL import Image

from ..core.config import ModelConfig
from ..core.models import BaseModel

logger = logging.getLogger(__name__)


class DiffusionModel(BaseModel):
    """Base class for diffusion models using Hugging Face diffusers."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the diffusion model.
        
        Args:
            config: Model configuration containing model name, device, etc.
        """
        super().__init__(config)
        self.pipeline: Optional[DiffusionPipeline] = None
        self.scheduler_map = {
            "ddpm": DDPMScheduler,
            "ddim": DDIMScheduler,
            "dpm": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
            "lms": LMSDiscreteScheduler,
        }
    
    async def load(self) -> None:
        """Load the diffusion model pipeline."""
        if self.is_loaded:
            return
            
        logger.info(f"Loading diffusion model: {self.config.name}")
        
        try:
            # Load pipeline in a thread to avoid blocking
            pipeline = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._load_pipeline
            )
            
            self.pipeline = pipeline
            self._is_loaded = True
            logger.info(f"Successfully loaded model: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.name}: {e}")
            raise
    
    def _load_pipeline(self) -> DiffusionPipeline:
        """Load the pipeline synchronously (called in executor)."""
        pipeline_kwargs = {
            "torch_dtype": torch.float16 if self.config.device == "cuda" else torch.float32,
            "device_map": "auto" if self.config.device == "cuda" else None,
        }
        
        # Load pipeline based on model type
        if "xl" in self.config.name.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.name, 
                **pipeline_kwargs
            )
        else:
            pipeline = HFStableDiffusionPipeline.from_pretrained(
                self.config.name,
                **pipeline_kwargs
            )
        
        # Move to device
        pipeline = pipeline.to(self.config.device)
        
        # Configure scheduler if specified
        if hasattr(self.config, 'scheduler') and self.config.scheduler:
            scheduler_class = self.scheduler_map.get(self.config.scheduler.lower())
            if scheduler_class:
                pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
        
        # Enable memory efficient attention if available
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.warning("xformers not available, using default attention")
        
        return pipeline
    
    async def unload(self) -> None:
        """Unload the model to free memory."""
        if not self.is_loaded:
            return
            
        logger.info(f"Unloading model: {self.config.name}")
        
        if self.pipeline:
            # Move to CPU to free GPU memory
            self.pipeline = self.pipeline.to("cpu")
            del self.pipeline
            self.pipeline = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info(f"Successfully unloaded model: {self.config.name}")
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images using the diffusion model.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to avoid certain features
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            progress_callback: Callback function for progress updates
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            List of generated PIL Images
        """
        if not self.is_loaded:
            await self.load()
        
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        # Prepare generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "generator": generator,
            **kwargs
        }
        
        # Add progress callback if supported
        if progress_callback and hasattr(self.pipeline, 'set_progress_bar_config'):
            self.pipeline.set_progress_bar_config(disable=True)
            generation_kwargs["callback"] = lambda step, timestep, latents: progress_callback(step, num_inference_steps)
        
        try:
            logger.info(f"Generating {num_images} image(s) with prompt: {prompt[:50]}...")
            
            # Generate images in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(**generation_kwargs)
            )
            
            images = result.images
            logger.info(f"Successfully generated {len(images)} image(s)")
            
            return images
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise


class StableDiffusionPipeline(DiffusionModel):
    """Stable Diffusion model implementation."""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", device: str = "auto"):
        """Initialize Stable Diffusion pipeline.
        
        Args:
            model_name: Hugging Face model name or path
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        config = ModelConfig(
            name=model_name,
            device=device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
            model_type="diffusion"
        )
        super().__init__(config)
    
    @asynccontextmanager
    async def batch_generation_context(self):
        """Context manager for batch generation with optimized memory usage."""
        await self.load()
        
        # Enable memory optimizations for batch generation
        if self.pipeline and hasattr(self.pipeline, 'enable_model_cpu_offload'):
            self.pipeline.enable_model_cpu_offload()
        
        try:
            yield self
        finally:
            # Cleanup after batch generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[List[Image.Image]]:
        """Generate images for multiple prompts in batches.
        
        Args:
            prompts: List of text prompts
            batch_size: Number of prompts to process at once
            progress_callback: Progress callback for overall progress
            **kwargs: Additional generation parameters
            
        Returns:
            List of image lists, one for each prompt
        """
        async with self.batch_generation_context():
            results = []
            total_prompts = len(prompts)
            
            for i in range(0, total_prompts, batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_results = []
                
                for j, prompt in enumerate(batch_prompts):
                    images = await self.generate(prompt, **kwargs)
                    batch_results.append(images)
                    
                    # Update progress
                    current_prompt = i + j + 1
                    if progress_callback:
                        progress_callback(current_prompt, total_prompts)
                
                results.extend(batch_results)
            
            return results


# Convenience functions for common models
async def create_stable_diffusion_v1_5(device: str = "auto") -> StableDiffusionPipeline:
    """Create a Stable Diffusion v1.5 pipeline."""
    return StableDiffusionPipeline("runwayml/stable-diffusion-v1-5", device)


async def create_stable_diffusion_v2_1(device: str = "auto") -> StableDiffusionPipeline:
    """Create a Stable Diffusion v2.1 pipeline."""
    return StableDiffusionPipeline("stabilityai/stable-diffusion-2-1", device)


async def create_stable_diffusion_xl(device: str = "auto") -> DiffusionModel:
    """Create a Stable Diffusion XL pipeline."""
    config = ModelConfig(
        name="stabilityai/stable-diffusion-xl-base-1.0",
        device=device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
        model_type="diffusion"
    )
    return DiffusionModel(config)
