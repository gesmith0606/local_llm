"""
Core image generation functionality combining Hugging Face and Ollama.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import torch
from PIL import Image
import numpy as np

from .config import Config, config
from .models import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Request parameters for image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    model_id: Optional[str] = None
    scheduler: Optional[str] = None
    enhance_prompt: bool = False


@dataclass
class GenerationResult:
    """Result of image generation."""
    images: List[Image.Image]
    prompt: str
    enhanced_prompt: Optional[str] = None
    parameters: Dict[str, Any] = None
    generation_time: float = 0.0
    metadata: Dict[str, Any] = None


class ImageGenerator:
    """Main image generation class combining HF and Ollama capabilities."""
    
    def __init__(self, config_instance: Optional[Config] = None):
        self.config = config_instance or config
        self.model_manager = ModelManager(self.config)
        self._ollama_client = None
        self._generation_cache: Dict[str, GenerationResult] = {}
        self.max_cache_size = 100
        
    async def generate(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> GenerationResult:
        """Generate images based on the request."""
        start_time = time.time()
        
        try:
            # Enhance prompt if requested
            enhanced_prompt = None
            if request.enhance_prompt:
                if progress_callback:
                    progress_callback(0, 100, "Enhancing prompt...")
                enhanced_prompt = await self._enhance_prompt(request.prompt)
                final_prompt = enhanced_prompt or request.prompt
            else:
                final_prompt = request.prompt
            
            # Check cache
            cache_key = self._get_cache_key(request, final_prompt)
            if cache_key in self._generation_cache:
                logger.info("Returning cached result")
                return self._generation_cache[cache_key]
            
            # Load model if not already loaded
            model_id = request.model_id or self.config.default_diffusion_model
            if progress_callback:
                progress_callback(10, 100, f"Loading model {model_id}...")
            
            pipe = self.model_manager.load_diffusion_model(model_id)
            
            # Set scheduler if specified
            if request.scheduler:
                self.model_manager.set_scheduler(model_id, request.scheduler)
            
            # Prepare generation parameters
            generation_params = {
                "prompt": final_prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "num_images_per_prompt": request.num_images,
            }
            
            if request.seed is not None:
                generator = torch.Generator(device=self.model_manager.device)
                generator.manual_seed(request.seed)
                generation_params["generator"] = generator
            
            # Custom callback for progress
            def pipeline_callback(step: int, timestep: int, latents: torch.FloatTensor):
                if progress_callback:
                    progress = 20 + int((step / request.num_inference_steps) * 70)
                    progress_callback(progress, 100, f"Generating... Step {step}/{request.num_inference_steps}")
            
            generation_params["callback"] = pipeline_callback
            generation_params["callback_steps"] = 1
            
            if progress_callback:
                progress_callback(20, 100, "Starting generation...")
            
            # Generate images
            with torch.no_grad():
                result = pipe(**generation_params)
            
            # Process results
            images = result.images
            generation_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(95, 100, "Processing results...")
            
            # Create result object
            generation_result = GenerationResult(
                images=images,
                prompt=final_prompt,
                enhanced_prompt=enhanced_prompt,
                parameters=generation_params,
                generation_time=generation_time,
                metadata={
                    "model_id": model_id,
                    "device": self.model_manager.device,
                    "timestamp": time.time(),
                    "seed": request.seed,
                }
            )
            
            # Cache result
            self._add_to_cache(cache_key, generation_result)
            
            if progress_callback:
                progress_callback(100, 100, "Complete!")
            
            logger.info(f"Generated {len(images)} images in {generation_time:.2f}s")
            return generation_result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if progress_callback:
                progress_callback(0, 100, f"Error: {str(e)}")
            raise
    
    async def generate_enhanced(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Quick generation with prompt enhancement."""
        request = GenerationRequest(
            prompt=prompt,
            model_id=model,
            enhance_prompt=True,
            **kwargs
        )
        return await self.generate(request)
    
    async def batch_generate(
        self,
        prompts: List[str],
        count: int = 1,
        **kwargs
    ) -> List[GenerationResult]:
        """Generate multiple images from multiple prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating batch {i+1}/{len(prompts)}: {prompt[:50]}...")
            request = GenerationRequest(
                prompt=prompt,
                num_images=count,
                **kwargs
            )
            result = await self.generate(request)
            results.append(result)
        return results
    
    async def _enhance_prompt(self, prompt: str) -> Optional[str]:
        """Enhance prompt using Ollama."""
        try:
            if self._ollama_client is None:
                from ..ollama.client import OllamaClient
                self._ollama_client = OllamaClient(self.config.ollama)
            
            enhanced = await self._ollama_client.enhance_prompt(prompt)
            logger.info(f"Enhanced prompt: {prompt} -> {enhanced}")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Prompt enhancement failed: {e}")
            return None
    
    def _get_cache_key(self, request: GenerationRequest, final_prompt: str) -> str:
        """Generate cache key for request."""
        cache_data = {
            "prompt": final_prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.num_inference_steps,
            "guidance": request.guidance_scale,
            "model": request.model_id or self.config.default_diffusion_model,
            "scheduler": request.scheduler,
            "seed": request.seed,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, result: GenerationResult):
        """Add result to cache with size management."""
        if len(self._generation_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._generation_cache))
            del self._generation_cache[oldest_key]
        
        self._generation_cache[key] = result
    
    def save_images(
        self,
        result: GenerationResult,
        output_dir: Optional[Path] = None,
        prefix: str = "generated"
    ) -> List[Path]:
        """Save generated images to disk."""
        if output_dir is None:
            output_dir = self.config.get_output_dir()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        timestamp = int(time.time())
        
        for i, image in enumerate(result.images):
            filename = f"{prefix}_{timestamp}_{i:03d}.png"
            filepath = output_dir / filename
            
            # Add metadata to image
            metadata = {
                "prompt": result.prompt,
                "parameters": str(result.parameters),
                "generation_time": result.generation_time,
                "model": result.metadata.get("model_id", "unknown"),
            }
            
            # Save with metadata
            image.save(filepath, "PNG", pnginfo=self._create_png_info(metadata))
            saved_paths.append(filepath)
            logger.info(f"Saved image: {filepath}")
        
        return saved_paths
    
    def _create_png_info(self, metadata: Dict[str, Any]):
        """Create PNG info for metadata."""
        from PIL import PngImagePlugin
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            pnginfo.add_text(key, str(value))
        return pnginfo
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generations."""
        return {
            "cache_size": len(self._generation_cache),
            "models_loaded": len(self.model_manager.get_loaded_models()),
            "memory_usage": self.model_manager.get_memory_usage(),
            "device": self.model_manager.device,
        }
    
    def clear_cache(self):
        """Clear generation cache."""
        self._generation_cache.clear()
        logger.info("Cleared generation cache")
    
    def cleanup(self):
        """Clean up resources."""
        self.clear_cache()
        self.model_manager.unload_all_models()
        logger.info("Cleaned up generator resources")
