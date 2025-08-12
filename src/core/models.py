"""
Model management for Hugging Face and Ollama models.
"""

import torch
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import gc

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

from .config import Config, config

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and unloading of AI models."""
    
    def __init__(self, config_instance: Optional[Config] = None):
        self.config = config_instance or config
        self._loaded_models: Dict[str, Any] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine the best device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by category."""
        return {
            "diffusion": [
                "stabilityai/stable-diffusion-xl-base-1.0",
                "stabilityai/stable-diffusion-2-1",
                "runwayml/stable-diffusion-v1-5",
                "prompthero/openjourney-v4",
                "wavymulder/Analog-Diffusion"
            ],
            "schedulers": [
                "DPMSolverMultistepScheduler",
                "EulerDiscreteScheduler", 
                "DDIMScheduler"
            ]
        }
    
    def load_diffusion_model(self, model_id: str, **kwargs) -> DiffusionPipeline:
        """Load a diffusion model."""
        if model_id in self._loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return self._loaded_models[model_id]
        
        logger.info(f"Loading diffusion model: {model_id}")
        
        try:
            # Determine pipeline type based on model
            if "xl" in model_id.lower():
                pipeline_class = StableDiffusionXLPipeline
            else:
                pipeline_class = StableDiffusionPipeline
            
            # Load model with optimizations
            pipe = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                use_safetensors=True,
                cache_dir=self.config.models_cache_dir,
                **kwargs
            )
            
            # Apply optimizations
            if self.device != "cpu":
                pipe = pipe.to(self.device)
                
                # Enable memory efficient attention if available
                if hasattr(pipe.unet, 'set_use_memory_efficient_attention_xformers'):
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                        logger.info("Enabled xformers memory efficient attention")
                    except Exception as e:
                        logger.warning(f"Could not enable xformers: {e}")
                
                # Enable CPU offloading for low VRAM
                if torch.cuda.is_available():
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    if memory_gb < 8:
                        pipe.enable_sequential_cpu_offload()
                        logger.info("Enabled sequential CPU offload for low VRAM")
            
            # Set default scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            
            self._loaded_models[model_id] = pipe
            self._model_info[model_id] = {
                "type": "diffusion",
                "device": self.device,
                "memory_usage": self._estimate_memory_usage(pipe)
            }
            
            logger.info(f"Successfully loaded {model_id}")
            return pipe
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def set_scheduler(self, model_id: str, scheduler_name: str):
        """Change the scheduler for a loaded model."""
        if model_id not in self._loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        pipe = self._loaded_models[model_id]
        
        scheduler_map = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "DDIMScheduler": DDIMScheduler
        }
        
        if scheduler_name not in scheduler_map:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        scheduler_class = scheduler_map[scheduler_name]
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
        logger.info(f"Changed scheduler for {model_id} to {scheduler_name}")
    
    def unload_model(self, model_id: str):
        """Unload a specific model to free memory."""
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            del self._model_info[model_id]
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model: {model_id}")
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        model_ids = list(self._loaded_models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
        logger.info("Unloaded all models")
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models."""
        return self._model_info.copy()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        info = {
            "device": self.device,
            "models_loaded": len(self._loaded_models)
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9
            })
        
        return info
    
    def _estimate_memory_usage(self, model) -> float:
        """Estimate memory usage of a model in GB."""
        try:
            if hasattr(model, 'unet'):
                # Count parameters in UNet (main component)
                params = sum(p.numel() for p in model.unet.parameters())
                # Rough estimate: 4 bytes per parameter (float32) or 2 bytes (float16)
                bytes_per_param = 2 if self.device != "cpu" else 4
                return (params * bytes_per_param) / 1e9
        except:
            pass
        return 0.0
    
    def __del__(self):
        """Clean up models when manager is destroyed."""
        self.unload_all_models()
