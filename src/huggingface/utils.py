"""Utility functions for Hugging Face model management."""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    scan_cache_dir,
    delete_revisions,
    HfApi,
)

logger = logging.getLogger(__name__)


async def download_model(
    model_name: str,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    progress_callback: Optional[callable] = None,
) -> Path:
    """Download a Hugging Face model asynchronously.
    
    Args:
        model_name: Name of the model to download
        cache_dir: Directory to cache the model
        force_download: Whether to force re-download
        progress_callback: Callback for download progress
        
    Returns:
        Path to the downloaded model
    """
    logger.info(f"Downloading model: {model_name}")
    
    try:
        # Download model files in executor to avoid blocking
        model_path = await asyncio.get_event_loop().run_in_executor(
            None,
            _download_model_sync,
            model_name,
            cache_dir,
            force_download,
        )
        
        logger.info(f"Successfully downloaded model to: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise


def _download_model_sync(
    model_name: str,
    cache_dir: Optional[Union[str, Path]],
    force_download: bool,
) -> Path:
    """Synchronous model download (called in executor)."""
    api = HfApi()
    
    # Get model info
    try:
        model_info = api.model_info(model_name)
        files = list_repo_files(model_name)
        
        # Download key model files
        essential_files = [
            f for f in files 
            if any(f.endswith(ext) for ext in ['.bin', '.safetensors', '.json', '.txt'])
        ]
        
        downloaded_files = []
        for file in essential_files:
            file_path = hf_hub_download(
                repo_id=model_name,
                filename=file,
                cache_dir=cache_dir,
                force_download=force_download,
            )
            downloaded_files.append(file_path)
        
        # Return the directory containing the model
        if downloaded_files:
            return Path(downloaded_files[0]).parent
        else:
            raise ValueError(f"No files downloaded for model {model_name}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_name}: {e}")


async def get_available_models() -> Dict[str, List[str]]:
    """Get a list of popular available diffusion models.
    
    Returns:
        Dictionary mapping model categories to model names
    """
    return {
        "stable_diffusion_v1": [
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1",
        ],
        "stable_diffusion_xl": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
        ],
        "specialty_models": [
            "dreamlike-art/dreamlike-diffusion-1.0",
            "prompthero/openjourney",
            "wavymulder/Analog-Diffusion",
            "nitrosocke/Arcane-Diffusion",
        ],
        "community_models": [
            "hakurei/waifu-diffusion",
            "naclbit/trinart_stable_diffusion_v2",
            "Fictiverse/Stable_Diffusion_PaperCut_Model",
        ]
    }


async def clear_cache(cache_dir: Optional[Union[str, Path]] = None) -> None:
    """Clear the Hugging Face model cache.
    
    Args:
        cache_dir: Specific cache directory to clear, or None for default
    """
    logger.info("Clearing Hugging Face model cache")
    
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            _clear_cache_sync,
            cache_dir
        )
        logger.info("Successfully cleared model cache")
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise


def _clear_cache_sync(cache_dir: Optional[Union[str, Path]]) -> None:
    """Synchronously clear cache (called in executor)."""
    if cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
    else:
        # Use HF's cache management
        cache_info = scan_cache_dir()
        if cache_info.repos:
            # Delete all cached models
            to_delete = []
            for repo in cache_info.repos:
                to_delete.extend(repo.revisions)
            
            if to_delete:
                delete_strategy = delete_revisions(*to_delete)
                delete_strategy.execute()


def get_model_size(model_name: str) -> Optional[int]:
    """Get the approximate size of a model in bytes.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Size in bytes, or None if unavailable
    """
    try:
        api = HfApi()
        files = api.list_repo_files(model_name)
        total_size = 0
        
        for file in files:
            try:
                file_info = api.get_hf_file_metadata(
                    hf_hub_url(model_name, file)
                )
                if hasattr(file_info, 'size'):
                    total_size += file_info.size
            except Exception:
                continue
        
        return total_size if total_size > 0 else None
        
    except Exception as e:
        logger.warning(f"Could not get size for model {model_name}: {e}")
        return None


def check_model_exists(model_name: str) -> bool:
    """Check if a model exists on Hugging Face Hub.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model exists, False otherwise
    """
    try:
        api = HfApi()
        api.model_info(model_name)
        return True
    except Exception:
        return False


async def get_model_requirements(model_name: str) -> Dict[str, str]:
    """Get the system requirements for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with requirement information
    """
    # Basic requirements based on model type
    requirements = {
        "python": ">=3.8",
        "torch": ">=1.11.0",
        "diffusers": ">=0.21.0",
        "transformers": ">=4.21.0",
    }
    
    # Estimate memory requirements based on model name
    if "xl" in model_name.lower():
        requirements["gpu_memory"] = ">=8GB"
        requirements["system_memory"] = ">=16GB"
    elif "stable-diffusion-2" in model_name.lower():
        requirements["gpu_memory"] = ">=6GB"
        requirements["system_memory"] = ">=12GB"
    else:
        requirements["gpu_memory"] = ">=4GB"
        requirements["system_memory"] = ">=8GB"
    
    return requirements


def get_recommended_settings(model_name: str, device: str = "cuda") -> Dict[str, any]:
    """Get recommended generation settings for a model.
    
    Args:
        model_name: Name of the model
        device: Target device
        
    Returns:
        Dictionary with recommended settings
    """
    base_settings = {
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512,
    }
    
    # Adjust settings based on model
    if "xl" in model_name.lower():
        base_settings.update({
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.0,
        })
    elif "stable-diffusion-2" in model_name.lower():
        base_settings.update({
            "width": 768,
            "height": 768,
            "guidance_scale": 7.5,
        })
    
    # Adjust for device capabilities
    if device == "cpu":
        base_settings.update({
            "num_inference_steps": 10,  # Fewer steps for CPU
            "width": 512,
            "height": 512,
        })
    
    return base_settings
