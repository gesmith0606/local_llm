"""
Configuration management for AI image generation.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    type: str  # "diffusion", "ollama"
    path: str
    device: str = "auto"
    precision: str = "fp16"
    cache_dir: Optional[str] = None
    

@dataclass
class GenerationConfig:
    """Configuration for image generation parameters."""
    default_size: tuple[int, int] = (1024, 1024)
    default_steps: int = 50
    default_guidance: float = 7.5
    default_scheduler: str = "DPMSolverMultistepScheduler"
    batch_size: int = 1
    safety_checker: bool = True


@dataclass
class OllamaConfig:
    """Configuration for Ollama integration."""
    host: str = "localhost"
    port: int = 11434
    timeout: int = 30
    model: str = "llama2"
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class UIConfig:
    """Configuration for user interfaces."""
    theme: str = "dark"
    auto_save: bool = True
    output_dir: str = "outputs"
    preview_size: tuple[int, int] = (512, 512)
    max_history: int = 100


class Config:
    """Main configuration class for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    
    def _load_config(self):
        """Load configuration from file or create defaults."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = self._get_default_config()
            self.save_config(config_data)
        
        self._parse_config(config_data)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "models": {
                "diffusion": {
                    "default": "stabilityai/stable-diffusion-xl-base-1.0",
                    "cache_dir": "./models/diffusion"
                },
                "ollama": {
                    "default": "llama2",
                    "host": "localhost",
                    "port": 11434
                }
            },
            "generation": {
                "default_size": [1024, 1024],
                "default_steps": 50,
                "default_guidance": 7.5,
                "batch_size": 1
            },
            "ollama": {
                "host": "localhost",
                "port": 11434,
                "timeout": 30,
                "model": "llama2"
            },
            "ui": {
                "theme": "dark",
                "auto_save": True,
                "output_dir": "outputs"
            }
        }
    
    def _parse_config(self, config_data: Dict[str, Any]):
        """Parse configuration data into structured objects."""
        # Models configuration
        models_data = config_data.get("models", {})
        self.default_diffusion_model = models_data.get("diffusion", {}).get("default")
        self.models_cache_dir = models_data.get("diffusion", {}).get("cache_dir")
        
        # Generation configuration
        gen_data = config_data.get("generation", {})
        self.generation = GenerationConfig(
            default_size=tuple(gen_data.get("default_size", [1024, 1024])),
            default_steps=gen_data.get("default_steps", 50),
            default_guidance=gen_data.get("default_guidance", 7.5),
            batch_size=gen_data.get("batch_size", 1)
        )
        
        # Ollama configuration
        ollama_data = config_data.get("ollama", {})
        self.ollama = OllamaConfig(
            host=ollama_data.get("host", "localhost"),
            port=ollama_data.get("port", 11434),
            timeout=ollama_data.get("timeout", 30),
            model=ollama_data.get("model", "llama2")
        )
        
        # UI configuration
        ui_data = config_data.get("ui", {})
        self.ui = UIConfig(
            theme=ui_data.get("theme", "dark"),
            auto_save=ui_data.get("auto_save", True),
            output_dir=ui_data.get("output_dir", "outputs")
        )
    
    def save_config(self, config_data: Optional[Dict[str, Any]] = None):
        """Save current configuration to file."""
        if config_data is None:
            config_data = self.to_dict()
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "models": {
                "diffusion": {
                    "default": self.default_diffusion_model,
                    "cache_dir": self.models_cache_dir
                }
            },
            "generation": {
                "default_size": list(self.generation.default_size),
                "default_steps": self.generation.default_steps,
                "default_guidance": self.generation.default_guidance,
                "batch_size": self.generation.batch_size
            },
            "ollama": {
                "host": self.ollama.host,
                "port": self.ollama.port,
                "timeout": self.ollama.timeout,
                "model": self.ollama.model
            },
            "ui": {
                "theme": self.ui.theme,
                "auto_save": self.ui.auto_save,
                "output_dir": self.ui.output_dir
            }
        }
    
    def get_output_dir(self) -> Path:
        """Get the output directory path."""
        output_dir = Path(self.ui.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


# Global configuration instance
config = Config()
