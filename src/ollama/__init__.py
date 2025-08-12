"""Ollama integration modules for local LLM capabilities."""

from .client import OllamaClient, OllamaModel
from .prompt_enhancer import PromptEnhancer, EnhancementConfig

__all__ = [
    "OllamaClient",
    "OllamaModel", 
    "PromptEnhancer",
    "EnhancementConfig",
]
