"""Prompt enhancement using Ollama LLMs."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from .client import OllamaClient, GenerationConfig

logger = logging.getLogger(__name__)


class EnhancementStyle(Enum):
    """Different styles of prompt enhancement."""
    DETAILED = "detailed"
    ARTISTIC = "artistic"
    PHOTOREALISTIC = "photorealistic"
    MINIMALIST = "minimalist"
    CREATIVE = "creative"
    TECHNICAL = "technical"


@dataclass
class EnhancementConfig:
    """Configuration for prompt enhancement."""
    style: EnhancementStyle = EnhancementStyle.DETAILED
    max_length: int = 500
    temperature: float = 0.7
    add_quality_tags: bool = True
    add_style_tags: bool = True
    preserve_original: bool = True
    enhancement_model: str = "llama2"


class PromptEnhancer:
    """Enhances user prompts using Ollama LLM for better image generation."""
    
    def __init__(
        self, 
        ollama_client: Optional[OllamaClient] = None,
        default_model: str = "llama2"
    ):
        """Initialize the prompt enhancer.
        
        Args:
            ollama_client: Ollama client instance
            default_model: Default model to use for enhancement
        """
        self.client = ollama_client or OllamaClient()
        self.default_model = default_model
        self._enhancement_cache: Dict[str, str] = {}
        
        # Quality and style tags for different enhancement styles
        self.quality_tags = {
            EnhancementStyle.DETAILED: [
                "highly detailed", "intricate", "sharp focus", "professional", 
                "4k", "8k", "high resolution", "masterpiece"
            ],
            EnhancementStyle.ARTISTIC: [
                "artistic", "beautiful", "elegant", "stylized", "expressive", 
                "oil painting", "digital art", "concept art"
            ],
            EnhancementStyle.PHOTOREALISTIC: [
                "photorealistic", "hyperrealistic", "photography", "realistic", 
                "natural lighting", "cinematic", "professional photography"
            ],
            EnhancementStyle.MINIMALIST: [
                "clean", "simple", "minimal", "elegant", "modern", "refined"
            ],
            EnhancementStyle.CREATIVE: [
                "creative", "imaginative", "unique", "innovative", "experimental", 
                "surreal", "fantasy", "sci-fi"
            ],
            EnhancementStyle.TECHNICAL: [
                "technical illustration", "blueprint", "schematic", "precise", 
                "engineering", "detailed diagram"
            ]
        }
    
    async def enhance_prompt(
        self,
        original_prompt: str,
        config: Optional[EnhancementConfig] = None
    ) -> str:
        """Enhance a user prompt for better image generation.
        
        Args:
            original_prompt: Original user prompt
            config: Enhancement configuration
            
        Returns:
            Enhanced prompt string
        """
        if config is None:
            config = EnhancementConfig()
        
        # Check cache first
        cache_key = f"{original_prompt}:{config.style.value}:{config.max_length}"
        if cache_key in self._enhancement_cache:
            return self._enhancement_cache[cache_key]
        
        try:
            # Check if Ollama is available
            if not await self.client.is_available():
                logger.warning("Ollama not available, using basic enhancement")
                return self._basic_enhancement(original_prompt, config)
            
            # Generate enhanced prompt using LLM
            enhanced = await self._llm_enhancement(original_prompt, config)
            
            # Apply additional enhancements
            enhanced = self._apply_style_enhancements(enhanced, config)
            
            # Cache the result
            self._enhancement_cache[cache_key] = enhanced
            
            logger.info(f"Enhanced prompt: '{original_prompt[:50]}...' -> '{enhanced[:50]}...'")
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return self._basic_enhancement(original_prompt, config)
    
    async def _llm_enhancement(
        self,
        prompt: str,
        config: EnhancementConfig
    ) -> str:
        """Use LLM to enhance the prompt."""
        system_prompt = self._create_system_prompt(config)
        
        # Create messages for chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Enhance this prompt for AI image generation: {prompt}"}
        ]
        
        generation_config = GenerationConfig(
            temperature=config.temperature,
            num_predict=config.max_length,
            stop=["Human:", "User:", "\n\n"]
        )
        
        try:
            # Try chat endpoint first
            enhanced = await self.client.chat(
                model=config.enhancement_model,
                messages=messages,
                config=generation_config
            )
        except Exception:
            # Fallback to generate endpoint
            full_prompt = f"{system_prompt}\n\nUser: Enhance this prompt for AI image generation: {prompt}\nAssistant:"
            enhanced = await self.client.generate_text(
                model=config.enhancement_model,
                prompt=full_prompt,
                config=generation_config
            )
        
        # Clean up the response
        enhanced = enhanced.strip()
        
        # Remove any conversational artifacts
        for artifact in ["Here's an enhanced version:", "Enhanced prompt:", "Assistant:", "Here is"]:
            if enhanced.startswith(artifact):
                enhanced = enhanced[len(artifact):].strip()
        
        return enhanced
    
    def _create_system_prompt(self, config: EnhancementConfig) -> str:
        """Create system prompt for LLM enhancement."""
        style_descriptions = {
            EnhancementStyle.DETAILED: "detailed and comprehensive descriptions with rich visual elements",
            EnhancementStyle.ARTISTIC: "artistic and creative language with emphasis on aesthetics",
            EnhancementStyle.PHOTOREALISTIC: "realistic and photographic terminology",
            EnhancementStyle.MINIMALIST: "clean and simple descriptions focusing on essential elements",
            EnhancementStyle.CREATIVE: "imaginative and unique descriptions with creative flair",
            EnhancementStyle.TECHNICAL: "precise and technical descriptions with scientific accuracy"
        }
        
        style_desc = style_descriptions.get(config.style, "detailed and helpful")
        
        return f"""You are an expert at enhancing prompts for AI image generation. Your task is to take user prompts and make them more effective for creating high-quality images.

Guidelines:
- Create {style_desc}
- Keep the original intent and subject matter
- Add relevant visual details, lighting, composition, and style elements
- Use specific and descriptive language
- Avoid overly complex or contradictory instructions
- Maximum length: {config.max_length} characters
- Focus on visual elements that will improve image quality

Enhance the following prompt by making it more descriptive and effective for AI image generation. Provide only the enhanced prompt without explanations."""
    
    def _apply_style_enhancements(
        self,
        prompt: str,
        config: EnhancementConfig
    ) -> str:
        """Apply style-specific enhancements to the prompt."""
        enhanced = prompt
        
        # Add quality tags if enabled
        if config.add_quality_tags:
            quality_tags = self.quality_tags.get(config.style, [])
            if quality_tags:
                # Add a few relevant quality tags
                selected_tags = quality_tags[:3]
                enhanced += f", {', '.join(selected_tags)}"
        
        # Add style-specific modifiers
        if config.add_style_tags:
            style_modifiers = self._get_style_modifiers(config.style)
            if style_modifiers:
                enhanced += f", {style_modifiers}"
        
        # Preserve original prompt if requested
        if config.preserve_original and not prompt.lower() in enhanced.lower():
            enhanced = f"{prompt}, {enhanced}"
        
        return enhanced
    
    def _get_style_modifiers(self, style: EnhancementStyle) -> str:
        """Get style-specific modifiers."""
        modifiers = {
            EnhancementStyle.DETAILED: "highly detailed, intricate details",
            EnhancementStyle.ARTISTIC: "artistic style, beautiful composition",
            EnhancementStyle.PHOTOREALISTIC: "photorealistic, natural lighting",
            EnhancementStyle.MINIMALIST: "minimalist style, clean composition",
            EnhancementStyle.CREATIVE: "creative interpretation, unique perspective",
            EnhancementStyle.TECHNICAL: "technical precision, accurate representation"
        }
        return modifiers.get(style, "")
    
    def _basic_enhancement(
        self,
        prompt: str,
        config: EnhancementConfig
    ) -> str:
        """Basic enhancement without LLM (fallback)."""
        enhanced = prompt
        
        # Add basic quality improvements
        if "detailed" not in enhanced.lower():
            enhanced += ", highly detailed"
        
        if "quality" not in enhanced.lower() and "4k" not in enhanced.lower():
            enhanced += ", high quality"
        
        # Apply style enhancements
        enhanced = self._apply_style_enhancements(enhanced, config)
        
        return enhanced
    
    async def batch_enhance(
        self,
        prompts: List[str],
        config: Optional[EnhancementConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """Enhance multiple prompts in batch.
        
        Args:
            prompts: List of prompts to enhance
            config: Enhancement configuration
            progress_callback: Optional progress callback
            
        Returns:
            List of enhanced prompts
        """
        if config is None:
            config = EnhancementConfig()
        
        enhanced_prompts = []
        total = len(prompts)
        
        for i, prompt in enumerate(prompts):
            enhanced = await self.enhance_prompt(prompt, config)
            enhanced_prompts.append(enhanced)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return enhanced_prompts
    
    def clear_cache(self):
        """Clear the enhancement cache."""
        self._enhancement_cache.clear()
        logger.info("Enhancement cache cleared")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models for enhancement.
        
        Returns:
            List of model names
        """
        try:
            models = await self.client.list_models()
            return [model.name for model in models]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    async def suggest_enhancements(
        self,
        prompt: str,
        num_suggestions: int = 3
    ) -> List[str]:
        """Generate multiple enhancement suggestions for a prompt.
        
        Args:
            prompt: Original prompt
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of enhancement suggestions
        """
        suggestions = []
        
        # Generate suggestions with different styles
        styles = list(EnhancementStyle)
        selected_styles = styles[:num_suggestions] if len(styles) >= num_suggestions else styles * (num_suggestions // len(styles) + 1)
        
        for i in range(num_suggestions):
            style = selected_styles[i % len(selected_styles)]
            config = EnhancementConfig(
                style=style,
                temperature=0.8 + (i * 0.1),  # Vary temperature for diversity
            )
            
            enhanced = await self.enhance_prompt(prompt, config)
            suggestions.append(enhanced)
        
        return suggestions[:num_suggestions]


# Convenience functions
async def enhance_prompt_simple(
    prompt: str,
    style: str = "detailed",
    ollama_url: str = "http://localhost:11434"
) -> str:
    """Simple function to enhance a prompt.
    
    Args:
        prompt: Original prompt
        style: Enhancement style
        ollama_url: Ollama service URL
        
    Returns:
        Enhanced prompt
    """
    client = OllamaClient(base_url=ollama_url)
    enhancer = PromptEnhancer(client)
    
    try:
        style_enum = EnhancementStyle(style.lower())
    except ValueError:
        style_enum = EnhancementStyle.DETAILED
    
    config = EnhancementConfig(style=style_enum)
    
    async with client:
        enhanced = await enhancer.enhance_prompt(prompt, config)
    
    return enhanced
