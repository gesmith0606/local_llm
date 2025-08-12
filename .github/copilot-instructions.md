<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AI Image Generation Project - Copilot Instructions

## Project Overview
This is an AI-powered image generation project that combines Hugging Face's diffusion models with Ollama's local LLM capabilities.

## Key Technologies
- **Hugging Face Diffusers**: For image generation models (Stable Diffusion, DALL-E, etc.)
- **Ollama**: For local LLM integration and prompt enhancement
- **PyQt6/Tkinter**: For GUI interfaces
- **Gradio**: For web-based interfaces
- **PIL/Pillow**: For image processing
- **asyncio**: For asynchronous operations

## Code Style Guidelines
- Use async/await patterns for I/O operations
- Implement proper error handling for API calls
- Add type hints for all functions
- Use dataclasses for configuration objects
- Follow PEP 8 style guidelines
- Add comprehensive docstrings

## Model Integration Patterns
- Use context managers for model loading/unloading
- Implement caching for frequently used models
- Add progress callbacks for long-running operations
- Use proper resource cleanup for GPU memory

## UI Development
- Create responsive interfaces that work on different screen sizes
- Implement proper loading states and progress indicators
- Add keyboard shortcuts for common operations
- Use modern UI patterns and accessibility features
