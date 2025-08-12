#!/usr/bin/env python3
"""
Example script demonstrating the AI Image Generation project capabilities.

This script shows how to use the core components for:
1. Loading configuration
2. Creating image generation models
3. Enhancing prompts with Ollama
4. Generating images with Hugging Face models
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


async def example_basic_generation():
    """Example of basic image generation without dependencies."""
    print("=== Basic Image Generation Example ===")
    print("This example shows the project structure and workflow.")
    print("Note: Requires actual dependencies to run fully.\n")
    
    # This would be the actual workflow once dependencies are installed:
    print("1. Load configuration:")
    print("   config = load_config('config.yaml')")
    
    print("\n2. Initialize generator:")
    print("   generator = ImageGenerator(config)")
    
    print("\n3. Generate image:")
    print("   image = await generator.generate_image('a beautiful sunset')")
    
    print("\n4. Save result:")
    print("   await save_image(image, 'output/sunset.png')")
    
    print("\nProject is ready for development!")


async def example_with_enhancement():
    """Example showing prompt enhancement workflow."""
    print("\n=== Prompt Enhancement Example ===")
    print("This shows how Ollama integration would work:")
    
    print("\n1. Initialize Ollama client:")
    print("   client = OllamaClient('http://localhost:11434')")
    
    print("\n2. Create prompt enhancer:")
    print("   enhancer = PromptEnhancer(client)")
    
    print("\n3. Enhance prompt:")
    print("   original = 'a cat'")
    print("   enhanced = await enhancer.enhance_prompt(original)")
    print("   # Result: 'a beautiful fluffy cat with detailed fur, warm lighting, high quality, 4k'")
    
    print("\n4. Generate with enhanced prompt:")
    print("   image = await generator.generate_image(enhanced)")


def show_project_structure():
    """Display the current project structure."""
    print("\n=== Project Structure ===")
    print("""
AI Image Generation Project
├── .github/
│   └── copilot-instructions.md    # AI assistant guidance
├── .vscode/
│   └── tasks.json                 # VS Code build tasks
├── src/
│   ├── core/
│   │   ├── config.py             # Configuration management
│   │   ├── generator.py          # Main image generation logic
│   │   └── models.py             # Model loading and management
│   ├── huggingface/
│   │   ├── diffusion.py          # Stable Diffusion integration
│   │   └── utils.py              # HF utility functions
│   ├── ollama/
│   │   ├── client.py             # Ollama API client
│   │   └── prompt_enhancer.py    # LLM prompt enhancement
│   ├── utils/
│   │   └── image_utils.py        # Image processing utilities
│   └── cli/
│       └── main.py               # Command-line interface
├── config.yaml                   # Main configuration file
├── requirements.txt               # Python dependencies
├── main.py                       # Main entry point
├── test_setup.py                 # Setup verification
└── README.md                     # Project documentation
    """)


def show_next_steps():
    """Show the next steps for project development."""
    print("\n=== Next Steps ===")
    print("""
1. INSTALL DEPENDENCIES:
   .\venv\Scripts\python.exe -m pip install -r requirements.txt

2. START OLLAMA (Optional for prompt enhancement):
   - Download and install Ollama from https://ollama.ai
   - Run: ollama serve
   - Pull a model: ollama pull llama2

3. TEST THE SETUP:
   .\venv\Scripts\python.exe test_setup.py

4. GENERATE YOUR FIRST IMAGE:
   .\venv\Scripts\python.exe main.py generate "a beautiful landscape"

5. DEVELOPMENT OPTIONS:
   - Add GUI interface (PyQt6/Tkinter)
   - Add web interface (Gradio/Streamlit)
   - Add API server (FastAPI)
   - Extend model support
   - Add custom schedulers and samplers

6. CONFIGURATION:
   - Edit config.yaml to customize settings
   - Set your preferred models and parameters
   - Configure output directories and formats
    """)


async def main():
    """Main example function."""
    print("AI Image Generation Project - Examples & Workflow")
    print("=" * 60)
    
    await example_basic_generation()
    await example_with_enhancement()
    show_project_structure()
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("🎨 Ready to create amazing AI-generated images! 🎨")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
