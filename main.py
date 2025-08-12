#!/usr/bin/env python3
"""
AI Image Generation Project - Entry Point

This script serves as the main entry point for the AI image generation project.
It provides a simple command-line interface for generating images using 
Hugging Face diffusion models enhanced with Ollama LLM capabilities.
"""

import sys
import asyncio
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from cli.main import main
except ImportError as e:
    print(f"Error importing CLI module: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
