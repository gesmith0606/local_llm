#!/usr/bin/env python3
"""
Test script to verify the basic project setup and imports.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test basic imports to verify project structure."""
    print("Testing AI Image Generation Project Setup...")
    print("=" * 50)
    
    # Test core modules
    try:
        from core.config import Config, ModelConfig
        print("✓ Core config module imported successfully")
    except ImportError as e:
        print(f"✗ Core config import failed: {e}")
    
    try:
        from core.models import BaseModel
        print("✓ Core models module imported successfully")
    except ImportError as e:
        print(f"✗ Core models import failed: {e}")
    
    try:
        from core.generator import ImageGenerator
        print("✓ Core generator module imported successfully")
    except ImportError as e:
        print(f"✗ Core generator import failed: {e}")
    
    # Test integration modules
    try:
        from huggingface.diffusion import DiffusionModel
        print("✓ Hugging Face diffusion module imported successfully")
    except ImportError as e:
        print(f"✗ Hugging Face diffusion import failed: {e}")
    
    try:
        from ollama.client import OllamaClient
        print("✓ Ollama client module imported successfully")
    except ImportError as e:
        print(f"✗ Ollama client import failed: {e}")
    
    try:
        from ollama.prompt_enhancer import PromptEnhancer
        print("✓ Ollama prompt enhancer module imported successfully")
    except ImportError as e:
        print(f"✗ Ollama prompt enhancer import failed: {e}")
    
    # Test utility modules
    try:
        from utils.image_utils import ImageProcessor
        print("✓ Image utils module imported successfully")
    except ImportError as e:
        print(f"✗ Image utils import failed: {e}")
    
    print("\n" + "=" * 50)
    print("Note: Import errors are expected until dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    print("=" * 50)


def test_config():
    """Test configuration loading."""
    print("\nTesting Configuration Loading...")
    print("-" * 30)
    
    config_path = project_root / "config.yaml"
    if config_path.exists():
        print(f"✓ Configuration file found: {config_path}")
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"✓ Configuration loaded successfully")
            print(f"  - Ollama URL: {config_data.get('ollama', {}).get('base_url', 'Not set')}")
            print(f"  - Default model: {config_data.get('models', {}).get('default_diffusion_model', 'Not set')}")
            print(f"  - Output directory: {config_data.get('output', {}).get('directory', 'Not set')}")
        except ImportError:
            print("✗ PyYAML not installed - cannot load config")
        except Exception as e:
            print(f"✗ Error loading config: {e}")
    else:
        print(f"✗ Configuration file not found: {config_path}")


def check_dependencies():
    """Check if key dependencies are available."""
    print("\nChecking Dependencies...")
    print("-" * 25)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("diffusers", "Hugging Face Diffusers"),
        ("transformers", "Hugging Face Transformers"),
        ("PIL", "Pillow"),
        ("aiohttp", "aiohttp"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} is available")
        except ImportError:
            print(f"✗ {name} is NOT available")


def main():
    """Main test function."""
    print("AI Image Generation Project - Setup Verification")
    print("=" * 60)
    
    test_imports()
    test_config()
    check_dependencies()
    
    print("\n" + "=" * 60)
    print("Setup verification complete!")
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start Ollama service if using prompt enhancement")
    print("3. Run: python main.py generate 'your prompt here'")
    print("=" * 60)


if __name__ == "__main__":
    main()
