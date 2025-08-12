#!/usr/bin/env python3
"""
Launch script for different interfaces of the AI Image Generation project.

This script provides a unified entry point to launch:
- GUI interface (Tkinter)
- Web interface (Gradio)
- CLI interface
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def launch_gui(config_path: str = None):
    """Launch the GUI interface."""
    try:
        from gui.main_window import ImageGeneratorGUI
        
        print("🖥️  Launching GUI interface...")
        app = ImageGeneratorGUI(config_path)
        app.run()
        
    except ImportError as e:
        print(f"❌ Error launching GUI: {e}")
        print("Make sure tkinter is available in your Python installation.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


def launch_web(config_path: str = None, host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """Launch the web interface."""
    try:
        from web.gradio_app import launch_gradio_app
        
        print("🌐 Launching web interface...")
        launch_gradio_app(config_path, host, port, share)
        
    except ImportError as e:
        print(f"❌ Error launching web interface: {e}")
        print("Install gradio: pip install gradio")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


def launch_cli(args):
    """Launch the CLI interface."""
    try:
        from cli.main import main
        
        # Remove 'cli' from args and pass the rest to CLI
        cli_args = args[1:] if len(args) > 1 else []
        return main(cli_args)
        
    except ImportError as e:
        print(f"❌ Error launching CLI: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="AI Image Generator - Multiple Interface Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py gui                    # Launch GUI interface
  python launcher.py web                    # Launch web interface
  python launcher.py web --port 8080        # Launch web on port 8080
  python launcher.py web --share            # Launch web with public sharing
  python launcher.py cli generate "a cat"   # Use CLI to generate image
  python launcher.py cli --help             # Show CLI help
        """
    )
    
    parser.add_argument(
        "interface",
        choices=["gui", "web", "cli"],
        help="Interface to launch"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    # Web-specific arguments
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web interface (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link for web interface"
    )
    
    # Parse known args to allow CLI passthrough
    args, unknown = parser.parse_known_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"⚠️  Configuration file not found: {config_path}")
        print("Using default configuration...")
        args.config = None
    
    # Launch appropriate interface
    if args.interface == "gui":
        return launch_gui(args.config)
    
    elif args.interface == "web":
        return launch_web(args.config, args.host, args.port, args.share)
    
    elif args.interface == "cli":
        # Reconstruct CLI args
        cli_args = sys.argv[2:]  # Skip 'launcher.py' and 'cli'
        return launch_cli(cli_args)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
