"""Main CLI entry point for the AI image generation project."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ..core.config import load_config
from ..core.generator import ImageGenerator
from ..utils.logging_utils import setup_logging
from .commands import GenerateCommand, EnhanceCommand, ConfigCommand

logger = logging.getLogger(__name__)


def create_cli_parser() -> argparse.ArgumentParser:
    """Create the main CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-image-gen",
        description="AI-powered image generation using Hugging Face and Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ai-image-gen generate "a beautiful sunset over mountains"
  ai-image-gen generate "a cat wearing a hat" --model "runwayml/stable-diffusion-v1-5" --steps 30
  ai-image-gen enhance "simple prompt" --style artistic
  ai-image-gen config --show
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate images from text prompts",
        aliases=["gen", "g"]
    )
    GenerateCommand.add_arguments(generate_parser)
    
    # Enhance command
    enhance_parser = subparsers.add_parser(
        "enhance",
        help="Enhance text prompts using Ollama",
        aliases=["enh", "e"]
    )
    EnhanceCommand.add_arguments(enhance_parser)
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        aliases=["cfg", "c"]
    )
    ConfigCommand.add_arguments(config_parser)
    
    return parser


async def run_command(args: argparse.Namespace) -> int:
    """Run the specified command."""
    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        config = load_config(config_path)
        
        # Initialize generator
        generator = ImageGenerator(config)
        
        # Execute command
        if args.command in ["generate", "gen", "g"]:
            command = GenerateCommand(generator)
            return await command.execute(args)
        elif args.command in ["enhance", "enh", "e"]:
            command = EnhanceCommand(generator)
            return await command.execute(args)
        elif args.command in ["config", "cfg", "c"]:
            command = ConfigCommand(config)
            return await command.execute(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.DEBUG
    elif not args.quiet:
        log_level = logging.INFO
    
    setup_logging(level=log_level, console=True)
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 0
    
    # Run async command
    try:
        return asyncio.run(run_command(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
