"""Logging utilities for the AI image generation project."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    max_size: str = "10MB",
    backup_count: int = 5,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        max_size: Maximum log file size
        backup_count: Number of backup files to keep
        console: Whether to log to console
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max_size
        max_bytes = parse_size_string(max_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def parse_size_string(size_str: str) -> int:
    """Parse size string to bytes.
    
    Args:
        size_str: Size string like "10MB", "1GB", etc.
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    # Extract number and unit
    import re
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    
    number, unit = match.groups()
    number = float(number)
    
    # Convert to bytes
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
        '': 1,  # Default to bytes
    }
    
    multiplier = multipliers.get(unit, 1)
    return int(number * multiplier)


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.RESET}"
            )
        return super().format(record)


def setup_colored_logging(level: Union[str, int] = logging.INFO) -> logging.Logger:
    """Setup colored console logging.
    
    Args:
        level: Logging level
        
    Returns:
        Configured logger
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    colored_formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    
    return logger


class ProgressLogger:
    """Logger for progress tracking."""
    
    def __init__(self, logger: logging.Logger, total: int, prefix: str = "Progress"):
        self.logger = logger
        self.total = total
        self.prefix = prefix
        self.current = 0
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100
        self.logger.info(f"{self.prefix}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self):
        """Mark as finished."""
        self.logger.info(f"{self.prefix}: Complete!")


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


async def log_async_function_call(func):
    """Decorator to log async function calls."""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Async {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Async {func.__name__} failed with error: {e}")
            raise
    return wrapper
