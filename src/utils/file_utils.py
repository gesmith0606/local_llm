"""File utilities for the AI image generation project."""

import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def create_output_directory(base_path: Union[str, Path], subdir: Optional[str] = None) -> Path:
    """Create output directory structure.
    
    Args:
        base_path: Base output directory
        subdir: Optional subdirectory name
        
    Returns:
        Created directory path
    """
    base_path = Path(base_path)
    
    if subdir:
        output_dir = base_path / subdir
    else:
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_path / f"generation_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for filesystem compatibility.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Replace multiple spaces/underscores with single underscore
    filename = re.sub(r'[_\s]+', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure not empty
    if not filename:
        filename = "unnamed"
    
    # Truncate if too long
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename


def get_unique_filename(directory: Union[str, Path], base_name: str, extension: str = "") -> Path:
    """Get a unique filename in the directory.
    
    Args:
        directory: Target directory
        base_name: Base filename
        extension: File extension (with or without dot)
        
    Returns:
        Unique file path
    """
    directory = Path(directory)
    
    # Ensure extension starts with dot
    if extension and not extension.startswith('.'):
        extension = f".{extension}"
    
    # Sanitize base name
    base_name = sanitize_filename(base_name)
    
    # Try the original name first
    filepath = directory / f"{base_name}{extension}"
    if not filepath.exists():
        return filepath
    
    # Add counter if file exists
    counter = 1
    while True:
        filepath = directory / f"{base_name}_{counter:03d}{extension}"
        if not filepath.exists():
            return filepath
        counter += 1


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """Get file size in megabytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in MB
    """
    filepath = Path(filepath)
    if filepath.exists():
        return filepath.stat().st_size / (1024 * 1024)
    return 0.0


def clean_old_files(directory: Union[str, Path], max_age_days: int = 7, pattern: str = "*") -> int:
    """Clean old files from directory.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    deleted_count = 0
    for filepath in directory.glob(pattern):
        if filepath.is_file():
            file_age = current_time - filepath.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    filepath.unlink()
                    deleted_count += 1
                except OSError:
                    pass  # Skip files that can't be deleted
    
    return deleted_count


def copy_file_with_metadata(source: Union[str, Path], destination: Union[str, Path]) -> bool:
    """Copy file preserving metadata.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        shutil.copy2(source, destination)
        return True
    except (OSError, shutil.Error):
        return False


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
