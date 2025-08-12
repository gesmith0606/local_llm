"""Main GUI window for the AI image generation project."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import json

from .components import PromptFrame, SettingsFrame, PreviewFrame, ProgressFrame


class ImageGeneratorGUI:
    """Main GUI application for AI image generation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the GUI application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        
        # Setup main window
        self.root = tk.Tk()
        self.root.title("AI Image Generator")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Setup async loop for GUI
        self.loop = None
        self.generator = None
        
        # Initialize components
        self._setup_gui()
        self._setup_menu()
        self._setup_bindings()
        
        # Status variables
        self.generation_running = False
        self.current_images = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            # Return default config if loading fails
            return {
                "generation": {
                    "default_width": 512,
                    "default_height": 512,
                    "default_steps": 20,
                    "default_guidance_scale": 7.5,
                },
                "output": {
                    "directory": "./output",
                    "format": "PNG",
                }
            }
    
    def _setup_gui(self):
        """Setup the main GUI layout."""
        # Create main paned window
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)
        
        # Right panel for preview
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=2)
        
        # Setup left panel components
        self._setup_left_panel()
        
        # Setup right panel components
        self._setup_right_panel()
    
    def _setup_left_panel(self):
        """Setup the left control panel."""
        # Notebook for different tabs
        self.notebook = ttk.Notebook(self.left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Prompt tab
        self.prompt_frame = PromptFrame(self.notebook, self)
        self.notebook.add(self.prompt_frame, text="Prompt")
        
        # Settings tab
        self.settings_frame = SettingsFrame(self.notebook, self, self.config)
        self.notebook.add(self.settings_frame, text="Settings")
        
        # Progress tab
        self.progress_frame = ProgressFrame(self.notebook, self)
        self.notebook.add(self.progress_frame, text="Progress")
    
    def _setup_right_panel(self):
        """Setup the right preview panel."""
        self.preview_frame = PreviewFrame(self.right_frame, self)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)
    
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Generation", command=self.new_generation)
        file_menu.add_command(label="Open Output Folder", command=self.open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save Current Images", command=self.save_current_images)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Prompt", command=self.clear_prompt)
        edit_menu.add_command(label="Reset Settings", command=self.reset_settings)
        
        # Models menu
        models_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Models", menu=models_menu)
        models_menu.add_command(label="Refresh Model List", command=self.refresh_models)
        models_menu.add_command(label="Download Model", command=self.download_model)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
    
    def _setup_bindings(self):
        """Setup keyboard bindings."""
        self.root.bind('<Control-n>', lambda e: self.new_generation())
        self.root.bind('<Control-o>', lambda e: self.open_output_folder())
        self.root.bind('<Control-s>', lambda e: self.save_current_images())
        self.root.bind('<F5>', lambda e: self.refresh_models())
        self.root.bind('<Control-Return>', lambda e: self.start_generation())
    
    def start_generation(self):
        """Start image generation in background thread."""
        if self.generation_running:
            messagebox.showwarning("Generation Running", "A generation is already in progress.")
            return
        
        # Get parameters from GUI
        params = self.get_generation_params()
        
        if not params['prompt'].strip():
            messagebox.showerror("Error", "Please enter a prompt.")
            return
        
        # Start generation in background
        self.generation_running = True
        self.progress_frame.start_progress()
        
        thread = threading.Thread(target=self._run_generation_thread, args=(params,))
        thread.daemon = True
        thread.start()
    
    def _run_generation_thread(self, params: Dict[str, Any]):
        """Run generation in background thread."""
        try:
            # Simulate generation process
            import time
            import random
            
            # Update progress
            self.root.after(0, lambda: self.progress_frame.update_status("Initializing models..."))
            time.sleep(1)
            
            self.root.after(0, lambda: self.progress_frame.update_status("Enhancing prompt..."))
            time.sleep(0.5)
            
            self.root.after(0, lambda: self.progress_frame.update_status("Generating images..."))
            
            # Simulate generation steps
            for step in range(params['steps']):
                progress = (step + 1) / params['steps'] * 100
                self.root.after(0, lambda p=progress: self.progress_frame.update_progress(p))
                time.sleep(0.1)
            
            # Simulate saving
            self.root.after(0, lambda: self.progress_frame.update_status("Saving images..."))
            time.sleep(0.5)
            
            # Complete
            self.root.after(0, self._generation_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self._generation_error(str(e)))
    
    def _generation_complete(self):
        """Handle generation completion."""
        self.generation_running = False
        self.progress_frame.complete_progress()
        
        # Show placeholder images in preview
        self.preview_frame.show_placeholder_images(self.get_generation_params()['num_images'])
        
        messagebox.showinfo("Success", "Image generation completed!")
    
    def _generation_error(self, error_msg: str):
        """Handle generation error."""
        self.generation_running = False
        self.progress_frame.reset_progress()
        messagebox.showerror("Generation Error", f"An error occurred: {error_msg}")
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get generation parameters from GUI."""
        return {
            'prompt': self.prompt_frame.get_prompt(),
            'negative_prompt': self.prompt_frame.get_negative_prompt(),
            'width': self.settings_frame.get_width(),
            'height': self.settings_frame.get_height(),
            'steps': self.settings_frame.get_steps(),
            'guidance_scale': self.settings_frame.get_guidance_scale(),
            'num_images': self.settings_frame.get_num_images(),
            'seed': self.settings_frame.get_seed(),
            'enhance_prompt': self.prompt_frame.get_enhance_enabled(),
        }
    
    def new_generation(self):
        """Start a new generation session."""
        if self.generation_running:
            if messagebox.askyesno("Stop Generation", "Stop current generation and start new?"):
                self.stop_generation()
            else:
                return
        
        self.clear_prompt()
        self.preview_frame.clear_preview()
        self.progress_frame.reset_progress()
    
    def stop_generation(self):
        """Stop current generation."""
        self.generation_running = False
        self.progress_frame.reset_progress()
    
    def clear_prompt(self):
        """Clear the prompt text."""
        self.prompt_frame.clear_prompt()
    
    def reset_settings(self):
        """Reset settings to defaults."""
        self.settings_frame.reset_to_defaults()
    
    def open_output_folder(self):
        """Open the output folder."""
        output_dir = self.config.get('output', {}).get('directory', './output')
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open folder in system file manager
        import subprocess
        import sys
        
        try:
            if sys.platform == "win32":
                subprocess.run(['explorer', str(output_path)])
            elif sys.platform == "darwin":
                subprocess.run(['open', str(output_path)])
            else:
                subprocess.run(['xdg-open', str(output_path)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")
    
    def save_current_images(self):
        """Save current images."""
        if not self.current_images:
            messagebox.showinfo("No Images", "No images to save.")
            return
        
        # This would implement actual saving
        messagebox.showinfo("Save Images", "Image saving functionality would be implemented here.")
    
    def refresh_models(self):
        """Refresh the model list."""
        messagebox.showinfo("Refresh Models", "Model list refresh would be implemented here.")
    
    def download_model(self):
        """Download a new model."""
        model_name = tk.simpledialog.askstring("Download Model", "Enter model name:")
        if model_name:
            messagebox.showinfo("Download Model", f"Would download model: {model_name}")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """AI Image Generator

A powerful tool for generating images using:
- Hugging Face Diffusion Models
- Ollama LLM Enhancement
- Modern GUI Interface

Version: 1.0.0
"""
        messagebox.showinfo("About", about_text)
    
    def show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts_text = """Keyboard Shortcuts:

Ctrl+N    - New Generation
Ctrl+O    - Open Output Folder
Ctrl+S    - Save Current Images
F5        - Refresh Model List
Ctrl+Enter - Start Generation
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


class MainWindow(ImageGeneratorGUI):
    """Alias for ImageGeneratorGUI for backward compatibility."""
    pass


if __name__ == "__main__":
    app = ImageGeneratorGUI()
    app.run()
