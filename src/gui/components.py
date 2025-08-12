"""GUI components for the AI image generation interface."""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional, Dict, Any
from PIL import Image, ImageTk
import io


class PromptFrame(ttk.Frame):
    """Frame for prompt input and enhancement controls."""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Setup the prompt input widgets."""
        # Main prompt
        ttk.Label(self, text="Prompt:").pack(anchor=tk.W, pady=(0, 5))
        
        self.prompt_text = scrolledtext.ScrolledText(
            self, height=4, wrap=tk.WORD
        )
        self.prompt_text.pack(fill=tk.X, pady=(0, 10))
        
        # Negative prompt
        ttk.Label(self, text="Negative Prompt:").pack(anchor=tk.W, pady=(0, 5))
        
        self.negative_prompt_text = scrolledtext.ScrolledText(
            self, height=2, wrap=tk.WORD
        )
        self.negative_prompt_text.pack(fill=tk.X, pady=(0, 10))
        
        # Enhancement controls
        enhancement_frame = ttk.LabelFrame(self, text="Prompt Enhancement")
        enhancement_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            enhancement_frame,
            text="Enhance prompt with Ollama",
            variable=self.enhance_var
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        style_frame = ttk.Frame(enhancement_frame)
        style_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(style_frame, text="Style:").pack(side=tk.LEFT)
        
        self.style_var = tk.StringVar(value="detailed")
        style_combo = ttk.Combobox(
            style_frame,
            textvariable=self.style_var,
            values=["detailed", "artistic", "photorealistic", "minimalist", "creative", "technical"],
            state="readonly",
            width=15
        )
        style_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Action buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="Generate Images",
            command=self.app.start_generation
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Enhance Prompt",
            command=self._enhance_prompt
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_prompt
        ).pack(side=tk.RIGHT)
    
    def get_prompt(self) -> str:
        """Get the current prompt text."""
        return self.prompt_text.get(1.0, tk.END).strip()
    
    def get_negative_prompt(self) -> str:
        """Get the current negative prompt text."""
        return self.negative_prompt_text.get(1.0, tk.END).strip()
    
    def get_enhance_enabled(self) -> bool:
        """Get whether prompt enhancement is enabled."""
        return self.enhance_var.get()
    
    def get_enhancement_style(self) -> str:
        """Get the selected enhancement style."""
        return self.style_var.get()
    
    def set_prompt(self, text: str):
        """Set the prompt text."""
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(1.0, text)
    
    def clear_prompt(self):
        """Clear both prompt fields."""
        self.prompt_text.delete(1.0, tk.END)
        self.negative_prompt_text.delete(1.0, tk.END)
    
    def _enhance_prompt(self):
        """Enhance the current prompt."""
        current_prompt = self.get_prompt()
        if not current_prompt.strip():
            messagebox.showwarning("No Prompt", "Please enter a prompt to enhance.")
            return
        
        # Simulate enhancement
        enhanced = f"{current_prompt}, {self.get_enhancement_style()} style, high quality, detailed"
        self.set_prompt(enhanced)
        messagebox.showinfo("Enhanced", "Prompt has been enhanced!")


class SettingsFrame(ttk.Frame):
    """Frame for generation settings."""
    
    def __init__(self, parent, app, config: Dict[str, Any]):
        super().__init__(parent)
        self.app = app
        self.config = config
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Setup the settings widgets."""
        # Model selection
        model_frame = ttk.LabelFrame(self, text="Model")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.model_var = tk.StringVar(value="runwayml/stable-diffusion-v1-5")
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=[
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2-1",
                "stabilityai/stable-diffusion-xl-base-1.0"
            ],
            width=30
        )
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Image dimensions
        dims_frame = ttk.LabelFrame(self, text="Image Dimensions")
        dims_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(dims_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.width_var = tk.IntVar(value=self.config.get('generation', {}).get('default_width', 512))
        width_spin = ttk.Spinbox(dims_frame, from_=256, to=2048, increment=64, textvariable=self.width_var, width=10)
        width_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(dims_frame, text="Height:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.height_var = tk.IntVar(value=self.config.get('generation', {}).get('default_height', 512))
        height_spin = ttk.Spinbox(dims_frame, from_=256, to=2048, increment=64, textvariable=self.height_var, width=10)
        height_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Generation parameters
        gen_frame = ttk.LabelFrame(self, text="Generation Parameters")
        gen_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(gen_frame, text="Steps:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.steps_var = tk.IntVar(value=self.config.get('generation', {}).get('default_steps', 20))
        steps_spin = ttk.Spinbox(gen_frame, from_=1, to=150, textvariable=self.steps_var, width=10)
        steps_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(gen_frame, text="Guidance Scale:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.guidance_var = tk.DoubleVar(value=self.config.get('generation', {}).get('default_guidance_scale', 7.5))
        guidance_spin = ttk.Spinbox(gen_frame, from_=1.0, to=20.0, increment=0.5, textvariable=self.guidance_var, width=10)
        guidance_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(gen_frame, text="Num Images:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.num_images_var = tk.IntVar(value=1)
        num_images_spin = ttk.Spinbox(gen_frame, from_=1, to=10, textvariable=self.num_images_var, width=10)
        num_images_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(gen_frame, text="Seed:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.seed_var = tk.StringVar()
        seed_entry = ttk.Entry(gen_frame, textvariable=self.seed_var, width=12)
        seed_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Random seed button
        ttk.Button(gen_frame, text="Random", command=self._generate_random_seed).grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
    
    def get_width(self) -> int:
        return self.width_var.get()
    
    def get_height(self) -> int:
        return self.height_var.get()
    
    def get_steps(self) -> int:
        return self.steps_var.get()
    
    def get_guidance_scale(self) -> float:
        return self.guidance_var.get()
    
    def get_num_images(self) -> int:
        return self.num_images_var.get()
    
    def get_seed(self) -> Optional[int]:
        try:
            seed_text = self.seed_var.get().strip()
            return int(seed_text) if seed_text else None
        except ValueError:
            return None
    
    def get_model(self) -> str:
        return self.model_var.get()
    
    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.width_var.set(512)
        self.height_var.set(512)
        self.steps_var.set(20)
        self.guidance_var.set(7.5)
        self.num_images_var.set(1)
        self.seed_var.set("")
        self.model_var.set("runwayml/stable-diffusion-v1-5")
    
    def _generate_random_seed(self):
        """Generate a random seed."""
        import random
        self.seed_var.set(str(random.randint(0, 2**32 - 1)))


class PreviewFrame(ttk.Frame):
    """Frame for image preview."""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.images = []
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Setup the preview widgets."""
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="Generated Images", font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT)
        
        ttk.Button(title_frame, text="Save All", command=self._save_all).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(title_frame, text="Clear", command=self.clear_preview).pack(side=tk.RIGHT)
        
        # Preview canvas with scrollbar
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Scrollable frame
        self.preview_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.preview_frame, anchor=tk.NW)
        
        self.preview_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
    
    def _on_frame_configure(self, event):
        """Handle frame resize."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def show_placeholder_images(self, count: int):
        """Show placeholder images."""
        self.clear_preview()
        
        for i in range(count):
            # Create placeholder image
            img_frame = ttk.LabelFrame(self.preview_frame, text=f"Image {i+1}")
            img_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Placeholder label
            placeholder_label = ttk.Label(img_frame, text=f"Generated Image {i+1}\n(512x512)", anchor=tk.CENTER)
            placeholder_label.pack(pady=20)
            
            # Action buttons
            btn_frame = ttk.Frame(img_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(btn_frame, text="Save", command=lambda idx=i: self._save_image(idx)).pack(side=tk.LEFT, padx=2)
            ttk.Button(btn_frame, text="Copy", command=lambda idx=i: self._copy_image(idx)).pack(side=tk.LEFT, padx=2)
            ttk.Button(btn_frame, text="View Full", command=lambda idx=i: self._view_full(idx)).pack(side=tk.LEFT, padx=2)
    
    def clear_preview(self):
        """Clear the preview area."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        self.images.clear()
    
    def _save_all(self):
        """Save all images."""
        messagebox.showinfo("Save All", "Save all images functionality would be implemented here.")
    
    def _save_image(self, index: int):
        """Save a specific image."""
        messagebox.showinfo("Save Image", f"Save image {index+1} functionality would be implemented here.")
    
    def _copy_image(self, index: int):
        """Copy image to clipboard."""
        messagebox.showinfo("Copy Image", f"Copy image {index+1} to clipboard functionality would be implemented here.")
    
    def _view_full(self, index: int):
        """View image in full size."""
        messagebox.showinfo("View Full", f"View image {index+1} in full size functionality would be implemented here.")


class ProgressFrame(ttk.Frame):
    """Frame for progress monitoring."""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Setup the progress widgets."""
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self, textvariable=self.status_var, font=("TkDefaultFont", 10))
        status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100, length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Progress percentage
        self.percentage_var = tk.StringVar(value="0%")
        percentage_label = ttk.Label(self, textvariable=self.percentage_var)
        percentage_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Log area
        ttk.Label(self, text="Generation Log:").pack(anchor=tk.W, pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(self, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Generation", command=self._stop_generation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)
    
    def start_progress(self):
        """Start progress monitoring."""
        self.progress_var.set(0)
        self.percentage_var.set("0%")
        self.status_var.set("Starting...")
        self.stop_button.config(state=tk.NORMAL)
        self._log_message("Generation started")
    
    def update_progress(self, percentage: float):
        """Update progress bar."""
        self.progress_var.set(percentage)
        self.percentage_var.set(f"{percentage:.1f}%")
    
    def update_status(self, status: str):
        """Update status message."""
        self.status_var.set(status)
        self._log_message(status)
    
    def complete_progress(self):
        """Mark progress as complete."""
        self.progress_var.set(100)
        self.percentage_var.set("100%")
        self.status_var.set("Complete!")
        self.stop_button.config(state=tk.DISABLED)
        self._log_message("Generation completed successfully")
    
    def reset_progress(self):
        """Reset progress to initial state."""
        self.progress_var.set(0)
        self.percentage_var.set("0%")
        self.status_var.set("Ready")
        self.stop_button.config(state=tk.DISABLED)
    
    def _stop_generation(self):
        """Stop the current generation."""
        self.app.stop_generation()
        self._log_message("Generation stopped by user")
    
    def _log_message(self, message: str):
        """Add message to log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _clear_log(self):
        """Clear the log area."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
