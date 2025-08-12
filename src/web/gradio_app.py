"""Gradio web interface for the AI image generation project."""

import gradio as gr
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
import json
import time
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class GradioImageGenerator:
    """Gradio interface wrapper for image generation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Gradio interface.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        self.generator = None
        self.current_model = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return {
                "generation": {
                    "default_width": 512,
                    "default_height": 512,
                    "default_steps": 20,
                    "default_guidance_scale": 7.5,
                },
                "web": {
                    "host": "127.0.0.1",
                    "port": 7860,
                    "share": False,
                }
            }
    
    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        enhance_prompt: bool = True,
        enhancement_style: str = "detailed",
        model_name: str = "runwayml/stable-diffusion-v1-5",
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[List[Image.Image], str, Dict[str, Any]]:
        """Generate images using the specified parameters.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            num_images: Number of images to generate
            seed: Random seed
            enhance_prompt: Whether to enhance prompt
            enhancement_style: Enhancement style
            model_name: Model to use
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (images, enhanced_prompt, metadata)
        """
        try:
            # Simulate generation process
            progress(0, desc="Initializing...")
            time.sleep(0.5)
            
            enhanced_prompt_text = prompt
            if enhance_prompt:
                progress(0.1, desc="Enhancing prompt...")
                enhanced_prompt_text = self._enhance_prompt(prompt, enhancement_style)
                time.sleep(0.5)
            
            progress(0.2, desc="Loading model...")
            time.sleep(0.5)
            
            # Simulate image generation
            images = []
            for i in range(num_images):
                progress(0.2 + (0.7 * (i + 1) / num_images), desc=f"Generating image {i+1}/{num_images}...")
                
                # Create placeholder image
                image = self._create_placeholder_image(width, height, f"Generated Image {i+1}")
                images.append(image)
                time.sleep(1)  # Simulate generation time
            
            progress(0.9, desc="Finalizing...")
            time.sleep(0.2)
            
            # Create metadata
            metadata = {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt_text,
                "negative_prompt": negative_prompt,
                "model": model_name,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "num_images": num_images,
                "enhancement_style": enhancement_style if enhance_prompt else None,
            }
            
            progress(1.0, desc="Complete!")
            
            return images, enhanced_prompt_text, metadata
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise gr.Error(f"Generation failed: {str(e)}")
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Simulate prompt enhancement."""
        style_enhancements = {
            "detailed": "highly detailed, intricate, sharp focus, professional",
            "artistic": "artistic, beautiful, elegant, stylized, expressive",
            "photorealistic": "photorealistic, hyperrealistic, photography, natural lighting",
            "minimalist": "clean, simple, minimal, elegant, modern",
            "creative": "creative, imaginative, unique, innovative, experimental",
            "technical": "technical illustration, precise, engineering, detailed diagram"
        }
        
        enhancement = style_enhancements.get(style, "high quality")
        return f"{prompt}, {enhancement}, masterpiece, 4k"
    
    def _create_placeholder_image(self, width: int, height: int, text: str) -> Image.Image:
        """Create a placeholder image."""
        # Create a simple colored rectangle with text
        import random
        colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 200)]
        color = random.choice(colors)
        
        image = Image.new('RGB', (width, height), color)
        
        # Add text if PIL supports it
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Get text size and center it
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill=(0, 0, 0), font=font)
        except ImportError:
            pass  # Skip text if ImageDraw not available
        
        return image
    
    def get_model_list(self) -> List[str]:
        """Get list of available models."""
        return [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "dreamlike-art/dreamlike-diffusion-1.0",
            "prompthero/openjourney",
        ]
    
    def save_generation_history(self, metadata: Dict[str, Any]) -> str:
        """Save generation history."""
        try:
            history_file = Path("generation_history.json")
            
            # Load existing history
            history = []
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new entry
            import datetime
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            history.append(metadata)
            
            # Keep only last 100 entries
            history = history[-100:]
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            return "Generation saved to history"
        except Exception as e:
            return f"Failed to save history: {e}"


def create_gradio_interface(config_path: Optional[str] = None) -> gr.Blocks:
    """Create the Gradio interface.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Gradio Blocks interface
    """
    generator = GradioImageGenerator(config_path)
    
    with gr.Blocks(
        title="AI Image Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .generate-button { background: linear-gradient(45deg, #6366f1, #8b5cf6) !important; }
        .preview-gallery { min-height: 400px; }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🎨 AI Image Generator
            Generate stunning images using Hugging Face diffusion models enhanced with Ollama LLM capabilities.
            """,
            elem_classes=["main-container"]
        )
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Prompt")
                    
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=3,
                        max_lines=5
                    )
                    
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid in the image...",
                        lines=2,
                        max_lines=3
                    )
                    
                    with gr.Row():
                        enhance_prompt = gr.Checkbox(
                            label="Enhance prompt with Ollama",
                            value=True
                        )
                        enhancement_style = gr.Dropdown(
                            choices=["detailed", "artistic", "photorealistic", "minimalist", "creative", "technical"],
                            value="detailed",
                            label="Enhancement Style"
                        )
                
                with gr.Group():
                    gr.Markdown("### Model & Settings")
                    
                    model_dropdown = gr.Dropdown(
                        choices=generator.get_model_list(),
                        value="runwayml/stable-diffusion-v1-5",
                        label="Model"
                    )
                    
                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            step=64,
                            value=512,
                            label="Width"
                        )
                        height_slider = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            step=64,
                            value=512,
                            label="Height"
                        )
                    
                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=1,
                            maximum=150,
                            step=1,
                            value=20,
                            label="Inference Steps"
                        )
                        guidance_slider = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            step=0.5,
                            value=7.5,
                            label="Guidance Scale"
                        )
                    
                    with gr.Row():
                        num_images_slider = gr.Slider(
                            minimum=1,
                            maximum=4,
                            step=1,
                            value=1,
                            label="Number of Images"
                        )
                        seed_input = gr.Number(
                            label="Seed (optional)",
                            precision=0
                        )
                
                generate_button = gr.Button(
                    "🎨 Generate Images",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-button"]
                )
            
            # Right column - Results
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### Generated Images")
                    
                    image_gallery = gr.Gallery(
                        label="Generated Images",
                        show_label=False,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        height="auto",
                        elem_classes=["preview-gallery"]
                    )
                    
                    with gr.Row():
                        save_button = gr.Button("💾 Save Images")
                        download_button = gr.DownloadButton(
                            "📥 Download All",
                            visible=False
                        )
                
                with gr.Group():
                    gr.Markdown("### Enhanced Prompt")
                    enhanced_prompt_output = gr.Textbox(
                        label="Enhanced Prompt",
                        interactive=False,
                        lines=3
                    )
                
                with gr.Group():
                    gr.Markdown("### Generation Info")
                    metadata_output = gr.JSON(
                        label="Generation Metadata",
                        visible=False
                    )
                    
                    status_output = gr.Textbox(
                        label="Status",
                        value="Ready to generate!",
                        interactive=False
                    )
        
        # Event handlers
        generate_button.click(
            fn=generator.generate_images,
            inputs=[
                prompt_input,
                negative_prompt_input,
                width_slider,
                height_slider,
                steps_slider,
                guidance_slider,
                num_images_slider,
                seed_input,
                enhance_prompt,
                enhancement_style,
                model_dropdown
            ],
            outputs=[
                image_gallery,
                enhanced_prompt_output,
                metadata_output
            ],
            show_progress=True
        )
        
        save_button.click(
            fn=generator.save_generation_history,
            inputs=[metadata_output],
            outputs=[status_output]
        )
        
        # Add examples
        gr.Examples(
            examples=[
                ["A beautiful sunset over mountains with orange and pink clouds"],
                ["A futuristic city with flying cars and neon lights"],
                ["A cute cat wearing a wizard hat, digital art"],
                ["An ancient castle on a cliff overlooking the ocean"],
                ["A serene lake with cherry blossoms in spring"]
            ],
            inputs=[prompt_input]
        )
    
    return interface


def launch_gradio_app(
    config_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    debug: bool = False
) -> None:
    """Launch the Gradio application.
    
    Args:
        config_path: Path to configuration file
        host: Host address
        port: Port number
        share: Whether to create public link
        debug: Enable debug mode
    """
    interface = create_gradio_interface(config_path)
    
    print(f"🚀 Launching AI Image Generator on {host}:{port}")
    print(f"📝 Open your browser and navigate to: http://{host}:{port}")
    
    interface.launch(
        server_name=host,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    launch_gradio_app()
