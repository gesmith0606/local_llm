# 🎨 AI Image Generation with Hugging Face & Ollama

A sophisticated image generation application that combines Hugging Face's state-of-the-art diffusion models with Ollama's local LLM capabilities for intelligent prompt enhancement and creative AI-powered image generation.

## ✨ Features

### 🤖 AI-Powered Generation
- **Multiple Model Support**: Stable Diffusion, DALL-E, Midjourney-style models
- **Smart Prompt Enhancement**: Ollama integration for automatic prompt improvement
- **Batch Generation**: Generate multiple variations efficiently
- **Style Transfer**: Apply artistic styles to generated images

### 🎯 User Interfaces
- **GUI Application**: User-friendly desktop interface with PyQt6
- **Web Interface**: Browser-based interface with Gradio
- **CLI Tools**: Command-line tools for automation and batch processing
- **API Server**: RESTful API for integration with other applications

### ⚡ Performance & Quality
- **GPU Acceleration**: CUDA support for faster generation
- **Memory Optimization**: Efficient model loading and unloading
- **Quality Controls**: Configurable generation parameters
- **Progress Tracking**: Real-time generation progress and ETA

## 🚀 Quick Start

### Prerequisites
```bash
# Clone and setup
git clone <repository-url>
cd local_llm
pip install -r requirements.txt

# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download
```

### Basic Usage

1. **GUI Application**:
   ```bash
   python -m src.gui.main_window
   ```

2. **Web Interface**:
   ```bash
   python -m src.web.gradio_app
   ```

3. **CLI Generation**:
   ```bash
   python -m src.cli.generate --prompt "A beautiful sunset over mountains" --model stable-diffusion-xl
   ```

## 📁 Project Structure

```
local_llm/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── models.py          # Model loading and management
│   │   └── generator.py       # Core image generation logic
│   ├── huggingface/
│   │   ├── __init__.py
│   │   ├── diffusion.py       # Diffusion model integration
│   │   └── utils.py           # HF utilities
│   ├── ollama/
│   │   ├── __init__.py
│   │   ├── client.py          # Ollama API client
│   │   └── prompt_enhancer.py # Prompt enhancement logic
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py     # Main GUI application
│   │   └── components/        # GUI components
│   ├── web/
│   │   ├── __init__.py
│   │   └── gradio_app.py      # Web interface
│   ├── cli/
│   │   ├── __init__.py
│   │   └── generate.py        # CLI tools
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py     # Image processing utilities
│       └── logging_config.py  # Logging configuration
├── tests/
├── examples/
├── docs/
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd local_llm
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Ollama (Optional for prompt enhancement):**
   - Download and install Ollama from [https://ollama.ai](https://ollama.ai)
   - Start Ollama service: `ollama serve`
   - Pull a model: `ollama pull llama2`

### Quick Test

```bash
# Test the setup
python test_setup.py

# View examples and project structure
python examples.py
```

## 🖥️ Multiple Interfaces

This project supports three different interfaces:

### 1. GUI Interface (Tkinter)
Modern desktop application with intuitive controls:

```bash
# Launch GUI
python launcher.py gui

# Or directly
python -m src.gui.main_window
```

**Features:**
- Visual prompt editor with enhancement options
- Real-time settings adjustment
- Image preview gallery
- Progress monitoring with logs
- Batch generation support
- Keyboard shortcuts

### 2. Web Interface (Gradio)
Browser-based interface for easy access:

```bash
# Launch web interface
python launcher.py web

# Launch on custom port
python launcher.py web --port 8080

# Launch with public sharing
python launcher.py web --share
```

**Features:**
- Responsive web design
- Interactive sliders and controls
- Live image gallery
- Generation history
- Mobile-friendly interface
- Public sharing capabilities

### 3. Command Line Interface (CLI)
Powerful CLI for automation and scripting:

```bash
# Generate an image
python launcher.py cli generate "a beautiful sunset over mountains"

# Generate with specific parameters
python launcher.py cli generate "a cute cat" --model "runwayml/stable-diffusion-v1-5" --steps 30 --width 768 --height 768

# Enhance a prompt
python launcher.py cli enhance "simple prompt" --style artistic

# Show configuration
python launcher.py cli config --show

# Get help
python launcher.py cli --help
```

**Features:**
- Batch processing
- Scriptable automation
- Configuration management
- Model downloading
- Progress tracking

generator = ImageGenerator()
result = await generator.generate_enhanced(
    prompt="sunset mountains",
    enhance_prompt=True,
    model="stable-diffusion-xl"
)
```

### Batch Generation
```python
prompts = ["cat in space", "futuristic city", "abstract art"]
results = await generator.batch_generate(prompts, count=3)
```

### Style Transfer
```python
result = await generator.generate_with_style(
    prompt="portrait of a woman",
    style="oil painting",
    strength=0.8
)
```

## ⚙️ Configuration

Configure the application through `config.yaml`:

```yaml
models:
  default_diffusion: "stabilityai/stable-diffusion-xl-base-1.0"
  default_ollama: "llama2"

generation:
  default_size: [1024, 1024]
  default_steps: 50
  default_guidance: 7.5

ollama:
  host: "localhost"
  port: 11434
  timeout: 30

ui:
  theme: "dark"
  auto_save: true
  output_dir: "outputs"
```

## 🛠️ Development

### Setup Development Environment
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Run Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

## 📊 Supported Models

### Hugging Face Models
- Stable Diffusion XL
- Stable Diffusion 2.1
- DALL-E Mini
- DeepFloyd IF
- ControlNet variants

### Ollama Models
- Llama 2/3
- Mistral
- CodeLlama
- Vicuna
- Any custom models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
- [Ollama](https://ollama.ai/)
- [Documentation](./docs/)

---

**Status**: 🟢 Ready for AI-powered image generation!
