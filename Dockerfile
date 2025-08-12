# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for models and output
RUN mkdir -p /app/models /app/output /app/logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models

# Expose port for web interface
EXPOSE 7860

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command (can be overridden)
CMD ["python", "launcher.py", "web", "--host", "0.0.0.0", "--port", "7860"]
