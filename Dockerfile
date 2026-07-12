# Use the official PyTorch image with CUDA 12.1 and cuDNN 8
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables to non-interactive to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and other CV tools
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
# (Relies on .dockerignore to exclude large weights, datasets, runs, and .venv)
COPY . /app

# Install the package in editable mode with all optional dependencies
RUN pip install --no-cache-dir -e ".[train,measurement,annotator,dev]"

# Set PYTHONPATH so that 'src' is discoverable without requiring activation
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Provide a default command. 
# Users will typically override this (e.g., to run bash or a specific python module)
CMD ["/bin/bash"]
