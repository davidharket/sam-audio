# Use a PyTorch base image with CUDA support for RunPod
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies (FFmpeg is required for audio processing)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone the specific repository
RUN git clone https://github.com/davidharket/sam-audio.git .

# Install Python dependencies
# We install the repo in editable mode to ensure sam_audio is in the path
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir huggingface_hub accelerate

# Create a directory for outputs
RUN mkdir -p /app/outputs

# Copy your inference script into the container
COPY inference.py /app/inference.py

# Set the Hugging Face token environment variable placeholder
# (You will pass this via RunPod's env variables)
ENV HF_TOKEN=""

# Default command (keeps container alive for manual use or runs a script)
CMD ["python3", "inference.py"]
