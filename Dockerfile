# Use the proven RunPod base image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update --yes --quiet && \
    apt-get install --yes --quiet --no-install-recommends \
      ffmpeg \
      git \
      libsndfile1 \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Clone the SAM-Audio repository
# We clone this first so we can install it in the next step
RUN git clone https://github.com/davidharket/sam-audio.git .

# 3. Install Python dependencies
# We install the repository itself (-e .) plus the HF requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir huggingface_hub accelerate

# 4. Create output directory
RUN mkdir -p /app/outputs

# 5. Copy your inference script (ensure inference.py is in your build folder)
COPY inference.py /app/inference.py

# Default command
CMD ["python3", "inference.py"]
