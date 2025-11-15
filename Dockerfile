# Use NVIDIA CUDA base image with runtime and Python support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1
ENV PORT=8009

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies
RUN pip install fastapi uvicorn numpy numba

# Create app directory
WORKDIR /app

# Copy your FastAPI application code
COPY . /app

# Expose FastAPI port
EXPOSE ${PORT}

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009"]