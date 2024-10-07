# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory in the container
WORKDIR /app

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
  python3.10 \
  python3-pip \
  python3.10-venv \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run the application
CMD ["python", "main.py"]