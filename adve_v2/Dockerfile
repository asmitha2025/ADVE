FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git

# App code
COPY . .

# Download models at build time (not at runtime)
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')"
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Data directory
RUN mkdir -p /data/uploads /data/main_index

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda

CMD ["python", "-m", "uvicorn", "adve.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
