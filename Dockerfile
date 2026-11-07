# CPU image for the demo app (Hugging Face Spaces / Render / any Docker host).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/mpl
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install -r requirements.txt && \
    pip uninstall -y opencv-python || true && pip install --force-reinstall --no-deps opencv-python-headless

# pre-fetch the CLIP ViT-B/16 (LAION-2B) weights so the first request is not slow
ENV HF_HOME=/app/.hf
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')"

COPY . .
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD ["python", "app.py"]
