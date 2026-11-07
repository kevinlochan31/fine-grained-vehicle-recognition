"""Deploy AutoLens (FastAPI app + web UI) on Modal: https://modal.com

    pip install modal
    modal token new                 # one-time browser sign-in
    modal deploy modal_app.py       # prints the public https URL

The container is CPU-only (2 cores, 6 GB RAM). The CLIP ViT-B/16 weights are downloaded while the image
is built, so a cold start only has to load the models from disk. A warm container answers in ~1-3 s;
after 20 minutes without traffic it scales to zero, and the next request cold-starts (~30-60 s).
To keep it warm during judging, set KEEP_WARM=1 when deploying: `KEEP_WARM=1 modal deploy modal_app.py`.
"""
import os

import modal

IGNORE = [".git", ".git/**", ".venv", ".venv/**", "**/__pycache__", "**/__pycache__/**", "*.pyc", "runs", "runs/**",
          "*.log", ".claude", ".claude/**", "results/failures", "results/failures/**", "yolo11*.pt", "~$*",
          "docs/*.docx", "docs/*.pptx"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements("requirements.txt")
    .run_commands("pip uninstall -y opencv-python || true", "pip install --force-reinstall --no-deps opencv-python-headless")
    .env({"HF_HOME": "/root/.hf", "YOLO_CONFIG_DIR": "/tmp/Ultralytics", "MPLCONFIGDIR": "/tmp/mpl",
          "CVH_WORK_DIR": "/tmp/cvh", "OMP_NUM_THREADS": "2"})
    .run_commands("python -c \"import open_clip; open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')\"")
    .add_local_dir(".", remote_path="/app", ignore=IGNORE)
)

app = modal.App("autolens", image=image)


@app.function(cpu=2.0, memory=6144, timeout=600, scaledown_window=1200,
              min_containers=1 if os.environ.get("KEEP_WARM") == "1" else 0)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(label="autolens")
def web():
    import sys
    os.chdir("/app")
    sys.path.insert(0, "/app")
    import importlib
    return importlib.import_module("app").app      # app.py loads and warms the models at import
