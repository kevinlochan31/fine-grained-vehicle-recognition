"""Deploy the app to a public Hugging Face Space (Docker SDK) - no git required.

1. Create a free account at https://huggingface.co and a *write* access token
   (Settings -> Access Tokens).
2. Log in once on this machine (the token is stored locally, never in the repo):
       hf auth login
3. Deploy:
       python -m scripts.deploy_hf --space <your-username>/fine-grained-cars
The Space builds the Dockerfile and serves app.py at
https://huggingface.co/spaces/<your-username>/fine-grained-cars
"""
from __future__ import annotations

import argparse

from huggingface_hub import HfApi

from src import config as C

IGNORE = ["runs/*", "*.log", "__pycache__/*", "**/__pycache__/*", ".gradio/*", "yolo11*.pt",
          "results/failures/*", ".venv/*", "*.pyc", "docs/*.pptx", ".git/*", ".claude/*", "~$*"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="username/space-name")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(C.PROJECT_ROOT), repo_id=a.space, repo_type="space",
                      ignore_patterns=IGNORE, commit_message="Deploy P5 vehicle recognition app")
    print(f"Uploaded. Build logs / app: https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
