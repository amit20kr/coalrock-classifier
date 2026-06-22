"""
download_model.py — HuggingFace Hub weight downloader.
Runs BEFORE uvicorn starts (see Render start command below).

Render start command:
    python download_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT

Why this exists:
  - pcvit_expert.pth is 3.6 MB. Too large for git without LFS.
  - Render's free-tier filesystem is ephemeral: it resets on every new deploy.
  - HuggingFace Hub provides free model hosting with versioned downloads.

Setup checklist (one-time):
  1. pip install huggingface_hub
  2. Create a free HuggingFace account
  3. Create a new Model repo: https://huggingface.co/new
  4. Upload pcvit_expert.pth to the repo
  5. Update REPO_ID below with your actual "username/repo-name"
"""
import os
import sys
from pathlib import Path


REPO_ID    = "amit2-0kr/spectra-sense"        # HuggingFace repo for model weights
FILENAME   = "pcvit_expert.pth"
LOCAL_DIR  = Path("models")
MODEL_PATH = LOCAL_DIR / FILENAME


def download_model() -> None:
    if MODEL_PATH.exists():
        print(f"[download_model] Model already present at {MODEL_PATH}. Skipping.")
        return

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[download_model] ERROR: huggingface_hub not installed.")
        print("  Add it to backend/requirements.txt and redeploy.")
        sys.exit(1)

    print(f"[download_model] Downloading {FILENAME} from {REPO_ID} ...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(LOCAL_DIR),
    )
    print(f"[download_model] Done. Saved to {downloaded}")


if __name__ == "__main__":
    download_model()
