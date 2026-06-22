"""
main.py — FastAPI Inference Engine
Exposes /predict and /health endpoints.
Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

import torch
from model import PCViT
from pipeline import (
    process_csv_to_tensor,
    load_label_encoder,
    FUEL_CLASSES,
    XRF_PASSPORT,
)

app = FastAPI(
    title="Spectra-Sense Inference Engine",
    description="PCViT-based real-time coal-gangue classification via NIR spectroscopy.",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────────────────────
# SERVER BOOT: load all assets into RAM once at startup.
# Running on Render free tier → CPU only. map_location='cpu' is not optional.
# weights_only=True: security hardening for PyTorch 2.x (no arbitrary code exec).
# ──────────────────────────────────────────────────────────────────────────────
_ASSETS_DIR = Path("assets")
_MODEL_DIR  = Path("models")

_le_path    = _ASSETS_DIR / "label_encoder.pkl"
_model_path = _MODEL_DIR  / "pcvit_expert.pth"

# Fail loudly at startup, not silently at first request.
if not _le_path.exists():
    raise FileNotFoundError(
        f"label_encoder.pkl not found at {_le_path}. "
        "Did you commit it to the repo? It is required for class name mapping."
    )
if not _model_path.exists():
    raise FileNotFoundError(
        f"pcvit_expert.pth not found at {_model_path}. "
        "Run `python download_model.py` first, or check your Render deploy command."
    )

device = torch.device("cpu")
le     = load_label_encoder(str(_le_path))

model  = PCViT(num_classes=len(le.classes_))
model.load_state_dict(
    torch.load(str(_model_path), map_location=device, weights_only=True)
)
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# GANGUE OVERRIDE: the industrial safety net.
# If ANY single scan in the batch scores > 0.85 gangue probability, the batch
# is flagged REJECT regardless of what the mean probability says.
# Mean of [45 coal, 5 rock scans] can mask the rock — max-pool cannot.
# ──────────────────────────────────────────────────────────────────────────────
GANGUE_OVERRIDE_THRESHOLD = 0.85
gangue_indices = [i for i, cls in enumerate(le.classes_) if cls not in FUEL_CLASSES]


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Lightweight liveness check. Render's health-check URL should point here."""
    return {
        "status": "ok",
        "model": "PCViT",
        "num_classes": len(le.classes_),
        "device": str(device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a raw spectrometer CSV, run it through the full pipeline,
    and return a structured prediction payload.
    """
    try:
        contents = await file.read()

        # ── Data Transformation ────────────────────────────────────────────
        tensor, snv_plot, d2_plot, wavelengths = process_csv_to_tensor(contents)
        # tensor shape: (N_scans, 3, 1500)

        # ── Model Inference ────────────────────────────────────────────────
        with torch.no_grad():
            logits = model(tensor)           # (N, 24) raw logits
            probs  = torch.softmax(logits, dim=1)  # (N, 24) probabilities

        # ── Gangue Override Safety Net ─────────────────────────────────────
        gangue_probs_per_sample = probs[:, gangue_indices]  # (N, 12)

        # Max-pool: even one heavily gangue-flagged scan trips the override.
        max_gangue_prob = gangue_probs_per_sample.max().item()

        # Mean-pool: determines the dominant predicted class across the batch.
        mean_probs           = probs.mean(dim=0)            # (24,)
        top5_probs, top5_idx = mean_probs.topk(5)

        predicted_id    = top5_idx[0].item()
        predicted_class = str(le.inverse_transform([predicted_id])[0])
        confidence      = float(top5_probs[0].item())

        # The ultimate industrial logic check
        is_gangue = (predicted_class not in FUEL_CLASSES) or \
                    (max_gangue_prob > GANGUE_OVERRIDE_THRESHOLD)

        if is_gangue and predicted_class in FUEL_CLASSES:
            # Mean said "coal" but a single scan screamed "rock" at >85% confidence.
            # This is the safety net doing exactly its job. Stop the conveyor.
            predicted_class = "Gangue Contamination Detected"
            verdict         = "REJECT (Safety Override)"
        elif is_gangue:
            verdict = "REJECT"
        else:
            verdict = "ACCEPT"

        # ── Package Response ───────────────────────────────────────────────
        top_5_list = [
            {
                "class":       str(le.inverse_transform([idx.item()])[0]),
                "probability": float(prob.item()),
            }
            for prob, idx in zip(top5_probs, top5_idx)
        ]

        passport_data = XRF_PASSPORT.get(
            predicted_class,
            {"Origin": "Standard Mining Library", "Geochemistry": {}}
        )

        return {
            "verdict":          verdict,
            "is_gangue":        is_gangue,
            "predicted_class":  predicted_class,
            "confidence":       confidence,
            "top_5":            top_5_list,
            "digital_passport": passport_data,
            "spectral_channels": {
                "wavelengths":       wavelengths,
                "snv":               snv_plot,
                "second_derivative": d2_plot,
            },
        }

    except ValueError as e:
        # Known pipeline errors (bad CSV format, wrong band count, etc.)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Unexpected errors — surface the message for debugging on Render logs.
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
