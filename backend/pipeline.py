"""
pipeline.py — The Data Layer
Wavelength Assassin + 3-Channel Physics Tensor Constructor
"""
import io
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder

EXPECTED_BANDS = 1500

# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: 12 FUEL classes vs 12 GANGUE classes
# This list is the boundary the safety net enforces.
# ──────────────────────────────────────────────────────────────────────────────
FUEL_CLASSES = [
    "Anthracite coal grade 1",
    "Anthracite coal grade 2",
    "Lean coal",
    "Lean-thin coal",
    "Thin coal",
    "Coking coal",
    "0.3 coking coal",
    "Fat coal",
    "Gas-fat coal",
    "Gas coal",
    "Lignite coal grade 1",
    "Lignite coal grade 2",
]

# ──────────────────────────────────────────────────────────────────────────────
# XRF PASSPORT — nested structure (Origin + Geochemistry)
# Geochemistry values sourced from backend/assets/XRF_database.csv (real XRF measurements).
# Frontend calls: passport_data.get('Origin', ...) and passport_data.get('Geochemistry', {})
# ──────────────────────────────────────────────────────────────────────────────
XRF_PASSPORT = {
    # ── FUEL CLASSES (coal — only SiO2, Al2O3, Fe2O3 measurable via XRF) ─────
    "Anthracite coal grade 1": {
        "Origin": "Jincheng Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "2.63%", "Al2O3": "1.01%", "Fe2O3": "0.87%"
        }
    },
    "Anthracite coal grade 2": {
        "Origin": "Yangquan Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "4.61%", "Al2O3": "1.17%", "Fe2O3": "0.81%"
        }
    },
    "Lean coal": {
        "Origin": "Fengfeng Coal Mine, Hebei Province",
        "Geochemistry": {
            "SiO2": "6.18%", "Al2O3": "1.40%", "Fe2O3": "1.19%"
        }
    },
    "Lean-thin coal": {
        "Origin": "Huainan Coal Mine, Anhui Province",
        "Geochemistry": {
            "SiO2": "6.46%", "Al2O3": "0.98%", "Fe2O3": "0.44%"
        }
    },
    "Thin coal": {
        "Origin": "Pingdingshan Coal Mine, Henan Province",
        "Geochemistry": {
            "SiO2": "7.13%", "Al2O3": "1.20%", "Fe2O3": "0.54%"
        }
    },
    "Coking coal": {
        "Origin": "Liuzhi Coal Mine, Guizhou Province",
        "Geochemistry": {
            "SiO2": "3.05%", "Al2O3": "0.84%", "Fe2O3": "1.08%"
        }
    },
    "0.3 coking coal": {
        "Origin": "Xishan Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "2.48%", "Al2O3": "1.23%", "Fe2O3": "0.89%"
        }
    },
    "Fat coal": {
        "Origin": "Kailuan Coal Mine, Hebei Province",
        "Geochemistry": {
            "SiO2": "3.96%", "Al2O3": "1.00%", "Fe2O3": "1.25%"
        }
    },
    "Gas-fat coal": {
        "Origin": "Shenmu Coal Mine, Shaanxi Province",
        "Geochemistry": {
            "SiO2": "2.92%", "Al2O3": "1.68%", "Fe2O3": "0.93%"
        }
    },
    "Gas coal": {
        "Origin": "Hanjing Coal Mine, Shaanxi Province",
        "Geochemistry": {
            "SiO2": "2.05%", "Al2O3": "2.09%", "Fe2O3": "0.98%"
        }
    },
    "Lignite coal grade 1": {
        "Origin": "Baotou Coal Mine, Inner Mongolia",
        "Geochemistry": {
            "SiO2": "17.20%", "Al2O3": "1.07%", "Fe2O3": "0.86%"
        }
    },
    "Lignite coal grade 2": {
        "Origin": "Yimin Coal Mine, Inner Mongolia",
        "Geochemistry": {
            "SiO2": "21.05%", "Al2O3": "1.75%", "Fe2O3": "1.03%"
        }
    },
    # ── GANGUE CLASSES (full XRF elemental profiles) ─────────────────────────
    "Carbonaceous shale": {
        "Origin": "Dongfeng Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "31.11%", "Al2O3": "18.19%", "Fe2O3": "1.66%",
            "K2O": "1.05%", "MgO": "0.45%", "Na2O": "0.64%",
            "CaO": "0.15%", "Substrate": "46.25%"
        }
    },
    "Black shale 1": {
        "Origin": "Xuzhou Coal Mine, Jiangsu Province",
        "Geochemistry": {
            "SiO2": "47.02%", "Al2O3": "30.21%", "Fe2O3": "3.64%",
            "K2O": "0.64%", "MgO": "0.60%", "Na2O": "1.62%",
            "CaO": "2.58%", "Substrate": "13.24%"
        }
    },
    "Black shale 2": {
        "Origin": "Datong Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "42.36%", "Al2O3": "27.83%", "Fe2O3": "2.74%",
            "K2O": "1.27%", "MgO": "0.22%", "Na2O": "0.08%",
            "CaO": "0.30%", "Substrate": "24.80%"
        }
    },
    "Arenaceous shale 1": {
        "Origin": "Xinglongzhuang Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "64.51%", "Al2O3": "15.40%", "Fe2O3": "2.77%",
            "K2O": "2.29%", "MgO": "1.73%", "Na2O": "0.12%",
            "CaO": "2.73%", "Substrate": "9.89%"
        }
    },
    "Arenaceous shale 2": {
        "Origin": "Baodian Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "64.17%", "Al2O3": "14.96%", "Fe2O3": "2.93%",
            "K2O": "1.90%", "MgO": "1.45%",
            "CaO": "2.61%", "Substrate": "11.35%"
        }
    },
    "Medium-grained sandstone": {
        "Origin": "Jining Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "79.17%", "Al2O3": "8.72%", "Fe2O3": "1.95%",
            "K2O": "1.14%", "MgO": "0.52%", "Na2O": "0.53%",
            "CaO": "0.37%", "Substrate": "7.26%"
        }
    },
    "Clay": {
        "Origin": "Yanzhou Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "40.32%", "Al2O3": "30.02%", "Fe2O3": "4.44%",
            "K2O": "13.26%", "MgO": "1.00%", "Na2O": "1.63%",
            "CaO": "3.55%", "Substrate": "5.36%"
        }
    },
    "Fine-grained sandstone": {
        "Origin": "Zibo Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "77.47%", "Al2O3": "9.04%", "Fe2O3": "1.76%",
            "K2O": "1.10%", "MgO": "1.30%", "Na2O": "0.16%",
            "CaO": "2.48%", "Substrate": "6.17%"
        }
    },
    "Siltstone 1": {
        "Origin": "Huaibei Coal Mine, Anhui Province",
        "Geochemistry": {
            "SiO2": "56.13%", "Al2O3": "13.21%", "Fe2O3": "4.50%",
            "K2O": "2.47%", "MgO": "1.42%", "Na2O": "1.42%",
            "CaO": "7.82%", "Substrate": "12.73%"
        }
    },
    "Siltstone 2": {
        "Origin": "Huainan Coal Mine, Anhui Province",
        "Geochemistry": {
            "SiO2": "82.06%", "Al2O3": "1.66%", "Fe2O3": "3.51%",
            "K2O": "1.37%", "MgO": "2.48%", "Na2O": "0.77%",
            "CaO": "2.69%", "Substrate": "4.87%"
        }
    },
    "Argillaceous limestone 1": {
        "Origin": "Feicheng Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "13.50%", "Al2O3": "11.22%", "Fe2O3": "3.06%",
            "K2O": "2.07%", "MgO": "16.70%",
            "CaO": "26.56%", "Substrate": "26.40%"
        }
    },
    "Argillaceous limestone 2": {
        "Origin": "Zaozhuang Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "17.28%", "Al2O3": "12.08%", "Fe2O3": "2.59%",
            "K2O": "1.89%", "MgO": "4.93%", "Na2O": "1.22%",
            "CaO": "28.90%", "Substrate": "30.56%"
        }
    },
}


def load_label_encoder(path: str) -> LabelEncoder:
    with open(path, "rb") as f:
        return pickle.load(f)


def _snv(arr: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate: per-sample z-score normalization.
    Maps every sample to zero mean and unit variance.
    Strips out albedo and sensor-distance effects entirely.
    Input/Output: (N, L)
    """
    mean = arr.mean(axis=1, keepdims=True)
    std  = arr.std(axis=1,  keepdims=True)
    return (arr - mean) / (std + 1e-8)


def _savgol_deriv(arr: np.ndarray, order: int) -> np.ndarray:
    """
    Savitzky-Golay derivative. window=11, polyorder=2.
    1st order: highlights slope/edges at absorption bands.
    2nd order: turns the 2200nm Al-OH dip into a sharp spike.
    Input/Output: (N, L)
    """
    deriv = savgol_filter(arr, window_length=11, polyorder=2, deriv=order, axis=1)
    # Crop to EXPECTED_BANDS to handle edge-effect padding
    return deriv[:, :EXPECTED_BANDS]


def process_csv_to_tensor(file_bytes: bytes):
    """
    The full 8-step Wavelength Assassin + Physics Tensor pipeline.

    Accepts raw CSV bytes (from UploadFile.read() in FastAPI).

    Returns:
        tensor      : torch.FloatTensor of shape (N, 3, 1500) — model input
        snv_plot    : list[float], length 1500 — SNV channel of sample 0 for plot
        d2_plot     : list[float], length 1500 — 2nd deriv of sample 0 for plot
        wavelengths : list[float], length 1500 — x-axis values (1000–2499 nm)
    """
    # ── Step 1: Schema-agnostic ingestion ─────────────────────────────────────
    # header=None because row 0 might be metadata, not column headers.
    df_raw = pd.read_csv(io.BytesIO(file_bytes), header=None, low_memory=False)
    df_num = df_raw.apply(pd.to_numeric, errors='coerce')

    # Keep only rows where more than 50% of values survived numeric conversion.
    # A metadata row like "Block with fractured surface" → almost 100% NaN after coerce.
    # A spectral row → almost 0% NaN after coerce.
    valid_mask = df_num.notna().sum(axis=1) > (df_raw.shape[1] * 0.5)
    vals = df_num[valid_mask].values.astype(np.float32)

    if vals.size == 0:
        raise ValueError("No valid numeric spectral data found in the uploaded file.")

    # ── Step 2: Wavelength Assassin ───────────────────────────────────────────
    # Reflectance values are always between 0.0 and 1.0 (or near it after normalization).
    # The wavelength axis holds values like 1000, 1001, ... 2499.
    # Physics fact: reflectance NEVER exceeds 900. So if vals[0, 0] >= 900,
    # that row IS the wavelength axis — kill it.
    if vals[0, 0] >= 900:
        vals = vals[1:]

    if vals.size == 0:
        raise ValueError("File contained only a wavelength axis. No spectral samples found.")

    # ── Step 3: Transpose detection (distance heuristic) ─────────────────────
    # The spectral axis must be exactly EXPECTED_BANDS long.
    # Whichever dimension is closer to 1500 is the wavelength axis.
    # If rows are closer → rows are bands, cols are samples → transpose needed.
    if abs(vals.shape[0] - EXPECTED_BANDS) < abs(vals.shape[1] - EXPECTED_BANDS):
        vals = vals.T  # (N_bands, N_samples) → (N_samples, N_bands)

    # ── Step 4: Crop to exactly 1500 bands ───────────────────────────────────
    if vals.shape[1] > EXPECTED_BANDS:
        vals = vals[:, :EXPECTED_BANDS]
    elif vals.shape[1] < EXPECTED_BANDS:
        raise ValueError(
            f"Only {vals.shape[1]} spectral bands found. Expected {EXPECTED_BANDS}. "
            "File may be from a different instrument or is corrupted."
        )

    # Remove flat-line samples (dead sensor rows)
    row_stds = vals.std(axis=1)
    vals = vals[row_stds > 1e-5]

    if vals.shape[0] == 0:
        raise ValueError("All spectral samples are flat-line. Sensor output may be invalid.")

    # ── Step 5: SNV normalization ─────────────────────────────────────────────
    s0 = _snv(vals)

    # ── Step 6: 1st Savitzky-Golay Derivative ────────────────────────────────
    s1 = _savgol_deriv(s0, order=1)

    # ── Step 7: 2nd Savitzky-Golay Derivative ────────────────────────────────
    s2 = _savgol_deriv(s0, order=2)

    # ── Step 8: Stack and tensorize → (N, 3, 1500) ───────────────────────────
    stacked = np.stack([s0, s1, s2], axis=1).astype(np.float32)
    tensor  = torch.FloatTensor(stacked)

    # Build the wavelength x-axis for frontend plotting
    wavelengths = list(np.linspace(1000, 2499, EXPECTED_BANDS).round(2))

    # Return sample 0's SNV and 2nd derivative channels for the spectral plot
    snv_plot = s0[0].tolist()
    d2_plot  = s2[0].tolist()

    return tensor, snv_plot, d2_plot, wavelengths
