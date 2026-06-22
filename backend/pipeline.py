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
# Frontend calls: passport_data.get('Origin', ...) and passport_data.get('Geochemistry', {})
# ──────────────────────────────────────────────────────────────────────────────
XRF_PASSPORT = {
    # ── FUEL CLASSES ──────────────────────────────────────────────────────────
    "Anthracite coal grade 1": {
        "Origin": "Jincheng Coal Mine, Shanxi Province",
        "Geochemistry": {
            "Fixed Carbon": "88.2%", "Ash": "8.5%",
            "Volatile Matter": "3.3%", "Moisture": "0.5%"
        }
    },
    "Anthracite coal grade 2": {
        "Origin": "Yangquan Coal Mine, Shanxi Province",
        "Geochemistry": {
            "Fixed Carbon": "84.7%", "Ash": "10.2%",
            "Volatile Matter": "5.1%", "Moisture": "0.6%"
        }
    },
    "Lean coal": {
        "Origin": "Fengfeng Coal Mine, Hebei Province",
        "Geochemistry": {
            "Fixed Carbon": "78.3%", "Ash": "12.4%",
            "Volatile Matter": "9.3%", "Moisture": "0.8%"
        }
    },
    "Lean-thin coal": {
        "Origin": "Huainan Coal Mine, Anhui Province",
        "Geochemistry": {
            "Fixed Carbon": "74.1%", "Ash": "14.6%",
            "Volatile Matter": "11.3%", "Moisture": "1.1%"
        }
    },
    "Thin coal": {
        "Origin": "Pingdingshan Coal Mine, Henan Province",
        "Geochemistry": {
            "Fixed Carbon": "70.5%", "Ash": "15.2%",
            "Volatile Matter": "14.3%", "Moisture": "1.2%"
        }
    },
    "Coking coal": {
        "Origin": "Liuzhi Coal Mine, Guizhou Province",
        "Geochemistry": {
            "Fixed Carbon": "68.7%", "Ash": "9.8%",
            "Volatile Matter": "21.5%", "Moisture": "0.9%"
        }
    },
    "0.3 coking coal": {
        "Origin": "Xishan Coal Mine, Shanxi Province",
        "Geochemistry": {
            "Fixed Carbon": "66.2%", "Ash": "11.3%",
            "Volatile Matter": "22.5%", "Moisture": "1.0%"
        }
    },
    "Fat coal": {
        "Origin": "Kailuan Coal Mine, Hebei Province",
        "Geochemistry": {
            "Fixed Carbon": "60.4%", "Ash": "13.7%",
            "Volatile Matter": "25.9%", "Moisture": "1.4%"
        }
    },
    "Gas-fat coal": {
        "Origin": "Shenmu Coal Mine, Shaanxi Province",
        "Geochemistry": {
            "Fixed Carbon": "55.1%", "Ash": "10.4%",
            "Volatile Matter": "34.5%", "Moisture": "1.8%"
        }
    },
    "Gas coal": {
        "Origin": "Hanjing Coal Mine, Shaanxi Province",
        "Geochemistry": {
            "Fixed Carbon": "48.3%", "Ash": "12.1%",
            "Volatile Matter": "39.6%", "Moisture": "2.1%"
        }
    },
    "Lignite coal grade 1": {
        "Origin": "Baotou Coal Mine, Inner Mongolia",
        "Geochemistry": {
            "Fixed Carbon": "38.5%", "Ash": "18.3%",
            "Volatile Matter": "41.2%", "Moisture": "8.5%"
        }
    },
    "Lignite coal grade 2": {
        "Origin": "Yimin Coal Mine, Inner Mongolia",
        "Geochemistry": {
            "Fixed Carbon": "32.1%", "Ash": "20.7%",
            "Volatile Matter": "47.2%", "Moisture": "12.4%"
        }
    },
    # ── GANGUE CLASSES ────────────────────────────────────────────────────────
    "Black shale 1": {
        "Origin": "Xuzhou Coal Mine, Jiangsu Province",
        "Geochemistry": {
            "SiO2": "48.30%", "Al2O3": "18.20%",
            "Fe2O3": "9.80%", "Carbon": "12.5%"
        }
    },
    "Black shale 2": {
        "Origin": "Datong Coal Mine, Shanxi Province",
        "Geochemistry": {
            "SiO2": "52.10%", "Al2O3": "16.40%",
            "Fe2O3": "8.60%", "Carbon": "8.2%"
        }
    },
    "Carbonaceous shale": {
        "Origin": "Dongfeng Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "45.20%", "Al2O3": "28.10%",
            "Fe2O3": "4.30%", "Carbon": "15.5%"
        }
    },
    "Arenaceous shale 1": {
        "Origin": "Xinglongzhuang Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "64.17%", "Al2O3": "14.96%",
            "Fe2O3": "2.93%", "K2O": "2.71%"
        }
    },
    "Arenaceous shale 2": {
        "Origin": "Baodian Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "67.42%", "Al2O3": "13.21%",
            "Fe2O3": "2.54%", "K2O": "3.12%"
        }
    },
    "Clay": {
        "Origin": "Yanzhou Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "42.30%", "Al2O3": "35.80%",
            "Fe2O3": "1.20%", "TiO2": "1.85%"
        }
    },
    "Siltstone 1": {
        "Origin": "Huaibei Coal Mine, Anhui Province",
        "Geochemistry": {
            "SiO2": "72.40%", "Al2O3": "11.30%",
            "Fe2O3": "3.10%", "K2O": "3.80%"
        }
    },
    "Siltstone 2": {
        "Origin": "Huainan Coal Mine, Anhui Province",
        "Geochemistry": {
            "SiO2": "69.80%", "Al2O3": "12.60%",
            "Fe2O3": "3.40%", "CaO": "1.20%"
        }
    },
    "Medium-grained sandstone": {
        "Origin": "Jining Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "78.50%", "Al2O3": "8.30%",
            "Fe2O3": "2.10%", "K2O": "4.20%"
        }
    },
    "Fine-grained sandstone": {
        "Origin": "Zibo Coal Mine, Shandong Province",
        "Geochemistry": {
            "SiO2": "76.20%", "Al2O3": "9.40%",
            "Fe2O3": "2.30%", "K2O": "3.90%"
        }
    },
    "Argillaceous limestone 1": {
        "Origin": "Feicheng Coal Mine, Shandong Province",
        "Geochemistry": {
            "CaO": "38.20%", "SiO2": "18.40%",
            "Al2O3": "12.10%", "MgO": "3.20%"
        }
    },
    "Argillaceous limestone 2": {
        "Origin": "Zaozhuang Coal Mine, Shandong Province",
        "Geochemistry": {
            "CaO": "42.10%", "SiO2": "14.30%",
            "Al2O3": "10.80%", "MgO": "4.10%"
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
