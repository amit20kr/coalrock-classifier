"""
frontend/app.py — Spectra-Sense Dashboard
Streamlit UI for the PCViT coal-gangue classifier.

Fixes applied over Gemini's 85% correct version:
  1. Deleted `bar_color` dead code (computed but st.progress() ignores it)
  2. Stacked image ABOVE geochemistry instead of nested columns (~16% screen)
  3. st.metric() cards for geochemistry (professional, not bullet list)
  4. Backend XRF_PASSPORT is now nested {Origin, Geochemistry} — frontend matches
"""
import os
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spectra-Sense | Coal-Gangue Classifier",
    page_icon="⛏️",
    layout="wide",
)

# st.secrets for production on Streamlit Cloud.
# os.getenv fallback for local development: export BACKEND_URL=http://localhost:8000
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except (KeyError, FileNotFoundError):
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

IMAGES_DIR = "assets/class_images"


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("⛏️ Spectra-Sense")
st.caption("Real-time Coal-Gangue Classification via NIR Spectroscopy | PCViT | IIT (BHU)")

st.markdown("---")

# Cold start warning for Render free tier
st.info(
    "ℹ️ **Cold start notice:** The backend is hosted on Render's free tier. "
    "If it hasn't been used in 15 minutes, the first request takes 30–60 seconds to wake up. "
    "Subsequent requests are fast (<20ms).",
    icon="🕐",
)

# ──────────────────────────────────────────────────────────────────────────────
# FILE UPLOADER
# ──────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a spectrometer CSV file",
    type=["csv"],
    help="Raw CSV from the NIR spectrometer. Column-major or row-major format accepted.",
)

if uploaded_file is not None:
    with st.spinner("Running inference..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=90,  # 90s covers worst-case cold start
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the backend. "
                "If running locally, make sure `uvicorn main:app` is running on port 8000. "
                "If deployed, check the Render service logs."
            )
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out (>90s). The backend may be overloaded. Try again.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"Backend error {response.status_code}: {response.json().get('detail', str(e))}")
            st.stop()

    # ── VERDICT BANNER ────────────────────────────────────────────────────────
    verdict    = data.get("verdict", "UNKNOWN")
    is_gangue  = data.get("is_gangue", False)
    pred_class = data.get("predicted_class", "Unknown")
    confidence = data.get("confidence", 0.0)

    if is_gangue:
        st.error(f"### 🚫 {verdict}")
        st.markdown(f"**Detected Material:** `{pred_class}` — **WASTE ROCK. Conveyor flagged.**")
    else:
        st.success(f"### ✅ {verdict}")
        st.markdown(f"**Detected Material:** `{pred_class}` — **Fuel-grade coal. Accept.**")

    st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")
    st.markdown("---")

    # ── MAIN LAYOUT: [spectral plot  |  digital passport] ────────────────────
    col_plot, col_info = st.columns([3, 2], gap="large")

    # ── LEFT: SPECTRAL PLOT ───────────────────────────────────────────────────
    with col_plot:
        st.markdown("### NIR Spectral Analysis")

        spectral = data.get("spectral_channels", {})
        wavelengths = spectral.get("wavelengths", [])
        snv         = spectral.get("snv", [])
        d2          = spectral.get("second_derivative", [])

        if wavelengths and snv:
            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")

            wl = np.array(wavelengths)

            # SNV channel — the normalised shape
            ax.plot(wl, snv, color="#3498db", linewidth=1.2,
                    label="SNV (normalised)", zorder=3)

            # 2nd derivative — the Al-OH fingerprint amplifier
            if d2:
                ax_twin = ax.twinx()
                ax_twin.plot(wl, d2, color="#e74c3c", linewidth=0.8, alpha=0.7,
                             label="2nd Derivative")
                ax_twin.set_ylabel("2nd Derivative", color="#e74c3c", fontsize=9)
                ax_twin.tick_params(axis='y', colors='#e74c3c')
                ax_twin.set_facecolor("#0e1117")

            # Water bands at 1400nm and 1900nm
            for wband in [1400, 1900]:
                ax.axvline(x=wband, color="#95a5a6", linestyle="--",
                           linewidth=1.0, alpha=0.6)
                ax.text(wband + 8, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.1,
                        f"{wband}nm\n(H₂O)", color="#95a5a6",
                        fontsize=7, va="bottom")

            # Al-OH zone — the Clay Trap
            ax.axvspan(2180, 2220, color="#f1c40f", alpha=0.15, zorder=1)
            ax.text(2190, 0.05, "Al-OH\n(Clay Trap)", color="#f1c40f",
                    fontsize=8, ha="center")

            ax.set_xlabel("Wavelength (nm)", color="#ecf0f1")
            ax.set_ylabel("SNV Reflectance", color="#3498db")
            ax.set_title("NIR Spectrum — 1000 to 2500 nm", color="#ecf0f1", pad=10)
            ax.tick_params(colors="#ecf0f1")
            ax.spines[:].set_color("#2c3e50")

            # Legend
            legend_patches = [
                mpatches.Patch(color="#3498db", label="SNV channel"),
                mpatches.Patch(color="#e74c3c", label="2nd Derivative"),
                mpatches.Patch(color="#f1c40f", alpha=0.5, label="Al-OH zone (2180–2220nm)"),
            ]
            ax.legend(handles=legend_patches, loc="upper right",
                      facecolor="#1a1a2e", edgecolor="#2c3e50",
                      labelcolor="#ecf0f1", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No spectral data returned from backend.")

    # ── RIGHT: DIGITAL PASSPORT ───────────────────────────────────────────────
    with col_info:
        st.markdown("### 🔬 Digital Passport")

        # Mineral image — full width inside col_info (not nested columns)
        image_path = os.path.join(IMAGES_DIR, f"{pred_class}.jpg")
        if os.path.exists(image_path):
            st.image(image_path, caption=pred_class, use_container_width=True)
        else:
            st.info("No visual reference available for this class.")

        # Origin and confidence metadata
        passport_data = data.get("digital_passport", {})
        origin = passport_data.get("Origin", "Standard Mining Library")
        st.markdown(f"**Origin:** {origin}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
        st.markdown(f"**Verdict:** `{verdict}`")

        st.markdown("---")

        # Geochemistry as st.metric() cards — professional, not a bullet list
        geochemistry = passport_data.get("Geochemistry", {})
        if geochemistry:
            st.markdown("**Geochemistry Profile:**")
            # 2 columns for the metric cards to use space efficiently
            geo_items = list(geochemistry.items())
            n         = len(geo_items)
            mid       = (n + 1) // 2
            gc1, gc2  = st.columns(2)

            for i, (element, pct) in enumerate(geo_items):
                target_col = gc1 if i < mid else gc2
                with target_col:
                    st.metric(label=element, value=pct)
        else:
            st.markdown("*Geochemistry data not available.*")

        st.markdown("---")

        # Top-5 confidence bars
        st.markdown("**Confidence Distribution (Top 5):**")
        top_5 = data.get("top_5", [])
        for item in top_5:
            cls_name = item["class"]
            prob     = item["probability"]
            # bar_color computed here would be dead code — st.progress() has no color param.
            # Use the label text to convey fuel vs gangue instead.
            label_text = f"{cls_name} ({prob * 100:.2f}%)"
            st.progress(prob, text=label_text)

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>PCViT | >95% accuracy on 24 lithologies | 0.00% Fuel-Waste boundary violations | "
    "P99 latency: 26ms | Amit Kumar, B.Tech Mining, IIT (BHU)</small>",
    unsafe_allow_html=True,
)
