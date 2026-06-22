# SPECTRA-SENSE: SYSTEM ARCHITECTURE & STRICT LOGIC RULES
**Project:** Real-time Coal vs. Gangue classification via NIR Spectroscopy (1000 to 2500nm).
**Objective:** Halt the conveyor belt if Al-OH chemical bonds (indicating waste rock) are detected.
You are acting as a Senior Machine Learning Engineer assisting with a Hackathon deployment. 

## CORE ARCHITECTURE RULES (DO NOT MODIFY WITHOUT ASKING)
1.  **The Physics:** We classify coal vs gangue using NIR spectroscopy (1000–2500nm). Gangue is detected via Al-OH bonds at exactly 2200nm. The model is a custom 900k-parameter PCViT (Pyramid Convolutional Vision Transformer).
2.  **The Safety Net (main.py):** We NEVER trust the batch mean blindly. If a single scan in a batch has >0.85 probability for a gangue class, `is_gangue = True` is triggered to stop the conveyor.
3.  **The Data Layer (pipeline.py):** Reflectance values never exceed 900. If `vals[0,0] >= 900`, that row is the wavelength axis and MUST be dropped.
4.  **Hardware Restrictions:** The Render backend runs on a CPU. `torch.load()` MUST include `map_location='cpu'`. 
5.  **State Dictionary:** The PCViT attributes must strictly remain `stem`, `cls`, `pos`, `tf`, and `head`. Renaming them will break the loaded `pcvit_expert.pth` weights.
## 1. The Deployment Architecture
* **Backend:** FastAPI running PyTorch (CPU only). Model weights are hosted on HuggingFace and downloaded at boot via `download_model.py`.
* **Frontend:** Streamlit dashboard displaying 2nd Derivative plots and a 24-class Geochemistry Digital Passport.
* **Model:** PCViT (Pre-Convolutional Vision Transformer). 903,576 parameters.

## 2. Immutable Engineering Rules (DO NOT MODIFY)
If asked to edit `pipeline.py` or `main.py`, you must respect these hardcoded heuristics:
* **The Wavelength Assassin:** Raw CSVs often include a wavelength axis. If `vals[0, 0] >= 900`, that row is the X-axis and must be dropped.
* **The Orientation Heuristic:** If `abs(rows - 1500) < abs(cols - 1500)`, the matrix must be transposed.
* **The Gangue Override (main.py):** We do not trust mean-pooling for safety. If *any* single scan in a batch hits a gangue class probability of > 0.85, the `is_gangue` boolean must trip to True, overriding the mean prediction.

## 3. PyTorch State Dictionary Constraints
The `pcvit_expert.pth` model was trained with specific layer names. `model.py` must strictly retain the keys: `stem`, `cls`, `pos`, `tf`, and `head`. Modifying these keys will trigger a dictionary mapping crash during deployment.
