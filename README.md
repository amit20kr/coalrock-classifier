# Spectra-Sense — PCViT Coal-Gangue Classifier

Real-time classification of coal vs. waste rock (gangue) using NIR spectroscopy (1000–2500nm) and a Pyramid Convolutional Vision Transformer.

**Team:** Amit Kumar | Roll 23155012 | B.Tech Mining Engineering, IIT (BHU)  
**Supervisor:** Dr. Aishwarya Mishra  
**Hackathon:** Bharat Academix CodeQuest (Unstop) — Round 2

---

## The Physics

Coal is amorphous carbon. Its NIR spectrum is flat.  
Gangue (shale, clay) contains Kaolinite and Illite with Al-OH bonds that absorb light at exactly ~2200nm — creating an unmistakable spectral dip. The model is engineered to hunt this chemical signature, not pixel colour.

---

## Quick Start (Local)

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python download_model.py      # pulls pcvit_expert.pth from HuggingFace
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Check it's alive: http://localhost:8000/health

### 2. Frontend

```bash
cd frontend
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

Then open http://localhost:8501 and upload any raw NIR spectrometer `.csv` file.

---

## Demo — The Killer Move

Rename `arenaceous_shale_1.csv` to `PURE_ANTHRACITE_A_GRADE.csv`.  
Upload it. The model outputs **REJECT**.

The model ignores filenames. It reads the Al-OH chemical bond at 2200nm.  
The file says "anthracite". The chemistry says "rock".

---

## Architecture

```
(Batch, 3, 1500)       3-channel NIR tensor [SNV | 1st Deriv | 2nd Deriv]
       ↓
[CNN Stem]             3 conv layers, stride-2 → compresses 1500 → 375 tokens
       ↓
[CLS Token + Pos Emb]  Prepend learnable summary vector
       ↓
[Transformer × 4]      4 heads, 4 layers — global context across all 375 tokens
       ↓
[CLS → Linear]         24-class logit vector
```

Model size: 903,576 parameters | 3.6 MB | <20ms CPU inference

---

## Key Results

| Metric | Value |
|---|---|
| 24-class accuracy | >95% |
| Fuel-Waste boundary violations | 0.00% |
| P99 CPU inference latency | 26ms |
| SNR robustness (noise injection) | Degrades gracefully to 76% at 26dB |

---

## Deployment (Render)

Start command:
```
python download_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set environment variable `BACKEND_URL` in Streamlit Cloud secrets.  
Free tier cold starts take 30–60s. The UI shows a notice.
