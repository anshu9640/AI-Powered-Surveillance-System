# AI-Powered Surveillance (UCSD) — Hackathon Solution

A small, practical pipeline that learns **normal crowd behavior** from the UCSD dataset and flags:
- **Loitering**
- **Unusual motion**
- **Abandoned objects**

It blends YOLOv8 detection, a lightweight centroid tracker, simple rules, and a tiny 3D autoencoder. A Streamlit dashboard explains **why** each alert fired (Semantic / Motion / Loiter / Abandon).

---

## What this repo contains

- `train/` – train the 3D autoencoder on normal clips  
- `inference/` – run the detector + rules + motion model and save alerts  
- `models/` – model code (weights are not committed)  
- `dashboard/` – Streamlit app to browse/inspect alerts  
- `fusion.py`, `utils.py` – tracking, rules, helpers  
- `requirements.txt` – Python deps

> **Note:** The **UCSD Anomaly Detection Dataset** is **not** included. Please download it yourself and keep it outside the repo folder.

---

## Quickstart (Windows, VS Code)

```powershell
# 1) create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) train the motion model (point to your UCSD train folder)
python -m train.train_behavior --train_dir "C:\path\to\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Train" --epochs 10

# 4) run inference on a test sequence (folder of frames or a video)
python -m inference.run_stream --video "C:\path\to\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test001" --out out_peds1 `
  --det_conf 0.50 `
  --abandon_labels backpack,handbag,suitcase `
  --abandon_seconds 5

# 5) open the dashboard and pick the output folder (e.g., out_peds1)
streamlit run .\dashboard\app.py
