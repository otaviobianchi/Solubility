# Numerical Optimization vs ML — Solubility Space (Streamlit)

⚠️ Research use only. This app is intended for methodological comparison and visualization and is **not** a predictive device.

## What it does
- Upload Excel dataset
- Select columns: δd, δp, δh, solubility (0, 0.5, 1), optional group
- Numerical Optimization sphere (RED=1) using probabilistic RED→p mapping
- Calibrated ML models (XGBoost, RandomForest, SVM-RBF)
- Cross-validation: LOGO (if group) or LOO
- 3D plots: (A) numerical optimization sphere, (B) ML shell-sphere, (C) overlay
- Downloads: results Excel + HTML figure ZIP

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
