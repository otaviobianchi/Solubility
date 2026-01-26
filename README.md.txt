# Numerical Optimization (HSP) vs ML — Streamlit App

Research use only. This app reproduces the same paper pipeline:
- Numerical Optimization (multi-DF + multi-optimizer) to fit center (δd, δp, δh) and radius R0
- Probabilistic mapping RED -> p
- ML models with probability calibration
- CV: LOGO (if group column exists) or LOO
- 3D Plotly: Numerical Optimization sphere (RED=1) + ML shell-spheres + overlay
- Full Excel export

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
