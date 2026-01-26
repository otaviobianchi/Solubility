import io
import zipfile
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import plotly.graph_objects as go


# =========================
# PAGE
# =========================
st.set_page_config(
    page_title="Numerical Optimization vs ML — Solubility Space",
    layout="wide"
)

st.title("Numerical Optimization vs Machine Learning in Solubility Space")
st.markdown("""
⚠️ **Research use only.**  
This app is intended for methodological comparison and visualization.  
It is **not** a predictive device and must **not** be used as a standalone decision tool.
""")

warnings.filterwarnings("ignore")
np.random.seed(42)


# =========================
# SETTINGS (defaults)
# =========================
UNIT_DEFAULT = "MPa\u00b9\u2044\u00b2"

# Probabilistic mapping (RED → p)
K_PROB_DEFAULT = 6.0
REG_R0_DEFAULT = 0.05

# ML shell sphere settings
ISO_LEVELS_DEFAULT = [0.50, 0.80]
GRID_PAD_DEFAULT = 1.5
GRID_N_DEFAULT = 28
SHELL_DELTA_DEFAULT = 0.05

# Calibration
CALIBRATION_MIN_N_ISO_DEFAULT = 60

# Bounds for numerical optimization: (δd, δp, δh, R0)
BOUNDS_DEFAULT = [(10, 25), (0, 25), (0, 25), (2, 25)]


# =========================
# Sidebar
# =========================
st.sidebar.header("Controls")

MODE = st.sidebar.radio(
    "Mode",
    ["Paper mode (fast & reproducible)", "Exploratory mode (slower)"],
    help="Paper mode uses faster optimization settings for Streamlit Cloud stability."
)

UNIT = st.sidebar.text_input("Units label", value=UNIT_DEFAULT)

K_PROB = st.sidebar.number_input("K (RED→p logistic)", min_value=0.5, max_value=50.0, value=float(K_PROB_DEFAULT), step=0.5)
REG_R0 = st.sidebar.number_input("R0 regularization (λ)", min_value=0.0, max_value=1.0, value=float(REG_R0_DEFAULT), step=0.01)

ISO_LEVELS = st.sidebar.multiselect(
    "ML iso-levels for shell spheres",
    options=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
    default=ISO_LEVELS_DEFAULT
)
GRID_PAD = st.sidebar.number_input("Grid padding", min_value=0.0, max_value=10.0, value=float(GRID_PAD_DEFAULT), step=0.1)
GRID_N = st.sidebar.number_input("Grid resolution (N)", min_value=12, max_value=60, value=int(GRID_N_DEFAULT), step=1)
SHELL_DELTA = st.sidebar.number_input("Shell delta (Δ)", min_value=0.01, max_value=0.30, value=float(SHELL_DELTA_DEFAULT), step=0.01)

CALIBRATION_MIN_N_ISO = st.sidebar.number_input("Min N for isotonic", min_value=10, max_value=500, value=int(CALIBRATION_MIN_N_ISO_DEFAULT), step=5)

if MODE.startswith("Paper"):
    OPT_MAXITER_DE = 250
    OPT_MAXITER_LOCAL = 1200
else:
    OPT_MAXITER_DE = 800
    OPT_MAXITER_LOCAL = 3000

st.sidebar.divider()
run_button = st.sidebar.button("Run analysis", type="primary")


# =========================
# Helpers — math & models
# =========================
def hansen_distance(d_d, d_p, d_h, dp, pp, hp):
    return np.sqrt(4*(d_d - dp)**2 + (d_p - pp)**2 + (d_h - hp)**2)

def prob_from_red(RED, k=6.0):
    RED = np.asarray(RED, float)
    z = k*(1.0 - RED)
    z = np.clip(z, -60, 60)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 1e-12, 1.0-1e-12)

def red_values(df_local, params):
    dp, pp, hp, R0 = map(float, params)
    Ra = hansen_distance(df_local['delta_d'].values,
                         df_local['delta_p'].values,
                         df_local['delta_h'].values,
                         dp, pp, hp)
    return Ra / max(R0, 1e-6)

def df_logloss(df_local, x, weights=None, k=6.0, reg_ro=0.05, ro_ref=None):
    d_d = df_local['delta_d'].values
    d_p = df_local['delta_p'].values
    d_h = df_local['delta_h'].values
    y = df_local['solubility'].values.astype(int)

    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
    RED = Ra / R0
    p = prob_from_red(RED, k=k)

    ll = -(y*np.log(p) + (1-y)*np.log(1-p))
    if weights is None:
        loss = float(np.mean(ll))
    else:
        ww = np.asarray(weights, float)
        loss = float(np.sum(ww*ll) / np.sum(ww))

    if reg_ro and reg_ro > 0:
        if ro_ref is None:
            ro_ref = float(np.median(Ra))
        loss = loss + float(reg_ro)*((R0 - ro_ref)/max(ro_ref, 1e-6))**2
    return float(loss)

def fit_numerical_optimization(df_local, w_local, bounds):
    """
    Fast + robust: Differential Evolution (global) + local refinement (Powell).
    Objective: weighted logloss on probabilistic RED mapping (+ R0 regularization).
    """
    d_d = df_local['delta_d'].values
    d_p = df_local['delta_p'].values
    d_h = df_local['delta_h'].values

    Ra_ref = hansen_distance(d_d, d_p, d_h, np.median(d_d), np.median(d_p), np.median(d_h))
    ro_ref = float(np.median(Ra_ref))

    def obj(x):
        return df_logloss(df_local, x, weights=w_local, k=K_PROB, reg_ro=REG_R0, ro_ref=ro_ref)

    # Global
    res_de = differential_evolution(
        obj, bounds, maxiter=OPT_MAXITER_DE, polish=False, seed=42, disp=False
    )

    # Local refine
    res_local = minimize(
        obj, res_de.x, method="Powell",
        options=dict(maxiter=OPT_MAXITER_LOCAL)
    )

    x_best = res_local.x if res_local.success else res_de.x
    dp, pp, hp, R0 = map(float, x_best)
    R0 = max(R0, 1e-6)
    return np.array([dp, pp, hp, R0], float), float(obj([dp, pp, hp, R0]))

def make_base_models(random_state=42, base_score=None):
    models = {}
    models["XGBoost"] = XGBClassifier(
        n_estimators=450, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=random_state,
        eval_metric='logloss',
        base_score=float(base_score) if base_score is not None else 0.5
    )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=700, random_state=random_state
    )
    models["SVM-RBF"] = SVC(
        C=2.0, gamma="scale", probability=True, random_state=random_state
    )
    return models

def calibrate_model(base_estimator, X_tr, y_tr, w_tr):
    """
    isotonic if N >= threshold; else sigmoid.
    Tries sample_weight; falls back if not supported.
    """
    method = "isotonic" if len(y_tr) >= CALIBRATION_MIN_N_ISO else "sigmoid"
    try:
        cal = CalibratedClassifierCV(base_estimator, method=method, cv=3)
        try:
            cal.fit(X_tr, y_tr, sample_weight=w_tr)
        except TypeError:
            base_estimator.fit(X_tr, y_tr, sample_weight=w_tr)
            cal = CalibratedClassifierCV(base_estimator, method=method, cv="prefit")
            cal.fit(X_tr, y_tr)
        return cal
    except Exception:
        try:
            base_estimator.fit(X_tr, y_tr, sample_weight=w_tr)
        except Exception:
            base_estimator.fit(X_tr, y_tr)
        return base_estimator


# =========================
# Helpers — metrics
# =========================
def weighted_confusion(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sw = np.ones_like(y_true, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    TN = sw[(y_true==0) & (y_pred==0)].sum()
    FP = sw[(y_true==0) & (y_pred==1)].sum()
    FN = sw[(y_true==1) & (y_pred==0)].sum()
    TP = sw[(y_true==1) & (y_pred==1)].sum()
    return TN, FP, FN, TP

def safe_div(a, b):
    return float(a)/float(b) if float(b) != 0 else np.nan

def compute_metrics_row(model_name, y_true, p, thr=0.5, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p).astype(float), 1e-12, 1-1e-12)
    y_pred = (p >= float(thr)).astype(int)

    TN, FP, FN, TP = weighted_confusion(y_true, y_pred, sample_weight=sample_weight)
    precision = safe_div(TP, TP+FP)
    recall = safe_div(TP, TP+FN)
    specificity = safe_div(TN, TN+FP)
    f1 = safe_div(2*precision*recall, precision+recall)
    bal_acc = np.nan if (recall!=recall or specificity!=specificity) else 0.5*(recall+specificity)

    denom = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc = safe_div((TP*TN - FP*FN), denom)

    out = dict(
        Model=model_name,
        thr=float(thr),
        Accuracy=safe_div((TP+TN), (TP+TN+FP+FN)),
        BalancedAcc=bal_acc,
        Precision=precision,
        Recall=recall,
        Specificity=specificity,
        F1=f1,
        MCC=mcc,
        TN=float(TN), FP=float(FP), FN=float(FN), TP=float(TP),
        AUC_ROC=np.nan,
        AUC_PR=np.nan
    )
    if len(np.unique(y_true)) == 2:
        out["AUC_ROC"] = float(roc_auc_score(y_true, p))
        out["AUC_PR"] = float(average_precision_score(y_true, p))
    return out


# =========================
# Helpers — 3D shell-sphere
# =========================
def point_colors_from_yraw(y_raw):
    yraw = np.asarray(y_raw, float)
    colors = np.where(yraw == 1.0, "blue", np.where(yraw == 0.5, "orange", "red"))
    labels = np.where(yraw == 1.0, "soluble", np.where(yraw == 0.5, "partial", "insoluble"))
    return colors, labels

def fit_sphere_least_squares(P):
    P = np.asarray(P, float)
    if P.shape[0] < 10:
        raise ValueError("Too few points to fit a sphere (>=10).")
    x, y_, z = P[:,0], P[:,1], P[:,2]
    A = np.c_[2*x, 2*y_, 2*z, np.ones_like(x)]
    b = x**2 + y_**2 + z**2
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz, c0 = beta
    r2 = cx**2 + cy**2 + cz**2 + c0
    r2 = max(float(r2), 1e-12)
    r = float(np.sqrt(r2))
    return (float(cx), float(cy), float(cz)), r

def sphere_mesh(center, r, nu=80, nv=40):
    cx, cy, cz = center
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    xs = r*np.cos(uu)*np.sin(vv) + cx
    ys = r*np.sin(uu)*np.sin(vv) + cy
    zs = r*np.cos(vv) + cz
    return xs, ys, zs

def ml_shell_sphere_from_grid(df_local, model, iso=0.5, pad=1.5, n=28, shell_delta=0.05, min_points=120):
    mn = df_local[["delta_d","delta_p","delta_h"]].min().values.astype(float) - float(pad)
    mx = df_local[["delta_d","delta_p","delta_h"]].max().values.astype(float) + float(pad)

    xd = np.linspace(mn[0], mx[0], int(n))
    xp = np.linspace(mn[1], mx[1], int(n))
    xh = np.linspace(mn[2], mx[2], int(n))
    Xd, Xp, Xh = np.meshgrid(xd, xp, xh, indexing="ij")
    pts = np.c_[Xd.ravel(), Xp.ravel(), Xh.ravel()]

    p = np.clip(model.predict_proba(pts)[:, 1], 1e-12, 1-1e-12)

    iso_use = float(iso)
    delta = float(shell_delta)

    for _ in range(6):
        mask = (p >= iso_use) & (p <= min(1.0, iso_use + delta))
        Psel = pts[mask]
        if Psel.shape[0] >= min_points:
            center, r = fit_sphere_least_squares(Psel)
            return center, r, int(Psel.shape[0]), iso_use, delta

        delta = min(0.20, delta + 0.03)
        iso_use = max(0.05, iso_use - 0.03)

    mask = (p >= float(iso))
    Psel = pts[mask]
    if Psel.shape[0] < 10:
        raise ValueError("Not enough grid points. Increase GRID_N/GRID_PAD or reduce iso.")
    center, r = fit_sphere_least_squares(Psel)
    return center, r, int(Psel.shape[0]), float(iso), -1.0


# =========================
# Plotly style (no opacity duplication)
# =========================
def pretty_scene(unit=UNIT_DEFAULT):
    return dict(
        xaxis=dict(title=f"δd ({unit})", showbackground=True, backgroundcolor="rgba(245,245,245,1)",
                   gridcolor="rgba(180,180,180,0.35)", zerolinecolor="rgba(120,120,120,0.25)"),
        yaxis=dict(title=f"δp ({unit})", showbackground=True, backgroundcolor="rgba(245,245,245,1)",
                   gridcolor="rgba(180,180,180,0.35)", zerolinecolor="rgba(120,120,120,0.25)"),
        zaxis=dict(title=f"δh ({unit})", showbackground=True, backgroundcolor="rgba(245,245,245,1)",
                   gridcolor="rgba(180,180,180,0.35)", zerolinecolor="rgba(120,120,120,0.25)"),
        aspectmode="data",
        camera=dict(eye=dict(x=1.35, y=1.25, z=0.95))
    )

def surface_style(base_rgb):
    return dict(
        showscale=False,
        colorscale=[[0.0, base_rgb], [1.0, base_rgb]],
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.55, fresnel=0.08),
        lightposition=dict(x=100, y=200, z=100)
    )

def points_style(colors):
    return dict(
        size=5,
        color=colors,
        opacity=0.95,
        line=dict(width=0.6, color="rgba(0,0,0,0.35)")
    )


# =========================
# Upload + column mapping UI
# =========================
st.header("1) Upload dataset")

uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])
df_raw = None
sheet_name = None

if uploaded is not None:
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("Select sheet", xls.sheet_names)
    df_raw = xls.parse(sheet_name)
    st.success(f"Loaded sheet: {sheet_name}  |  rows={df_raw.shape[0]} cols={df_raw.shape[1]}")
    st.dataframe(df_raw.head(15), use_container_width=True)

if df_raw is None:
    st.info("Upload an Excel file to continue.")
    st.stop()

st.header("2) Select columns")

col_delta_d = st.selectbox("δd column", df_raw.columns)
col_delta_p = st.selectbox("δp column", df_raw.columns)
col_delta_h = st.selectbox("δh column", df_raw.columns)
col_solub  = st.selectbox("Solubility label column (0, 0.5, 1)", df_raw.columns)
col_group  = st.selectbox("Group column (optional, for LOGO)", ["(none)"] + list(df_raw.columns))

use_group = (col_group != "(none)")

st.caption("Labels: 0 = insoluble, 0.5 = partial, 1 = soluble. Partial is treated as positive with weight 0.5.")

# Build base table
df = df_raw[[col_delta_d, col_delta_p, col_delta_h, col_solub] + ([col_group] if use_group else [])].copy()
df.columns = ["delta_d", "delta_p", "delta_h", "solubility"] + (["group"] if use_group else [])

df[["delta_d","delta_p","delta_h","solubility"]] = df[["delta_d","delta_p","delta_h","solubility"]].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["delta_d","delta_p","delta_h","solubility"]).drop_duplicates().reset_index(drop=True)

y_raw = df["solubility"].astype(float).values
y = (y_raw >= 0.5).astype(int)
w = np.where(y_raw == 0.5, 0.5, 1.0).astype(float)
df["solubility"] = y

st.write(f"Samples: **{len(df)}** | Positives(bin): **{int(y.sum())}** | Negatives: **{int(len(df)-y.sum())}** | Partials(0.5): **{int((y_raw==0.5).sum())}**")
if use_group:
    groups = df["group"].astype(str).values
    st.write(f"CV scheme: **LOGO** | groups: **{len(np.unique(groups))}**")
else:
    groups = None
    st.write("CV scheme: **LOO** (no group selected)")


# =========================
# Methods (paper-ready)
# =========================
st.header("3) Methods (paper-ready text)")

cv_label = "LOGO" if use_group else "LOO"

with st.expander("Open methods text"):
    st.markdown(f"""
**Numerical Optimization (RED=1 sphere)**  
A solubility sphere was obtained via numerical optimization by minimizing a **weighted log-loss**
objective derived from the Hansen-like distance (Ra) and radius (R0). Distances were converted to
RED (= Ra/R0) and mapped to probabilities using a logistic transformation with **K = {K_PROB:.3g}**.

**Machine Learning (calibrated probabilities)**  
Supervised classifiers were trained using (δd, δp, δh). Predicted probabilities were calibrated
(isotonic if N ≥ {CALIBRATION_MIN_N_ISO}, otherwise sigmoid).  

**ML shell-sphere extraction**  
A geometric sphere was fitted to grid points lying in an isoprobability **shell**:
p ∈ [iso, iso+Δ], with Δ = {SHELL_DELTA:.3g}. If insufficient shell points are available, the method
relaxes Δ upward and iso downward.

**Validation**  
Cross-validation uses **{cv_label}**. Partial labels (0.5) are treated as positive (y=1) with weight 0.5.
""")


# =========================
# Run analysis
# =========================
if not run_button:
    st.info("Adjust settings if needed, then click **Run analysis** in the sidebar.")
    st.stop()

st.header("4) Results")

# Data for ML
X = df[["delta_d","delta_p","delta_h"]].values

# 4.1 Numerical optimization (global on full data)
with st.spinner("Running numerical optimization (global fit)…"):
    pars_best, best_obj = fit_numerical_optimization(df, w, BOUNDS_DEFAULT)

dp, pp, hp, R0 = map(float, pars_best)
st.subheader("4.1 Numerical Optimization (global)")
st.write(f"Center: δd={dp:.4f} {UNIT} | δp={pp:.4f} {UNIT} | δh={hp:.4f} {UNIT}")
st.write(f"Radius: R0={R0:.4f} {UNIT}")
st.write(f"Objective (weighted logloss + reg): {best_obj:.6f}")

RED_all = red_values(df, pars_best)
p_numopt_in = prob_from_red(RED_all, k=K_PROB)

# 4.2 ML (in-sample calibrated)
p0_in = float(np.clip(np.average(y, weights=w), 1e-12, 1-1e-12))
base_models = make_base_models(42, base_score=p0_in)

proba_ml_in: Dict[str, np.ndarray] = {}
fitted_ml_in = {}

with st.spinner("Training calibrated ML models (in-sample)…"):
    for name, est in base_models.items():
        try:
            fitted = calibrate_model(est, X, y, w)
            fitted_ml_in[name] = fitted
            if hasattr(fitted, "predict_proba"):
                proba_ml_in[name] = fitted.predict_proba(X)[:, 1]
        except Exception:
            pass

st.subheader("4.2 ML (in-sample calibrated)")
st.write(f"Trained models: **{', '.join(list(proba_ml_in.keys()))}**")

# choose best ML by AUC_PR in-sample (simple)
best_ml_name = None
best_ml_score = -np.inf
if len(np.unique(y)) == 2 and len(proba_ml_in) > 0:
    for name, p in proba_ml_in.items():
        score = average_precision_score(y, p)
        if score > best_ml_score:
            best_ml_score = score
            best_ml_name = name
else:
    best_ml_name = list(proba_ml_in.keys())[0] if len(proba_ml_in) else "XGBoost"

st.write(f"Best ML (in-sample by PR-AUC): **{best_ml_name}**")

# 4.3 Cross-validation: LOGO or LOO
if use_group:
    splitter = LeaveOneGroupOut()
    splits = list(splitter.split(X, y, groups=groups))
else:
    splitter = LeaveOneOut()
    splits = list(splitter.split(X, y))

p_numopt_cv = np.zeros(len(y), dtype=float)
p_ml_cv = {name: np.zeros(len(y), dtype=float) for name in base_models.keys()}

with st.spinner(f"Running {cv_label} cross-validation (this can take a while)…"):
    for (tr_idx, te_idx) in splits:
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)
        w_tr = w[tr_idx]
        y_tr = df_tr["solubility"].values.astype(int)

        # Numerical optimization inside fold (train only)
        try:
            pars_fold, _ = fit_numerical_optimization(df_tr, w_tr, BOUNDS_DEFAULT)
        except Exception:
            pars_fold = pars_best

        RED_te = red_values(df_te, pars_fold)
        p_numopt_cv[te_idx] = prob_from_red(RED_te, k=K_PROB)

        # ML inside fold (train only)
        X_tr = df_tr[["delta_d","delta_p","delta_h"]].values
        X_te = df_te[["delta_d","delta_p","delta_h"]].values

        if len(np.unique(y_tr)) < 2:
            prev = float(np.clip(np.average(y, weights=w), 1e-12, 1-1e-12))
            for name in base_models.keys():
                p_ml_cv[name][te_idx] = prev
            continue

        p0_fold = float(np.clip(np.average(y_tr, weights=w_tr), 1e-12, 1-1e-12))
        fold_models = make_base_models(42, base_score=p0_fold)

        for name, est in fold_models.items():
            try:
                fitted = calibrate_model(est, X_tr, y_tr, w_tr)
                if hasattr(fitted, "predict_proba"):
                    p_ml_cv[name][te_idx] = fitted.predict_proba(X_te)[:, 1]
                else:
                    p_ml_cv[name][te_idx] = p0_fold
            except Exception:
                p_ml_cv[name][te_idx] = p0_fold

st.subheader(f"4.3 Metrics ({cv_label})")

# Build metric tables (unweighted + weighted)
rows_unw = []
rows_w = []

rows_unw.append(compute_metrics_row(f"NumericalOpt_IN (unweighted)", y, p_numopt_in, thr=0.5, sample_weight=None))
rows_w.append(compute_metrics_row(f"NumericalOpt_IN (weighted)", y, p_numopt_in, thr=0.5, sample_weight=w))

rows_unw.append(compute_metrics_row(f"NumericalOpt_{cv_label} (unweighted)", y, p_numopt_cv, thr=0.5, sample_weight=None))
rows_w.append(compute_metrics_row(f"NumericalOpt_{cv_label} (weighted)", y, p_numopt_cv, thr=0.5, sample_weight=w))

for name, p in proba_ml_in.items():
    rows_unw.append(compute_metrics_row(f"{name}_IN (unweighted)", y, p, thr=0.5, sample_weight=None))
    rows_w.append(compute_metrics_row(f"{name}_IN (weighted)", y, p, thr=0.5, sample_weight=w))

for name, p in p_ml_cv.items():
    rows_unw.append(compute_metrics_row(f"{name}_{cv_label} (unweighted)", y, p, thr=0.5, sample_weight=None))
    rows_w.append(compute_metrics_row(f"{name}_{cv_label} (weighted)", y, p, thr=0.5, sample_weight=w))

df_metrics_unw = pd.DataFrame(rows_unw).sort_values(by=["AUC_PR","AUC_ROC","MCC"], ascending=[False, False, False]).reset_index(drop=True)
df_metrics_w = pd.DataFrame(rows_w).sort_values(by=["AUC_PR","AUC_ROC","MCC"], ascending=[False, False, False]).reset_index(drop=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Unweighted metrics**")
    st.dataframe(df_metrics_unw, use_container_width=True)
with c2:
    st.markdown("**Weighted metrics (partial=0.5 weight)**")
    st.dataframe(df_metrics_w, use_container_width=True)


# =========================
# ROC plots (paper-friendly)
# =========================
st.subheader("4.4 ROC curves")

def roc_plot(y_true, curves: Dict[str, np.ndarray], title: str):
    fig = go.Figure()
    if len(np.unique(y_true)) != 2:
        st.warning("ROC requires both classes.")
        return None
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        aucv = roc_auc_score(y_true, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={aucv:.2f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        width=950, height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
    )
    return fig

curves_in = {"Numerical Optimization": p_numopt_in}
curves_cv = {f"Numerical Optimization ({cv_label})": p_numopt_cv}

for name, p in proba_ml_in.items():
    curves_in[f"{name} (cal)"] = p
for name, p in p_ml_cv.items():
    curves_cv[f"{name} ({cv_label}, cal)"] = p

fig_roc_in = roc_plot(y, curves_in, "ROC — In-sample")
fig_roc_cv = roc_plot(y, curves_cv, f"ROC — {cv_label}")

r1, r2 = st.columns(2)
with r1:
    if fig_roc_in is not None:
        st.plotly_chart(fig_roc_in, use_container_width=True)
with r2:
    if fig_roc_cv is not None:
        st.plotly_chart(fig_roc_cv, use_container_width=True)


# =========================
# 3D Figures
# =========================
st.subheader("4.5 3D figures (article-style)")

OPTI_COLOR = "rgb(60,110,220)"   # Numerical Optimization
ML_COLOR   = "rgb(140,80,200)"   # ML

colors_pts, labels_pts = point_colors_from_yraw(y_raw)

# Model for 3D: pick best_ml_name, train on all data and calibrate
p0_all = float(np.clip(np.average(y, weights=w), 1e-12, 1-1e-12))
full_base = make_base_models(42, base_score=p0_all)
if best_ml_name not in full_base:
    best_ml_name = "XGBoost"
model_for_3d = calibrate_model(full_base[best_ml_name], X, y, w)

# (A) Numerical Optimization sphere
xs_h, ys_h, zs_h = sphere_mesh((dp, pp, hp), R0)
figA = go.Figure()
figA.add_trace(go.Surface(
    x=xs_h, y=ys_h, z=zs_h,
    opacity=0.18,
    name="Numerical Optimization (RED=1)",
    legendgroup="opti",
    **surface_style(OPTI_COLOR)
))
figA.add_trace(go.Scatter3d(
    x=df["delta_d"], y=df["delta_p"], z=df["delta_h"],
    mode="markers",
    marker=points_style(colors_pts),
    text=labels_pts,
    name="Samples",
    legendgroup="pts"
))
figA.update_layout(
    title="3D (A) — Numerical Optimization sphere (RED=1)",
    scene=pretty_scene(UNIT),
    width=1100, height=820,
    margin=dict(l=10, r=10, t=70, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
)

st.plotly_chart(figA, use_container_width=True)

ml_rows = []
figs_BC = []

for iso in ISO_LEVELS:
    try:
        c_ml, r_ml, npts, iso_used, delta_used = ml_shell_sphere_from_grid(
            df, model_for_3d, iso=float(iso),
            pad=float(GRID_PAD), n=int(GRID_N),
            shell_delta=float(SHELL_DELTA), min_points=120
        )

        xs_ml, ys_ml, zs_ml = sphere_mesh(c_ml, r_ml)

        ml_rows.append(dict(
            ML_model=str(best_ml_name),
            iso_target=float(iso),
            iso_used=float(iso_used),
            shell_delta_used=float(delta_used),
            center_delta_d=float(c_ml[0]),
            center_delta_p=float(c_ml[1]),
            center_delta_h=float(c_ml[2]),
            R_ml=float(r_ml),
            n_grid_points_shell=int(npts),
            GRID_PAD=float(GRID_PAD),
            GRID_N=int(GRID_N),
            unit=str(UNIT)
        ))

        extra = f" | shell Δ={delta_used:.2f}" if delta_used > 0 else " | fallback: volume p≥iso"

        # (B) ML only
        figB = go.Figure()
        figB.add_trace(go.Surface(
            x=xs_ml, y=ys_ml, z=zs_ml,
            opacity=0.18,
            name=f"{best_ml_name} shell-sphere (p≈{iso:.2f})",
            legendgroup="ml",
            **surface_style(ML_COLOR)
        ))
        figB.add_trace(go.Scatter3d(
            x=df["delta_d"], y=df["delta_p"], z=df["delta_h"],
            mode="markers", marker=points_style(colors_pts),
            text=labels_pts, name="Samples", legendgroup="pts"
        ))
        figB.update_layout(
            title=f"3D (B) — ML shell-sphere p≈{iso:.2f}{extra}",
            scene=pretty_scene(UNIT),
            width=1100, height=820,
            margin=dict(l=10, r=10, t=70, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
        )

        # (C) Overlay
        figC = go.Figure()
        figC.add_trace(go.Surface(
            x=xs_h, y=ys_h, z=zs_h,
            opacity=0.12,
            name="Numerical Optimization (RED=1)",
            legendgroup="opti",
            **surface_style(OPTI_COLOR)
        ))
        figC.add_trace(go.Surface(
            x=xs_ml, y=ys_ml, z=zs_ml,
            opacity=0.12,
            name=f"ML shell-sphere (p≈{iso:.2f}) — {best_ml_name}",
            legendgroup="ml",
            **surface_style(ML_COLOR)
        ))
        figC.add_trace(go.Scatter3d(
            x=df["delta_d"], y=df["delta_p"], z=df["delta_h"],
            mode="markers", marker=points_style(colors_pts),
            text=labels_pts, name="Samples", legendgroup="pts"
        ))
        figC.update_layout(
            title=f"3D (C) — Overlay: Numerical Optimization vs ML (p≈{iso:.2f})",
            scene=pretty_scene(UNIT),
            width=1100, height=820,
            margin=dict(l=10, r=10, t=70, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
        )

        st.plotly_chart(figB, use_container_width=True)
        st.plotly_chart(figC, use_container_width=True)

        figs_BC.append((iso, figB, figC))

    except Exception as e:
        st.warning(f"ML shell-sphere failed at iso={iso}: {e}")

df_ml_spheres = pd.DataFrame(ml_rows)

if not df_ml_spheres.empty:
    st.markdown("**ML shell-sphere parameters**")
    st.dataframe(df_ml_spheres, use_container_width=True)


# =========================
# Downloads: Excel + HTML plots (ZIP)
# =========================
st.header("5) Downloads")

# Build per-sample output
out = df.copy()
out["y_raw"] = y_raw
out["w"] = w
out["RED"] = RED_all
out["p_numopt_in"] = p_numopt_in
out[f"p_numopt_{cv_label}"] = p_numopt_cv
for name, p in proba_ml_in.items():
    out[f"proba_{name}_in"] = p
for name, p in p_ml_cv.items():
    out[f"proba_{name}_{cv_label}"] = p

params_tbl = pd.DataFrame([{
    "center_delta_d": dp,
    "center_delta_p": pp,
    "center_delta_h": hp,
    "R0": R0,
    "unit": UNIT,
    "K_PROB": K_PROB,
    "REG_R0": REG_R0,
    "CV_scheme": cv_label,
    "group_col": (col_group if use_group else "None"),
    "best_ml_for_3d": best_ml_name,
    "ISO_LEVELS": str(ISO_LEVELS),
    "GRID_PAD": GRID_PAD,
    "GRID_N": GRID_N,
    "SHELL_DELTA": SHELL_DELTA
}])

# Excel in memory
excel_bytes = io.BytesIO()
with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Base_bin")
    out.to_excel(writer, index=False, sheet_name="Results_per_sample")
    df_metrics_unw.to_excel(writer, index=False, sheet_name="Metrics_unweighted")
    df_metrics_w.to_excel(writer, index=False, sheet_name="Metrics_weighted")
    params_tbl.to_excel(writer, index=False, sheet_name="Final_Params")
    if not df_ml_spheres.empty:
        df_ml_spheres.to_excel(writer, index=False, sheet_name="ML_ShellSpheres")
excel_bytes.seek(0)

st.download_button(
    "Download results Excel",
    data=excel_bytes.getvalue(),
    file_name=f"results_numopt_vs_ml_{cv_label}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ZIP with HTML plots
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr("fig_3D_A_numopt.html", figA.to_html(full_html=True, include_plotlyjs="cdn"))
    if fig_roc_in is not None:
        z.writestr("fig_ROC_in_sample.html", fig_roc_in.to_html(full_html=True, include_plotlyjs="cdn"))
    if fig_roc_cv is not None:
        z.writestr(f"fig_ROC_{cv_label}.html", fig_roc_cv.to_html(full_html=True, include_plotlyjs="cdn"))
    for iso, figB, figC in figs_BC:
        z.writestr(f"fig_3D_B_ml_iso_{iso}.html", figB.to_html(full_html=True, include_plotlyjs="cdn"))
        z.writestr(f"fig_3D_C_overlay_iso_{iso}.html", figC.to_html(full_html=True, include_plotlyjs="cdn"))
zip_buf.seek(0)

st.download_button(
    "Download figures (HTML) as ZIP",
    data=zip_buf.getvalue(),
    file_name="figures_html.zip",
    mime="application/zip"
)
