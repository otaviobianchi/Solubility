============================================================
#  Numerical Optimization (Hansen) vs ML (Classification) — FINAL (PAPER)
#  IMPLEMENTS IMPROVEMENTS 1, 2 and 3:
#   (1) Group-wise CV (LOGO / Leave-One-Group-Out) when a group column exists;
#       automatic fallback to LOO when not.
#   (2) Probability calibration (CalibratedClassifierCV) with sample_weight (when supported),
#       isotonic if N is sufficient, otherwise sigmoid.
#   (3) ML "sphere-like" fitted on the SHELL (p≈iso): uses points with iso <= p <= iso+Δ (shell),
#       with automatic relaxation if too few points.
#
#  Outputs:
#   - Full metric tables (IN and CV; weighted/unweighted)
#   - ROC curves (IN and CV) + PR curves (IN and CV) + calibration curves
#   - Plotly 3D (units): Numerical Optimization sphere (RED=1) + ML shell-sphere (p≈iso) + overlay
#   - Excel export (full)
#  Colab-ready
# ============================================================

# -----------------------------
# Install packages on Colab (if needed)
# -----------------------------
try:
    import plotly, openpyxl, xgboost, pygad
except ModuleNotFoundError:
    !pip install -q plotly openpyxl xgboost pygad

# Optional:
# !pip install -q lightgbm catboost shap

# -----------------------------
# Imports
# -----------------------------
import warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo

from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from xgboost import XGBClassifier

import plotly.graph_objects as go
import plotly.io as pio
from google.colab import files

import pygad

warnings.filterwarnings("ignore")
np.random.seed(42)
pio.renderers.default = "colab"
plt.rcParams["figure.dpi"] = 160

# ======================
# SETTINGS (PAPER)
# ======================
RUN_ROC_PLOTS = True
RUN_3D = True
RUN_PR_PLOTS = True
RUN_CALIB_PLOTS = True

# Probabilistic Numerical Optimization (RED -> p)
K_PROB = 6.0
REG_R0 = 0.05

# 3D ML “shell-sphere”
ISO_LEVELS = [0.50, 0.80]   # user can change (2 ML spheres for comparison)
GRID_PAD = 1.5
GRID_N = 28
SHELL_DELTA = 0.05          # shell: iso <= p <= iso+Δ (auto-relax if too few points)

# Reparam optimizers
NMS_RESTARTS = 1
COBYLA_RESTARTS = 1

# Units
UNIT = "MPa\u00b9\u2044\u00b2"

# Calibration
CALIBRATION_METHOD_HIGHN = "isotonic"  # if N sufficient
CALIBRATION_METHOD_LOWN  = "sigmoid"   # if N small
CALIBRATION_MIN_N_ISO = 60             # conservative threshold for isotonic

# ======================
# FIGURE STYLE HELPERS (ARTICLE-READY) — English axes/titles
# ======================
def _article_axes(ax):
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_alpha(0.6)

def plot_confusion_matrix_pretty(cm, title="Confusion Matrix", xlabel="Predicted", ylabel="True"):
    plt.figure(figsize=(5.6, 4.8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax = plt.gca()
    _article_axes(ax)
    plt.tight_layout()
    plt.show()

def plot_roc_pretty(y_true, curves, title="ROC Curve", subtitle=None):
    """
    curves: list of dicts: {"label": str, "p": probas, "lw": float}
    """
    if len(np.unique(y_true)) < 2:
        return
    plt.figure(figsize=(6.6, 5.2))
    for c in curves:
        fpr, tpr, _ = roc_curve(y_true, c["p"])
        plt.plot(fpr, tpr, label=f'{c["label"]} (AUC={auc(fpr,tpr):.2f})', lw=c.get("lw", 1.6))
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, alpha=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if subtitle:
        plt.title(f"{title}\n{subtitle}")
    else:
        plt.title(title)
    plt.legend()
    _article_axes(plt.gca())
    plt.tight_layout()
    plt.show()

def plot_pr_pretty(y_true, curves, title="Precision-Recall Curve", subtitle=None):
    if len(np.unique(y_true)) < 2:
        return
    plt.figure(figsize=(6.6, 5.2))
    for c in curves:
        prec, rec, _ = precision_recall_curve(y_true, c["p"])
        ap = average_precision_score(y_true, c["p"])
        plt.plot(rec, prec, label=f'{c["label"]} (AP={ap:.2f})', lw=c.get("lw", 1.6))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if subtitle:
        plt.title(f"{title}\n{subtitle}")
    else:
        plt.title(title)
    plt.legend()
    _article_axes(plt.gca())
    plt.tight_layout()
    plt.show()

def plot_calibration_pretty(y_true, probas_dict, title="Calibration (Reliability) Plot", subtitle=None, n_bins=10):
    """
    probas_dict: {label: probas}
    """
    if len(np.unique(y_true)) < 2:
        return
    plt.figure(figsize=(6.6, 5.2))
    for label, p in probas_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", lw=1.4, label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, alpha=0.8)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    if subtitle:
        plt.title(f"{title}\n{subtitle}")
    else:
        plt.title(title)
    plt.legend()
    _article_axes(plt.gca())
    plt.tight_layout()
    plt.show()

# ======================
# Upload and read Excel
# ======================
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

xls = pd.ExcelFile(file_name)
print("Available sheets:")
for i, aba in enumerate(xls.sheet_names):
    print(f"{i+1}. {aba}")
sheet_index = int(input("Choose sheet number: ")) - 1
df = xls.parse(xls.sheet_names[sheet_index])

print("\nAvailable columns:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

def escolher_coluna(desc, allow_skip=False):
    """
    allow_skip=True allows typing 0 for 'no column'
    """
    while True:
        try:
            idx = int(input(f"Column for {desc}" + (" (0 = none)" if allow_skip else "") + ": "))
            if allow_skip and idx == 0:
                return None
            if 1 <= idx <= len(df.columns):
                return df.columns[idx - 1]
        except:
            pass

delta_d_col = escolher_coluna("δd")
delta_p_col = escolher_coluna("δp")
delta_h_col = escolher_coluna("δh")
solub_col   = escolher_coluna("solubility (0, 0.5 or 1)")
group_col   = escolher_coluna("GROUP for LOGO (e.g., solvent/family/source)", allow_skip=True)

use_group = group_col is not None
cols = [delta_d_col, delta_p_col, delta_h_col, solub_col] + ([group_col] if use_group else [])
df_filtered = df[cols].copy()

new_cols = ['delta_d', 'delta_p', 'delta_h', 'solubility'] + (['group'] if use_group else [])
df_filtered.columns = new_cols
df_filtered[['delta_d','delta_p','delta_h','solubility']] = df_filtered[['delta_d','delta_p','delta_h','solubility']].apply(pd.to_numeric, errors='coerce')
df_filtered = df_filtered.dropna(subset=['delta_d','delta_p','delta_h','solubility']).drop_duplicates().reset_index(drop=True)

# ======================
# Label: PARTIAL -> binary + weights
# ======================
y_raw = df_filtered['solubility'].astype(float).values
y_bin = (y_raw >= 0.5).astype(int)                 # 0.5 -> 1
w = np.where(y_raw == 0.5, 0.5, 1.0).astype(float) # partial weighs 0.5
df_filtered['solubility'] = y_bin

n_total = len(df_filtered)
n_pos = int(np.sum(y_bin))
n_neg = n_total - n_pos
n_parcial = int(np.sum(y_raw == 0.5))
print(f"\nSamples: {n_total} | Positives(bin)={n_pos} | Negatives={n_neg} | Partial(0.5)={n_parcial}")

if use_group:
    groups = df_filtered['group'].astype(str).values
    n_groups = len(np.unique(groups))
    print(f"Group column for CV: '{group_col}' | #groups = {n_groups}")
else:
    groups = None
    print("CV: no group -> automatic fallback to Leave-One-Out (LOO)")

# ======================
# Numerical Optimization (Hansen): distance
# ======================
BOUNDS = [(10, 25), (0, 25), (0, 25), (2, 25)]  # (δd, δp, δh, R0)

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

# ======================
# Objective functions (>=5)
# ======================
def df_geom(d_d, d_p, d_h, y, x, weights=None):
    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra = hansen_distance(d_d, d_p, d_h, dp, pp, hp)

    A = np.ones_like(Ra, dtype=float)
    A[(Ra > R0) & (y == 1)] = np.exp(R0 - Ra[(Ra > R0) & (y == 1)])  # pos outside
    A[(Ra <= R0) & (y == 0)] = np.exp(Ra[(Ra <= R0) & (y == 0)] - R0) # neg inside

    if weights is None:
        gm = np.exp(np.mean(np.log(A + 1e-12)))
    else:
        ww = np.asarray(weights, float)
        gm = np.exp(np.sum(ww * np.log(A + 1e-12)) / np.sum(ww))
    return float(abs(gm - 1.0))

def df_logloss(d_d, d_p, d_h, y_bin, x, weights=None, k=6.0, reg_ro=0.05, ro_ref=None):
    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra  = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
    RED = Ra / R0
    p = prob_from_red(RED, k=k)

    y = np.asarray(y_bin).astype(int)
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

def df_brier(d_d, d_p, d_h, y_bin, x, weights=None, k=6.0, reg_ro=0.05, ro_ref=None):
    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra  = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
    RED = Ra / R0
    p = prob_from_red(RED, k=k)

    y = np.asarray(y_bin).astype(float)
    b = (p - y)**2

    if weights is None:
        loss = float(np.mean(b))
    else:
        ww = np.asarray(weights, float)
        loss = float(np.sum(ww*b) / np.sum(ww))

    if reg_ro and reg_ro > 0:
        if ro_ref is None:
            ro_ref = float(np.median(Ra))
        loss = loss + float(reg_ro)*((R0 - ro_ref)/max(ro_ref, 1e-6))**2
    return float(loss)

def df_hinge(d_d, d_p, d_h, y_bin, x, weights=None):
    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra  = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
    RED = Ra / R0
    y = np.asarray(y_bin).astype(int)

    pos = (y == 1)
    neg = (y == 0)
    loss_pos = np.maximum(0.0, RED[pos] - 1.0)
    loss_neg = np.maximum(0.0, 1.0 - RED[neg])

    loss_vec = np.concatenate([loss_pos, loss_neg], axis=0)
    if loss_vec.size == 0:
        return 0.0

    if weights is None:
        return float(np.mean(loss_vec))
    ww = np.asarray(weights, float)
    ww_vec = np.concatenate([ww[pos], ww[neg]], axis=0)
    return float(np.sum(ww_vec*loss_vec) / np.sum(ww_vec))

def df_softcount(d_d, d_p, d_h, y_bin, x, weights=None, beta=8.0):
    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra  = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
    RED = Ra / R0
    y = np.asarray(y_bin).astype(int)

    s_out = 1.0 / (1.0 + np.exp(-beta*(RED - 1.0)))  # ~1 outside
    err = np.where(y == 1, s_out, (1.0 - s_out))      # pos outside / neg inside

    if weights is None:
        return float(np.mean(err))
    ww = np.asarray(weights, float)
    return float(np.sum(ww*err) / np.sum(ww))

DF_LIST_TO_COMPARE = ["DF_GEOM","DF_LOGLOSS","DF_BRIER","DF_HINGE","DF_SOFTCOUNT"]

def z_to_x(z, bounds=BOUNDS):
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    s  = 1.0 / (1.0 + np.exp(-np.asarray(z, float)))
    return lo + s*(hi - lo)

def fit_by_methods(df_local, weights_local, df_name="DF_GEOM",
                   nms_restarts=1, cobyla_restarts=1):
    d_d = df_local['delta_d'].values
    d_p = df_local['delta_p'].values
    d_h = df_local['delta_h'].values
    yloc = df_local['solubility'].values.astype(int)

    Ra_ref = hansen_distance(d_d, d_p, d_h, np.median(d_d), np.median(d_p), np.median(d_h))
    ro_ref = float(np.median(Ra_ref))

    if df_name == "DF_GEOM":
        def DF(x):
            return df_geom(d_d, d_p, d_h, yloc, x, weights=weights_local)
    elif df_name == "DF_LOGLOSS":
        def DF(x):
            return df_logloss(d_d, d_p, d_h, yloc, x, weights=weights_local, k=K_PROB, reg_ro=REG_R0, ro_ref=ro_ref)
    elif df_name == "DF_BRIER":
        def DF(x):
            return df_brier(d_d, d_p, d_h, yloc, x, weights=weights_local, k=K_PROB, reg_ro=REG_R0, ro_ref=ro_ref)
    elif df_name == "DF_HINGE":
        def DF(x):
            return df_hinge(d_d, d_p, d_h, yloc, x, weights=weights_local)
    elif df_name == "DF_SOFTCOUNT":
        def DF(x):
            return df_softcount(d_d, d_p, d_h, yloc, x, weights=weights_local, beta=8.0)
    else:
        raise ValueError("Unknown DF.")

    starts = [
        ("start=median", [np.median(d_d), np.median(d_p), np.median(d_h), 10.0]),
        ("start=mean",   [np.mean(d_d),   np.mean(d_p),   np.mean(d_h),   12.0]),
    ]

    rows = []

    def _append_row(metodo, execucao, pars):
        pars = np.asarray(pars, float)
        pars[3] = max(float(pars[3]), 1e-6)

        RED = red_values(df_local, pars)
        p = prob_from_red(RED, k=K_PROB)

        AUCv = roc_auc_score(yloc, p) if len(np.unique(yloc)) == 2 else np.nan
        AUPRC = average_precision_score(yloc, p) if len(np.unique(yloc)) == 2 else np.nan

        df_geom_val = df_geom(d_d, d_p, d_h, yloc, pars, weights=weights_local)
        df_ll_val   = df_logloss(d_d, d_p, d_h, yloc, pars, weights=weights_local, k=K_PROB, reg_ro=REG_R0, ro_ref=ro_ref)

        rows.append(dict(
            DF=df_name,
            Método=metodo, Execucao=execucao,
            delta_d=float(pars[0]), delta_p=float(pars[1]), delta_h=float(pars[2]), R0=float(pars[3]),
            DF_GEOM=float(df_geom_val),
            DF_LOGLOSS=float(df_ll_val),
            DF_MAIN=float(DF(pars)),
            AUC_unweighted=float(AUCv) if AUCv==AUCv else np.nan,
            AUPRC_unweighted=float(AUPRC) if AUPRC==AUPRC else np.nan
        ))

    # Local
    for met in ['Powell','L-BFGS-B','TNC']:
        for tag, guess in starts:
            try:
                res = minimize(
                    DF, guess, method=met,
                    bounds=BOUNDS if met in ['L-BFGS-B','TNC'] else None,
                    options=dict(maxiter=2000)
                )
                _append_row(met, tag, res.x)
            except Exception:
                pass

    # Global
    try:
        res_de = differential_evolution(DF, BOUNDS, maxiter=800, polish=True, seed=42)
        _append_row('Differential Evolution', 'global', res_de.x)
    except Exception:
        pass

    try:
        res_da = dual_annealing(DF, bounds=np.array(BOUNDS, float), maxiter=600)
        _append_row('Dual Annealing', 'global', res_da.x)
    except Exception:
        pass

    try:
        res_sh = shgo(DF, BOUNDS, n=128, iters=3)
        _append_row('SHGO', 'global', res_sh.x)
    except Exception:
        pass

    # GA
    def fitness_func(ga_instance, solution, solution_idx):
        return float(-DF(solution))
    try:
        ga = pygad.GA(
            num_generations=180, num_parents_mating=12,
            fitness_func=fitness_func,
            sol_per_pop=40, num_genes=4,
            mutation_probability=0.15, crossover_probability=0.9,
            parent_selection_type="sss", keep_parents=2,
            stop_criteria=["saturate_50"],
            gene_space=[{'low': b[0], 'high': b[1]} for b in BOUNDS],
            random_seed=42
        )
        ga.run()
        _append_row('Genetic Algorithm', 'global', ga.best_solution()[0])
    except Exception:
        pass

    # Nelder–Mead (reparam)
    try:
        for r in range(nms_restarts):
            z0 = np.zeros(4) if r==0 else np.random.normal(0.0, 0.7, 4)
            tag = "z0=center" if r==0 else f"z0=rand#{r}"
            def DF_unconstrained(z):
                return DF(z_to_x(z, BOUNDS))
            res_nm = minimize(DF_unconstrained, z0, method='Nelder-Mead',
                              options=dict(maxiter=3000, maxfev=6000, xatol=1e-6, fatol=1e-9))
            _append_row('Nelder-Mead (reparam)', tag, z_to_x(res_nm.x, BOUNDS))
    except Exception:
        pass

    # COBYLA (reparam)
    try:
        for r in range(cobyla_restarts):
            z0 = np.zeros(4) if r==0 else np.random.normal(0.0, 0.7, 4)
            tag = "z0=center" if r==0 else f"z0=rand#{r}"
            def DF_unconstrained(z):
                return DF(z_to_x(z, BOUNDS))
            res_c = minimize(DF_unconstrained, z0, method='COBYLA',
                             options=dict(maxiter=3000, rhobeg=1.0, catol=1e-8))
            _append_row('COBYLA (reparam)', tag, z_to_x(res_c.x, BOUNDS))
    except Exception:
        pass

    df_params = pd.DataFrame(rows)
    if not df_params.empty:
        df_params = df_params.sort_values(
            by=['DF_MAIN','DF_LOGLOSS','DF_GEOM','AUPRC_unweighted','AUC_unweighted'],
            ascending=[True, True, True, False, False]
        ).reset_index(drop=True)
    return df_params

# ======================
# Select best Numerical Optimization global (compare >=5 DFs)
# ======================
all_runs = []
best_overall = None

for df_name in DF_LIST_TO_COMPARE:
    print(f"\n--- Running Numerical Optimization for {df_name} ---")
    runs = fit_by_methods(df_filtered, weights_local=w,
                          df_name=df_name,
                          nms_restarts=NMS_RESTARTS,
                          cobyla_restarts=COBYLA_RESTARTS)
    if runs.empty:
        print(f"[WARN] No results for {df_name}")
        continue

    all_runs.append(runs)
    row0 = runs.iloc[0].to_dict()
    print(f"Best ({df_name}): {row0['Método']} | DF_MAIN={row0['DF_MAIN']:.6f} | "
          f"δd={row0['delta_d']:.3f}, δp={row0['delta_p']:.3f}, δh={row0['delta_h']:.3f}, R0={row0['R0']:.3f}")

    if best_overall is None:
        best_overall = row0
    else:
        # global criterion (paper): prioritize DF_LOGLOSS and DF_GEOM + AUPRC
        if (row0['DF_LOGLOSS'], row0['DF_GEOM'], -row0['AUPRC_unweighted']) < (best_overall['DF_LOGLOSS'], best_overall['DF_GEOM'], -best_overall['AUPRC_unweighted']):
            best_overall = row0

df_all_runs = pd.concat(all_runs, ignore_index=True) if len(all_runs) else pd.DataFrame()
if best_overall is None:
    raise RuntimeError("No Numerical Optimization run returned results. Check data/columns.")

best_df_name = str(best_overall["DF"])
best_optimizer = str(best_overall["Método"])
dp, pp, hp, R0 = float(best_overall["delta_d"]), float(best_overall["delta_p"]), float(best_overall["delta_h"]), float(best_overall["R0"])

print("\n====================")
print("🏆 BEST (NUMERICAL OPTIMIZATION) GLOBAL")
print(f"DF: {best_df_name} | Optimizer: {best_optimizer}")
print(f"Center: δd={dp:.4f} {UNIT} | δp={pp:.4f} {UNIT} | δh={hp:.4f} {UNIT}")
print(f"Radius: R0={R0:.4f} {UNIT}")
print("====================\n")

# ======================
# Numerical Optimization in-sample
# ======================
y = df_filtered['solubility'].values.astype(int)
RED_all = red_values(df_filtered, (dp, pp, hp, R0))
p_numopt_in = prob_from_red(RED_all, k=K_PROB)

thr_numopt_in = 0.5
if len(np.unique(y)) == 2:
    fpr, tpr, thr = roc_curve(y, p_numopt_in)
    j = tpr - fpr
    thr_numopt_in = float(thr[int(np.argmax(j))])

pred_numopt_in = (p_numopt_in >= thr_numopt_in).astype(int)
cm_numopt_in = confusion_matrix(y, pred_numopt_in)

plot_confusion_matrix_pretty(
    cm_numopt_in,
    title=f"Numerical Optimization — Confusion Matrix (in-sample, thr={thr_numopt_in:.3f})",
    xlabel="Predicted label",
    ylabel="True label"
)

# ======================
# ML base models + CALIBRATION (Improvement #2)
# ======================
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

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=700, learning_rate=0.03,
            num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=random_state
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=800, depth=4, learning_rate=0.05,
            loss_function="Logloss",
            verbose=False, random_seed=random_state
        )
    except Exception:
        pass

    return models

def calibrate_model(base_estimator, X_tr, y_tr, w_tr):
    """
    Probability calibration using internal CV (3-fold) when possible.
    - isotonic if N>=CALIBRATION_MIN_N_ISO and both classes are present, else sigmoid.
    - if something fails, returns the uncalibrated estimator.
    """
    method = CALIBRATION_METHOD_HIGHN if len(y_tr) >= CALIBRATION_MIN_N_ISO else CALIBRATION_METHOD_LOWN
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

# ML data
X = df_filtered[['delta_d','delta_p','delta_h']].values
p0_in = float(np.clip(np.average(y, weights=w), 1e-12, 1-1e-12))
base_models = make_base_models(42, base_score=p0_in)

# In-sample (calibrated)
proba_ml_in = {}
fitted_ml_in = {}
for name, est in base_models.items():
    try:
        fitted = calibrate_model(est, X, y, w)
        fitted_ml_in[name] = fitted
        if hasattr(fitted, "predict_proba"):
            proba_ml_in[name] = fitted.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"[WARN] {name} failed in-sample: {e}")

# ROC / PR / Calibration (in-sample)
if RUN_ROC_PLOTS and (len(np.unique(y))==2):
    curves = [{"label": "Numerical Optimization", "p": p_numopt_in, "lw": 2.2}]
    for name, p in proba_ml_in.items():
        curves.append({"label": f"{name} (cal)", "p": p, "lw": 1.5})
    plot_roc_pretty(y, curves, title="ROC Curve", subtitle="In-sample (calibrated ML)")

if RUN_PR_PLOTS and (len(np.unique(y))==2):
    curves = [{"label": "Numerical Optimization", "p": p_numopt_in, "lw": 2.2}]
    for name, p in proba_ml_in.items():
        curves.append({"label": f"{name} (cal)", "p": p, "lw": 1.5})
    plot_pr_pretty(y, curves, title="Precision–Recall Curve", subtitle="In-sample (calibrated ML)")

if RUN_CALIB_PLOTS and (len(np.unique(y))==2):
    calib_dict = {"Numerical Optimization": p_numopt_in}
    # keep only a few lines if many models exist (still deterministic order)
    for name in sorted(proba_ml_in.keys()):
        calib_dict[f"{name} (cal)"] = proba_ml_in[name]
    plot_calibration_pretty(y, calib_dict, title="Calibration (Reliability) Plot", subtitle="In-sample", n_bins=10)

# ======================
# CV by group (LOGO) or LOO (Improvement #1)
# ======================
if use_group:
    splitter = LeaveOneGroupOut()
    splits = list(splitter.split(X, y, groups=groups))
else:
    splitter = LeaveOneOut()
    splits = list(splitter.split(X, y))

p_numopt_cv = np.zeros(len(y), dtype=float)
p_ml_cv = {name: np.zeros(len(y), dtype=float) for name in base_models.keys()}

# In CV we:
# - Numerical Optimization: refit on train (best_df_name) and predict on test
# - ML: calibrate on train and predict on test (with weights)
for (tr_idx, te_idx) in splits:
    df_tr = df_filtered.iloc[tr_idx].reset_index(drop=True)
    df_te = df_filtered.iloc[te_idx].reset_index(drop=True)

    w_tr = w[tr_idx]
    y_tr = df_tr['solubility'].values.astype(int)

    # Numerical Optimization train-only
    try:
        runs_cv = fit_by_methods(df_tr, weights_local=w_tr, df_name=best_df_name,
                                 nms_restarts=0, cobyla_restarts=0)
        if runs_cv.empty:
            pars_cv = np.array([dp, pp, hp, R0], float)
        else:
            r0 = runs_cv.iloc[0]
            pars_cv = np.array([r0["delta_d"], r0["delta_p"], r0["delta_h"], r0["R0"]], float)
    except Exception:
        pars_cv = np.array([dp, pp, hp, R0], float)

    RED_te = red_values(df_te, pars_cv)
    p_numopt_cv[te_idx] = prob_from_red(RED_te, k=K_PROB)

    # ML train-only (calibrated)
    X_tr = df_tr[['delta_d','delta_p','delta_h']].values
    X_te = df_te[['delta_d','delta_p','delta_h']].values

    # guard: train with 1 class (can happen in LOGO)
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

# ROC / PR / Calibration (CV)
cv_label = "LOGO" if use_group else "LOO"

if RUN_ROC_PLOTS and (len(np.unique(y))==2):
    curves = [{"label": f"Numerical Optimization ({cv_label})", "p": p_numopt_cv, "lw": 2.2}]
    for name, p in p_ml_cv.items():
        curves.append({"label": f"{name} ({cv_label}, cal)", "p": p, "lw": 1.5})
    plot_roc_pretty(y, curves, title="ROC Curve", subtitle=f"Cross-validation ({cv_label})")

if RUN_PR_PLOTS and (len(np.unique(y))==2):
    curves = [{"label": f"Numerical Optimization ({cv_label})", "p": p_numopt_cv, "lw": 2.2}]
    for name, p in p_ml_cv.items():
        curves.append({"label": f"{name} ({cv_label}, cal)", "p": p, "lw": 1.5})
    plot_pr_pretty(y, curves, title="Precision–Recall Curve", subtitle=f"Cross-validation ({cv_label})")

if RUN_CALIB_PLOTS and (len(np.unique(y))==2):
    calib_dict = {f"Numerical Optimization ({cv_label})": p_numopt_cv}
    for name in sorted(p_ml_cv.keys()):
        calib_dict[f"{name} ({cv_label}, cal)"] = p_ml_cv[name]
    plot_calibration_pretty(y, calib_dict, title="Calibration (Reliability) Plot", subtitle=f"Cross-validation ({cv_label})", n_bins=10)

# ======================
# Full metrics
# ======================
def _safe_div(a, b):
    return float(a) / float(b) if float(b) != 0 else np.nan

def weighted_confusion(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sw = np.ones_like(y_true, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    TN = sw[(y_true==0) & (y_pred==0)].sum()
    FP = sw[(y_true==0) & (y_pred==1)].sum()
    FN = sw[(y_true==1) & (y_pred==0)].sum()
    TP = sw[(y_true==1) & (y_pred==1)].sum()
    return TN, FP, FN, TP

def weighted_brier(y_true, p, sample_weight=None):
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(p).astype(float)
    if sample_weight is None:
        return float(np.mean((p - y_true)**2))
    sw = np.asarray(sample_weight, dtype=float)
    return float(np.sum(sw*(p - y_true)**2) / np.sum(sw))

def weighted_logloss(y_true, p, sample_weight=None, eps=1e-12):
    y_true = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(p).astype(float), eps, 1-eps)
    ll = -(y_true*np.log(p) + (1-y_true)*np.log(1-p))
    if sample_weight is None:
        return float(np.mean(ll))
    sw = np.asarray(sample_weight, dtype=float)
    return float(np.sum(sw*ll) / np.sum(sw))

def weighted_accuracy(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if sample_weight is None:
        return float(np.mean(y_true == y_pred))
    sw = np.asarray(sample_weight, dtype=float)
    return float(np.sum(sw*(y_true == y_pred)) / np.sum(sw))

def compute_metrics_full(model_name, y_true, p, thr=0.5, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p).astype(float), 1e-12, 1-1e-12)
    y_pred = (p >= float(thr)).astype(int)

    TN, FP, FN, TP = weighted_confusion(y_true, y_pred, sample_weight=sample_weight)

    precision = _safe_div(TP, TP+FP)
    recall = _safe_div(TP, TP+FN)
    specificity = _safe_div(TN, TN+FP)
    f1 = _safe_div(2*precision*recall, precision+recall)
    npv = _safe_div(TN, TN+FN)
    fpr = _safe_div(FP, FP+TN)
    fnr = _safe_div(FN, FN+TP)
    bal_acc = np.nan if (recall!=recall or specificity!=specificity) else 0.5*(recall+specificity)

    denom = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc = _safe_div((TP*TN - FP*FN), denom)

    out = dict(
        Model=model_name,
        thr=float(thr),
        LogLoss=weighted_logloss(y_true, p, sample_weight=sample_weight),
        Brier=weighted_brier(y_true, p, sample_weight=sample_weight),
        Accuracy=weighted_accuracy(y_true, y_pred, sample_weight=sample_weight),
        BalancedAcc=bal_acc,
        Precision=precision,
        Recall=recall,
        Specificity=specificity,
        F1=f1,
        NPV=npv,
        MCC=mcc,
        FPR=fpr,
        FNR=fnr,
        TN=float(TN), FP=float(FP), FN=float(FN), TP=float(TP),
        AUC_ROC_unweighted=np.nan,
        AUC_PR_unweighted=np.nan
    )
    if len(np.unique(y_true)) == 2:
        out["AUC_ROC_unweighted"] = float(roc_auc_score(y_true, p))
        out["AUC_PR_unweighted"] = float(average_precision_score(y_true, p))
    return out

def build_metrics_tables(y, w, p_in, thr_in, p_cv, thr_cv,
                         ml_in_dict, ml_cv_dict, thr_ml=0.5, cv_label="CV"):
    rows_in_unw, rows_in_w = [], []
    rows_cv_unw, rows_cv_w = [], []

    rows_in_unw.append(compute_metrics_full("NumericalOptimization_IN (unweighted)", y, p_in, thr=thr_in, sample_weight=None))
    rows_in_w.append(compute_metrics_full("NumericalOptimization_IN (weighted)", y, p_in, thr=thr_in, sample_weight=w))

    rows_cv_unw.append(compute_metrics_full(f"NumericalOptimization_{cv_label} (unweighted)", y, p_cv, thr=thr_cv, sample_weight=None))
    rows_cv_w.append(compute_metrics_full(f"NumericalOptimization_{cv_label} (weighted)", y, p_cv, thr=thr_cv, sample_weight=w))

    for k, p in ml_in_dict.items():
        rows_in_unw.append(compute_metrics_full(f"{k}_IN (unweighted)", y, p, thr=thr_ml, sample_weight=None))
        rows_in_w.append(compute_metrics_full(f"{k}_IN (weighted)", y, p, thr=thr_ml, sample_weight=w))

    for k, p in ml_cv_dict.items():
        rows_cv_unw.append(compute_metrics_full(f"{k}_{cv_label} (unweighted)", y, p, thr=thr_ml, sample_weight=None))
        rows_cv_w.append(compute_metrics_full(f"{k}_{cv_label} (weighted)", y, p, thr=thr_ml, sample_weight=w))

    df_in_unw = pd.DataFrame(rows_in_unw)
    df_in_w = pd.DataFrame(rows_in_w)
    df_cv_unw = pd.DataFrame(rows_cv_unw)
    df_cv_w = pd.DataFrame(rows_cv_w)

    sort_cols = ["LogLoss","Brier","AUC_PR_unweighted","AUC_ROC_unweighted","MCC"]
    asc = [True, True, False, False, False]

    for dfx in [df_in_unw, df_in_w, df_cv_unw, df_cv_w]:
        dfx.sort_values(by=sort_cols, ascending=asc, inplace=True)
        dfx.reset_index(drop=True, inplace=True)

    return df_in_unw, df_in_w, df_cv_unw, df_cv_w

thr_numopt_cv = 0.5  # paper-friendly (p(RED) >= 0.5)

df_metrics_in_unw, df_metrics_in_w, df_metrics_cv_unw, df_metrics_cv_w = build_metrics_tables(
    y=y, w=w,
    p_in=p_numopt_in, thr_in=thr_numopt_in,
    p_cv=p_numopt_cv, thr_cv=thr_numopt_cv,
    ml_in_dict=proba_ml_in,
    ml_cv_dict=p_ml_cv,
    thr_ml=0.5,
    cv_label=cv_label
)

print("\n=== METRICS IN-SAMPLE (UNWEIGHTED) ===")
print(df_metrics_in_unw.to_string(index=False, float_format="%.6f"))

print(f"\n=== METRICS {cv_label} (UNWEIGHTED) ===")
print(df_metrics_cv_unw.to_string(index=False, float_format="%.6f"))

# Best ML in CV (ranking)
ml_only = df_metrics_cv_unw[
    df_metrics_cv_unw["Model"].str.contains(f"_{cv_label}") &
    (~df_metrics_cv_unw["Model"].str.startswith("NumericalOptimization"))
]
if not ml_only.empty:
    best_ml_row = ml_only.iloc[0].to_dict()
    best_ml_name = best_ml_row["Model"].replace(f"_{cv_label} (unweighted)", "").replace(f"_{cv_label} (weighted)", "").replace(f"_{cv_label}", "")
else:
    best_ml_name = "XGBoost"

print(f"\n🏅 Best ML ({cv_label}, unweighted ranking): {best_ml_name}")

# ======================
# 3D: Numerical Optimization sphere + ML shell-sphere (Improvement #3)
# ======================

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
    """
    Fit sphere using points on the "shell" near iso:
      iso <= p <= iso + shell_delta
    If too few points, relax by increasing delta and decreasing iso slightly.
    """
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
    return center, r, int(Psel.shape[0]), float(iso), -1.0  # delta=-1 indicates fallback

# Model for 3D: train on all data and calibrate
p0_all = float(np.clip(np.average(y, weights=w), 1e-12, 1-1e-12))
full_base = make_base_models(42, base_score=p0_all)
if best_ml_name not in full_base:
    print(f"[3D] Model '{best_ml_name}' not available. Using XGBoost.")
    best_ml_name = "XGBoost"

model_for_3d = calibrate_model(full_base[best_ml_name], X, y, w)

ml_spheres_table_rows = []

# ---------- Plotly 3D ARTICLE STYLE ----------
def _pretty_scene(unit=UNIT):
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

def _pretty_layout(title):
    return dict(
        title=title,
        width=1100, height=820,
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1
        )
    )

OPTI_COLOR = "rgb(60,110,220)"   # Numerical Optimization (blue)
ML_COLOR   = "rgb(140,80,200)"   # ML (purple)

def _surface_style(base_rgb, opacity=0.16):
    return dict(
        opacity=opacity,
        showscale=False,
        colorscale=[[0.0, base_rgb],[1.0, base_rgb]],
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.55, fresnel=0.08),
        lightposition=dict(x=100, y=200, z=100)
    )

def _points_style(colors):
    return dict(
        size=5,
        color=colors,
        opacity=0.95,
        line=dict(width=0.6, color="rgba(0,0,0,0.35)")
    )

if RUN_3D:
    colors, labels = point_colors_from_yraw(y_raw)

    # (A) Numerical Optimization sphere (RED=1)  --- FIX: remove duplicate opacity/showscale
    xs_h, ys_h, zs_h = sphere_mesh((dp, pp, hp), R0)
    fig_h = go.Figure()
    fig_h.add_trace(go.Surface(
        x=xs_h, y=ys_h, z=zs_h,
        name="Numerical Optimization (RED=1)",
        legendgroup="opti",
        **_surface_style(OPTI_COLOR, opacity=0.18)
    ))
    fig_h.add_trace(go.Scatter3d(
        x=df_filtered["delta_d"], y=df_filtered["delta_p"], z=df_filtered["delta_h"],
        mode="markers", marker=_points_style(colors),
        text=labels, name="Samples", legendgroup="pts"
    ))
    fig_h.update_layout(**_pretty_layout(
        title=f"3D (A) — Numerical Optimization sphere (RED=1) | DF={best_df_name} | Optimizer={best_optimizer}"
    ))
    fig_h.update_layout(scene=_pretty_scene())
    fig_h.show()

    # (B)/(C) ML shell-spheres + overlay
    for iso in ISO_LEVELS:
        try:
            c_ml, r_ml, npts, iso_used, delta_used = ml_shell_sphere_from_grid(
                df_filtered, model_for_3d, iso=iso, pad=GRID_PAD, n=GRID_N,
                shell_delta=SHELL_DELTA, min_points=120
            )
            xs_ml, ys_ml, zs_ml = sphere_mesh(c_ml, r_ml)

            ml_spheres_table_rows.append({
                "ML_model": best_ml_name,
                "iso_target": float(iso),
                "iso_used": float(iso_used),
                "shell_delta_used": float(delta_used),
                "center_delta_d": float(c_ml[0]),
                "center_delta_p": float(c_ml[1]),
                "center_delta_h": float(c_ml[2]),
                "R_ml": float(r_ml),
                "unit": UNIT,
                "n_grid_points_shell": int(npts),
                "GRID_PAD": float(GRID_PAD),
                "GRID_N": int(GRID_N)
            })

            # (B) ML alone
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Surface(
                x=xs_ml, y=ys_ml, z=zs_ml,
                name=f"{best_ml_name} shell-sphere (p≈{iso:.2f})",
                legendgroup="ml",
                **_surface_style(ML_COLOR, opacity=0.18)
            ))
            fig_ml.add_trace(go.Scatter3d(
                x=df_filtered["delta_d"], y=df_filtered["delta_p"], z=df_filtered["delta_h"],
                mode="markers", marker=_points_style(colors),
                text=labels, name="Samples", legendgroup="pts"
            ))
            extra = f" | shell Δ={delta_used:.2f}" if delta_used > 0 else " | fallback: volume p≥iso"
            fig_ml.update_layout(**_pretty_layout(
                title=f"3D (B) — ML shell-sphere p≈{iso:.2f} | {best_ml_name} | R≈{r_ml:.3f} {UNIT}{extra}"
            ))
            fig_ml.update_layout(scene=_pretty_scene())
            fig_ml.show()

            # (C) Overlay with distinct tones
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Surface(
                x=xs_h, y=ys_h, z=zs_h,
                name="Numerical Optimization (RED=1)",
                legendgroup="opti",
                **_surface_style(OPTI_COLOR, opacity=0.12)
            ))
            fig_ov.add_trace(go.Surface(
                x=xs_ml, y=ys_ml, z=zs_ml,
                name=f"ML shell-sphere (p≈{iso:.2f}) — {best_ml_name}",
                legendgroup="ml",
                **_surface_style(ML_COLOR, opacity=0.12)
            ))
            fig_ov.add_trace(go.Scatter3d(
                x=df_filtered["delta_d"], y=df_filtered["delta_p"], z=df_filtered["delta_h"],
                mode="markers", marker=_points_style(colors),
                text=labels, name="Samples", legendgroup="pts"
            ))
            fig_ov.update_layout(**_pretty_layout(
                title=f"3D (C) — Overlay: Numerical Optimization (RED=1) vs ML shell-sphere (p≈{iso:.2f}) | {best_ml_name}"
            ))
            fig_ov.update_layout(scene=_pretty_scene())
            fig_ov.show()

        except Exception as e:
            print(f"[ML-SHELL-SPHERE][WARN] Failed at iso={iso:.2f}: {e}")
            print("  Suggestion: increase GRID_N (e.g., 32) and/or GRID_PAD (e.g., 2.0), or reduce ISO_LEVELS.")

df_ml_spheres = pd.DataFrame(ml_spheres_table_rows) if len(ml_spheres_table_rows) else pd.DataFrame()

# ======================
# Export Excel
# ======================
def round_df_numeric(df_in, ndigits=6):
    df_out = df_in.copy()
    for c in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[c]):
            df_out[c] = df_out[c].round(ndigits)
    return df_out

out = df_filtered.copy()
out['y_raw'] = y_raw
out['w'] = w
out['RED'] = RED_all
out['p_numopt_in'] = p_numopt_in
out['p_numopt_CV'] = p_numopt_cv
for name, p in proba_ml_in.items():
    out[f'proba_{name}_in'] = p
for name, p in p_ml_cv.items():
    out[f'proba_{name}_{cv_label}'] = p

out = round_df_numeric(out, ndigits=8)

df_params_numopt = pd.DataFrame({
    "Parameter": ["DF_best", "optimizer_best",
                  "delta_d_center", "delta_p_center", "delta_h_center", "R0", "unit",
                  "CV_scheme", "group_col"],
    "Value": [best_df_name, best_optimizer, dp, pp, hp, R0, UNIT,
              cv_label, (group_col if use_group else "None")]
})

df_final_settings = pd.DataFrame([
    {"Group":"NumOpt_prob", "Item":"K_PROB", "Value": K_PROB},
    {"Group":"NumOpt_prob", "Item":"REG_R0", "Value": REG_R0},
    {"Group":"CV", "Item":"scheme", "Value": cv_label},
    {"Group":"CV", "Item":"group_col", "Value": (group_col if use_group else "None")},
    {"Group":"3D_settings", "Item":"GRID_PAD", "Value": GRID_PAD},
    {"Group":"3D_settings", "Item":"GRID_N", "Value": GRID_N},
    {"Group":"3D_settings", "Item":"ISO_LEVELS", "Value": str(ISO_LEVELS)},
    {"Group":"3D_settings", "Item":"SHELL_DELTA", "Value": SHELL_DELTA},
    {"Group":"ML_visual", "Item":"best_ml_name_for_3d", "Value": str(best_ml_name)},
    {"Group":"Calib", "Item":"min_n_isotonic", "Value": CALIBRATION_MIN_N_ISO},
])

output_path = "/content/result_full_paper_LOGO_cal_shell.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_filtered.to_excel(writer, index=False, sheet_name="Base_bin")
    out.to_excel(writer, index=False, sheet_name="Results_per_sample")

    round_df_numeric(df_all_runs, 8).to_excel(writer, index=False, sheet_name="NumOpt_AllRuns")
    df_params_numopt.to_excel(writer, index=False, sheet_name="NumOpt_Final")

    df_metrics_in_unw.to_excel(writer, index=False, sheet_name="Metrics_IN_unweighted")
    df_metrics_in_w.to_excel(writer, index=False, sheet_name="Metrics_IN_weighted")
    df_metrics_cv_unw.to_excel(writer, index=False, sheet_name=f"Metrics_{cv_label}_unweighted")
    df_metrics_cv_w.to_excel(writer, index=False, sheet_name=f"Metrics_{cv_label}_weighted")

    if not df_ml_spheres.empty:
        df_ml_spheres.to_excel(writer, index=False, sheet_name="ML_ShellSpheres")
    df_final_settings.to_excel(writer, index=False, sheet_name="Final_Settings")

time.sleep(1.0)
files.download(output_path)
print(f"\n📦 Excel saved and downloaded: {output_path}")
