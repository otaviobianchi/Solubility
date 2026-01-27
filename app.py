# ============================================================
# APP.PY — PARTE 1/4 (COMPLETA)
#  ✅ Agora com 6 ABAS:
#   tab1 Data
#   tab2 Numerical Optimization (+ 3D rápido do NumOpt)
#   tab3 ML (in-sample)
#   tab4 Cross-Validation (LOGO/LOO)
#   tab5 3D (NumOpt + ML + overlay)
#   tab6 Export
#
#  ✅ Parte 1/4 inclui:
#   - imports + config + tabs
#   - session_state defaults
#   - carga de Excel + mapeamento de colunas (Tab 1)
#   - core numérico (Hansen distance, RED->p)
#   - objective functions + fit_by_methods()
#
#  ⚠️ Importante:
#   - Mantém "Numerical Optimization" (não Hansen no título do app)
#   - Corrige imports faltantes (roc_auc_score, average_precision_score)
# ============================================================

import warnings
import io
import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo

from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
np.random.seed(42)
import streamlit as st

st.set_page_config(page_title="Solubility App", layout="wide")

# =========================
# SIDEBAR — CACHE CONTROL
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Cache")
    if st.button("🧹 Limpar cache", use_container_width=True):
        st.cache_data.clear()
        for k in [
            "proba_ml_in", "p_numopt_cv", "p_ml_cv",
            "df_metrics_in_unw", "df_metrics_in_w",
            "df_metrics_cv_unw", "df_metrics_cv_w"
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Cache limpo.")
        st.rerun()


# -----------------------------
# Streamlit config + 6 TABS
# -----------------------------
st.set_page_config(page_title="Numerical Optimization vs ML — Solubility", layout="wide")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1) Data",
    "2) Numerical Optimization",
    "3) ML (in-sample)",
    "4) Cross-Validation",
    "5) 3D",
    "6) Export"
])

# -----------------------------
# Global settings (shared)
# -----------------------------
st.session_state.setdefault("UNIT", "MPa\u00b9\u2044\u00b2")  # MPa½
st.session_state.setdefault("K_PROB", 6.0)
st.session_state.setdefault("REG_R0", 0.05)
st.session_state.setdefault("NMS_RESTARTS", 1)
st.session_state.setdefault("COBYLA_RESTARTS", 1)
st.session_state.setdefault("speed_profile", "full")  # "full" or "fast"

# app metadata (export)
st.session_state.setdefault("app_version", "v6tabs_paper")
st.session_state.setdefault("run_timestamp", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

# -----------------------------
# Optional GA
# -----------------------------
try:
    import pygad
    _HAS_PYGAD = True
except Exception:
    _HAS_PYGAD = False

# -----------------------------
# Bounds (δd, δp, δh, R0)
# -----------------------------
BOUNDS = [(10, 25), (0, 25), (0, 25), (2, 25)]

# ============================================================
# CORE FUNCTIONS (Numerical Optimization)
# ============================================================
def hansen_distance(d_d, d_p, d_h, dp, pp, hp):
    d_d = np.asarray(d_d, float)
    d_p = np.asarray(d_p, float)
    d_h = np.asarray(d_h, float)
    return np.sqrt(4*(d_d - dp)**2 + (d_p - pp)**2 + (d_h - hp)**2)

def prob_from_red(RED, k=6.0):
    RED = np.asarray(RED, float)
    z = k*(1.0 - RED)
    z = np.clip(z, -60, 60)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 1e-12, 1.0-1e-12)

def red_values(df_local, params):
    dp, pp, hp, R0 = map(float, params)
    Ra = hansen_distance(
        df_local["delta_d"].values,
        df_local["delta_p"].values,
        df_local["delta_h"].values,
        dp, pp, hp
    )
    return Ra / max(R0, 1e-6)

# -----------------------------
# Objective functions
# -----------------------------
def df_geom(d_d, d_p, d_h, y, x, weights=None):
    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra = hansen_distance(d_d, d_p, d_h, dp, pp, hp)

    A = np.ones_like(Ra, dtype=float)
    A[(Ra > R0) & (y == 1)] = np.exp(R0 - Ra[(Ra > R0) & (y == 1)])       # pos outside
    A[(Ra <= R0) & (y == 0)] = np.exp(Ra[(Ra <= R0) & (y == 0)] - R0)     # neg inside

    if weights is None:
        gm = np.exp(np.mean(np.log(A + 1e-12)))
    else:
        ww = np.asarray(weights, float)
        gm = np.exp(np.sum(ww*np.log(A + 1e-12)) / np.sum(ww))
    return float(abs(gm - 1.0))

def df_logloss(d_d, d_p, d_h, y_bin, x, weights=None, k=6.0, reg_ro=0.05, ro_ref=None):
    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
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
    Ra = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
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
    Ra = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
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
    Ra = hansen_distance(d_d, d_p, d_h, dp, pp, hp)
    RED = Ra / R0
    y = np.asarray(y_bin).astype(int)

    s_out = 1.0 / (1.0 + np.exp(-beta*(RED - 1.0)))   # ~1 outside
    err = np.where(y == 1, s_out, (1.0 - s_out))      # pos outside / neg inside

    if weights is None:
        return float(np.mean(err))
    ww = np.asarray(weights, float)
    return float(np.sum(ww*err) / np.sum(ww))

# -----------------------------
# Reparam for unconstrained solvers
# -----------------------------
def z_to_x(z, bounds=BOUNDS):
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    s = 1.0 / (1.0 + np.exp(-np.asarray(z, float)))
    return lo + s*(hi - lo)

# -----------------------------
# fit_by_methods (main optimizer runner)
# -----------------------------
def fit_by_methods(
    df_local,
    weights_local,
    df_name="DF_GEOM",
    K_PROB=6.0,
    REG_R0=0.05,
    nms_restarts=1,
    cobyla_restarts=1,
    speed_profile="full"
):
    d_d = df_local["delta_d"].values.astype(float)
    d_p = df_local["delta_p"].values.astype(float)
    d_h = df_local["delta_h"].values.astype(float)
    yloc = df_local["solubility"].values.astype(int)

    Ra_ref = hansen_distance(d_d, d_p, d_h, np.median(d_d), np.median(d_p), np.median(d_h))
    ro_ref = float(np.median(Ra_ref))

    # choose objective
    if df_name == "DF_GEOM":
        def DF(x): return df_geom(d_d, d_p, d_h, yloc, x, weights=weights_local)
    elif df_name == "DF_LOGLOSS":
        def DF(x): return df_logloss(d_d, d_p, d_h, yloc, x, weights=weights_local, k=K_PROB, reg_ro=REG_R0, ro_ref=ro_ref)
    elif df_name == "DF_BRIER":
        def DF(x): return df_brier(d_d, d_p, d_h, yloc, x, weights=weights_local, k=K_PROB, reg_ro=REG_R0, ro_ref=ro_ref)
    elif df_name == "DF_HINGE":
        def DF(x): return df_hinge(d_d, d_p, d_h, yloc, x, weights=weights_local)
    elif df_name == "DF_SOFTCOUNT":
        def DF(x): return df_softcount(d_d, d_p, d_h, yloc, x, weights=weights_local, beta=8.0)
    else:
        raise ValueError("Unknown DF name.")

    # speed presets
    sp = str(speed_profile).lower().strip()
    if sp == "fast":
        maxiter_local = 900
        maxiter_de = 220
        maxiter_da = 220
        shgo_n = 64
        shgo_iters = 2
        ga_gen = 80
        ga_pop = 28
    else:
        maxiter_local = 2000
        maxiter_de = 700
        maxiter_da = 500
        shgo_n = 128
        shgo_iters = 3
        ga_gen = 180
        ga_pop = 40

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

        aucv = roc_auc_score(yloc, p) if len(np.unique(yloc)) == 2 else np.nan
        aupr = average_precision_score(yloc, p) if len(np.unique(yloc)) == 2 else np.nan

        rows.append(dict(
            DF=df_name,
            Método=str(metodo),
            Execucao=str(execucao),
            delta_d=float(pars[0]),
            delta_p=float(pars[1]),
            delta_h=float(pars[2]),
            R0=float(pars[3]),
            DF_GEOM=float(df_geom(d_d, d_p, d_h, yloc, pars, weights=weights_local)),
            DF_LOGLOSS=float(df_logloss(d_d, d_p, d_h, yloc, pars, weights=weights_local, k=K_PROB, reg_ro=REG_R0, ro_ref=ro_ref)),
            DF_MAIN=float(DF(pars)),
            AUC_unweighted=float(aucv) if aucv == aucv else np.nan,
            AUPRC_unweighted=float(aupr) if aupr == aupr else np.nan,
        ))

    # Local solvers
    for met in ["Powell", "L-BFGS-B", "TNC"]:
        for tag, guess in starts:
            try:
                res = minimize(
                    DF, guess, method=met,
                    bounds=BOUNDS if met in ["L-BFGS-B", "TNC"] else None,
                    options=dict(maxiter=maxiter_local)
                )
                _append_row(met, tag, res.x)
            except Exception:
                pass

    # Global solvers
    try:
        res_de = differential_evolution(DF, BOUNDS, maxiter=maxiter_de, polish=True, seed=42)
        _append_row("Differential Evolution", "global", res_de.x)
    except Exception:
        pass

    try:
        res_da = dual_annealing(DF, bounds=np.array(BOUNDS, float), maxiter=maxiter_da)
        _append_row("Dual Annealing", "global", res_da.x)
    except Exception:
        pass

    try:
        res_sh = shgo(DF, BOUNDS, n=shgo_n, iters=shgo_iters)
        _append_row("SHGO", "global", res_sh.x)
    except Exception:
        pass

    # GA (optional)
    if _HAS_PYGAD:
        def fitness_func(ga_instance, solution, solution_idx):
            return float(-DF(solution))
        try:
            ga = pygad.GA(
                num_generations=int(ga_gen),
                num_parents_mating=12,
                fitness_func=fitness_func,
                sol_per_pop=int(ga_pop),
                num_genes=4,
                mutation_probability=0.15,
                crossover_probability=0.9,
                parent_selection_type="sss",
                keep_parents=2,
                stop_criteria=["saturate_50"],
                gene_space=[{"low": b[0], "high": b[1]} for b in BOUNDS],
                random_seed=42
            )
            ga.run()
            _append_row("Genetic Algorithm", "global", ga.best_solution()[0])
        except Exception:
            pass

    # Nelder–Mead (reparam)
    try:
        for r in range(int(nms_restarts)):
            z0 = np.zeros(4) if r == 0 else np.random.normal(0.0, 0.7, 4)
            tag = "z0=center" if r == 0 else f"z0=rand#{r}"
            def DF_unconstrained(z):
                return DF(z_to_x(z, BOUNDS))
            res_nm = minimize(
                DF_unconstrained, z0, method="Nelder-Mead",
                options=dict(maxiter=3000, maxfev=6000, xatol=1e-6, fatol=1e-9)
            )
            _append_row("Nelder-Mead (reparam)", tag, z_to_x(res_nm.x, BOUNDS))
    except Exception:
        pass

    # COBYLA (reparam)
    try:
        for r in range(int(cobyla_restarts)):
            z0 = np.zeros(4) if r == 0 else np.random.normal(0.0, 0.7, 4)
            tag = "z0=center" if r == 0 else f"z0=rand#{r}"
            def DF_unconstrained(z):
                return DF(z_to_x(z, BOUNDS))
            res_c = minimize(
                DF_unconstrained, z0, method="COBYLA",
                options=dict(maxiter=2500, rhobeg=1.0, catol=1e-8)
            )
            _append_row("COBYLA (reparam)", tag, z_to_x(res_c.x, BOUNDS))
    except Exception:
        pass

    df_params = pd.DataFrame(rows)
    if df_params.empty:
        return df_params

    df_params = df_params.sort_values(
        by=["DF_MAIN", "DF_LOGLOSS", "DF_GEOM", "AUPRC_unweighted", "AUC_unweighted"],
        ascending=[True, True, True, False, False]
    ).reset_index(drop=True)

    return df_params

# ============================================================
# TAB 1 — DATA (upload + mapping)
# ============================================================
with tab1:
    st.title("Numerical Optimization vs ML — Solubility")
    st.caption("Upload an Excel file, select the sheet and columns, then proceed to the other tabs.")

    st.markdown("### Upload Excel")
    up = st.file_uploader("Upload .xlsx", type=["xlsx"])

    if up is None:
        st.info("Upload an Excel file to start.")
        st.stop()

    xls = pd.ExcelFile(up)
    sheet_name = st.selectbox("Sheet", xls.sheet_names, index=0)
    df = xls.parse(sheet_name)

    st.session_state["sheet_name"] = sheet_name

    st.markdown("### Column mapping")
    cols = list(df.columns)

    c1, c2, c3 = st.columns(3)
    with c1:
        delta_d_col = st.selectbox("δd column", cols, index=0)
        delta_p_col = st.selectbox("δp column", cols, index=1 if len(cols) > 1 else 0)
    with c2:
        delta_h_col = st.selectbox("δh column", cols, index=2 if len(cols) > 2 else 0)
        solub_col = st.selectbox("solubility column (0 / 0.5 / 1)", cols, index=3 if len(cols) > 3 else 0)
    with c3:
        group_col = st.selectbox("Group column (optional)", ["(none)"] + cols, index=0)

    use_group = (group_col != "(none)")

    # store mapping (for export metadata)
    colmap = dict(delta_d=delta_d_col, delta_p=delta_p_col, delta_h=delta_h_col, solubility=solub_col, group=(group_col if use_group else None))
    st.session_state["colmap"] = colmap
    st.session_state["group_col"] = (group_col if use_group else "None")

    # Build filtered base
    need_cols = [delta_d_col, delta_p_col, delta_h_col, solub_col] + ([group_col] if use_group else [])
    df_filtered = df[need_cols].copy()

    new_cols = ["delta_d", "delta_p", "delta_h", "solubility"] + (["group"] if use_group else [])
    df_filtered.columns = new_cols

    # numeric
    df_filtered[["delta_d", "delta_p", "delta_h", "solubility"]] = df_filtered[
        ["delta_d", "delta_p", "delta_h", "solubility"]
    ].apply(pd.to_numeric, errors="coerce")

    df_filtered = (
        df_filtered
        .dropna(subset=["delta_d", "delta_p", "delta_h", "solubility"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Store raw labels + weights
    y_raw = df_filtered["solubility"].astype(float).values
    y_bin = (y_raw >= 0.5).astype(int)                      # 0.5 -> 1
    w = np.where(y_raw == 0.5, 0.5, 1.0).astype(float)      # partial weights 0.5

    df_filtered["solubility"] = y_bin

    if use_group:
        groups = df_filtered["group"].astype(str).values
    else:
        groups = None

    # Save in session_state for other tabs
    st.session_state["df_filtered"] = df_filtered
    st.session_state["y_raw"] = y_raw
    st.session_state["w"] = w
    st.session_state["use_group"] = use_group
    st.session_state["groups"] = groups

    # Preview
    n_total = len(df_filtered)
    n_pos = int(np.sum(y_bin))
    n_neg = int(n_total - n_pos)
    n_part = int(np.sum(y_raw == 0.5))

    st.success(f"Loaded: N={n_total} | Positives(bin)={n_pos} | Negatives={n_neg} | Partial(0.5)={n_part}")
    if use_group:
        st.info(f"CV scheme: LOGO | group column: {group_col} | #groups={len(np.unique(groups))}")
    else:
        st.info("CV scheme: LOO (no group column)")

    st.dataframe(df_filtered.head(50), use_container_width=True)

# ============================================================
# Make required globals available for other parts
# ============================================================
df_filtered = st.session_state.get("df_filtered")
y_raw = st.session_state.get("y_raw")
w = st.session_state.get("w")
use_group = st.session_state.get("use_group", False)
groups = st.session_state.get("groups", None)
UNIT = st.session_state.get("UNIT", "MPa\u00b9\u2044\u00b2")
# ============================================================
# APP.PY — PARTE 2/4 (TAB 2 COMPLETA)
#  Aba 2: Numerical Optimization
#   ✅ roda NumOpt (comparando DFs e solvers)
#   ✅ mostra cards (δd, δp, δh, R0 + DF/optimizer)
#   ✅ métricas rápidas + confusion matrix (in-sample)
#   ✅ 2D map (mais abaixo)
#   ✅ 3D RÁPIDO DO NUMOPT AQUI (para não precisar ir até o fim)
#
#  Requisitos (Parte 1/4):
#   - df_filtered, y_raw, w, use_group, groups
#   - hansen_distance, red_values, prob_from_red, fit_by_methods
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, confusion_matrix

# -----------------------------
# Pretty axes
# -----------------------------
def _article_axes(ax):
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_alpha(0.6)

# -----------------------------
# Confusion matrix
# -----------------------------
def plot_confusion_matrix_pretty(cm, title="Confusion Matrix", xlabel="Predicted", ylabel="True"):
    fig = plt.figure(figsize=(5.6, 4.8), dpi=160)
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    _article_axes(plt.gca())
    plt.tight_layout()
    return fig

# -----------------------------
# 2D map (δd, δp) with δh fixed at center
# -----------------------------
def plot_numopt_map(df_local, center, R0, unit=None, bins=36, pad=1.0, k_prob=6.0):
    if unit is None:
        unit = st.session_state.get("UNIT", "MPa\u00b9\u2044\u00b2")  # MPa½

    dp, pp, hp = map(float, center)

    dmin = float(df_local["delta_d"].min() - pad)
    dmax = float(df_local["delta_d"].max() + pad)
    pmin = float(df_local["delta_p"].min() - pad)
    pmax = float(df_local["delta_p"].max() + pad)

    xd = np.linspace(dmin, dmax, int(bins))
    xp = np.linspace(pmin, pmax, int(bins))
    DD, PP = np.meshgrid(xd, xp, indexing="xy")
    HH = np.full_like(DD, hp)

    Ra = hansen_distance(DD, PP, HH, dp, pp, hp)
    RED = Ra / max(float(R0), 1e-6)
    P = prob_from_red(RED, k=float(k_prob))

    fig = plt.figure(figsize=(7.2, 5.6), dpi=160)
    im = plt.imshow(P, origin="lower", extent=[dmin, dmax, pmin, pmax], aspect="auto")
    plt.colorbar(im, label="p(RED)")
    plt.scatter(
        df_local["delta_d"], df_local["delta_p"],
        s=18, alpha=0.85, edgecolors="k", linewidths=0.3
    )

    plt.axvline(dp, linestyle="--", linewidth=1.0, alpha=0.6)
    plt.axhline(pp, linestyle="--", linewidth=1.0, alpha=0.6)

    plt.title("Numerical Optimization — 2D map (δh fixed at optimized center)")
    plt.xlabel(f"δd ({unit})")
    plt.ylabel(f"δp ({unit})")
    _article_axes(plt.gca())
    plt.tight_layout()
    return fig

# -----------------------------
# Plotly helpers (NumOpt 3D quick)
# -----------------------------
def sphere_mesh(center, r, nu=72, nv=36):
    cx, cy, cz = map(float, center)
    r = float(r)
    u = np.linspace(0, 2*np.pi, int(nu))
    v = np.linspace(0, np.pi, int(nv))
    uu, vv = np.meshgrid(u, v, indexing="xy")
    xs = r*np.cos(uu)*np.sin(vv) + cx
    ys = r*np.sin(uu)*np.sin(vv) + cy
    zs = r*np.cos(vv) + cz
    return xs, ys, zs

def point_colors_from_yraw(y_raw_local):
    yraw = np.asarray(y_raw_local, float)
    colors = np.where(yraw == 1.0, "blue", np.where(yraw == 0.5, "orange", "red"))
    labels = np.where(yraw == 1.0, "soluble", np.where(yraw == 0.5, "partial", "insoluble"))
    return colors, labels

def _pretty_scene(unit):
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

def _surface_style(base_rgb="rgb(60,110,220)", opacity=0.18):
    return dict(
        opacity=float(opacity),
        showscale=False,
        colorscale=[[0.0, base_rgb], [1.0, base_rgb]],
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

# -----------------------------
# Robust call to fit_by_methods
# -----------------------------
def _call_fit_by_methods(df_local, weights_local, df_name, nms_restarts=1, cobyla_restarts=1):
    try:
        return fit_by_methods(
            df_local=df_local,
            weights_local=weights_local,
            df_name=df_name,
            K_PROB=float(st.session_state.get("K_PROB", 6.0)),
            REG_R0=float(st.session_state.get("REG_R0", 0.05)),
            nms_restarts=int(nms_restarts),
            cobyla_restarts=int(cobyla_restarts),
            speed_profile=st.session_state.get("speed_profile", "full")
        )
    except TypeError:
        return fit_by_methods(
            df_local=df_local,
            weights_local=weights_local,
            df_name=df_name,
            nms_restarts=int(nms_restarts),
            cobyla_restarts=int(cobyla_restarts)
        )

# ============================================================
# TAB 2 — Numerical Optimization (+ 3D quick)
# ============================================================
with tab2:
    st.subheader("Numerical Optimization — Sphere Fit (Solubility Parameters)")

    if df_filtered is None or len(df_filtered) == 0:
        st.info("Go to Tab 1 and load data first.")
        st.stop()

    UNIT_LOCAL = st.session_state.get("UNIT", "MPa\u00b9\u2044\u00b2")

    # Controls
    c1, c2, c3, c4 = st.columns([1.05, 1.05, 1.05, 1.2])
    with c1:
        RUN_NUMOPT = st.checkbox("Run Numerical Optimization", value=True)
        RUN_CM = st.checkbox("Confusion matrix (in-sample)", value=True)
    with c2:
        K_PROB = st.number_input(
            "K_PROB (RED→p steepness)",
            min_value=1.0, max_value=20.0,
            value=float(st.session_state.get("K_PROB", 6.0)),
            step=0.5
        )
        REG_R0 = st.number_input(
            "REG_R0 (radius regularization)",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.get("REG_R0", 0.05)),
            step=0.01
        )
    with c3:
        speed_profile = st.radio("Speed profile", ["fast", "full"], index=1,
                                 help="fast is recommended for quick iteration; full for final runs.")
        st.session_state["speed_profile"] = str(speed_profile)
        show_map = st.checkbox("Show 2D map (placed below)", value=True)
    with c4:
        show_3d_quick = st.checkbox("Show 3D (NumOpt sphere) here", value=True)
        map_bins = st.slider("Map resolution (bins)", min_value=20, max_value=80, value=36, step=2)

    st.session_state["K_PROB"] = float(K_PROB)
    st.session_state["REG_R0"] = float(REG_R0)

    if not RUN_NUMOPT:
        st.stop()

    # Prepare data
    y = df_filtered["solubility"].values.astype(int)
    w_local = np.asarray(w, float)

    # Objective list
    DF_LIST_TO_COMPARE = ["DF_GEOM", "DF_LOGLOSS", "DF_BRIER", "DF_HINGE", "DF_SOFTCOUNT"]

    NMS_RESTARTS = int(st.session_state.get("NMS_RESTARTS", 1))
    COBYLA_RESTARTS = int(st.session_state.get("COBYLA_RESTARTS", 1))

    # Run optimization across DFs
    try:
        with st.spinner("Running Numerical Optimization (comparing objective functions and solvers)..."):
            all_runs = []
            best_overall = None

            for df_name in DF_LIST_TO_COMPARE:
                runs = _call_fit_by_methods(
                    df_local=df_filtered,
                    weights_local=w_local,
                    df_name=df_name,
                    nms_restarts=NMS_RESTARTS,
                    cobyla_restarts=COBYLA_RESTARTS
                )
                if runs is None or runs.empty:
                    continue

                all_runs.append(runs)
                row0 = runs.iloc[0].to_dict()

                if best_overall is None:
                    best_overall = row0
                else:
                    # Paper criterion
                    key_new = (
                        float(row0.get("DF_LOGLOSS", np.inf)),
                        float(row0.get("DF_GEOM", np.inf)),
                        -float(row0.get("AUPRC_unweighted", -np.inf))
                    )
                    key_old = (
                        float(best_overall.get("DF_LOGLOSS", np.inf)),
                        float(best_overall.get("DF_GEOM", np.inf)),
                        -float(best_overall.get("AUPRC_unweighted", -np.inf))
                    )
                    if key_new < key_old:
                        best_overall = row0

            if best_overall is None:
                st.error("No Numerical Optimization results. Check your data/columns.")
                st.stop()

            df_all_runs = pd.concat(all_runs, ignore_index=True) if len(all_runs) else pd.DataFrame()

    except Exception as e:
        st.error("Numerical Optimization failed inside fit_by_methods().")
        st.exception(e)
        st.stop()

    # Best solution
    best_df_name = str(best_overall["DF"])
    best_optimizer = str(best_overall["Método"])
    dp = float(best_overall["delta_d"])
    pp = float(best_overall["delta_p"])
    hp = float(best_overall["delta_h"])
    R0 = float(best_overall["R0"])

    # ✅ Highlight
    st.markdown("### ⭐ Solubility Parameters — Optimized by Numerical Optimization")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("δd (center)", f"{dp:.4f} {UNIT_LOCAL}")
    m2.metric("δp (center)", f"{pp:.4f} {UNIT_LOCAL}")
    m3.metric("δh (center)", f"{hp:.4f} {UNIT_LOCAL}")
    m4.metric("R0 (radius)", f"{R0:.4f} {UNIT_LOCAL}")

    mm1, mm2, mm3 = st.columns([1.4, 1.2, 1.4])
    mm1.metric("Best DF", best_df_name)
    mm2.metric("Optimizer", best_optimizer)
    mm3.metric("CV scheme", "LOGO" if use_group else "LOO")

    # Save for other tabs
    st.session_state["numopt_done"] = True
    st.session_state["best_df_name"] = best_df_name
    st.session_state["best_optimizer"] = best_optimizer
    st.session_state["dp"] = dp
    st.session_state["pp"] = pp
    st.session_state["hp"] = hp
    st.session_state["R0"] = R0
    st.session_state["df_all_runs"] = df_all_runs

    # In-sample probabilities
    RED_all = red_values(df_filtered, (dp, pp, hp, R0))
    p_numopt_in = prob_from_red(RED_all, k=float(K_PROB))

    thr_numopt_in = 0.5
    if len(np.unique(y)) == 2:
        fpr, tpr, thr = roc_curve(y, p_numopt_in)
        j = tpr - fpr
        thr_numopt_in = float(thr[int(np.argmax(j))])

    pred_numopt_in = (p_numopt_in >= thr_numopt_in).astype(int)
    cm_numopt_in = confusion_matrix(y, pred_numopt_in)

    st.session_state["RED_all"] = np.asarray(RED_all, float)
    st.session_state["p_numopt_in"] = np.asarray(p_numopt_in, float)
    st.session_state["thr_numopt_in"] = float(thr_numopt_in)

    # Quick in-sample summary
    cA, cB, cC = st.columns([1.25, 1.0, 1.0])
    with cA:
        st.markdown("#### In-sample summary (Numerical Optimization)")
        if len(np.unique(y)) == 2:
            st.metric("AUC-ROC", f"{roc_auc_score(y, p_numopt_in):.3f}")
            st.metric("AUC-PR", f"{average_precision_score(y, p_numopt_in):.3f}")
        st.metric("thr (Youden J)", f"{thr_numopt_in:.3f}")
    with cB:
        st.metric("Positives (bin)", f"{int(np.sum(y==1))}")
        st.metric("Negatives", f"{int(np.sum(y==0))}")
    with cC:
        st.metric("Partial (0.5)", f"{int(np.sum(np.asarray(y_raw, float)==0.5))}")
        st.metric("N total", f"{int(len(y))}")

    # Confusion matrix
    if RUN_CM:
        fig_cm = plot_confusion_matrix_pretty(
            cm_numopt_in,
            title=f"Numerical Optimization — Confusion Matrix (in-sample, thr={thr_numopt_in:.3f})",
            xlabel="Predicted label",
            ylabel="True label"
        )
        st.pyplot(fig_cm, clear_figure=True)

    # ✅ 3D QUICK NUMOPT HERE
    if show_3d_quick:
        st.markdown("---")
        st.markdown("## 3D (Quick) — Numerical Optimization sphere (RED=1)")
        st.caption("This is the NumOpt sphere only (fast preview). Full ML+overlay is in Tab 5.")

        colors, labels = point_colors_from_yraw(y_raw)
        xs_h, ys_h, zs_h = sphere_mesh((dp, pp, hp), R0)

        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(
            x=xs_h, y=ys_h, z=zs_h,
            name="Numerical Optimization (RED=1)",
            legendgroup="opti",
            **_surface_style("rgb(60,110,220)", opacity=0.18)
        ))
        fig3d.add_trace(go.Scatter3d(
            x=df_filtered["delta_d"], y=df_filtered["delta_p"], z=df_filtered["delta_h"],
            mode="markers",
            marker=_points_style(colors),
            text=labels,
            name="Samples",
            legendgroup="pts"
        ))
        fig3d.update_layout(
            title=f"NumOpt sphere (RED=1) | DF={best_df_name} | Optimizer={best_optimizer}",
            width=1100, height=780,
            margin=dict(l=10, r=10, t=70, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
        )
        fig3d.update_layout(scene=_pretty_scene(UNIT_LOCAL))
        st.plotly_chart(fig3d, use_container_width=True)

    # All runs
    with st.expander("Numerical Optimization — all runs (ranked)"):
        st.dataframe(df_all_runs, use_container_width=True)

    # 2D map placed below
    if show_map:
        st.markdown("---")
        st.markdown("## 2D Map (placed below) — Numerical Optimization")
        st.caption("Map in (δd, δp) with δh fixed at the optimized center. Colors represent p(RED).")
        fig_map = plot_numopt_map(
            df_filtered,
            center=(dp, pp, hp),
            R0=R0,
            unit=UNIT_LOCAL,
            bins=int(map_bins),
            pad=1.0,
            k_prob=float(K_PROB)
        )
        st.pyplot(fig_map, clear_figure=True)

# ============================================================
# APP.PY — PARTE 3/4 (TAB 3 + TAB 4) — CORRIGIDA ✅
#  ✅ Recoloca BARRA DE PROGRESSO + ETA (tempo estimado)
#  ✅ Mantém CACHE, mas com UI viva (cache por-modelo, não no loop inteiro)
#  ✅ Corrige o erro do CV com XGBoost:
#       NÃO usa float(...) em array -> usa atribuição vetorial p[te_idx] = ...
#  ✅ Evita crash em average_precision_score por inconsistência de shapes
#
#  Requisitos (Partes 1/4 e 2/4):
#   - df_filtered, y_raw, w, use_group, groups
#   - st.session_state["numopt_done"], dp, pp, hp, R0, p_numopt_in, thr_numopt_in
#   - funç: hansen_distance, red_values, prob_from_red, fit_by_methods
# ============================================================

import time
from time import perf_counter
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# XGBoost (se não existir no cloud, trate no make_base_models)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# ------------------------------------------------------------
# Plot helpers (matplotlib -> streamlit)
# ------------------------------------------------------------
def _article_axes(ax):
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_alpha(0.6)

def plot_roc_pretty(y_true, curves, title="ROC Curve", subtitle=None):
    if len(np.unique(y_true)) < 2:
        return None
    fig = plt.figure(figsize=(6.6, 5.2), dpi=160)
    from sklearn.metrics import roc_curve, auc
    ax = fig.gca()
    for c in curves:
        fpr, tpr, _ = roc_curve(y_true, c["p"])
        ax.plot(fpr, tpr, label=f'{c["label"]} (AUC={auc(fpr,tpr):.2f})', lw=c.get("lw", 1.6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, alpha=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.legend()
    _article_axes(ax)
    fig.tight_layout()
    return fig

def plot_pr_pretty(y_true, curves, title="Precision–Recall Curve", subtitle=None):
    if len(np.unique(y_true)) < 2:
        return None
    fig = plt.figure(figsize=(6.6, 5.2), dpi=160)
    from sklearn.metrics import precision_recall_curve, average_precision_score
    ax = fig.gca()
    for c in curves:
        prec, rec, _ = precision_recall_curve(y_true, c["p"])
        ap = average_precision_score(y_true, c["p"])
        ax.plot(rec, prec, label=f'{c["label"]} (AP={ap:.2f})', lw=c.get("lw", 1.6))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.legend()
    _article_axes(ax)
    fig.tight_layout()
    return fig

def plot_calibration_pretty(y_true, probas_dict, title="Calibration (Reliability) Plot", subtitle=None, n_bins=10):
    if len(np.unique(y_true)) < 2:
        return None
    fig = plt.figure(figsize=(6.6, 5.2), dpi=160)
    ax = fig.gca()
    for label, p in probas_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.4, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, alpha=0.8)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.legend()
    _article_axes(ax)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# ML builders + calibration
# ------------------------------------------------------------
CALIBRATION_METHOD_HIGHN = "isotonic"
CALIBRATION_METHOD_LOWN  = "sigmoid"

def make_base_models(random_state=42, base_score=None):
    models = {}

    # XGBoost (se disponível)
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=450, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=random_state,
            eval_metric="logloss",
            base_score=float(base_score) if base_score is not None else 0.5
        )

    models["RandomForest"] = RandomForestClassifier(n_estimators=700, random_state=random_state)
    models["SVM-RBF"] = SVC(C=2.0, gamma="scale", probability=True, random_state=random_state)
    return models

def calibrate_model(base_estimator, X_tr, y_tr, w_tr, min_n_iso=60):
    method = CALIBRATION_METHOD_HIGHN if len(y_tr) >= int(min_n_iso) else CALIBRATION_METHOD_LOWN
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
        # fallback: uncalibrated
        try:
            base_estimator.fit(X_tr, y_tr, sample_weight=w_tr)
        except Exception:
            base_estimator.fit(X_tr, y_tr)
        return base_estimator


# ------------------------------------------------------------
# Helpers: ranking seguro (evita crash se shape inconsistente)
# ------------------------------------------------------------
def safe_ap(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float).ravel()
    if len(p) != len(y) or len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


# ------------------------------------------------------------
# CACHE POR MODELO (permite barra/ETA no loop)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fit_predict_one_model_cached(model_name: str, X_csv: str, y_tuple: tuple, w_tuple: tuple, min_n_iso: int, base_score: float):
    import io
    X = pd.read_csv(io.StringIO(X_csv)).values
    y = np.asarray(y_tuple, int)
    w = np.asarray(w_tuple, float)

    base = make_base_models(42, base_score=base_score)
    if model_name not in base:
        return None

    fitted = calibrate_model(base[model_name], X, y, w, min_n_iso=int(min_n_iso))
    if hasattr(fitted, "predict_proba"):
        p = fitted.predict_proba(X)[:, 1]
        return np.asarray(p, float).ravel()
    return None


# ============================================================
# TAB 3 — ML (Calibrated) IN-SAMPLE  ✅ barra + ETA
# ============================================================
with tab3:
    st.subheader("ML Models (Calibrated) — In-sample")

    if not st.session_state.get("numopt_done", False):
        st.info("Run Tab 2 first (Numerical Optimization).")
        st.stop()

    y = df_filtered["solubility"].values.astype(int)
    X = df_filtered[["delta_d","delta_p","delta_h"]].values
    w_local = np.asarray(w, float)

    p_numopt_in = st.session_state.get("p_numopt_in", None)
    if p_numopt_in is None:
        st.info("Numerical Optimization probabilities not found. Run Tab 2 again.")
        st.stop()

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
    with c1:
        RUN_ROC = st.checkbox("ROC plot", value=True)
        RUN_PR = st.checkbox("PR plot", value=True)
        RUN_CAL = st.checkbox("Calibration plot", value=True)
    with c2:
        min_n_iso = st.number_input(
            "Min N for isotonic",
            min_value=20, max_value=300,
            value=int(st.session_state.get("CALIB_MIN_N_ISO", 60)),
            step=5
        )
    with c3:
        avail = list(make_base_models().keys())
        keep_models = st.multiselect(
            "Models",
            avail,
            default=avail
        )
        topN = st.number_input("Top-N ML lines in plots", min_value=1, max_value=5, value=2, step=1)

    run_ml_btn = st.button("▶ Train + Calibrate ML (in-sample)", type="primary", use_container_width=True)

    if run_ml_btn:
        if len(keep_models) == 0:
            st.error("Select at least one ML model.")
            st.stop()

        # prepara dados para cache
        X_csv = pd.DataFrame(X).to_csv(index=False)
        y_tuple = tuple(y.tolist())
        w_tuple = tuple(w_local.tolist())

        # base_score ponderado
        p0_in = float(np.clip(np.average(y, weights=w_local), 1e-12, 1-1e-12))

        st.caption("Treinando (com cache por modelo) — barra + ETA em tempo real.")
        prog = st.progress(0.0, text="Inicializando...")
        eta_box = st.empty()

        proba_ml_in = {}
        times = []
        t0_all = perf_counter()

        for i, name in enumerate(keep_models, start=1):
            t0 = perf_counter()
            p = fit_predict_one_model_cached(
                model_name=str(name),
                X_csv=X_csv,
                y_tuple=y_tuple,
                w_tuple=w_tuple,
                min_n_iso=int(min_n_iso),
                base_score=float(p0_in),
            )
            dt = perf_counter() - t0
            times.append(dt)

            if p is not None and len(p) == len(y):
                proba_ml_in[name] = p

            frac = i / len(keep_models)
            avg_t = float(np.mean(times)) if len(times) else 0.0
            remaining = max(0, len(keep_models) - i)
            eta_s = remaining * avg_t
            elapsed = perf_counter() - t0_all

            prog.progress(frac, text=f"Treinando {name} ({i}/{len(keep_models)}) — {dt:.1f}s")
            eta_box.info(f"⏱️ Tempo decorrido: {elapsed:.1f}s | ETA: {eta_s:.1f}s | Média/modelo: {avg_t:.1f}s")

        prog.empty()
        eta_box.empty()

        st.session_state["proba_ml_in"] = proba_ml_in
        st.session_state["CALIB_MIN_N_ISO"] = int(min_n_iso)

        if not proba_ml_in:
            st.error("Nenhum modelo ML retornou probabilidade (verifique dependências/labels).")
            st.stop()

        st.success(f"ML in-sample cached: {list(proba_ml_in.keys())}. Proceed to Tab 4 for CV.")

    proba_ml_in = st.session_state.get("proba_ml_in", {})
    if not proba_ml_in:
        st.info("No ML cached yet. Click the button above.")
        st.stop()

    # rank por AUPRC (seguro)
    ranked = []
    if len(np.unique(y)) == 2:
        for name, p in proba_ml_in.items():
            ap = safe_ap(y, p)
            if ap == ap:
                ranked.append((name, ap))
        ranked = sorted(ranked, key=lambda t: -t[1])
    else:
        ranked = [(k, 0.0) for k in proba_ml_in.keys()]

    keep_for_plot = [n for n, _ in ranked[: int(topN)]] if ranked else list(proba_ml_in.keys())[: int(topN)]

    st.write("Models cached:", list(proba_ml_in.keys()))
    st.write("Shown in plots:", keep_for_plot)

    if len(np.unique(y)) == 2 and RUN_ROC:
        curves = [{"label": "Numerical Optimization", "p": np.asarray(p_numopt_in, float).ravel(), "lw": 2.2}]
        for name in keep_for_plot:
            curves.append({"label": f"{name} (cal)", "p": np.asarray(proba_ml_in[name], float).ravel(), "lw": 1.5})
        fig = plot_roc_pretty(y, curves, title="ROC Curve", subtitle="In-sample (calibrated ML)")
        if fig: st.pyplot(fig, clear_figure=True)

    if len(np.unique(y)) == 2 and RUN_PR:
        curves = [{"label": "Numerical Optimization", "p": np.asarray(p_numopt_in, float).ravel(), "lw": 2.2}]
        for name in keep_for_plot:
            curves.append({"label": f"{name} (cal)", "p": np.asarray(proba_ml_in[name], float).ravel(), "lw": 1.5})
        fig = plot_pr_pretty(y, curves, title="Precision–Recall Curve", subtitle="In-sample (calibrated ML)")
        if fig: st.pyplot(fig, clear_figure=True)

    if len(np.unique(y)) == 2 and RUN_CAL:
        calib_dict = {"Numerical Optimization": np.asarray(p_numopt_in, float).ravel()}
        for name in keep_for_plot:
            calib_dict[f"{name} (cal)"] = np.asarray(proba_ml_in[name], float).ravel()
        fig = plot_calibration_pretty(y, calib_dict, title="Calibration (Reliability) Plot", subtitle="In-sample", n_bins=10)
        if fig: st.pyplot(fig, clear_figure=True)


# ============================================================
# TAB 4 — CV (LOGO/LOO) — Out-of-fold probabilities  ✅ corrige XGBoost
# ============================================================
with tab4:
    st.subheader("Cross-Validation (LOGO/LOO) — Out-of-fold probabilities")

    st.session_state.setdefault("p_numopt_cv", None)
    st.session_state.setdefault("p_ml_cv", None)
    st.session_state.setdefault("cv_label", None)
    st.session_state.setdefault("cv_last_error", "")

    if not st.session_state.get("numopt_done", False):
        st.info("Run Tab 2 first.")
        st.stop()

    if "proba_ml_in" not in st.session_state:
        st.info("Run Tab 3 first (ML in-sample).")
        st.stop()

    y = df_filtered["solubility"].values.astype(int)
    X = df_filtered[["delta_d","delta_p","delta_h"]].values
    w_local = np.asarray(w, float)

    best_df_name = st.session_state["best_df_name"]
    dp = float(st.session_state["dp"])
    pp = float(st.session_state["pp"])
    hp = float(st.session_state["hp"])
    R0 = float(st.session_state["R0"])
    K_PROB = float(st.session_state.get("K_PROB", 6.0))
    REG_R0 = float(st.session_state.get("REG_R0", 0.05))
    min_n_iso = int(st.session_state.get("CALIB_MIN_N_ISO", 60))

    proba_ml_in = st.session_state.get("proba_ml_in", {})
    model_names = list(proba_ml_in.keys())
    if len(model_names) == 0:
        st.error("No ML models available (did Tab 3 run successfully?).")
        st.stop()

    c1, c2, c3, c4 = st.columns([1.0, 1.05, 1.35, 1.0])
    with c1:
        thr_numopt_cv = st.number_input("NumOpt thr (CV)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
        thr_ml = st.number_input("ML thr (CV)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
    with c2:
        cv_mode = st.radio(
            "NumOpt in CV",
            ["Fast (global params only)", "Paper (refit per fold)"],
            index=0,
            help="Paper mode re-optimizes Numerical Optimization per fold (very slow)."
        )
        if cv_mode == "Paper (refit per fold)":
            st.warning("Paper mode is slow: it refits Numerical Optimization per fold.")
    with c3:
        max_folds = st.number_input(
            "Max folds (debug)",
            min_value=5, max_value=5000, value=200, step=25,
            help="Limits folds for quick tests. Increase for full CV."
        )
        cv_models = st.multiselect(
            "Models to run in CV",
            options=model_names,
            default=[st.session_state.get("best_ml_name", model_names[0])],
            help="CV cost scales with #models."
        )
        topN_cv = st.number_input("Top-N ML lines in CV plots", min_value=1, max_value=5, value=2, step=1)
        show_plots_cv = st.checkbox("Show CV ROC/PR/Calib plots", value=True)
    with c4:
        cv_nms = st.number_input("CV: NM restarts", min_value=0, max_value=2, value=0, step=1)
        cv_cobyla = st.number_input("CV: COBYLA restarts", min_value=0, max_value=2, value=0, step=1)

    if len(cv_models) == 0:
        st.warning("Select at least one model for CV.")
        st.stop()

    run_cv_btn = st.button("▶ Run CV (LOGO/LOO)", type="primary", use_container_width=True)

    if run_cv_btn:
        st.session_state["cv_last_error"] = ""
        try:
            # Splitter
            if use_group:
                splitter = LeaveOneGroupOut()
                splits = list(splitter.split(X, y, groups=groups))
                cv_label = "LOGO"
            else:
                splitter = LeaveOneOut()
                splits = list(splitter.split(X, y))
                cv_label = "LOO"

            if len(splits) == 0:
                raise RuntimeError("No CV splits generated (check groups/labels).")

            splits = splits[: int(max_folds)]

            p_numopt_cv = np.zeros(len(y), dtype=float)
            p_ml_cv = {name: np.zeros(len(y), dtype=float) for name in cv_models}

            prog = st.progress(0.0, text=f"Inicializando CV ({cv_label})...")
            eta_box = st.empty()
            times = []
            t0_all = perf_counter()

            with st.spinner(f"Running {cv_label} CV... folds={len(splits)}"):
                for i, (tr_idx, te_idx) in enumerate(splits, start=1):
                    t0 = perf_counter()

                    df_tr = df_filtered.iloc[tr_idx].reset_index(drop=True)
                    df_te = df_filtered.iloc[te_idx].reset_index(drop=True)

                    w_tr = w_local[tr_idx]
                    y_tr = df_tr["solubility"].values.astype(int)

                    # -------- NumOpt in CV --------
                    if cv_mode == "Fast (global params only)":
                        pars_cv = np.array([dp, pp, hp, R0], float)
                    else:
                        runs_cv = fit_by_methods(
                            df_local=df_tr,
                            weights_local=w_tr,
                            df_name=best_df_name,
                            K_PROB=float(K_PROB),
                            REG_R0=float(REG_R0),
                            nms_restarts=int(cv_nms),
                            cobyla_restarts=int(cv_cobyla),
                            speed_profile="fast"
                        )
                        if runs_cv is None or runs_cv.empty:
                            pars_cv = np.array([dp, pp, hp, R0], float)
                        else:
                            r0 = runs_cv.iloc[0]
                            pars_cv = np.array([r0["delta_d"], r0["delta_p"], r0["delta_h"], r0["R0"]], float)

                    RED_te = red_values(df_te, pars_cv)
                    p_numopt_cv[te_idx] = prob_from_red(RED_te, k=float(K_PROB)).ravel()

                    # -------- ML calibrated per fold --------
                    X_tr = df_tr[["delta_d","delta_p","delta_h"]].values
                    X_te = df_te[["delta_d","delta_p","delta_h"]].values

                    # guard: single-class fold
                    if len(np.unique(y_tr)) < 2:
                        prev = float(np.clip(np.average(y, weights=w_local), 1e-12, 1-1e-12))
                        for name in cv_models:
                            p_ml_cv[name][te_idx] = prev
                    else:
                        p0_fold = float(np.clip(np.average(y_tr, weights=w_tr), 1e-12, 1-1e-12))
                        fold_models = make_base_models(42, base_score=p0_fold)

                        for name in cv_models:
                            est = fold_models.get(name, None)
                            if est is None:
                                p_ml_cv[name][te_idx] = p0_fold
                                continue
                            fitted = calibrate_model(est, X_tr, y_tr, w_tr, min_n_iso=int(min_n_iso))

                            if hasattr(fitted, "predict_proba"):
                                p_te = np.asarray(fitted.predict_proba(X_te)[:, 1], float).ravel()
                                # ✅ CORREÇÃO CRÍTICA: atribuição vetorial (não float)
                                p_ml_cv[name][te_idx] = p_te
                            else:
                                p_ml_cv[name][te_idx] = p0_fold

                    dt = perf_counter() - t0
                    times.append(dt)
                    frac = i / len(splits)
                    avg_t = float(np.mean(times)) if times else 0.0
                    eta_s = max(0.0, (len(splits) - i) * avg_t)
                    elapsed = perf_counter() - t0_all
                    prog.progress(frac, text=f"CV {cv_label}: fold {i}/{len(splits)} — {dt:.2f}s")
                    eta_box.info(f"⏱️ Decorrido: {elapsed:.1f}s | ETA: {eta_s:.1f}s | Média/fold: {avg_t:.2f}s")

            prog.empty()
            eta_box.empty()

            # cache only after success
            st.session_state["p_numopt_cv"] = p_numopt_cv
            st.session_state["p_ml_cv"] = p_ml_cv
            st.session_state["cv_label"] = cv_label
            st.session_state["thr_numopt_cv"] = float(thr_numopt_cv)
            st.session_state["thr_ml"] = float(thr_ml)
            st.session_state["cv_mode_numopt"] = str(cv_mode)
            st.session_state["cv_models"] = list(cv_models)
            st.session_state["max_folds"] = int(max_folds)

            st.success(f"{cv_label} CV done and cached. (folds used: {len(splits)})")

        except Exception as e:
            st.session_state["cv_last_error"] = repr(e)
            st.error("CV failed before caching results.")
            st.exception(e)

    # ---- Display cached CV ----
    if st.session_state.get("p_numopt_cv", None) is None:
        st.info("No CV cached yet. Click Run.")
        if st.session_state.get("cv_last_error", ""):
            st.warning(f"Last CV error: {st.session_state['cv_last_error']}")
        st.stop()

    cv_label = st.session_state["cv_label"]
    p_numopt_cv = np.asarray(st.session_state["p_numopt_cv"], float).ravel()
    p_ml_cv = st.session_state["p_ml_cv"]
    best_ml_name = st.session_state.get("best_ml_name", model_names[0])

    st.markdown(f"### Cached CV ({cv_label})")
    st.write(f"CV models: {st.session_state.get('cv_models', [])} | Max folds used: {st.session_state.get('max_folds', '')}")

    # rank por AUPRC (seguro)
    ranked_cv = []
    if len(np.unique(y)) == 2:
        for name, p in p_ml_cv.items():
            ap = safe_ap(y, p)
            if ap == ap:
                ranked_cv.append((name, ap))
        ranked_cv = sorted(ranked_cv, key=lambda t: -t[1])
    else:
        ranked_cv = [(k, 0.0) for k in p_ml_cv.keys()]

    keep_cv_plot = [n for n, _ in ranked_cv[: int(topN_cv)]] if ranked_cv else list(p_ml_cv.keys())[: int(topN_cv)]
    if ranked_cv:
        st.write("Top models by AUPRC:", ranked_cv[: min(5, len(ranked_cv))])

    if show_plots_cv and len(np.unique(y)) == 2:
        curves = [{"label": f"Numerical Optimization ({cv_label})", "p": p_numopt_cv, "lw": 2.2}]
        for name in keep_cv_plot:
            curves.append({"label": f"{name} ({cv_label}, cal)", "p": np.asarray(p_ml_cv[name], float).ravel(), "lw": 1.5})

        fig = plot_roc_pretty(y, curves, title="ROC Curve", subtitle=f"Cross-validation ({cv_label})")
        if fig: st.pyplot(fig, clear_figure=True)

        fig = plot_pr_pretty(y, curves, title="Precision–Recall Curve", subtitle=f"Cross-validation ({cv_label})")
        if fig: st.pyplot(fig, clear_figure=True)

        calib_dict = {f"Numerical Optimization ({cv_label})": p_numopt_cv}
        for name in keep_cv_plot:
            calib_dict[f"{name} ({cv_label}, cal)"] = np.asarray(p_ml_cv[name], float).ravel()
        fig = plot_calibration_pretty(y, calib_dict, title="Calibration (Reliability) Plot",
                                      subtitle=f"Cross-validation ({cv_label})", n_bins=10)
        if fig: st.pyplot(fig, clear_figure=True)

# ============================================================
# APP.PY — PARTE 4/4 (TAB 5 + TAB 6) — COMPLETA (6 ABAS)
#  Tab 5: 3D Plotly — NumOpt sphere (RED=1) + ML shell/volume/quantile sphere + overlay
#  Tab 6: Export Excel — base + resultados por amostra + métricas + configs + metadados
#
#  ✅ CORREÇÃO-CHAVE (seu erro):
#   - A extração ML agora é ROBUSTA: se não existir p>=iso no grid (shell pts=0),
#     cai automaticamente em "quantile fallback" (top-q) e SEMPRE extrai uma esfera.
#   - Inclui "mode" na tabela ML_ShellSpheres: shell | volume | quantile(q=..)
#
#  ✅ Mantém:
#   - 6 abas
#   - Tons distintos (NumOpt azul, ML roxo)
#   - Export completo (inclui ML_ShellSpheres se existir)
#
#  REQUISITOS (Partes 1/4–3/4):
#   - df_filtered, y_raw, w, use_group, groups
#   - st.session_state["numopt_done"], dp, pp, hp, R0, best_df_name, best_optimizer
#   - funç: red_values, prob_from_red (NumOpt)
#   - funç: make_base_models, calibrate_model (da Parte 3/4)
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import io

# -----------------------------
# Helpers gerais
# -----------------------------
def _get_unit():
    return st.session_state.get("UNIT", "MPa\u00b9\u2044\u00b2")  # MPa½

def point_colors_from_yraw(y_raw_local):
    yraw = np.asarray(y_raw_local, float)
    colors = np.where(yraw == 1.0, "blue", np.where(yraw == 0.5, "orange", "red"))
    labels = np.where(yraw == 1.0, "soluble", np.where(yraw == 0.5, "partial", "insoluble"))
    return colors, labels

def fit_sphere_least_squares(P):
    P = np.asarray(P, float)
    if P.shape[0] < 10:
        raise ValueError("Too few points to fit a sphere (>=10).")
    x, y_, z = P[:, 0], P[:, 1], P[:, 2]
    A = np.c_[2*x, 2*y_, 2*z, np.ones_like(x)]
    b = x**2 + y_**2 + z**2
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz, c0 = beta
    r2 = cx**2 + cy**2 + cz**2 + c0
    r2 = max(float(r2), 1e-12)
    r = float(np.sqrt(r2))
    return (float(cx), float(cy), float(cz)), r

def sphere_mesh(center, r, nu=80, nv=40):
    cx, cy, cz = map(float, center)
    r = float(r)
    u = np.linspace(0, 2*np.pi, int(nu))
    v = np.linspace(0, np.pi, int(nv))
    uu, vv = np.meshgrid(u, v, indexing="xy")
    xs = r*np.cos(uu)*np.sin(vv) + cx
    ys = r*np.sin(uu)*np.sin(vv) + cy
    zs = r*np.cos(vv) + cz
    return xs, ys, zs

# -----------------------------
# Extração robusta da esfera ML (corrige "shell pts=0")
# -----------------------------
def ml_shell_sphere_from_grid(
    df_local, model,
    iso=0.5,
    pad=1.5,
    n=28,
    shell_delta=0.05,
    min_points=120,
    auto_iso=True,
    q_fallback=0.95,
    min_points_frac=0.010,
    max_relax_steps=10
):
    """
    Robust ML sphere extraction:
      1) tries shell: iso <= p <= iso+delta (relax delta / iso)
      2) tries volume: p >= iso
      3) if still empty and auto_iso=True: uses quantile fallback (top-q probabilities)
    Returns:
      (center, r, npts, iso_used, delta_used, mode)
    mode: "shell" | "volume" | "quantile(q=..)"
    """

    mn = df_local[["delta_d", "delta_p", "delta_h"]].min().values.astype(float) - float(pad)
    mx = df_local[["delta_d", "delta_p", "delta_h"]].max().values.astype(float) + float(pad)

    xd = np.linspace(mn[0], mx[0], int(n))
    xp = np.linspace(mn[1], mx[1], int(n))
    xh = np.linspace(mn[2], mx[2], int(n))

    Xd, Xp, Xh = np.meshgrid(xd, xp, xh, indexing="ij")
    pts = np.c_[Xd.ravel(), Xp.ravel(), Xh.ravel()]

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model has no predict_proba(). Ensure calibration returns a proba-capable estimator.")

    p = np.clip(model.predict_proba(pts)[:, 1], 1e-12, 1 - 1e-12)

    # min_points adaptativo ao grid
    gridN = int(pts.shape[0])
    min_pts_auto = max(30, int(min_points_frac * gridN))
    min_points_use = int(max(min_points, min_pts_auto))

    iso_use = float(iso)
    delta = float(shell_delta)

    # 1) shell com relax
    for _ in range(int(max_relax_steps)):
        mask = (p >= iso_use) & (p <= min(1.0, iso_use + delta))
        Psel = pts[mask]
        if Psel.shape[0] >= int(min_points_use):
            center, r = fit_sphere_least_squares(Psel)
            return center, r, int(Psel.shape[0]), float(iso_use), float(delta), "shell"

        delta = min(0.35, delta + 0.04)
        iso_use = max(0.01, iso_use - 0.05)

    # 2) volume p >= iso (alvo)
    mask = (p >= float(iso))
    Psel = pts[mask]
    if Psel.shape[0] >= 10:
        center, r = fit_sphere_least_squares(Psel)
        return center, r, int(Psel.shape[0]), float(iso), -1.0, "volume"

    # 3) quantile fallback (garante pontos)
    if auto_iso:
        q = float(q_fallback)
        q = min(max(q, 0.50), 0.995)
        thr_q = float(np.quantile(p, q))
        mask = (p >= thr_q)
        Psel = pts[mask]

        tries = 0
        while Psel.shape[0] < 10 and tries < 8:
            q = max(0.50, q - 0.05)
            thr_q = float(np.quantile(p, q))
            mask = (p >= thr_q)
            Psel = pts[mask]
            tries += 1

        if Psel.shape[0] < 10:
            raise ValueError(
                "Not enough grid points even after quantile fallback. "
                "Model may be nearly constant on the grid. "
                "Try increasing GRID_PAD further or check ML training."
            )

        center, r = fit_sphere_least_squares(Psel)
        return center, r, int(Psel.shape[0]), float(thr_q), -2.0, f"quantile(q={q:.2f})"

    raise ValueError(
        f"Not enough grid points for shell or volume. "
        f"Try increasing GRID_N and GRID_PAD or using lower ISO_LEVELS. "
        f"(GRID_N={n}, GRID_PAD={pad})"
    )

# -----------------------------
# Plotly style (tons distintos)
# -----------------------------
OPTI_COLOR = "rgb(60,110,220)"   # NumOpt (azul)
ML_COLOR   = "rgb(140,80,200)"   # ML (roxo)

def _pretty_scene(unit=None):
    if unit is None:
        unit = _get_unit()
    return dict(
        xaxis=dict(
            title=f"δd ({unit})", showbackground=True, backgroundcolor="rgba(245,245,245,1)",
            gridcolor="rgba(180,180,180,0.35)", zerolinecolor="rgba(120,120,120,0.25)"
        ),
        yaxis=dict(
            title=f"δp ({unit})", showbackground=True, backgroundcolor="rgba(245,245,245,1)",
            gridcolor="rgba(180,180,180,0.35)", zerolinecolor="rgba(120,120,120,0.25)"
        ),
        zaxis=dict(
            title=f"δh ({unit})", showbackground=True, backgroundcolor="rgba(245,245,245,1)",
            gridcolor="rgba(180,180,180,0.35)", zerolinecolor="rgba(120,120,120,0.25)"
        ),
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

def _surface_style(base_rgb, opacity=0.16):
    return dict(
        opacity=float(opacity),
        showscale=False,
        colorscale=[[0.0, base_rgb], [1.0, base_rgb]],
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

# ============================================================
# TAB 5 — 3D
# ============================================================
with tab5:
    st.subheader("3D (Plotly): Numerical Optimization sphere + ML sphere + overlay")

    if not st.session_state.get("numopt_done", False):
        st.info("Run Tab 2 first (Numerical Optimization).")
        st.stop()

    unit = _get_unit()

    dp = float(st.session_state["dp"])
    pp = float(st.session_state["pp"])
    hp = float(st.session_state["hp"])
    R0 = float(st.session_state["R0"])
    best_df_name = st.session_state.get("best_df_name", "")
    best_optimizer = st.session_state.get("best_optimizer", "")

    # highlight NumOpt
    st.markdown("### ⭐ Solubility Parameters — Numerical Optimization (Optimized)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("δd (center)", f"{dp:.4f} {unit}")
    m2.metric("δp (center)", f"{pp:.4f} {unit}")
    m3.metric("δh (center)", f"{hp:.4f} {unit}")
    m4.metric("R0 (radius)", f"{R0:.4f} {unit}")
    mm1, mm2, mm3 = st.columns([1.4, 1.2, 1.4])
    mm1.metric("Best DF", str(best_df_name))
    mm2.metric("Optimizer", str(best_optimizer))
    mm3.metric("CV scheme", "LOGO" if use_group else "LOO")

    # controls
    c1, c2, c3 = st.columns([1.1, 1.0, 1.1])
    with c1:
        RUN_3D = st.checkbox("Enable 3D plots", value=True)
        show_points = st.checkbox("Show sample points", value=True)
        auto_iso = st.checkbox("Auto-ISO fallback (quantile)", value=True,
                               help="If no points reach iso, extract sphere from top-q probability region.")
    with c2:
        ISO_LEVELS = st.multiselect(
            "ISO_LEVELS (p≈iso)",
            options=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90],
            default=[0.50]
        )
        if len(ISO_LEVELS) == 0:
            ISO_LEVELS = [0.50]
        Q_FALLBACK = st.slider("Quantile fallback (top-q)", 0.70, 0.99, float(st.session_state.get("Q_FALLBACK", 0.95)), 0.01)
    with c3:
        GRID_PAD = st.slider("GRID_PAD", 0.5, 4.0, float(st.session_state.get("GRID_PAD", 1.8)), 0.1)
        GRID_N = st.slider("GRID_N", 16, 44, int(st.session_state.get("GRID_N", 28)), 1)
        SHELL_DELTA = st.slider("SHELL_DELTA", 0.01, 0.35, float(st.session_state.get("SHELL_DELTA", 0.08)), 0.01)
        MIN_POINTS = st.number_input("MIN_POINTS (shell)", min_value=10, max_value=2000, value=int(st.session_state.get("MIN_POINTS", 120)), step=10)

    st.session_state["GRID_PAD"] = float(GRID_PAD)
    st.session_state["GRID_N"] = int(GRID_N)
    st.session_state["ISO_LEVELS"] = [float(x) for x in ISO_LEVELS]
    st.session_state["SHELL_DELTA"] = float(SHELL_DELTA)
    st.session_state["MIN_POINTS"] = int(MIN_POINTS)
    st.session_state["Q_FALLBACK"] = float(Q_FALLBACK)

    if not RUN_3D:
        st.stop()

    # Need ML config
    if ("proba_ml_in" not in st.session_state) and ("best_ml_name" not in st.session_state):
        st.info("Run Tab 3 (ML in-sample) or Tab 4 (CV) to define ML model for 3D.")
        st.stop()

    best_ml_name = st.session_state.get("best_ml_name", None)
    if best_ml_name is None:
        proba_ml_in = st.session_state.get("proba_ml_in", {})
        best_ml_name = list(proba_ml_in.keys())[0] if len(proba_ml_in) else "XGBoost"
        st.session_state["best_ml_name"] = best_ml_name

    # Build calibrated ML model on full data
    y_bin = df_filtered["solubility"].values.astype(int)
    X_all = df_filtered[["delta_d","delta_p","delta_h"]].values
    w_local = np.asarray(w, float)
    min_n_iso = int(st.session_state.get("CALIB_MIN_N_ISO", 60))

    p0_all = float(np.clip(np.average(y_bin, weights=w_local), 1e-12, 1-1e-12))
    full_base = make_base_models(42, base_score=p0_all)
    if best_ml_name not in full_base:
        st.warning(f"3D: Model '{best_ml_name}' not available; using XGBoost.")
        best_ml_name = "XGBoost"
        st.session_state["best_ml_name"] = best_ml_name

    model_for_3d = calibrate_model(full_base[best_ml_name], X_all, y_bin, w_local, min_n_iso=min_n_iso)

    colors, labels = point_colors_from_yraw(y_raw)

    # (A) NumOpt sphere
    xs_h, ys_h, zs_h = sphere_mesh((dp, pp, hp), R0)
    fig_h = go.Figure()
    fig_h.add_trace(go.Surface(
        x=xs_h, y=ys_h, z=zs_h,
        name="Numerical Optimization (RED=1)",
        legendgroup="opti",
        **_surface_style(OPTI_COLOR, opacity=0.18)
    ))
    if show_points:
        fig_h.add_trace(go.Scatter3d(
            x=df_filtered["delta_d"], y=df_filtered["delta_p"], z=df_filtered["delta_h"],
            mode="markers", marker=_points_style(colors),
            text=labels, name="Samples", legendgroup="pts"
        ))
    fig_h.update_layout(**_pretty_layout(
        title=f"3D (A) — NumOpt sphere (RED=1) | DF={best_df_name} | Optimizer={best_optimizer}"
    ))
    fig_h.update_layout(scene=_pretty_scene(unit=unit))
    st.plotly_chart(fig_h, use_container_width=True)

    ml_spheres_rows = []

    # (B)/(C) ML spheres per iso
    for iso in ISO_LEVELS:
        try:
            c_ml, r_ml, npts, iso_used, delta_used, mode_used = ml_shell_sphere_from_grid(
                df_filtered, model_for_3d,
                iso=float(iso),
                pad=float(GRID_PAD),
                n=int(GRID_N),
                shell_delta=float(SHELL_DELTA),
                min_points=int(MIN_POINTS),
                auto_iso=bool(auto_iso),
                q_fallback=float(Q_FALLBACK)
            )

            xs_ml, ys_ml, zs_ml = sphere_mesh(c_ml, r_ml)

            ml_spheres_rows.append({
                "ML_model": str(best_ml_name),
                "iso_target": float(iso),
                "iso_used": float(iso_used),
                "mode": str(mode_used),
                "shell_delta_used": float(delta_used),
                "center_delta_d": float(c_ml[0]),
                "center_delta_p": float(c_ml[1]),
                "center_delta_h": float(c_ml[2]),
                "R_ml": float(r_ml),
                "unit": str(unit),
                "n_grid_points_used": int(npts),
                "GRID_PAD": float(GRID_PAD),
                "GRID_N": int(GRID_N),
                "SHELL_DELTA": float(SHELL_DELTA),
                "MIN_POINTS": int(MIN_POINTS),
                "Q_FALLBACK": float(Q_FALLBACK)
            })

            # (B) ML only
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Surface(
                x=xs_ml, y=ys_ml, z=zs_ml,
                name=f"{best_ml_name} sphere ({mode_used})",
                legendgroup="ml",
                **_surface_style(ML_COLOR, opacity=0.18)
            ))
            if show_points:
                fig_ml.add_trace(go.Scatter3d(
                    x=df_filtered["delta_d"], y=df_filtered["delta_p"], z=df_filtered["delta_h"],
                    mode="markers", marker=_points_style(colors),
                    text=labels, name="Samples", legendgroup="pts"
                ))

            extra = f" | mode={mode_used}"
            if delta_used > 0:
                extra += f" | shell Δ={delta_used:.2f}"
            elif delta_used == -1.0:
                extra += " | fallback: volume p≥iso"
            elif delta_used == -2.0:
                extra += " | fallback: quantile"

            fig_ml.update_layout(**_pretty_layout(
                title=f"3D (B) — ML sphere | target iso={iso:.2f} | used={iso_used:.3f} | R≈{r_ml:.3f} {unit}{extra}"
            ))
            fig_ml.update_layout(scene=_pretty_scene(unit=unit))
            st.plotly_chart(fig_ml, use_container_width=True)

            # (C) Overlay
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Surface(
                x=xs_h, y=ys_h, z=zs_h,
                name="NumOpt (RED=1)",
                legendgroup="opti",
                **_surface_style(OPTI_COLOR, opacity=0.12)
            ))
            fig_ov.add_trace(go.Surface(
                x=xs_ml, y=ys_ml, z=zs_ml,
                name=f"ML sphere ({mode_used}) — {best_ml_name}",
                legendgroup="ml",
                **_surface_style(ML_COLOR, opacity=0.12)
            ))
            if show_points:
                fig_ov.add_trace(go.Scatter3d(
                    x=df_filtered["delta_d"], y=df_filtered["delta_p"], z=df_filtered["delta_h"],
                    mode="markers", marker=_points_style(colors),
                    text=labels, name="Samples", legendgroup="pts"
                ))
            fig_ov.update_layout(**_pretty_layout(
                title=f"3D (C) — Overlay: NumOpt vs ML | target iso={iso:.2f} | mode={mode_used} | {best_ml_name}"
            ))
            fig_ov.update_layout(scene=_pretty_scene(unit=unit))
            st.plotly_chart(fig_ov, use_container_width=True)

        except Exception as e:
            st.warning(f"[ML-SPHERE] Failed at iso={iso:.2f}: {e}")
            st.info("Dica: habilite Auto-ISO (quantile), aumente GRID_PAD, ou use iso menores (0.10–0.40).")

    df_ml_spheres = pd.DataFrame(ml_spheres_rows)
    st.session_state["df_ml_spheres"] = df_ml_spheres

    if df_ml_spheres.empty:
        st.warning("No ML spheres extracted (all iso levels failed). Try Auto-ISO or lower ISO_LEVELS.")
    else:
        st.markdown("### ⭐ ML Sphere — Fitted Parameters (Highlight)")
        df2 = df_ml_spheres.copy()
        df2["n_grid_points_used"] = pd.to_numeric(df2["n_grid_points_used"], errors="coerce").fillna(0).astype(int)
        best_row = df2.sort_values("n_grid_points_used", ascending=False).iloc[0].to_dict()

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("ML model", str(best_row.get("ML_model","")))
        s2.metric("mode", str(best_row.get("mode","")))
        s3.metric("iso_used", f"{float(best_row.get('iso_used', np.nan)):.3f}")
        s4.metric("grid points", f"{int(best_row.get('n_grid_points_used', 0))}")

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("δd_ml", f"{float(best_row.get('center_delta_d', np.nan)):.4f} {unit}")
        t2.metric("δp_ml", f"{float(best_row.get('center_delta_p', np.nan)):.4f} {unit}")
        t3.metric("δh_ml", f"{float(best_row.get('center_delta_h', np.nan)):.4f} {unit}")
        t4.metric("R_ml", f"{float(best_row.get('R_ml', np.nan)):.4f} {unit}")

        with st.expander("ML spheres table"):
            st.dataframe(df_ml_spheres, use_container_width=True)

# ============================================================
# TAB 6 — Export Excel
# ============================================================
with tab6:
    st.subheader("Export Excel (full)")

    if not st.session_state.get("numopt_done", False):
        st.info("Run Tab 2 first.")
        st.stop()

    unit = _get_unit()

    # NumOpt
    dp = float(st.session_state["dp"])
    pp = float(st.session_state["pp"])
    hp = float(st.session_state["hp"])
    R0 = float(st.session_state["R0"])
    best_df_name = st.session_state.get("best_df_name", "")
    best_optimizer = st.session_state.get("best_optimizer", "")

    K_PROB = float(st.session_state.get("K_PROB", 6.0))
    REG_R0 = float(st.session_state.get("REG_R0", 0.05))

    p_numopt_in = st.session_state.get("p_numopt_in", None)
    RED_all = st.session_state.get("RED_all", None)
    thr_numopt_in = float(st.session_state.get("thr_numopt_in", 0.5))

    # CV
    p_numopt_cv = st.session_state.get("p_numopt_cv", None)
    p_ml_cv = st.session_state.get("p_ml_cv", {})
    cv_label = st.session_state.get("cv_label", "CV")
    cv_mode = st.session_state.get("cv_mode_numopt", "")

    thr_numopt_cv = float(st.session_state.get("thr_numopt_cv", 0.5))
    thr_ml = float(st.session_state.get("thr_ml", 0.5))

    # ML in-sample
    proba_ml_in = st.session_state.get("proba_ml_in", {})

    # tables
    df_all_runs = st.session_state.get("df_all_runs", pd.DataFrame())
    df_ml_spheres = st.session_state.get("df_ml_spheres", pd.DataFrame())

    df_metrics_in_unw = st.session_state.get("df_metrics_in_unw", pd.DataFrame())
    df_metrics_in_w   = st.session_state.get("df_metrics_in_w", pd.DataFrame())
    df_metrics_cv_unw = st.session_state.get("df_metrics_cv_unw", pd.DataFrame())
    df_metrics_cv_w   = st.session_state.get("df_metrics_cv_w", pd.DataFrame())

    best_ml_name = st.session_state.get("best_ml_name", "")

    st.markdown("## ✅ Final Highlight — Numerical Optimization vs ML")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⭐ Numerical Optimization (optimized)")
        st.write(f"**Center:** δd={dp:.4f}, δp={pp:.4f}, δh={hp:.4f} ({unit})")
        st.write(f"**Radius:** R0={R0:.4f} ({unit})")
        st.write(f"**Best DF:** {best_df_name}")
        st.write(f"**Optimizer:** {best_optimizer}")
        st.write(f"**CV scheme:** {cv_label} | **NumOpt CV mode:** {cv_mode}")

    with c2:
        st.markdown(f"### ⭐ Best ML (CV): {best_ml_name if best_ml_name else '—'}")
        if (df_metrics_cv_unw is not None) and (not df_metrics_cv_unw.empty) and best_ml_name:
            row = df_metrics_cv_unw[df_metrics_cv_unw["Model"].str.contains(f"{best_ml_name}_{cv_label}", regex=False)]
            if not row.empty:
                r = row.iloc[0].to_dict()
                st.write(f"**AUC-ROC:** {float(r.get('AUC_ROC_unweighted', np.nan)):.3f}")
                st.write(f"**AUC-PR:**  {float(r.get('AUC_PR_unweighted', np.nan)):.3f}")
                st.write(f"**MCC:**     {float(r.get('MCC', np.nan)):.3f}")
                st.write(f"**LogLoss:**  {float(r.get('LogLoss', np.nan)):.3f}")
            else:
                st.info("Best-ML row not found in CV metrics table.")
        else:
            st.info("Run Tab 4 (CV) to populate Best-ML metrics.")

    # results per sample
    out = df_filtered.copy()
    out["y_raw"] = np.asarray(y_raw, float)
    out["w"] = np.asarray(w, float)

    if RED_all is not None:
        out["RED"] = np.asarray(RED_all, float)
    if p_numopt_in is not None:
        out["p_numopt_in"] = np.asarray(p_numopt_in, float)
    if p_numopt_cv is not None:
        out[f"p_numopt_{cv_label}"] = np.asarray(p_numopt_cv, float)

    for name, p in proba_ml_in.items():
        out[f"proba_{name}_in"] = np.asarray(p, float)
    if isinstance(p_ml_cv, dict) and len(p_ml_cv):
        for name, p in p_ml_cv.items():
            out[f"proba_{name}_{cv_label}"] = np.asarray(p, float)

    # params sheet
    df_params_numopt = pd.DataFrame({
        "Parameter": [
            "DF_best", "optimizer_best",
            "delta_d_center", "delta_p_center", "delta_h_center", "R0", "unit",
            "CV_scheme", "CV_mode_NumOpt",
            "thr_numopt_in", "thr_numopt_cv", "thr_ml",
            "best_ml_name"
        ],
        "Value": [
            best_df_name, best_optimizer,
            dp, pp, hp, R0, unit,
            cv_label, cv_mode,
            thr_numopt_in, thr_numopt_cv, thr_ml,
            best_ml_name
        ]
    })

    # metadata
    n_total = int(len(df_filtered))
    n_pos = int(np.sum(df_filtered["solubility"].values.astype(int)))
    n_neg = int(n_total - n_pos)
    n_partial = int(np.sum(np.asarray(y_raw, float) == 0.5))
    n_groups = int(len(np.unique(groups))) if use_group else 0

    colmap = st.session_state.get("colmap", {})
    df_meta = pd.DataFrame([
        {"Item":"app_version", "Value": st.session_state.get("app_version","")},
        {"Item":"run_timestamp_utc", "Value": st.session_state.get("run_timestamp","")},
        {"Item":"sheet_name", "Value": st.session_state.get("sheet_name","")},
        {"Item":"colmap", "Value": str(colmap)},
        {"Item":"N_total", "Value": n_total},
        {"Item":"N_pos_bin", "Value": n_pos},
        {"Item":"N_neg", "Value": n_neg},
        {"Item":"N_partial_0.5", "Value": n_partial},
        {"Item":"use_group", "Value": bool(use_group)},
        {"Item":"group_col", "Value": st.session_state.get("group_col","None")},
        {"Item":"N_groups", "Value": n_groups},
        {"Item":"GRID_PAD", "Value": st.session_state.get("GRID_PAD","")},
        {"Item":"GRID_N", "Value": st.session_state.get("GRID_N","")},
        {"Item":"ISO_LEVELS", "Value": str(st.session_state.get("ISO_LEVELS",""))},
        {"Item":"SHELL_DELTA", "Value": st.session_state.get("SHELL_DELTA","")},
        {"Item":"MIN_POINTS", "Value": st.session_state.get("MIN_POINTS","")},
        {"Item":"Q_FALLBACK", "Value": st.session_state.get("Q_FALLBACK","")},
        {"Item":"K_PROB", "Value": K_PROB},
        {"Item":"REG_R0", "Value": REG_R0},
        {"Item":"min_n_isotonic", "Value": int(st.session_state.get("CALIB_MIN_N_ISO", 60))},
    ])

    filename = st.text_input("Output filename (.xlsx)", value=f"result_full_paper_{cv_label}_cal_shell.xlsx")
    export_btn = st.button("⬇️ Build Excel + Download", type="primary", use_container_width=True)

    if export_btn:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df_filtered.to_excel(writer, index=False, sheet_name="Base_bin")
            out.to_excel(writer, index=False, sheet_name="Results_per_sample")

            if df_all_runs is not None and not df_all_runs.empty:
                df_all_runs.to_excel(writer, index=False, sheet_name="NumOpt_AllRuns")
            df_params_numopt.to_excel(writer, index=False, sheet_name="NumOpt_Final")

            if df_metrics_in_unw is not None and not df_metrics_in_unw.empty:
                df_metrics_in_unw.to_excel(writer, index=False, sheet_name="Metrics_IN_unweighted")
            if df_metrics_in_w is not None and not df_metrics_in_w.empty:
                df_metrics_in_w.to_excel(writer, index=False, sheet_name="Metrics_IN_weighted")
            if df_metrics_cv_unw is not None and not df_metrics_cv_unw.empty:
                df_metrics_cv_unw.to_excel(writer, index=False, sheet_name=f"Metrics_{cv_label}_unweighted")
            if df_metrics_cv_w is not None and not df_metrics_cv_w.empty:
                df_metrics_cv_w.to_excel(writer, index=False, sheet_name=f"Metrics_{cv_label}_weighted")

            if df_ml_spheres is not None and not df_ml_spheres.empty:
                df_ml_spheres.to_excel(writer, index=False, sheet_name="ML_ShellSpheres")

            df_meta.to_excel(writer, index=False, sheet_name="Run_Metadata")

        bio.seek(0)
        st.download_button(
            label="Download Excel",
            data=bio,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# FIM — PARTE 4/4
