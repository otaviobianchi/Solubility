# ============================================================
# BLOCO 1/4 — TOTAL (CORE + TABS + DATA LOADING/PREPROCESS)
#  ✅ Define tudo que os Blocos 2/3/4 precisam existir:
#   - st.set_page_config + tab1..tab4
#   - UNIT em st.session_state
#   - Upload de Excel, escolha de sheet/colunas (UI)
#   - df_filtered com colunas padronizadas: delta_d, delta_p, delta_h, solubility (+group opcional)
#   - y_raw, y_bin, w, use_group, groups
#   - Funções core: hansen_distance, prob_from_red, red_values
#   - DFs: df_geom, df_logloss, df_brier, df_hinge, df_softcount
#   - z_to_x, fit_by_methods
#
#  Observação:
#   - Mantém o termo "Numerical Optimization" (não Hansen no título do app)
# ============================================================

import warnings
import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo

warnings.filterwarnings("ignore")
np.random.seed(42)

# -----------------------------
# Streamlit config + TABS
# -----------------------------
st.set_page_config(page_title="Numerical Optimization vs ML — Solubility", layout="wide")

tab1, tab2, tab3, tab4 = st.tabs([
    "1) Data",
    "2) Numerical Optimization",
    "3) Cross-Validation",
    "4) 3D + Export"
])

# -----------------------------
# Global settings (shared)
# -----------------------------
if "UNIT" not in st.session_state:
    st.session_state["UNIT"] = "MPa\u00b9\u2044\u00b2"  # MPa½

# Defaults used later (can be overwritten by UI in other blocks)
st.session_state.setdefault("K_PROB", 6.0)
st.session_state.setdefault("REG_R0", 0.05)
st.session_state.setdefault("NMS_RESTARTS", 1)
st.session_state.setdefault("COBYLA_RESTARTS", 1)
st.session_state.setdefault("speed_profile", "full")  # "full" or "fast"

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
    d_d = np.asarray(d_d, float); d_p = np.asarray(d_p, float); d_h = np.asarray(d_h, float)
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
    st.caption("Upload an Excel file, select the sheet and columns, then proceed to the tabs.")

    st.markdown("### Upload Excel")
    up = st.file_uploader("Upload .xlsx", type=["xlsx"])

    if up is None:
        st.info("Upload an Excel file to start.")
        st.stop()

    xls = pd.ExcelFile(up)
    sheet_name = st.selectbox("Sheet", xls.sheet_names, index=0)
    df = xls.parse(sheet_name)

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

    # Build filtered base
    need_cols = [delta_d_col, delta_p_col, delta_h_col, solub_col] + ([group_col] if use_group else [])
    df_filtered = df[need_cols].copy()

    new_cols = ["delta_d", "delta_p", "delta_h", "solubility"] + (["group"] if use_group else [])
    df_filtered.columns = new_cols

    # numeric
    df_filtered[["delta_d", "delta_p", "delta_h", "solubility"]] = df_filtered[
        ["delta_d", "delta_p", "delta_h", "solubility"]
    ].apply(pd.to_numeric, errors="coerce")

    df_filtered = df_filtered.dropna(subset=["delta_d", "delta_p", "delta_h", "solubility"]).drop_duplicates().reset_index(drop=True)

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
# Make required globals available for Bloco 2/3/4
# (so you can paste blocks without re-wiring)
# ============================================================
df_filtered = st.session_state.get("df_filtered")
y_raw = st.session_state.get("y_raw")
w = st.session_state.get("w")
use_group = st.session_state.get("use_group", False)
groups = st.session_state.get("groups", None)
UNIT = st.session_state.get("UNIT", "MPa\u00b9\u2044\u00b2")
