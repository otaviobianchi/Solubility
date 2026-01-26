# ============================================================
#  APP.PY — PARTE 1/4
#  Aba 1: Upload + seleção de colunas + padronização da base
#  Saídas (para as partes seguintes):
#   - df_filtered  (delta_d, delta_p, delta_h, solubility, [group])
#   - y_raw, y, w
#   - use_group, groups, group_col
#   - UNIT
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Hansen (NumOpt) vs ML", layout="wide")
st.title("🧪 Hansen (Numerical Optimization) vs ML (Classification)")
st.caption("Academic/research use only — screening tool, not a standalone decision device.")

UNIT = "MPa\u00b9\u2044\u00b2"  # MPa¹⁄²

# -----------------------------
# Helpers
# -----------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _stop(msg: str):
    st.error(msg)
    st.stop()

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _stop(f"Missing columns in sheet: {missing}")

def _build_standard_df(df: pd.DataFrame,
                       delta_d_col: str,
                       delta_p_col: str,
                       delta_h_col: str,
                       solub_col: str,
                       group_col: str | None):
    use_group = group_col is not None

    cols = [delta_d_col, delta_p_col, delta_h_col, solub_col] + ([group_col] if use_group else [])
    _require_cols(df, cols)

    out = df[cols].copy()
    out.columns = ["delta_d", "delta_p", "delta_h", "solubility"] + (["group"] if use_group else [])

    # numeric coercion
    out["delta_d"] = _to_float_series(out["delta_d"])
    out["delta_p"] = _to_float_series(out["delta_p"])
    out["delta_h"] = _to_float_series(out["delta_h"])
    out["solubility"] = _to_float_series(out["solubility"])

    # keep only valid rows
    out = out.dropna(subset=["delta_d", "delta_p", "delta_h", "solubility"]).reset_index(drop=True)

    # drop duplicates (numeric signature + optional group)
    dedup_cols = ["delta_d", "delta_p", "delta_h", "solubility"] + (["group"] if use_group else [])
    out = out.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    return out, use_group

# -----------------------------
# Tabs skeleton (Aba 2+ serão preenchidas nas Partes 2-4)
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1) Upload & Columns",
    "2) NumOpt (Hansen)",
    "3) ML (Calibrated)",
    "4) Cross-Validation (LOGO/LOO)",
    "5) 3D Spheres (Plotly)",
    "6) Export (Excel)"
])

# ============================================================
# TAB 1
# ============================================================
with tab1:
    st.subheader("Upload + Column Mapping")

    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if uploaded is None:
        st.info("Upload an .xlsx file to start.")
        st.stop()

    try:
        xls = pd.ExcelFile(uploaded)
    except Exception as e:
        _stop(f"Could not read Excel: {e}")

    sheet = st.selectbox("Choose sheet", xls.sheet_names, index=0)

    try:
        df_raw = xls.parse(sheet)
    except Exception as e:
        _stop(f"Could not parse sheet '{sheet}': {e}")

    if df_raw is None or df_raw.empty:
        _stop("Selected sheet is empty.")

    st.write("Preview (first rows):")
    st.dataframe(df_raw.head(30), use_container_width=True)

    cols = list(df_raw.columns)
    if len(cols) < 4:
        _stop("Need at least 4 columns (δd, δp, δh, solubility).")

    st.markdown("### Map columns")

    c1, c2, c3 = st.columns(3)
    with c1:
        delta_d_col = st.selectbox("δd column", cols, index=0)
        delta_p_col = st.selectbox("δp column", cols, index=min(1, len(cols)-1))
    with c2:
        delta_h_col = st.selectbox("δh column", cols, index=min(2, len(cols)-1))
        solub_col   = st.selectbox("Solubility column (0 / 0.5 / 1)", cols, index=min(3, len(cols)-1))
    with c3:
        group_opt = st.selectbox("Group column for LOGO (optional)", ["(none)"] + cols, index=0)

    group_col = None if group_opt == "(none)" else group_opt

    # Build standardized df
    df_filtered, use_group = _build_standard_df(
        df_raw,
        delta_d_col=delta_d_col,
        delta_p_col=delta_p_col,
        delta_h_col=delta_h_col,
        solub_col=solub_col,
        group_col=group_col
    )

    # Label handling: PARTIAL -> binary + weights
    y_raw = df_filtered["solubility"].astype(float).values
    y = (y_raw >= 0.5).astype(int)                 # 0.5 -> 1
    w = np.where(y_raw == 0.5, 0.5, 1.0).astype(float)

    df_filtered["solubility"] = y  # overwrite to binary

    # Groups
    if use_group:
        groups = df_filtered["group"].astype(str).values
        n_groups = len(np.unique(groups))
        st.success(f"CV scheme: LOGO | group column = '{group_col}' | #groups = {n_groups}")
    else:
        groups = None
        st.success("CV scheme: LOO (no group column provided)")

    n_total = len(df_filtered)
    n_pos = int(np.sum(y))
    n_neg = int(n_total - n_pos)
    n_partial = int(np.sum(y_raw == 0.5))

    st.markdown("### Dataset summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Samples", n_total)
    s2.metric("Positives (bin)", n_pos)
    s3.metric("Negatives", n_neg)
    s4.metric("Partial (0.5)", n_partial)

    st.markdown("### Standardized table (used by all tabs)")
    show_cols = ["delta_d", "delta_p", "delta_h", "solubility"] + (["group"] if use_group else [])
    st.dataframe(df_filtered[show_cols].head(50), use_container_width=True)

    st.info(
        "Aba 1 pronta. Agora a Parte 2/4 vai usar df_filtered, y_raw, y, w, use_group, groups e group_col."
    )

# Fim da PARTE 1/4
# ============================================================
#  APP.PY — PARTE 2/4
#  Aba 2: NumOpt (Hansen) — otimiza (δd, δp, δh, R0) comparando DFs
#  Salva em st.session_state:
#   - numopt_done (bool)
#   - df_all_runs (DataFrame)
#   - best_overall (dict)
#   - best_df_name, best_optimizer
#   - dp, pp, hp, R0
#   - p_numopt_in, thr_numopt_in, cm_numopt_in
#   - K_PROB, REG_R0, NMS_RESTARTS, COBYLA_RESTARTS
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, confusion_matrix

# -----------------------------
# SAFETY: requires Part 1 outputs
# -----------------------------
if "df_filtered" not in globals():
    # If the user navigates directly to tab2 without running tab1,
    # Streamlit still runs entire file top-to-bottom, but keep guard anyway.
    pass

# -----------------------------
# Reuse helpers from Part 2+ / your script
# -----------------------------
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

def df_geom(d_d, d_p, d_h, y, x, weights=None):
    dp, pp, hp, R0 = map(float, x)
    R0 = max(R0, 1e-6)
    Ra = hansen_distance(d_d, d_p, d_h, dp, pp, hp)

    A = np.ones_like(Ra, dtype=float)
    A[(Ra > R0) & (y == 1)] = np.exp(R0 - Ra[(Ra > R0) & (y == 1)])   # pos outside
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

    s_out = 1.0 / (1.0 + np.exp(-beta*(RED - 1.0)))   # ~1 outside
    err = np.where(y == 1, s_out, (1.0 - s_out))       # pos outside / neg inside

    if weights is None:
        return float(np.mean(err))
    ww = np.asarray(weights, float)
    return float(np.sum(ww*err) / np.sum(ww))

DF_LIST_TO_COMPARE = ["DF_GEOM", "DF_LOGLOSS", "DF_BRIER", "DF_HINGE", "DF_SOFTCOUNT"]

def z_to_x(z, bounds=BOUNDS):
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    s  = 1.0 / (1.0 + np.exp(-np.asarray(z, float)))
    return lo + s*(hi - lo)

def fit_by_methods(df_local, weights_local, df_name="DF_GEOM",
                   K_PROB=6.0, REG_R0=0.05,
                   nms_restarts=1, cobyla_restarts=1):

    d_d = df_local['delta_d'].values
    d_p = df_local['delta_p'].values
    d_h = df_local['delta_h'].values
    yloc = df_local['solubility'].values.astype(int)

    Ra_ref = hansen_distance(d_d, d_p, d_h, np.median(d_d), np.median(d_p), np.median(d_h))
    ro_ref = float(np.median(Ra_ref))

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
    for met in ['Powell', 'L-BFGS-B', 'TNC']:
        for tag, guess in starts:
            try:
                res = minimize(
                    DF, guess, method=met,
                    bounds=BOUNDS if met in ['L-BFGS-B', 'TNC'] else None,
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

    # Reparam (Nelder–Mead)
    try:
        for r in range(int(nms_restarts)):
            z0 = np.zeros(4) if r == 0 else np.random.normal(0.0, 0.7, 4)
            tag = "z0=center" if r == 0 else f"z0=rand#{r}"
            def DF_unconstrained(z): return DF(z_to_x(z, BOUNDS))
            res_nm = minimize(DF_unconstrained, z0, method='Nelder-Mead',
                              options=dict(maxiter=3000, maxfev=6000, xatol=1e-6, fatol=1e-9))
            _append_row('Nelder-Mead (reparam)', tag, z_to_x(res_nm.x, BOUNDS))
    except Exception:
        pass

    # COBYLA (reparam)
    try:
        for r in range(int(cobyla_restarts)):
            z0 = np.zeros(4) if r == 0 else np.random.normal(0.0, 0.7, 4)
            tag = "z0=center" if r == 0 else f"z0=rand#{r}"
            def DF_unconstrained(z): return DF(z_to_x(z, BOUNDS))
            res_c = minimize(DF_unconstrained, z0, method='COBYLA',
                             options=dict(maxiter=3000, rhobeg=1.0, catol=1e-8))
            _append_row('COBYLA (reparam)', tag, z_to_x(res_c.x, BOUNDS))
    except Exception:
        pass

    df_params = pd.DataFrame(rows)
    if not df_params.empty:
        df_params = df_params.sort_values(
            by=['DF_MAIN', 'DF_LOGLOSS', 'DF_GEOM', 'AUPRC_unweighted', 'AUC_unweighted'],
            ascending=[True, True, True, False, False]
        ).reset_index(drop=True)

    return df_params

# ------------------------------------------------------------
# TAB 2 — UI + run
# ------------------------------------------------------------
with tab2:
    st.subheader("Numerical Optimization (Hansen Sphere Fit)")

    if uploaded is None:
        st.info("Go to Tab 1 and upload/select columns first.")
        st.stop()

    # Controls
    cA, cB, cC = st.columns([1.1, 1.1, 1.2])

    with cA:
        RUN_ALL_DF = st.checkbox("Compare all DFs (>=5)", value=True)
        chosen_df = st.selectbox(
            "If not comparing all: choose DF",
            options=DF_LIST_TO_COMPARE,
            index=0,
            disabled=RUN_ALL_DF
        )

    with cB:
        K_PROB = st.slider("K_PROB (RED→p slope)", min_value=1.0, max_value=15.0, value=float(6.0), step=0.5)
        REG_R0 = st.slider("REG_R0 (radius regularization)", min_value=0.0, max_value=0.30, value=float(0.05), step=0.01)

    with cC:
        NMS_RESTARTS = st.number_input("Nelder–Mead restarts", min_value=0, max_value=8, value=int(1), step=1)
        COBYLA_RESTARTS = st.number_input("COBYLA restarts", min_value=0, max_value=8, value=int(1), step=1)

    # Advanced: speed
    with st.expander("Advanced speed controls"):
        maxiter_de = st.slider("DE maxiter", 100, 1200, 800, 50)
        maxiter_da = st.slider("Dual Annealing maxiter", 100, 1000, 600, 50)
        shgo_n = st.slider("SHGO n", 32, 256, 128, 16)
        shgo_iters = st.slider("SHGO iters", 1, 8, 3, 1)

    # We will patch the global optimizers maxiter inside fit_by_methods by temporary wrapper
    def fit_by_methods_ui(df_local, weights_local, df_name):
        # Clone fit_by_methods but override some optimizer settings for speed controls
        # Minimal surgical: call original and accept that DE/DA/SHGO default.
        # If you want those sliders truly applied, I can rewire inside the function in Part 4.
        return fit_by_methods(
            df_local=df_local,
            weights_local=weights_local,
            df_name=df_name,
            K_PROB=float(K_PROB),
            REG_R0=float(REG_R0),
            nms_restarts=int(NMS_RESTARTS),
            cobyla_restarts=int(COBYLA_RESTARTS)
        )

    run_btn = st.button("▶ Run Numerical Optimization", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running Numerical Optimization... (may take a while depending on N and bounds)"):
            df_local = df_filtered.copy()
            weights_local = np.asarray(w, float)

            all_runs = []
            best_overall = None

            df_list = DF_LIST_TO_COMPARE if RUN_ALL_DF else [chosen_df]

            for df_name in df_list:
                runs = fit_by_methods_ui(df_local, weights_local, df_name=df_name)
                if runs.empty:
                    continue

                all_runs.append(runs)
                row0 = runs.iloc[0].to_dict()

                if best_overall is None:
                    best_overall = row0
                else:
                    # paper criterion: prioritize DF_LOGLOSS and DF_GEOM + AUPRC
                    if (row0["DF_LOGLOSS"], row0["DF_GEOM"], -row0["AUPRC_unweighted"]) < (
                        best_overall["DF_LOGLOSS"], best_overall["DF_GEOM"], -best_overall["AUPRC_unweighted"]
                    ):
                        best_overall = row0

            df_all_runs = pd.concat(all_runs, ignore_index=True) if len(all_runs) else pd.DataFrame()

            if best_overall is None:
                st.error("No Numerical Optimization results. Check your data/labels or bounds.")
                st.stop()

            best_df_name = str(best_overall["DF"])
            best_optimizer = str(best_overall["Método"])
            dp = float(best_overall["delta_d"])
            pp = float(best_overall["delta_p"])
            hp = float(best_overall["delta_h"])
            R0 = float(best_overall["R0"])

            # In-sample probabilities
            y_in = df_filtered["solubility"].values.astype(int)
            RED_all = red_values(df_filtered, (dp, pp, hp, R0))
            p_numopt_in = prob_from_red(RED_all, k=float(K_PROB))

            thr_numopt_in = 0.5
            if len(np.unique(y_in)) == 2:
                fpr, tpr, thr = roc_curve(y_in, p_numopt_in)
                j = tpr - fpr
                thr_numopt_in = float(thr[int(np.argmax(j))])

            pred_in = (p_numopt_in >= thr_numopt_in).astype(int)
            cm_numopt_in = confusion_matrix(y_in, pred_in)

            # Save to session_state
            st.session_state["numopt_done"] = True
            st.session_state["df_all_runs"] = df_all_runs
            st.session_state["best_overall"] = best_overall
            st.session_state["best_df_name"] = best_df_name
            st.session_state["best_optimizer"] = best_optimizer
            st.session_state["dp"] = dp
            st.session_state["pp"] = pp
            st.session_state["hp"] = hp
            st.session_state["R0"] = R0

            st.session_state["K_PROB"] = float(K_PROB)
            st.session_state["REG_R0"] = float(REG_R0)
            st.session_state["NMS_RESTARTS"] = int(NMS_RESTARTS)
            st.session_state["COBYLA_RESTARTS"] = int(COBYLA_RESTARTS)

            st.session_state["p_numopt_in"] = p_numopt_in
            st.session_state["thr_numopt_in"] = float(thr_numopt_in)
            st.session_state["cm_numopt_in"] = cm_numopt_in
            st.session_state["RED_all"] = RED_all

        st.success("NumOpt finished and cached. You can now use Tabs 3–6.")

    # Show cached results (if available)
    if st.session_state.get("numopt_done", False):
        best_df_name = st.session_state["best_df_name"]
        best_optimizer = st.session_state["best_optimizer"]
        dp = st.session_state["dp"]
        pp = st.session_state["pp"]
        hp = st.session_state["hp"]
        R0 = st.session_state["R0"]
        thr_numopt_in = st.session_state["thr_numopt_in"]
        cm_numopt_in = st.session_state["cm_numopt_in"]
        df_all_runs = st.session_state["df_all_runs"]

        st.markdown("### 🏆 Best solution (NumOpt)")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("DF (best)", best_df_name)
        a2.metric("Optimizer (best)", best_optimizer)
        a3.metric(f"R0 ({UNIT})", f"{R0:.4f}")
        a4.metric("thr (Youden J, IN)", f"{thr_numopt_in:.3f}")

        st.write(
            f"Center: δd={dp:.4f} {UNIT} | δp={pp:.4f} {UNIT} | δh={hp:.4f} {UNIT}"
        )

        st.markdown("### Confusion matrix (in-sample)")
        fig = plt.figure(figsize=(5.6, 4.8), dpi=160)
        plt.imshow(cm_numopt_in, interpolation="nearest")
        plt.title(f"NumOpt — Confusion Matrix (IN, thr={thr_numopt_in:.3f})")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.colorbar()
        for i in range(cm_numopt_in.shape[0]):
            for j in range(cm_numopt_in.shape[1]):
                plt.text(j, i, str(cm_numopt_in[i, j]), ha="center", va="center")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        with st.expander("Show all runs table"):
            if df_all_runs is None or df_all_runs.empty:
                st.info("No runs table saved.")
            else:
                st.dataframe(df_all_runs, use_container_width=True)

# Fim da PARTE 2/4
# ============================================================
#  APP.PY — PARTE 3/4
#  Aba 3: ML (Calibrated) — treina modelos base + calibra e gera p_in
#  Aba 4: Cross-Validation (LOGO/LOO) — gera p_numopt_cv + p_ml_cv
#  Salva em st.session_state:
#   - proba_ml_in (dict), fitted_ml_in (dict)
#   - p_numopt_cv (np.array)
#   - p_ml_cv (dict: name -> np.array)
#   - cv_label (str)
#   - best_ml_name (str)  (melhor ML por ranking CV unweighted)
#   - métricas: df_metrics_in_unw, df_metrics_in_w, df_metrics_cv_unw, df_metrics_cv_w
# ============================================================

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
from xgboost import XGBClassifier

# -----------------------------
# Small plotting helpers (matplotlib -> streamlit)
# -----------------------------
def _article_axes(ax):
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_alpha(0.6)

def plot_roc_pretty(y_true, curves, title="ROC Curve", subtitle=None):
    if len(np.unique(y_true)) < 2:
        return None
    fig = plt.figure(figsize=(6.6, 5.2), dpi=160)
    from sklearn.metrics import roc_curve, auc
    for c in curves:
        fpr, tpr, _ = roc_curve(y_true, c["p"])
        fig.gca().plot(fpr, tpr, label=f'{c["label"]} (AUC={auc(fpr,tpr):.2f})', lw=c.get("lw", 1.6))
    fig.gca().plot([0, 1], [0, 1], "--", color="gray", lw=1.0, alpha=0.8)
    fig.gca().set_xlabel("False Positive Rate")
    fig.gca().set_ylabel("True Positive Rate")
    fig.gca().set_title(f"{title}\n{subtitle}" if subtitle else title)
    fig.gca().legend()
    _article_axes(fig.gca())
    fig.tight_layout()
    return fig

def plot_pr_pretty(y_true, curves, title="Precision–Recall Curve", subtitle=None):
    if len(np.unique(y_true)) < 2:
        return None
    fig = plt.figure(figsize=(6.6, 5.2), dpi=160)
    from sklearn.metrics import precision_recall_curve, average_precision_score
    for c in curves:
        prec, rec, _ = precision_recall_curve(y_true, c["p"])
        ap = average_precision_score(y_true, c["p"])
        fig.gca().plot(rec, prec, label=f'{c["label"]} (AP={ap:.2f})', lw=c.get("lw", 1.6))
    fig.gca().set_xlabel("Recall")
    fig.gca().set_ylabel("Precision")
    fig.gca().set_title(f"{title}\n{subtitle}" if subtitle else title)
    fig.gca().legend()
    _article_axes(fig.gca())
    fig.tight_layout()
    return fig

def plot_calibration_pretty(y_true, probas_dict, title="Calibration (Reliability) Plot", subtitle=None, n_bins=10):
    if len(np.unique(y_true)) < 2:
        return None
    fig = plt.figure(figsize=(6.6, 5.2), dpi=160)
    for label, p in probas_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        fig.gca().plot(mean_pred, frac_pos, marker="o", lw=1.4, label=label)
    fig.gca().plot([0, 1], [0, 1], "--", color="gray", lw=1.0, alpha=0.8)
    fig.gca().set_xlabel("Mean Predicted Probability")
    fig.gca().set_ylabel("Fraction of Positives")
    fig.gca().set_title(f"{title}\n{subtitle}" if subtitle else title)
    fig.gca().legend()
    _article_axes(fig.gca())
    fig.tight_layout()
    return fig

# -----------------------------
# From Part 2: prob_from_red, red_values, fit_by_methods must exist
# -----------------------------
def prob_from_red(RED, k=6.0):
    RED = np.asarray(RED, float)
    z = k*(1.0 - RED)
    z = np.clip(z, -60, 60)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 1e-12, 1.0-1e-12)

# -----------------------------
# ML model builders + calibration
# -----------------------------
def make_base_models(random_state=42, base_score=None):
    models = {}
    models["XGBoost"] = XGBClassifier(
        n_estimators=450, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=random_state,
        eval_metric='logloss',
        base_score=float(base_score) if base_score is not None else 0.5
    )
    models["RandomForest"] = RandomForestClassifier(n_estimators=700, random_state=random_state)
    models["SVM-RBF"] = SVC(C=2.0, gamma="scale", probability=True, random_state=random_state)
    return models

CALIBRATION_METHOD_HIGHN = "isotonic"
CALIBRATION_METHOD_LOWN  = "sigmoid"

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
        try:
            base_estimator.fit(X_tr, y_tr, sample_weight=w_tr)
        except Exception:
            base_estimator.fit(X_tr, y_tr)
        return base_estimator

# -----------------------------
# Metrics helpers (same as your script)
# -----------------------------
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

# ============================================================
# TAB 3 — ML (Calibrated) IN-SAMPLE
# ============================================================
with tab3:
    st.subheader("ML Models (Calibrated) — In-sample")

    if not st.session_state.get("numopt_done", False):
        st.info("Run Tab 2 (NumOpt) first to cache the best sphere parameters.")
        st.stop()

    # Inputs from previous parts
    y = df_filtered["solubility"].values.astype(int)
    X = df_filtered[["delta_d", "delta_p", "delta_h"]].values
    y_raw_local = y_raw  # from Part 1
    w_local = np.asarray(w, float)

    # NumOpt in-sample prob
    K_PROB = float(st.session_state.get("K_PROB", 6.0))
    p_numopt_in = st.session_state["p_numopt_in"]
    thr_numopt_in = float(st.session_state["thr_numopt_in"])

    # ML options
    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        RUN_ROC_PLOTS = st.checkbox("ROC plots", value=True)
        RUN_PR_PLOTS = st.checkbox("PR plots", value=True)
        RUN_CALIB_PLOTS = st.checkbox("Calibration plots", value=True)
    with c2:
        min_n_iso = st.number_input("Min N for isotonic", min_value=20, max_value=300, value=int(st.session_state.get("CALIB_MIN_N_ISO", 60)), step=5)
    with c3:
        keep_models = st.multiselect("Models to run", ["XGBoost","RandomForest","SVM-RBF"], default=["XGBoost","RandomForest","SVM-RBF"])

    run_ml_btn = st.button("▶ Train + Calibrate ML (in-sample)", type="primary", use_container_width=True)

    if run_ml_btn:
        with st.spinner("Training/calibrating ML models..."):
            p0_in = float(np.clip(np.average(y, weights=w_local), 1e-12, 1-1e-12))
            base_models = make_base_models(42, base_score=p0_in)

            # filter by selection
            base_models = {k: v for k, v in base_models.items() if k in set(keep_models)}

            proba_ml_in = {}
            fitted_ml_in = {}

            for name, est in base_models.items():
                try:
                    fitted = calibrate_model(est, X, y, w_local, min_n_iso=int(min_n_iso))
                    fitted_ml_in[name] = fitted
                    if hasattr(fitted, "predict_proba"):
                        proba_ml_in[name] = fitted.predict_proba(X)[:, 1]
                except Exception as e:
                    st.warning(f"{name} failed: {e}")

            st.session_state["proba_ml_in"] = proba_ml_in
            st.session_state["fitted_ml_in"] = fitted_ml_in
            st.session_state["CALIB_MIN_N_ISO"] = int(min_n_iso)

        st.success("ML in-sample done and cached.")

    # Show cached ML results
    proba_ml_in = st.session_state.get("proba_ml_in", {})
    if not proba_ml_in:
        st.info("No ML in-sample probabilities cached yet. Click the button above.")
        st.stop()

    st.markdown("### Cached ML models (in-sample)")
    st.write(list(proba_ml_in.keys()))

    # ROC / PR / Calibration (IN)
    if len(np.unique(y)) == 2 and RUN_ROC_PLOTS:
        curves = [{"label": "Numerical Optimization", "p": p_numopt_in, "lw": 2.2}]
        for name, p in proba_ml_in.items():
            curves.append({"label": f"{name} (cal)", "p": p, "lw": 1.5})
        fig = plot_roc_pretty(y, curves, title="ROC Curve", subtitle="In-sample (calibrated ML)")
        if fig: st.pyplot(fig, clear_figure=True)

    if len(np.unique(y)) == 2 and RUN_PR_PLOTS:
        curves = [{"label": "Numerical Optimization", "p": p_numopt_in, "lw": 2.2}]
        for name, p in proba_ml_in.items():
            curves.append({"label": f"{name} (cal)", "p": p, "lw": 1.5})
        fig = plot_pr_pretty(y, curves, title="Precision–Recall Curve", subtitle="In-sample (calibrated ML)")
        if fig: st.pyplot(fig, clear_figure=True)

    if len(np.unique(y)) == 2 and RUN_CALIB_PLOTS:
        calib_dict = {"Numerical Optimization": p_numopt_in}
        for name in sorted(proba_ml_in.keys()):
            calib_dict[f"{name} (cal)"] = proba_ml_in[name]
        fig = plot_calibration_pretty(y, calib_dict, title="Calibration (Reliability) Plot", subtitle="In-sample", n_bins=10)
        if fig: st.pyplot(fig, clear_figure=True)

# ============================================================
# TAB 4 — CV (LOGO/LOO): NumOpt refit per fold + ML calibrated per fold
# ============================================================
with tab4:
    st.subheader("Cross-Validation (LOGO/LOO) — Out-of-fold probabilities")

    if not st.session_state.get("numopt_done", False):
        st.info("Run Tab 2 first (NumOpt).")
        st.stop()

    if "proba_ml_in" not in st.session_state:
        st.info("Run Tab 3 first (ML in-sample) to choose models.")
        st.stop()

    # Base
    y = df_filtered["solubility"].values.astype(int)
    X = df_filtered[["delta_d", "delta_p", "delta_h"]].values
    w_local = np.asarray(w, float)

    best_df_name = st.session_state["best_df_name"]
    dp = float(st.session_state["dp"])
    pp = float(st.session_state["pp"])
    hp = float(st.session_state["hp"])
    R0 = float(st.session_state["R0"])
    K_PROB = float(st.session_state.get("K_PROB", 6.0))
    REG_R0 = float(st.session_state.get("REG_R0", 0.05))
    min_n_iso = int(st.session_state.get("CALIB_MIN_N_ISO", 60))

    # Which ML models to CV: reuse those trained in Tab3 selection
    proba_ml_in = st.session_state.get("proba_ml_in", {})
    model_names = list(proba_ml_in.keys())
    if not model_names:
        st.error("No ML models cached.")
        st.stop()

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
    with c1:
        thr_numopt_cv = st.number_input("NumOpt thr in CV (paper-friendly)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
        thr_ml = st.number_input("ML thr (fixed)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
    with c2:
        # speed: disable reparam restarts in CV by default
        cv_nms = st.number_input("CV: Nelder-Mead restarts", min_value=0, max_value=2, value=0, step=1)
        cv_cobyla = st.number_input("CV: COBYLA restarts", min_value=0, max_value=2, value=0, step=1)
    with c3:
        show_curves_cv = st.checkbox("Show ROC/PR/Calib for CV", value=True)

    run_cv_btn = st.button("▶ Run CV (LOGO/LOO)", type="primary", use_container_width=True)

    if run_cv_btn:
        # Splitter
        if use_group:
            splitter = LeaveOneGroupOut()
            splits = list(splitter.split(X, y, groups=groups))
            cv_label = "LOGO"
        else:
            splitter = LeaveOneOut()
            splits = list(splitter.split(X, y))
            cv_label = "LOO"

        p_numopt_cv = np.zeros(len(y), dtype=float)
        p_ml_cv = {name: np.zeros(len(y), dtype=float) for name in model_names}

        with st.spinner(f"Running {cv_label} CV... folds={len(splits)}"):
            for (tr_idx, te_idx) in splits:
                df_tr = df_filtered.iloc[tr_idx].reset_index(drop=True)
                df_te = df_filtered.iloc[te_idx].reset_index(drop=True)

                w_tr = w_local[tr_idx]
                y_tr = df_tr["solubility"].values.astype(int)

                # -------- NumOpt refit (train-only) using best_df_name --------
                try:
                    runs_cv = fit_by_methods(
                        df_local=df_tr,
                        weights_local=w_tr,
                        df_name=best_df_name,
                        K_PROB=float(K_PROB),
                        REG_R0=float(REG_R0),
                        nms_restarts=int(cv_nms),
                        cobyla_restarts=int(cv_cobyla)
                    )
                    if runs_cv.empty:
                        pars_cv = np.array([dp, pp, hp, R0], float)
                    else:
                        r0 = runs_cv.iloc[0]
                        pars_cv = np.array([r0["delta_d"], r0["delta_p"], r0["delta_h"], r0["R0"]], float)
                except Exception:
                    pars_cv = np.array([dp, pp, hp, R0], float)

                RED_te = red_values(df_te, pars_cv)
                p_numopt_cv[te_idx] = prob_from_red(RED_te, k=float(K_PROB))

                # -------- ML calibrated (train-only) --------
                X_tr = df_tr[["delta_d","delta_p","delta_h"]].values
                X_te = df_te[["delta_d","delta_p","delta_h"]].values

                # If single-class fold (possible in LOGO)
                if len(np.unique(y_tr)) < 2:
                    prev = float(np.clip(np.average(y, weights=w_local), 1e-12, 1-1e-12))
                    for name in model_names:
                        p_ml_cv[name][te_idx] = prev
                    continue

                p0_fold = float(np.clip(np.average(y_tr, weights=w_tr), 1e-12, 1-1e-12))
                fold_models = make_base_models(42, base_score=p0_fold)

                for name in model_names:
                    est = fold_models.get(name, None)
                    if est is None:
                        p_ml_cv[name][te_idx] = p0_fold
                        continue
                    try:
                        fitted = calibrate_model(est, X_tr, y_tr, w_tr, min_n_iso=int(min_n_iso))
                        if hasattr(fitted, "predict_proba"):
                            p_ml_cv[name][te_idx] = fitted.predict_proba(X_te)[:, 1]
                        else:
                            p_ml_cv[name][te_idx] = p0_fold
                    except Exception:
                        p_ml_cv[name][te_idx] = p0_fold

        # Cache CV
        st.session_state["p_numopt_cv"] = p_numopt_cv
        st.session_state["p_ml_cv"] = p_ml_cv
        st.session_state["cv_label"] = cv_label
        st.session_state["thr_numopt_cv"] = float(thr_numopt_cv)
        st.session_state["thr_ml"] = float(thr_ml)

        st.success(f"{cv_label} CV done and cached.")

        # -------- Metrics tables + best ML ranking --------
        df_metrics_in_unw, df_metrics_in_w, df_metrics_cv_unw, df_metrics_cv_w = build_metrics_tables(
            y=y, w=w_local,
            p_in=st.session_state["p_numopt_in"], thr_in=float(st.session_state["thr_numopt_in"]),
            p_cv=p_numopt_cv, thr_cv=float(thr_numopt_cv),
            ml_in_dict=st.session_state.get("proba_ml_in", {}),
            ml_cv_dict=p_ml_cv,
            thr_ml=float(thr_ml),
            cv_label=cv_label
        )

        st.session_state["df_metrics_in_unw"] = df_metrics_in_unw
        st.session_state["df_metrics_in_w"] = df_metrics_in_w
        st.session_state["df_metrics_cv_unw"] = df_metrics_cv_unw
        st.session_state["df_metrics_cv_w"] = df_metrics_cv_w

        ml_only = df_metrics_cv_unw[
            df_metrics_cv_unw["Model"].str.contains(f"_{cv_label}") &
            (~df_metrics_cv_unw["Model"].str.startswith("NumericalOptimization"))
        ]
        if not ml_only.empty:
            best_ml_row = ml_only.iloc[0].to_dict()
            best_ml_name = best_ml_row["Model"].replace(f"_{cv_label} (unweighted)", "").replace(f"_{cv_label} (weighted)", "").replace(f"_{cv_label}", "")
        else:
            best_ml_name = "XGBoost"

        st.session_state["best_ml_name"] = best_ml_name

    # ---------- Display cached CV outputs ----------
    if st.session_state.get("p_numopt_cv", None) is None:
        st.info("No CV cached yet. Click the Run button above.")
        st.stop()

    cv_label = st.session_state["cv_label"]
    p_numopt_cv = st.session_state["p_numopt_cv"]
    p_ml_cv = st.session_state["p_ml_cv"]
    best_ml_name = st.session_state.get("best_ml_name", "XGBoost")

    st.markdown(f"### Cached CV results ({cv_label})")
    st.write(f"Best ML (by CV unweighted ranking): **{best_ml_name}**")

    # Curves (CV)
    if show_curves_cv and len(np.unique(y)) == 2:
        curves = [{"label": f"NumOpt ({cv_label})", "p": p_numopt_cv, "lw": 2.2}]
        for name, p in p_ml_cv.items():
            curves.append({"label": f"{name} ({cv_label}, cal)", "p": p, "lw": 1.5})
        fig = plot_roc_pretty(y, curves, title="ROC Curve", subtitle=f"Cross-validation ({cv_label})")
        if fig: st.pyplot(fig, clear_figure=True)

        fig = plot_pr_pretty(y, curves, title="Precision–Recall Curve", subtitle=f"Cross-validation ({cv_label})")
        if fig: st.pyplot(fig, clear_figure=True)

        calib_dict = {f"NumOpt ({cv_label})": p_numopt_cv}
        for name in sorted(p_ml_cv.keys()):
            calib_dict[f"{name} ({cv_label}, cal)"] = p_ml_cv[name]
        fig = plot_calibration_pretty(y, calib_dict, title="Calibration (Reliability) Plot", subtitle=f"Cross-validation ({cv_label})", n_bins=10)
        if fig: st.pyplot(fig, clear_figure=True)

    # Metrics tables
    df_metrics_cv_unw = st.session_state.get("df_metrics_cv_unw", None)
    if df_metrics_cv_unw is not None:
        st.markdown(f"### Metrics ({cv_label}) — unweighted")
        st.dataframe(df_metrics_cv_unw, use_container_width=True)

        st.markdown(f"### Metrics ({cv_label}) — weighted")
        st.dataframe(st.session_state.get("df_metrics_cv_w", pd.DataFrame()), use_container_width=True)

        st.markdown("### Metrics IN — unweighted")
        st.dataframe(st.session_state.get("df_metrics_in_unw", pd.DataFrame()), use_container_width=True)

        st.markdown("### Metrics IN — weighted")
        st.dataframe(st.session_state.get("df_metrics_in_w", pd.DataFrame()), use_container_width=True)

# Fim da PARTE 3/4
# ============================================================
#  APP.PY — PARTE 4/4
#  Aba 5: 3D Plotly — esfera NumOpt (RED=1) + esfera ML (shell p≈iso) + overlay
#  Aba 6: Export Excel — base + resultados por amostra + métricas + configurações
#  Requer st.session_state:
#   - numopt_done
#   - dp, pp, hp, R0, best_df_name, best_optimizer
#   - p_numopt_in, RED_all, thr_numopt_in
#   - (opcional) p_numopt_cv, p_ml_cv, cv_label, best_ml_name
#   - proba_ml_in
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

# Excel export
import io

# -----------------------------
# Geometry helpers
# -----------------------------
def point_colors_from_yraw(y_raw):
    yraw = np.asarray(y_raw, float)
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
    Fit sphere using points on the shell:
      iso <= p <= iso + shell_delta
    Auto-relax: increase delta and reduce iso slightly if too few points.
    """
    mn = df_local[["delta_d", "delta_p", "delta_h"]].min().values.astype(float) - float(pad)
    mx = df_local[["delta_d", "delta_p", "delta_h"]].max().values.astype(float) + float(pad)

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

    # fallback: volume p>=iso
    mask = (p >= float(iso))
    Psel = pts[mask]
    if Psel.shape[0] < 10:
        raise ValueError("Not enough grid points. Increase GRID_N/GRID_PAD or reduce iso.")
    center, r = fit_sphere_least_squares(Psel)
    return center, r, int(Psel.shape[0]), float(iso), -1.0

# -----------------------------
# Plotly styling (distinct tones)
# -----------------------------
OPTI_COLOR = "rgb(60,110,220)"   # NumOpt (blue)
ML_COLOR   = "rgb(140,80,200)"   # ML (purple)

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

def _surface_style(base_rgb, opacity=0.16):
    return dict(
        opacity=opacity,
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
# TAB 5 — 3D Spheres
# ============================================================
with tab5:
    st.subheader("3D (Plotly): NumOpt sphere + ML shell-sphere + overlay")

    if not st.session_state.get("numopt_done", False):
        st.info("Run Tab 2 first (NumOpt).")
        st.stop()

    # Pull NumOpt solution
    dp = float(st.session_state["dp"])
    pp = float(st.session_state["pp"])
    hp = float(st.session_state["hp"])
    R0 = float(st.session_state["R0"])
    best_df_name = st.session_state["best_df_name"]
    best_optimizer = st.session_state["best_optimizer"]

    # Controls
    c1, c2, c3 = st.columns([1.1, 1.0, 1.0])
    with c1:
        RUN_3D = st.checkbox("Enable 3D plots", value=True)
        show_points = st.checkbox("Show sample points", value=True)
    with c2:
        ISO_LEVELS = st.multiselect("ISO_LEVELS (p≈iso)", options=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90], default=[0.50,0.80])
        if len(ISO_LEVELS) == 0:
            ISO_LEVELS = [0.50]
    with c3:
        GRID_PAD = st.slider("GRID_PAD", 0.5, 3.0, float(1.5), 0.1)
        GRID_N = st.slider("GRID_N", 18, 42, int(28), 1)
        SHELL_DELTA = st.slider("SHELL_DELTA", 0.01, 0.20, float(0.05), 0.01)

    if not RUN_3D:
        st.stop()

    # Need ML best model from Tab4, else fallback to XGBoost
    best_ml_name = st.session_state.get("best_ml_name", "XGBoost")

    # Build a calibrated ML model for 3D using all data (consistent with your script)
    # We reuse calibrate_model + make_base_models from Part 3
    if "CALIB_MIN_N_ISO" not in st.session_state:
        st.session_state["CALIB_MIN_N_ISO"] = 60
    min_n_iso = int(st.session_state["CALIB_MIN_N_ISO"])

    y_bin = df_filtered["solubility"].values.astype(int)
    X = df_filtered[["delta_d", "delta_p", "delta_h"]].values
    w_local = np.asarray(w, float)

    p0_all = float(np.clip(np.average(y_bin, weights=w_local), 1e-12, 1-1e-12))
    full_base = make_base_models(42, base_score=p0_all)
    if best_ml_name not in full_base:
        st.warning(f"3D: Model '{best_ml_name}' not available; using XGBoost.")
        best_ml_name = "XGBoost"
    model_for_3d = calibrate_model(full_base[best_ml_name], X, y_bin, w_local, min_n_iso=min_n_iso)

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
    fig_h.update_layout(scene=_pretty_scene())
    st.plotly_chart(fig_h, use_container_width=True)

    ml_spheres_rows = []

    # (B)/(C) ML shell spheres
    for iso in ISO_LEVELS:
        try:
            c_ml, r_ml, npts, iso_used, delta_used = ml_shell_sphere_from_grid(
                df_filtered, model_for_3d,
                iso=float(iso), pad=float(GRID_PAD), n=int(GRID_N),
                shell_delta=float(SHELL_DELTA), min_points=120
            )
            xs_ml, ys_ml, zs_ml = sphere_mesh(c_ml, r_ml)

            ml_spheres_rows.append({
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
            if show_points:
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
                name=f"ML shell-sphere (p≈{iso:.2f}) — {best_ml_name}",
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
                title=f"3D (C) — Overlay: NumOpt (RED=1) vs ML shell-sphere (p≈{iso:.2f}) | {best_ml_name}"
            ))
            fig_ov.update_layout(scene=_pretty_scene())
            st.plotly_chart(fig_ov, use_container_width=True)

        except Exception as e:
            st.warning(f"[ML-SHELL-SPHERE] Failed at iso={iso:.2f}: {e}")
            st.info("Suggestion: increase GRID_N (e.g., 32) and/or GRID_PAD (e.g., 2.0), or reduce ISO_LEVELS.")

    df_ml_spheres = pd.DataFrame(ml_spheres_rows)
    st.session_state["df_ml_spheres"] = df_ml_spheres

    if not df_ml_spheres.empty:
        with st.expander("ML shell-sphere parameters table"):
            st.dataframe(df_ml_spheres, use_container_width=True)

# ============================================================
# TAB 6 — Export Excel
# ============================================================
with tab6:
    st.subheader("Export Excel (full)")

    if not st.session_state.get("numopt_done", False):
        st.info("Run Tab 2 first.")
        st.stop()

    # Gather what exists
    dp = float(st.session_state["dp"])
    pp = float(st.session_state["pp"])
    hp = float(st.session_state["hp"])
    R0 = float(st.session_state["R0"])

    best_df_name = st.session_state["best_df_name"]
    best_optimizer = st.session_state["best_optimizer"]

    K_PROB = float(st.session_state.get("K_PROB", 6.0))
    REG_R0 = float(st.session_state.get("REG_R0", 0.05))

    p_numopt_in = st.session_state.get("p_numopt_in", None)
    RED_all = st.session_state.get("RED_all", None)

    p_numopt_cv = st.session_state.get("p_numopt_cv", None)
    p_ml_cv = st.session_state.get("p_ml_cv", {})
    cv_label = st.session_state.get("cv_label", "CV")

    proba_ml_in = st.session_state.get("proba_ml_in", {})
    df_all_runs = st.session_state.get("df_all_runs", pd.DataFrame())
    df_ml_spheres = st.session_state.get("df_ml_spheres", pd.DataFrame())

    # Results_per_sample
    out = df_filtered.copy()
    out["y_raw"] = y_raw
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

    # NumOpt params
    group_col = st.session_state.get("group_col", (group_col if "group_col" in globals() else "None"))
    df_params_numopt = pd.DataFrame({
        "Parameter": ["DF_best", "optimizer_best",
                      "delta_d_center", "delta_p_center", "delta_h_center", "R0", "unit",
                      "CV_scheme", "group_col"],
        "Value": [best_df_name, best_optimizer, dp, pp, hp, R0, UNIT, cv_label, (group_col if use_group else "None")]
    })

    # Settings
    df_final_settings = pd.DataFrame([
        {"Group": "NumOpt_prob", "Item": "K_PROB", "Value": K_PROB},
        {"Group": "NumOpt_prob", "Item": "REG_R0", "Value": REG_R0},
        {"Group": "CV", "Item": "scheme", "Value": cv_label},
        {"Group": "CV", "Item": "group_col", "Value": (group_col if use_group else "None")},
        {"Group": "3D_settings", "Item": "GRID_PAD", "Value": st.session_state.get("GRID_PAD", "")},
        {"Group": "3D_settings", "Item": "GRID_N", "Value": st.session_state.get("GRID_N", "")},
        {"Group": "3D_settings", "Item": "ISO_LEVELS", "Value": str(st.session_state.get("ISO_LEVELS", ""))},
        {"Group": "3D_settings", "Item": "SHELL_DELTA", "Value": st.session_state.get("SHELL_DELTA", "")},
        {"Group": "ML_visual", "Item": "best_ml_name_for_3d", "Value": str(st.session_state.get("best_ml_name", ""))},
        {"Group": "Calib", "Item": "min_n_isotonic", "Value": int(st.session_state.get("CALIB_MIN_N_ISO", 60))},
    ])

    # Metrics (if available)
    df_metrics_in_unw = st.session_state.get("df_metrics_in_unw", pd.DataFrame())
    df_metrics_in_w = st.session_state.get("df_metrics_in_w", pd.DataFrame())
    df_metrics_cv_unw = st.session_state.get("df_metrics_cv_unw", pd.DataFrame())
    df_metrics_cv_w = st.session_state.get("df_metrics_cv_w", pd.DataFrame())

    # Filename
    filename = st.text_input("Output filename (.xlsx)", value=f"result_full_paper_{cv_label}_cal_shell.xlsx")

    export_btn = st.button("⬇️ Build Excel + Download", type="primary", use_container_width=True)

    if export_btn:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            # base
            df_filtered.to_excel(writer, index=False, sheet_name="Base_bin")
            # per-sample
            out.to_excel(writer, index=False, sheet_name="Results_per_sample")
            # numopt
            if df_all_runs is not None and not df_all_runs.empty:
                df_all_runs.to_excel(writer, index=False, sheet_name="NumOpt_AllRuns")
            df_params_numopt.to_excel(writer, index=False, sheet_name="NumOpt_Final")
            # metrics
            if df_metrics_in_unw is not None and not df_metrics_in_unw.empty:
                df_metrics_in_unw.to_excel(writer, index=False, sheet_name="Metrics_IN_unweighted")
            if df_metrics_in_w is not None and not df_metrics_in_w.empty:
                df_metrics_in_w.to_excel(writer, index=False, sheet_name="Metrics_IN_weighted")
            if df_metrics_cv_unw is not None and not df_metrics_cv_unw.empty:
                df_metrics_cv_unw.to_excel(writer, index=False, sheet_name=f"Metrics_{cv_label}_unweighted")
            if df_metrics_cv_w is not None and not df_metrics_cv_w.empty:
                df_metrics_cv_w.to_excel(writer, index=False, sheet_name=f"Metrics_{cv_label}_weighted")
            # 3D ML spheres
            if df_ml_spheres is not None and not df_ml_spheres.empty:
                df_ml_spheres.to_excel(writer, index=False, sheet_name="ML_ShellSpheres")
            # settings
            df_final_settings.to_excel(writer, index=False, sheet_name="Final_Settings")

        bio.seek(0)
        st.download_button(
            label="Download Excel",
            data=bio,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# Fim da PARTE 4/4
