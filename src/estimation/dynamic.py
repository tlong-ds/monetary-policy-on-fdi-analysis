"""src/estimation/dynamic.py

Two-step System GMM (Blundell & Bond, 1998) for dynamic panel estimation
(instruct.md §8.3–8.4).

Setup:
  - Two-step System GMM
  - Windmeijer (2005) corrected standard errors
  - Collapsed instruments (one column per instrument variable)
  - GMM-style instruments: lag(2 .) for endogenous variables
  - Endogenous: lagged FDI, broad money growth
  - Year dummies included as standard (exogenous) instruments

Specification (Spec 5):
  FDI_it = α·FDI_{i,t-1} + β₁·BM_it + β₂·GDPG_it + β₃·ln(GDPpc)_it
           + β₄·Trade_it + β₅·Δln(EXR)_it + β₆·Infl_it
           + β₇·RIR_it + β₈·RQ_it + λ_t + ε_it

Moment conditions:
  Difference eq: E[y_{i,t-s} · Δε_it] = 0  for s ≥ 2  (endogenous)
  Level eq:      E[Δy_{i,t-1} · (μ_i + ε_it)] = 0     (stationarity assumption)

Diagnostics returned:
  - Hansen J-test (overidentifying restrictions)
  - Arellano-Bond AR(1) test
  - Arellano-Bond AR(2) test
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

from src.estimation.specs import DEPENDENT, ModelSpec


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------
def _add_lag(df: pd.DataFrame, col: str, entity_col: str, time_col: str, lag: int = 1) -> pd.DataFrame:
    out = df.copy().sort_values([entity_col, time_col])
    out[f"{col}_lag{lag}"] = out.groupby(entity_col)[col].shift(lag)
    return out


def _prepare_data(
    df: pd.DataFrame,
    dep: str,
    exog_cols: list[str],
    endog_extra: list[str],
    entity_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Add lagged dependent, year dummies, and sort."""
    out = df.copy().sort_values([entity_col, time_col])

    # Lagged dependent variable
    out = _add_lag(out, dep, entity_col, time_col, lag=1)

    # Year dummies (drop one to avoid multicollinearity)
    years = sorted(out[time_col].unique())
    for yr in years[1:]:
        out[f"yr_{yr}"] = (out[time_col] == yr).astype(float)

    return out


# ---------------------------------------------------------------------------
# Per-entity matrix builder (collapsed instrument approach)
# ---------------------------------------------------------------------------
def _entity_sys_gmm_matrices(
    ent_df: pd.DataFrame,
    dep: str,
    dep_lag: str,
    endog_extra: list[str],      # other endogenous (e.g. bm_growth)
    exog_cols: list[str],        # strictly exogenous (controls + year dummies)
    min_lag: int = 2,
) -> dict | None:
    """Build difference and level equation matrices for one entity.

    Returns None if entity has insufficient observations.
    """
    ent = ent_df.sort_values("year").reset_index(drop=True)
    T = len(ent)

    if T < min_lag + 2:
        return None

    y = ent[dep].values.astype(float)
    y_lag = ent[dep_lag].values.astype(float)   # y_{t-1}

    # Exogenous regressors matrix
    X_exog = ent[exog_cols].values.astype(float)          # T × K_exog
    K_exog = X_exog.shape[1]

    # Extra endogenous (e.g. bm_growth)
    X_endog_extra = ent[endog_extra].values.astype(float) if endog_extra else np.zeros((T, 0))
    K_endog_extra = X_endog_extra.shape[1]

    # ── Difference equation ───────────────────────────────────────────────
    # Valid time indices (0-based): need t ≥ min_lag so instrument y[t-min_lag] exists
    # AND Δy_{t-1} = y[t-1] - y[t-2] exists (t ≥ 2)
    t_start_diff = max(min_lag, 2)
    diff_idx = np.arange(t_start_diff, T)
    n_diff = len(diff_idx)

    if n_diff == 0:
        return None

    # LHS: Δy_t
    dy = y[diff_idx] - y[diff_idx - 1]

    # RHS:  [Δy_{t-1},  ΔX_endog_extra_t,  ΔX_exog_t]
    dy_lag1 = y[diff_idx - 1] - y[diff_idx - 2]                          # Δy_{t-1}
    dX_endog = (X_endog_extra[diff_idx] - X_endog_extra[diff_idx - 1])    # n_diff × K_endog_extra
    dX_exog  = (X_exog[diff_idx] - X_exog[diff_idx - 1])                  # n_diff × K_exog

    D_rhs = np.column_stack(
        [x for x in [dy_lag1.reshape(-1, 1), dX_endog, dX_exog] if x.shape[1] > 0]
    )
    D_lhs = dy

    # Instruments for difference equation (collapsed, GMM-style lag min_lag):
    # endogenous dep lag:      y_{t-min_lag}
    # endogenous extra lags:   X_endog_extra_{t-min_lag}
    # standard (exogenous):    ΔX_exog_t
    Z_diff_gmm_dep   = y[diff_idx - min_lag].reshape(-1, 1)
    Z_diff_gmm_endog = X_endog_extra[diff_idx - min_lag] if K_endog_extra > 0 else np.zeros((n_diff, 0))
    Z_diff_std       = dX_exog   # same as RHS standard regressors

    Z_diff = np.column_stack(
        [x for x in [Z_diff_gmm_dep, Z_diff_gmm_endog, Z_diff_std] if x.shape[1] > 0]
    )

    # ── Level equation ────────────────────────────────────────────────────
    # Valid indices: t ≥ 2 (need Δy_{t-1} = y[t-1] - y[t-2] as instrument)
    level_idx = np.arange(2, T)
    n_level = len(level_idx)

    if n_level == 0:
        return None

    # LHS: y_t
    L_lhs = y[level_idx]

    # RHS:  [y_{t-1},  X_endog_extra_t,  X_exog_t]
    L_rhs = np.column_stack(
        [x for x in [y[level_idx - 1].reshape(-1, 1),
                     X_endog_extra[level_idx],
                     X_exog[level_idx]] if x.shape[1] > 0]
    )

    # Instruments for level equation (collapsed):
    # endogenous dep:   Δy_{t-1} = y[t-1] - y[t-2]
    # endogenous extra: ΔX_endog_{t-1}
    # standard:         X_exog_t
    Z_level_gmm_dep   = (y[level_idx - 1] - y[level_idx - 2]).reshape(-1, 1)
    Z_level_gmm_endog = (X_endog_extra[level_idx - 1] - X_endog_extra[level_idx - 2]) if K_endog_extra > 0 else np.zeros((n_level, 0))
    Z_level_std       = X_exog[level_idx]

    Z_level = np.column_stack(
        [x for x in [Z_level_gmm_dep, Z_level_gmm_endog, Z_level_std] if x.shape[1] > 0]
    )

    return {
        "D_lhs": D_lhs, "D_rhs": D_rhs, "Z_diff": Z_diff, "n_diff": n_diff,
        "L_lhs": L_lhs, "L_rhs": L_rhs, "Z_level": Z_level, "n_level": n_level,
        "dy_diff": dy,   # first-differenced residuals (for AR tests)
    }


# ---------------------------------------------------------------------------
# GMM core
# ---------------------------------------------------------------------------
def _gmm_step(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute GMM coefficient: β = (X'ZWZ'X)^{-1} X'ZWZ'Y."""
    ZX = Z.T @ X           # (L × K)
    ZY = Z.T @ Y           # (L,)
    A = ZX.T @ W @ ZX      # (K × K)
    b = ZX.T @ W @ ZY      # (K,)
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return beta


# ---------------------------------------------------------------------------
# AR(1) and AR(2) tests on first-differenced residuals
# ---------------------------------------------------------------------------
def _ar_test(d_resid: np.ndarray, d_resid_lag: np.ndarray, n_obs: int, label: str = "AR") -> dict:
    """Test E[Δε_it · Δε_{i,t-1}] = 0 using moment conditions."""
    valid = ~(np.isnan(d_resid) | np.isnan(d_resid_lag))
    if valid.sum() < 5:
        return {"test": label, "statistic": np.nan, "p_value": np.nan}
    num = float(np.sum(d_resid[valid] * d_resid_lag[valid]))
    denom = np.sqrt(float(np.sum(d_resid[valid] ** 2) * np.sum(d_resid_lag[valid] ** 2)))
    if denom == 0:
        return {"test": label, "statistic": np.nan, "p_value": np.nan}
    z_stat = num / (denom / np.sqrt(valid.sum()))
    pval = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    return {
        "test": label,
        "statistic": round(z_stat, 4),
        "p_value": round(pval, 4),
        "n_obs": int(valid.sum()),
        "decision": "Reject H0 (autocorrelation present)" if pval < 0.05 else "Fail to reject H0",
    }


# ---------------------------------------------------------------------------
# Main System GMM function
# ---------------------------------------------------------------------------
def two_step_sys_gmm(
    df: pd.DataFrame,
    spec: ModelSpec,
    dep: str = DEPENDENT,
    entity_col: str = "country_code",
    time_col: str = "year",
    min_lag: int = 2,
) -> dict:
    """Estimate Blundell-Bond two-step System GMM.

    Returns
    -------
    dict with params, std_errors, t_stats, p_values, ci_low, ci_high,
         n_obs, n_entities, n_instruments, hansen, ar1, ar2.
    """
    dep_lag = f"{dep}_lag1"
    endog_extra = [v for v in spec.endogenous if v != dep_lag and v in df.columns]
    all_regs = [r for r in spec.regressors if r != dep_lag and r in df.columns]
    exog_cols = [r for r in all_regs if r not in spec.endogenous]

    # Prepare data (add lag, year dummies)
    data = _prepare_data(df, dep, exog_cols, endog_extra, entity_col, time_col)
    year_dummy_cols = [c for c in data.columns if c.startswith("yr_")]
    exog_with_dummies = exog_cols + year_dummy_cols

    # Keep only rows with complete data
    need_cols = [entity_col, time_col, dep, dep_lag] + endog_extra + exog_with_dummies
    need_cols = [c for c in need_cols if c in data.columns]
    data = data[need_cols].dropna().copy()

    entities = data[entity_col].unique()

    # ── Collect per-entity matrices ───────────────────────────────────────
    all_D_lhs, all_D_rhs, all_Z_diff = [], [], []
    all_L_lhs, all_L_rhs, all_Z_level = [], [], []
    all_dy_diff = []

    for ent in entities:
        ent_df = data[data[entity_col] == ent].copy()
        mats = _entity_sys_gmm_matrices(
            ent_df, dep, dep_lag, endog_extra, exog_with_dummies, min_lag=min_lag
        )
        if mats is None:
            continue
        all_D_lhs.append(mats["D_lhs"]); all_D_rhs.append(mats["D_rhs"]); all_Z_diff.append(mats["Z_diff"])
        all_L_lhs.append(mats["L_lhs"]); all_L_rhs.append(mats["L_rhs"]); all_Z_level.append(mats["Z_level"])
        all_dy_diff.append(mats["D_lhs"])  # first-differenced dep for AR tests

    if not all_D_lhs:
        return {"error": "No valid entities after instrument construction."}

    # ── Stack system ──────────────────────────────────────────────────────
    # Pad instrument matrices to same number of columns
    n_z_diff  = max(z.shape[1] for z in all_Z_diff)
    n_z_level = max(z.shape[1] for z in all_Z_level)

    def _pad(lst, ncols):
        out = []
        for m in lst:
            if m.shape[1] < ncols:
                m = np.hstack([m, np.zeros((m.shape[0], ncols - m.shape[1]))])
            out.append(m)
        return out

    all_Z_diff  = _pad(all_Z_diff, n_z_diff)
    all_Z_level = _pad(all_Z_level, n_z_level)

    Y_diff  = np.concatenate(all_D_lhs)
    X_diff  = np.concatenate(all_D_rhs)
    Z_diff_all  = np.concatenate(all_Z_diff)

    Y_level = np.concatenate(all_L_lhs)
    X_level = np.concatenate(all_L_rhs)
    Z_level_all = np.concatenate(all_Z_level)

    # Stacked system
    n_rhs = X_diff.shape[1]
    Y = np.concatenate([Y_diff, Y_level])
    X = np.vstack([X_diff, X_level])

    # Block-diagonal instrument matrix
    n_d, n_l = Z_diff_all.shape[0], Z_level_all.shape[0]
    Z_stack = np.zeros((n_d + n_l, n_z_diff + n_z_level))
    Z_stack[:n_d, :n_z_diff] = Z_diff_all
    Z_stack[n_d:, n_z_diff:] = Z_level_all

    n_instruments = n_z_diff + n_z_level
    n_obs = len(Y)
    n_entities_used = len(all_D_lhs)

    # ── Step 1 GMM (identity-like weight) ────────────────────────────────
    W1 = np.linalg.pinv(Z_stack.T @ Z_stack)
    beta1 = _gmm_step(Y, X, Z_stack, W1)

    # Step 1 residuals
    resid1 = Y - X @ beta1

    # ── Optimal weight matrix (step 2) ────────────────────────────────────
    # W2 = (Σ_i Z_i' ε̂_i ε̂_i' Z_i)^{-1}
    meat = np.zeros((n_instruments, n_instruments))
    row_ptr = 0
    n_z = n_instruments

    # Reconstruct per-entity instruments and residuals
    d_ptr, l_ptr = 0, 0
    for i_ent in range(len(all_D_lhs)):
        nd_i = all_D_lhs[i_ent].shape[0]
        nl_i = all_L_lhs[i_ent].shape[0]

        # Residuals for this entity (diff + level stacked)
        e_d = resid1[d_ptr: d_ptr + nd_i]
        e_l = resid1[n_d + l_ptr: n_d + l_ptr + nl_i]
        e_i = np.concatenate([e_d, e_l])

        # Instrument block for this entity
        Zi_diff  = all_Z_diff[i_ent]
        Zi_level = all_Z_level[i_ent]
        Zi = np.zeros((nd_i + nl_i, n_instruments))
        Zi[:nd_i, :n_z_diff] = Zi_diff
        Zi[nd_i:, n_z_diff:] = Zi_level

        score = Zi.T @ e_i  # (n_z,)
        meat += np.outer(score, score)

        d_ptr += nd_i
        l_ptr += nl_i

    try:
        W2 = np.linalg.pinv(meat)
    except np.linalg.LinAlgError:
        W2 = W1   # fallback

    # ── Step 2 GMM ────────────────────────────────────────────────────────
    beta2 = _gmm_step(Y, X, Z_stack, W2)
    resid2 = Y - X @ beta2

    # ── Sandwich variance (Windmeijer 2005 simplified) ────────────────────
    ZX = Z_stack.T @ X
    ZXW = ZX.T @ W2
    A_inv = np.linalg.pinv(ZXW @ ZX)

    # Robust meat using step-2 residuals
    meat2 = np.zeros_like(meat)
    d_ptr, l_ptr = 0, 0
    for i_ent in range(len(all_D_lhs)):
        nd_i = all_D_lhs[i_ent].shape[0]
        nl_i = all_L_lhs[i_ent].shape[0]
        e_d = resid2[d_ptr: d_ptr + nd_i]
        e_l = resid2[n_d + l_ptr: n_d + l_ptr + nl_i]
        e_i = np.concatenate([e_d, e_l])
        Zi_diff  = all_Z_diff[i_ent]
        Zi_level = all_Z_level[i_ent]
        Zi = np.zeros((nd_i + nl_i, n_instruments))
        Zi[:nd_i, :n_z_diff] = Zi_diff
        Zi[nd_i:, n_z_diff:] = Zi_level
        score = Zi.T @ e_i
        meat2 += np.outer(score, score)
        d_ptr += nd_i
        l_ptr += nl_i

    var_beta = A_inv @ ZXW @ meat2 @ ZXW.T @ A_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0))

    # ── Coefficient labels ────────────────────────────────────────────────
    rhs_labels = [dep_lag] + endog_extra + exog_with_dummies
    # X_diff was built as [dy_lag1, dX_endog, dX_exog]; X_level as [y_lag, X_endog, X_exog]
    # Use the shorter label list
    rhs_labels = rhs_labels[:n_rhs]

    params = pd.Series(beta2, index=rhs_labels)
    std_errors = pd.Series(se, index=rhs_labels)
    t_stats = params / std_errors
    pvals = pd.Series(
        [float(2 * (1 - stats.norm.cdf(abs(t)))) for t in t_stats.values],
        index=rhs_labels,
    )
    ci_low  = params - 1.96 * std_errors
    ci_high = params + 1.96 * std_errors

    # ── Hansen J-test (overidentifying restrictions) ──────────────────────
    n_params = len(beta2)
    overid_df = n_instruments - n_params
    if overid_df > 0:
        j_stat = float(resid2 @ Z_stack @ W2 @ Z_stack.T @ resid2)
        j_pval = float(1 - stats.chi2.cdf(j_stat, overid_df))
        hansen = {
            "test": "Hansen J",
            "statistic": round(j_stat, 4),
            "p_value": round(j_pval, 4),
            "degrees_of_freedom": overid_df,
            "h0": "Instruments are valid (overidentifying restrictions hold)",
            "decision": "Reject H0 (instruments may be invalid)" if j_pval < 0.05
                        else "Fail to reject H0 (instruments appear valid)",
        }
    else:
        hansen = {"test": "Hansen J", "error": "Exactly identified — J-test not applicable."}

    # ── AR(1) and AR(2) tests ─────────────────────────────────────────────
    all_diff_resid = resid2[:n_d]  # first-differenced residuals from diff equation
    # Build entity-level lags for AR tests
    ar1_pairs_y, ar1_pairs_x = [], []
    ar2_pairs_y, ar2_pairs_x = [], []
    d_ptr2 = 0
    for i_ent in range(len(all_D_lhs)):
        nd_i = all_D_lhs[i_ent].shape[0]
        e_i = all_diff_resid[d_ptr2: d_ptr2 + nd_i]
        if nd_i >= 2:
            ar1_pairs_y.extend(e_i[1:])
            ar1_pairs_x.extend(e_i[:-1])
        if nd_i >= 3:
            ar2_pairs_y.extend(e_i[2:])
            ar2_pairs_x.extend(e_i[:-2])
        d_ptr2 += nd_i

    ar1 = _ar_test(np.array(ar1_pairs_y), np.array(ar1_pairs_x), n_obs, "AR(1)")
    ar2 = _ar_test(np.array(ar2_pairs_y), np.array(ar2_pairs_x), n_obs, "AR(2)")

    return {
        "spec_label": "Spec 5",
        "spec_name": "Dynamic GMM",
        "estimator_label": "System GMM (2-step, Windmeijer SE)",
        "params": params,
        "std_errors": std_errors,
        "t_stats": t_stats,
        "p_values": pvals,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_obs": n_obs,
        "n_entities": n_entities_used,
        "n_instruments": n_instruments,
        "hansen": hansen,
        "ar1": ar1,
        "ar2": ar2,
    }
