"""src/diagnostics/serial_correlation.py

Wooldridge (2002) test for first-order serial correlation in panel data
(instruct.md §9.6).

Method (Wooldridge 2002, Chapter 10.6.3):
  1. First-difference the model: ΔY_it = ΔX_it β + Δε_it.
  2. Estimate by pooled OLS; save residuals ũ_it = ΔY_it − ΔX_it β̂.
  3. Regress ũ_it on ũ_{i,t-1} (within-entity lag of first-differenced residuals).
  4. Under H0 (no serial correlation in levels), the OLS estimator of the
     coefficient on ũ_{i,t-1} should equal −0.5.
  5. Test H0: α = −0.5 using a standard t-test (cluster-robust SE by entity).

A significant result (reject H0) indicates AR(1) serial correlation in errors,
justifying cluster-robust standard errors.

Note: This is a pure-Python implementation; no rpy2 / R required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import statsmodels.formula.api as smf


def wooldridge_test(
    df: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    entity_col: str = "country_code",
    time_col: str = "year",
) -> dict:
    """Wooldridge (2002) test for AR(1) serial correlation in panel errors.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (long format).
    dependent : str
        Dependent variable column.
    regressors : list[str]
        Regressors for the first-differenced model.
    entity_col, time_col : str
        Panel identifiers.

    Returns
    -------
    dict with keys: statistic (t), p_value, coef_on_lag, decision.
    """
    cols = [entity_col, time_col, dependent] + regressors
    sub = df[cols].dropna().copy().sort_values([entity_col, time_col])

    if sub.empty:
        return {"error": "No complete observations after dropping NaNs."}

    # Step 1: First-difference all variables within entity
    fd_cols = [dependent] + regressors
    sub_fd = sub.copy()
    for col in fd_cols:
        sub_fd[col] = sub_fd.groupby(entity_col)[col].diff()

    sub_fd = sub_fd.dropna().copy()

    if len(sub_fd) < len(regressors) + 10:
        return {"error": "Insufficient first-differenced observations."}

    # Step 2: Pooled OLS on first-differenced model (no constant — FD removes it)
    X = sub_fd[regressors].values
    y = sub_fd[dependent].values

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        return {"error": f"First-difference OLS failed: {exc}"}

    sub_fd = sub_fd.copy()
    sub_fd["_resid_fd"] = y - X @ beta

    # Step 3: Lag residuals within entity
    sub_fd["_resid_fd_lag1"] = sub_fd.groupby(entity_col)["_resid_fd"].shift(1)
    sub_fd = sub_fd.dropna(subset=["_resid_fd_lag1"]).copy()

    if len(sub_fd) < 5:
        return {"error": "Too few observations for lagged residual regression."}

    # Step 4: Regress ũ_it on ũ_{i,t-1} (no constant)
    y2 = sub_fd["_resid_fd"].values
    X2 = sub_fd["_resid_fd_lag1"].values.reshape(-1, 1)

    try:
        alpha_hat, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
        alpha_hat = float(alpha_hat[0])
    except np.linalg.LinAlgError as exc:
        return {"error": f"Lag regression failed: {exc}"}

    # Step 5: Cluster-robust SE by entity for t-test H0: α = -0.5
    sub_fd = sub_fd.copy()
    sub_fd["_yhat2"] = alpha_hat * sub_fd["_resid_fd_lag1"]
    sub_fd["_e2"] = sub_fd["_resid_fd"] - sub_fd["_yhat2"]

    entities = sub_fd[entity_col].unique()
    N = len(entities)

    # Meat of cluster-robust sandwich
    meat = 0.0
    for ent in entities:
        mask = sub_fd[entity_col] == ent
        xi = sub_fd.loc[mask, "_resid_fd_lag1"].values.reshape(-1, 1)
        ei = sub_fd.loc[mask, "_e2"].values.reshape(-1, 1)
        score = xi.T @ ei  # (1 × 1)
        meat += float(score @ score.T)

    bread = float((X2.T @ X2))  # scalar
    if bread == 0:
        return {"error": "Zero bread matrix — collinear regressors."}

    bread_inv = 1.0 / bread
    # Small-sample correction
    correction = (N / (N - 1))
    var_alpha = correction * bread_inv * meat * bread_inv
    se_alpha = float(np.sqrt(var_alpha))

    if se_alpha == 0:
        return {"error": "Zero standard error."}

    # t-statistic testing H0: α = -0.5
    t_stat = float((alpha_hat - (-0.5)) / se_alpha)
    dof = N - 1
    pval = float(2 * t_dist.sf(abs(t_stat), df=dof))

    return {
        "test": "Wooldridge (2002)",
        "coef_on_lagged_fd_resid": round(alpha_hat, 4),
        "h0_value": -0.5,
        "statistic": round(t_stat, 4),
        "p_value": round(pval, 4),
        "degrees_of_freedom": dof,
        "n_entities": N,
        "n_obs": len(sub_fd),
        "h0": "No first-order serial correlation (α = −0.5)",
        "decision": "Reject H0 (serial correlation present)" if pval < 0.05 else "Fail to reject H0 (no serial correlation)",
    }
