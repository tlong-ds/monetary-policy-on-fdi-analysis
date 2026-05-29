"""src/diagnostics/heteroskedasticity.py

Modified Wald test for group-wise heteroskedasticity (instruct.md §9.5).

Tests H0: σ²_i = σ² for all i (homoskedastic across countries).
Procedure (Greene 2000, §11.5):
  1. Estimate the within (FE) model; collect group residuals.
  2. Compute group-specific variance σ̂²_i for each country i.
  3. Compute the pooled variance σ̂² = Σ_i Σ_t ê²_it / Σ_i (T_i - K).
  4. Test statistic: W = Σ_i T_i * (σ̂²_i / σ̂² - 1)²
     Under H0: W ~ χ²(N-1), where N = number of entities.

A significant p-value indicates heteroskedasticity across panels,
justifying cluster-robust or Driscoll-Kraay standard errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _within_demean(df: pd.DataFrame, cols: list[str], entity_col: str) -> pd.DataFrame:
    """Within-transform (demean) columns by entity."""
    out = df.copy()
    for col in cols:
        out[col] = out[col] - out.groupby(entity_col)[col].transform("mean")
    return out


def modified_wald_test(
    df: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    entity_col: str = "country_code",
    time_col: str = "year",
) -> dict:
    """Run the Modified Wald test for group-wise heteroskedasticity.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (long format).
    dependent : str
        Dependent variable column name.
    regressors : list[str]
        Regressor column names (excluding entity/time dummies).
    entity_col, time_col : str
        Panel dimension identifiers.

    Returns
    -------
    dict with keys: statistic, p_value, df, n_entities, decision.
    """
    cols = [entity_col, time_col, dependent] + regressors
    sub = df[cols].dropna().copy()

    if sub.empty:
        return {"error": "No complete observations after dropping NaNs."}

    entities = sub[entity_col].unique()
    N = len(entities)
    if N < 2:
        return {"error": "Need at least 2 entities."}

    K = len(regressors)

    # Within-demean all variables (equivalent to FE demeaning)
    demean_cols = [dependent] + regressors
    sub_w = _within_demean(sub, demean_cols, entity_col)

    # OLS on demeaned data (no constant) → FE residuals
    X = sub_w[regressors].values
    y = sub_w[dependent].values

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        return {"error": f"OLS failed: {exc}"}

    resid = y - X @ beta
    sub_w = sub_w.copy()
    sub_w["_resid"] = resid

    # Group-specific variances σ̂²_i = Σ_t ê²_it / (T_i - K)
    group_stats = sub_w.groupby(entity_col).apply(
        lambda g: pd.Series(
            {
                "T_i": len(g),
                "ss_i": float((g["_resid"] ** 2).sum()),
            }
        )
    ).reset_index()

    group_stats["sigma2_i"] = group_stats["ss_i"] / (group_stats["T_i"] - K).clip(lower=1)

    # Pooled variance σ̂² = Σ_i ss_i / (Σ_i (T_i - K))
    total_ss = group_stats["ss_i"].sum()
    total_df = (group_stats["T_i"] - K).clip(lower=1).sum()
    sigma2_pool = total_ss / total_df if total_df > 0 else np.nan

    if np.isnan(sigma2_pool) or sigma2_pool <= 0:
        return {"error": "Pooled variance is zero or undefined."}

    # Modified Wald statistic W = Σ_i T_i * (σ̂²_i / σ̂² - 1)²
    W = float(
        (group_stats["T_i"] * ((group_stats["sigma2_i"] / sigma2_pool) - 1) ** 2).sum()
    )
    dof = N - 1
    pval = float(1 - chi2.cdf(W, dof))

    return {
        "test": "Modified Wald",
        "statistic": round(W, 4),
        "p_value": round(pval, 4),
        "degrees_of_freedom": dof,
        "n_entities": N,
        "pooled_variance": round(float(sigma2_pool), 6),
        "h0": "Homoskedasticity across groups (σ²_i = σ² for all i)",
        "decision": "Reject H0 (heteroskedastic)" if pval < 0.05 else "Fail to reject H0 (homoskedastic)",
    }
