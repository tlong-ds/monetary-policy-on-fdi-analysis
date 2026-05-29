"""src/diagnostics/cross_section.py

Pesaran (2004) CD test for cross-sectional dependence (instruct.md §9.7).

Tests H0: no cross-sectional dependence (errors are cross-sectionally independent).

Procedure:
  1. Estimate FE model; collect residuals.
  2. For each pair (i, j), compute pairwise residual correlation ρ̂_ij
     using the T observations they share.
  3. CD statistic = √(2T / N(N−1)) × Σ_{i<j} √T_ij × ρ̂_ij  ~ N(0,1).

A significant p-value indicates cross-sectional dependence, which may bias
standard errors if not corrected (e.g., Driscoll-Kraay SE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def pesaran_cd_test(
    df: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    entity_col: str = "country_code",
    time_col: str = "year",
) -> dict:
    """Pesaran (2004) CD test for cross-sectional dependence.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (long format).
    dependent : str
        Dependent variable column.
    regressors : list[str]
        Regressors (used to compute FE residuals via within-demeaning).
    entity_col, time_col : str
        Panel dimension identifiers.

    Returns
    -------
    dict with keys: statistic, p_value, n_pairs, avg_pair_obs, decision.
    """
    cols = [entity_col, time_col, dependent] + regressors
    sub = df[cols].dropna().copy().sort_values([entity_col, time_col])

    if sub.empty:
        return {"error": "No complete observations after dropping NaNs."}

    # Within-demean (FE)
    demean_cols = [dependent] + regressors
    for col in demean_cols:
        sub[col] = sub[col] - sub.groupby(entity_col)[col].transform("mean")

    # OLS residuals from within model
    X = sub[regressors].values
    y = sub[dependent].values
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        return {"error": f"OLS failed: {exc}"}

    sub = sub.copy()
    sub["_resid"] = y - X @ beta

    # Pivot residuals: rows = year, cols = entity
    pivot = sub.pivot_table(index=time_col, columns=entity_col, values="_resid")
    entities = list(pivot.columns)
    N = len(entities)

    if N < 2:
        return {"error": "Need at least 2 entities for CD test."}

    terms: list[float] = []
    pair_obs: list[int] = []

    for i_idx in range(N):
        for j_idx in range(i_idx + 1, N):
            ei = entities[i_idx]
            ej = entities[j_idx]
            pair = pivot[[ei, ej]].dropna()
            T_ij = len(pair)
            if T_ij < 3:
                continue
            rho = float(pair[ei].corr(pair[ej]))
            if np.isnan(rho):
                continue
            terms.append(np.sqrt(T_ij) * rho)
            pair_obs.append(T_ij)

    n_pairs = len(terms)
    if n_pairs == 0:
        return {"error": "No valid cross-sectional pairs with ≥ 3 shared observations."}

    # CD statistic
    cd_stat = float(np.sqrt(2.0 / (N * (N - 1))) * sum(terms))
    pval = float(2 * (1 - norm.cdf(abs(cd_stat))))

    return {
        "test": "Pesaran CD",
        "statistic": round(cd_stat, 4),
        "p_value": round(pval, 4),
        "n_entities": N,
        "n_pairs": n_pairs,
        "avg_pair_obs": round(float(np.mean(pair_obs)), 1),
        "h0": "No cross-sectional dependence",
        "decision": "Reject H0 (cross-sectional dependence)" if pval < 0.05 else "Fail to reject H0 (no CSD)",
    }
