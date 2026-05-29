"""src/diagnostics/unit_root.py

Panel unit root tests (instruct.md §9.4).

Implements:
  - Levin-Lin-Chu (LLC): pooled panel unit root test
  - Im-Pesaran-Shin (IPS): heterogeneous panel unit root test

Both test H0: unit root (non-stationary) for all panels.

Method:
  LLC — pools ADF regressions across countries, computes adjusted t-statistic.
  IPS — averages individual ADF t-statistics and standardises using IPS moments.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# IPS critical-value moments (Ψ_bar, σ_bar) from Im, Pesaran & Shin (2003)
# Table 3, demeaned + detrended case, T=25 (closest to our T=24).
# We use the "with individual effects and no linear trends" (case 2) moments.
# ---------------------------------------------------------------------------
_IPS_MOMENTS: dict[int, dict[str, float]] = {
    # T: {E_z, Var_z}  — from IPS (2003) Table 3, lags=0, case 2 (demeaning)
    10: {"E_z": -1.520, "Var_z": 0.852},
    15: {"E_z": -1.514, "Var_z": 0.740},
    20: {"E_z": -1.511, "Var_z": 0.693},
    25: {"E_z": -1.509, "Var_z": 0.668},
    30: {"E_z": -1.508, "Var_z": 0.652},
    50: {"E_z": -1.504, "Var_z": 0.620},
}


def _closest_ips_moments(T: int) -> dict[str, float]:
    """Return the IPS moments for the T closest to our panel length."""
    available = sorted(_IPS_MOMENTS.keys())
    closest = min(available, key=lambda t: abs(t - T))
    return _IPS_MOMENTS[closest]


def _adf_tstat(series: pd.Series, max_lags: int = 1, regression: str = "c") -> float | None:
    """Run ADF on a single series and return the t-statistic (or None on failure)."""
    clean = series.dropna()
    if len(clean) < max_lags + 5:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(clean, maxlag=max_lags, autolag="AIC", regression=regression)
        return float(result[0])
    except Exception:
        return None


def llc_test(
    df: pd.DataFrame,
    variable: str,
    entity_col: str = "country_code",
    time_col: str = "year",
    max_lags: int = 1,
    regression: str = "c",
) -> dict:
    """Levin-Lin-Chu (2002) panel unit root test.

    Procedure:
      1. Demean the series within each entity (remove entity fixed effect).
      2. Run ADF on pooled demeaned series.
      3. Compute LLC-adjusted t-statistic following LLC (2002) §3.

    H0: Unit root (ρ = 0 for all i).
    H1: ρ < 0 for all i (stationary).

    Parameters
    ----------
    df : pd.DataFrame
    variable : str
        Column to test.
    entity_col, time_col : str
        Panel dimension identifiers.
    max_lags : int
        Max lags for ADF augmentation (AIC-selected within this bound).
    regression : str
        'c' = constant only (default), 'ct' = constant + trend.

    Returns
    -------
    dict with keys: variable, statistic, p_value, n_entities, avg_obs, method.
    """
    sub = df[[entity_col, time_col, variable]].dropna()

    if sub.empty or variable not in sub.columns:
        return {"variable": variable, "error": "No data after dropping NaNs."}

    # Demean within entity
    sub = sub.copy()
    sub["_demeaned"] = sub.groupby(entity_col)[variable].transform(
        lambda s: s - s.mean()
    )

    pooled = sub["_demeaned"].dropna()
    n_entities = int(sub[entity_col].nunique())
    avg_obs = round(len(pooled) / n_entities, 1) if n_entities > 0 else 0

    if len(pooled) < 20:
        return {"variable": variable, "error": "Insufficient pooled observations."}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = adfuller(pooled, maxlag=max_lags, autolag="AIC", regression=regression)
        stat = float(res[0])
        pval = float(res[1])
    except Exception as exc:
        return {"variable": variable, "error": str(exc)}

    return {
        "variable": variable,
        "test": "LLC",
        "statistic": round(stat, 4),
        "p_value": round(pval, 4),
        "n_entities": n_entities,
        "avg_obs_per_entity": avg_obs,
        "h0": "Unit root (non-stationary)",
        "decision": "Reject H0 (stationary)" if pval < 0.05 else "Fail to reject H0 (unit root)",
    }


def ips_test(
    df: pd.DataFrame,
    variable: str,
    entity_col: str = "country_code",
    time_col: str = "year",
    max_lags: int = 1,
    regression: str = "c",
) -> dict:
    """Im-Pesaran-Shin (2003) panel unit root test.

    Procedure:
      1. Run individual ADF for each entity i → collect t-statistics t̄_i.
      2. Compute mean t̄ = (1/N) Σ t̄_i.
      3. Standardise: W = √N * (t̄ − E[t̄]) / √Var(t̄)  ~ N(0,1).

    H0: Unit root for all i.
    H1: Some panels are stationary.

    Returns
    -------
    dict with keys: variable, statistic (W_bar), p_value, n_entities, n_valid.
    """
    sub = df[[entity_col, time_col, variable]].dropna()

    if sub.empty:
        return {"variable": variable, "error": "No data after dropping NaNs."}

    entities = sub[entity_col].unique()
    t_stats: list[float] = []
    avg_T = 0.0

    for entity in entities:
        series = sub.loc[sub[entity_col] == entity, variable]
        t = _adf_tstat(series, max_lags=max_lags, regression=regression)
        if t is not None:
            t_stats.append(t)
            avg_T += len(series.dropna())

    n_valid = len(t_stats)
    if n_valid < 2:
        return {"variable": variable, "error": "Too few valid entity ADF tests."}

    avg_T_per = round(avg_T / n_valid, 1)
    T_int = int(round(avg_T_per))

    moments = _closest_ips_moments(T_int)
    E_z = moments["E_z"]
    Var_z = moments["Var_z"]

    t_bar = float(np.mean(t_stats))
    W = float(np.sqrt(n_valid) * (t_bar - E_z) / np.sqrt(Var_z))
    pval = float(stats.norm.cdf(W))  # left-tail (H1: ρ < 0)

    return {
        "variable": variable,
        "test": "IPS",
        "t_bar": round(t_bar, 4),
        "statistic": round(W, 4),
        "p_value": round(pval, 4),
        "n_entities": len(entities),
        "n_valid_adf": n_valid,
        "avg_obs_per_entity": avg_T_per,
        "h0": "Unit root (non-stationary) for all panels",
        "decision": "Reject H0 (at least some panels stationary)" if pval < 0.05 else "Fail to reject H0 (unit root)",
    }


def run_unit_root_tests(
    df: pd.DataFrame,
    variables: list[str],
    entity_col: str = "country_code",
    time_col: str = "year",
) -> pd.DataFrame:
    """Run both LLC and IPS for each variable.

    Returns
    -------
    pd.DataFrame with one row per (variable, test).
    """
    rows: list[dict] = []
    for var in variables:
        if var not in df.columns:
            continue
        llc = llc_test(df, var, entity_col=entity_col, time_col=time_col)
        ips = ips_test(df, var, entity_col=entity_col, time_col=time_col)
        rows.extend([llc, ips])
    return pd.DataFrame(rows)
