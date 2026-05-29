"""src/diagnostics/descriptive.py

Descriptive statistics and correlation matrix (instruct.md §9.1–9.2).

Outputs:
  - Summary table: mean, median, SD, min, max per variable
  - Pairwise Pearson correlation matrix among explanatory variables
  - Flagged pairs with |r| > 0.7 (potential collinearity concern)
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# Variables included in model diagnostics (instruct.md §4 / §10)
MODEL_VARIABLES: list[str] = [
    "fdi_pct_gdp",
    "bm_growth",
    "real_interest_rate",
    "gdp_growth",
    "ln_gdppc",
    "trade_pct_gdp",
    "d_ln_exr",
    "inflation",
    "rq",
    "rl",
    "wgi_composite",
]

VARIABLE_LABELS: dict[str, str] = {
    "fdi_pct_gdp":        "FDI (% GDP)",
    "bm_growth":          "Broad Money Growth (%)",
    "real_interest_rate": "Real Interest Rate (%)",
    "gdp_growth":         "GDP Growth (%)",
    "ln_gdppc":           "ln(GDP per Capita)",
    "trade_pct_gdp":      "Trade (% GDP)",
    "d_ln_exr":           "Δln(Exchange Rate) (%)",
    "inflation":          "Inflation (%)",
    "rq":                 "Regulatory Quality",
    "rl":                 "Rule of Law",
    "wgi_composite":      "WGI Composite",
}


def descriptive_stats(
    df: pd.DataFrame,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Compute mean, median, SD, min, max for each variable.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (long format).
    variables : list[str] | None
        Subset of columns to summarise. Defaults to MODEL_VARIABLES.

    Returns
    -------
    pd.DataFrame with columns: variable, label, n, mean, median, std, min, max.
    """
    vars_ = variables or MODEL_VARIABLES
    present = [v for v in vars_ if v in df.columns]

    rows: list[dict] = []
    for var in present:
        series = df[var].dropna()
        rows.append(
            {
                "variable": var,
                "label": VARIABLE_LABELS.get(var, var),
                "n": int(len(series)),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )

    return pd.DataFrame(rows)


def correlation_matrix(
    df: pd.DataFrame,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (long format).
    variables : list[str] | None
        Columns to include. Defaults to MODEL_VARIABLES (excl. dependent).

    Returns
    -------
    pd.DataFrame (square correlation matrix, variable names as index & columns).
    """
    # Exclude dependent variable from correlation matrix of explanatory vars
    explanatory = [
        v for v in (variables or MODEL_VARIABLES)
        if v != "fdi_pct_gdp" and v in df.columns
    ]
    corr = df[explanatory].corr(method="pearson")
    corr.index = [VARIABLE_LABELS.get(v, v) for v in corr.index]
    corr.columns = [VARIABLE_LABELS.get(v, v) for v in corr.columns]
    return corr


def flag_high_correlations(
    corr: pd.DataFrame,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Return pairs with |r| > threshold (excluding diagonal).

    Parameters
    ----------
    corr : pd.DataFrame
        Output from correlation_matrix().
    threshold : float
        Absolute correlation threshold. Default 0.7.

    Returns
    -------
    pd.DataFrame with columns: var1, var2, correlation.
    """
    rows: list[dict] = []
    cols = corr.columns.tolist()
    for i, v1 in enumerate(cols):
        for v2 in cols[i + 1:]:
            r = corr.loc[v1, v2]
            if abs(r) >= threshold:
                rows.append({"var1": v1, "var2": v2, "correlation": round(float(r), 4)})
    return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False)
