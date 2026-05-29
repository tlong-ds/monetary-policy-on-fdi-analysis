"""src/data/clean.py

Missing data handling, winsorization, and audit outputs (instruct.md §5).

Rules:
  - Dependent variable (fdi_pct_gdp): drop observations where missing.
  - Independent variables:
      * If missing share < 5%  → listwise deletion (no imputation)
      * If missing share ≥ 5%  → within-country linear interpolation
                                  (macro controls only, NOT WGI indicators)
  - WGI indicators (va, pv, ge, rq, rl, cc, wgi_composite): never interpolated.
  - Winsorize all continuous variables at p1 / p99, applied AFTER transformations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import WINSOR_BOUNDS

# Variables that must NEVER be interpolated (instruct.md §5)
_WGI_COLS = {"va", "pv", "ge", "rq", "rl", "cc", "wgi_composite"}

# Macro controls eligible for interpolation (everything else that is continuous)
_MACRO_CONTROLS = [
    "bm_growth",
    "real_interest_rate",
    "trade_pct_gdp",
    "gdp_growth",
    "ln_gdppc",
    "d_ln_exr",
    "inflation",
]


def drop_missing_fdi(df: pd.DataFrame) -> pd.DataFrame:
    """Drop observations with missing dependent variable (instruct.md §5)."""
    before = len(df)
    out = df.dropna(subset=["fdi_pct_gdp"]).copy()
    dropped = before - len(out)
    if dropped > 0:
        print(f"[clean] Dropped {dropped} rows with missing fdi_pct_gdp.")
    return out.reset_index(drop=True)


def handle_missing_controls(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the instruct.md missing-data rule to each control variable.

    Returns
    -------
    df_out : pd.DataFrame
        Panel with imputation applied where eligible.
    log : pd.DataFrame
        Audit log: variable, missing_share, handling, filled_count.
    """
    out = df.copy().sort_values(["country_code", "year"])
    log_rows: list[dict] = []

    candidate_cols = [c for c in _MACRO_CONTROLS if c in out.columns]

    for col in candidate_cols:
        n_total = len(out)
        n_missing_before = int(out[col].isna().sum())
        share = n_missing_before / n_total if n_total > 0 else 0.0

        if n_missing_before == 0:
            handling = "complete — no action"
            n_filled = 0
        elif share < 0.05:
            # Listwise deletion — no imputation; missing obs will be dropped
            # during model estimation (dropna on regressor set)
            handling = "listwise_deletion (share < 5%)"
            n_filled = 0
        else:
            # Within-country linear interpolation, then edge fill
            handling = "within_country_interpolate + edge_fill (share ≥ 5%)"
            imputed = out.groupby("country_code")[col].transform(
                lambda s: s.interpolate(method="linear", limit_area="inside")
                           .ffill()
                           .bfill()
            )
            out[col] = imputed
            n_filled = n_missing_before - int(out[col].isna().sum())

        log_rows.append(
            {
                "variable": col,
                "missing_before": n_missing_before,
                "missing_share": round(share, 4),
                "handling": handling,
                "filled_count": n_filled,
                "missing_after": int(out[col].isna().sum()),
            }
        )

    return out.reset_index(drop=True), pd.DataFrame(log_rows)


def winsorize_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Winsorize all continuous variables at p1/p99 (instruct.md §5).

    Applied AFTER all transformations.  Creates in-place clipped values
    (no separate _winsorized column — the main column is winsorized).

    Returns
    -------
    df_out : pd.DataFrame
        Panel with winsorized values.
    log : pd.DataFrame
        Audit log: variable, lower_bound, upper_bound, n_clipped.
    """
    out = df.copy()
    lo_q, hi_q = WINSOR_BOUNDS

    # All continuous numeric columns (exclude IDs and year)
    skip = {"country_code", "country", "year"}
    continuous_cols = [
        c for c in out.select_dtypes(include=[np.number]).columns
        if c not in skip
    ]

    log_rows: list[dict] = []
    for col in continuous_cols:
        series = out[col].dropna()
        if len(series) == 0:
            continue
        lo_val = float(series.quantile(lo_q))
        hi_val = float(series.quantile(hi_q))
        n_clipped = int(((out[col] < lo_val) | (out[col] > hi_val)).sum())
        out[col] = out[col].clip(lower=lo_val, upper=hi_val)
        log_rows.append(
            {
                "variable": col,
                "lower_bound_p01": lo_val,
                "upper_bound_p99": hi_val,
                "n_clipped": n_clipped,
            }
        )

    return out.reset_index(drop=True), pd.DataFrame(log_rows)


def build_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-variable missing count and share for all panel columns."""
    n_total = len(df)
    rows = []
    for col in df.columns:
        if col in {"country_code", "country", "year"}:
            continue
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "variable": col,
                "n_obs": n_total,
                "n_non_missing": n_total - n_missing,
                "n_missing": n_missing,
                "missing_share": round(n_missing / n_total, 4) if n_total > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)
