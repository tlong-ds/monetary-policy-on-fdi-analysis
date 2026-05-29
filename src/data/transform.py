"""src/data/transform.py

Variable construction and transformations (instruct.md §4).

Rules:
  - ln_gdppc:      ln(GDP_current_usd / population)  — natural log of GDP per capita
  - d_ln_exr:      Δln(EXR) × 100 = first diff of log exchange rate × 100
                   Positive value → local currency depreciation vs USD
  - wgi_composite: (rq + rl + ge + cc) / 4  — optional composite index
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_ln_gdppc(df: pd.DataFrame) -> pd.DataFrame:
    """Add ln_gdppc: natural log of GDP per capita.

    Requires columns: gdp_current_usd, population.
    Derived as: gdppc = gdp_current_usd / population, then ln(gdppc).
    """
    out = df.copy()
    if "gdp_current_usd" not in out.columns:
        raise KeyError("Column 'gdp_current_usd' not found.")
    if "population" not in out.columns:
        raise KeyError("Column 'population' not found. Merge HC/pop data first.")

    gdppc = np.where(
        (out["population"] > 0) & (out["gdp_current_usd"] > 0),
        out["gdp_current_usd"] / out["population"],
        np.nan,
    )
    out["gdppc"] = gdppc
    out["ln_gdppc"] = np.where(
        out["gdppc"] > 0,
        np.log(out["gdppc"]),
        np.nan,
    )
    return out


def add_delta_ln_exr(df: pd.DataFrame) -> pd.DataFrame:
    """Add d_ln_exr: first difference of log exchange rate × 100.

    Δln(EXR_it) = [ln(EXR_it) − ln(EXR_i,t−1)] × 100
    Positive = local currency depreciation (instruct.md §4).
    First observation per country is NaN by construction.
    """
    out = df.copy().sort_values(["country_code", "year"])
    ln_exr = np.where(
        out["exr_lcu_usd"] > 0,
        np.log(out["exr_lcu_usd"]),
        np.nan,
    )
    out["_ln_exr"] = ln_exr
    out["d_ln_exr"] = (
        out.groupby("country_code")["_ln_exr"]
        .transform(lambda s: s.diff())
        * 100
    )
    out = out.drop(columns=["_ln_exr"])
    return out


def add_wgi_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Add wgi_composite = (rq + rl + ge + cc) / 4 (instruct.md §4).

    NaN if any of the four components is missing for that observation.
    """
    out = df.copy()
    required = ["rq", "rl", "ge", "cc"]
    missing_cols = [c for c in required if c not in out.columns]
    if missing_cols:
        raise KeyError(
            f"Cannot build WGI composite — missing columns: {missing_cols}"
        )
    out["wgi_composite"] = out[required].mean(axis=1, skipna=False)
    return out


def apply_all_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all variable transformations in sequence.

    Expected input columns (post-merge):
      gdp_current_usd, population, exr_lcu_usd, rq, rl, ge, cc

    Added columns:
      gdppc, ln_gdppc, d_ln_exr, wgi_composite
    """
    df = add_ln_gdppc(df)
    df = add_delta_ln_exr(df)
    df = add_wgi_composite(df)
    return df
