"""src/diagnostics/multicollinearity.py

Variance Inflation Factor (VIF) test (instruct.md §9.3).

Interpretation:
  VIF < 5   → acceptable
  5–10      → moderate concern
  > 10      → problematic

Special attention: inflation and real interest rate may exhibit multicollinearity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def _vif_flag(vif: float) -> str:
    if np.isnan(vif):
        return "n/a"
    if vif < 5:
        return "acceptable"
    if vif <= 10:
        return "moderate concern"
    return "problematic"


def compute_vif(
    df: pd.DataFrame,
    regressors: list[str],
    label: str = "full_model",
) -> pd.DataFrame:
    """Compute VIF for each regressor.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (long format).
    regressors : list[str]
        Column names of the regressors to check.
    label : str
        Model label to attach to output rows.

    Returns
    -------
    pd.DataFrame with columns: model, variable, vif, flag.
    """
    present = [r for r in regressors if r in df.columns]
    if len(present) < 2:
        return pd.DataFrame(columns=["model", "variable", "vif", "flag"])

    exog = df[present].dropna()
    if len(exog) < len(present) + 1:
        return pd.DataFrame(columns=["model", "variable", "vif", "flag"])

    # Add constant for VIF computation
    exog_c = add_constant(exog, has_constant="add")
    exog_arr = exog_c.values

    rows: list[dict] = []
    # VIF index matches columns in exog_c; skip the constant (index 0)
    for i, var in enumerate(present, start=1):
        try:
            vif_val = float(variance_inflation_factor(exog_arr, i))
        except Exception:
            vif_val = float("nan")
        rows.append(
            {
                "model": label,
                "variable": var,
                "vif": round(vif_val, 3),
                "flag": _vif_flag(vif_val),
            }
        )

    return pd.DataFrame(rows)


def vif_for_specs(
    df: pd.DataFrame,
    specs: dict[str, list[str]],
) -> pd.DataFrame:
    """Run VIF for multiple model specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    specs : dict[str, list[str]]
        Mapping of {spec_label: [regressor_columns]}.

    Returns
    -------
    pd.DataFrame (concatenated VIF results for all specs).
    """
    frames: list[pd.DataFrame] = []
    for label, regressors in specs.items():
        frames.append(compute_vif(df, regressors, label=label))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
