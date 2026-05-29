"""src/robustness/checks.py

Robustness checks (instruct.md §11), all anchored to the Full model (Spec 4):
  BASE_CONTROLS + real_interest_rate + rq

Checks implemented:
  R1 — Institutional   : Replace RQ → RL
  R2 — Monetary        : Replace bm_growth → bm_growth_lag1
  R3 — Sample          : Exclude Singapore
  R4 — Crisis          : Exclude GFC (2008–2009) and COVID (2020–2021)
  R5 — Dynamic         : System GMM (Spec 5 re-estimated here for comparison)
  R6 — Composite WGI   : Replace RQ → wgi_composite

Each check returns the same result dict structure as estimate_static_spec()
or two_step_sys_gmm(), augmented with 'check_label' and 'check_name'.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.estimation.specs import (
    BASE_CONTROLS,
    DEPENDENT,
    ModelSpec,
    SPECS,
    SPEC_MAP,
    VARIABLE_LABELS,
)
from src.estimation.static import estimate_static_spec
from src.estimation.dynamic import two_step_sys_gmm


# ---------------------------------------------------------------------------
# Crisis and sample filter constants (instruct.md §11.3 – §11.4)
# ---------------------------------------------------------------------------
GFC_YEARS = [2008, 2009]
COVID_YEARS = [2020, 2021]
CRISIS_YEARS = GFC_YEARS + COVID_YEARS
EXCLUDE_SINGAPORE = ["SGP"]


# ---------------------------------------------------------------------------
# RobustnessCheck descriptor
# ---------------------------------------------------------------------------
@dataclass
class RobustnessCheck:
    label: str                               # e.g. "R1"
    name: str                                # e.g. "Rule of Law (RL)"
    description: str                         # short explanation
    spec: ModelSpec                          # (possibly modified) model spec
    df_modifier: Callable | None = None      # transform applied to full df before filter
    df_filter: Callable | None = None        # row-level filter applied after modifier


# ---------------------------------------------------------------------------
# Helper: build a modified Spec 4 variant
# ---------------------------------------------------------------------------
def _spec4_variant(
    label: str,
    name: str,
    regressors: list[str],
    estimator: str = "static",
    endogenous: list[str] | None = None,
    note: str = "",
) -> ModelSpec:
    return ModelSpec(
        label=label,
        name=name,
        regressors=regressors,
        estimator=estimator,
        endogenous=endogenous or [],
        note=note,
    )


def _add_bm_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Add bm_growth_lag1 column (used for R2)."""
    out = df.copy().sort_values(["country_code", "year"])
    out["bm_growth_lag1"] = out.groupby("country_code")["bm_growth"].shift(1)
    return out


# ---------------------------------------------------------------------------
# All robustness checks — ordered as in instruct.md §11
# ---------------------------------------------------------------------------
ROBUSTNESS_CHECKS: list[RobustnessCheck] = [

    # R1 — Replace RQ with RL (instruct.md §11.1)
    RobustnessCheck(
        label="R1",
        name="Rule of Law (RL)",
        description="Replace Regulatory Quality (RQ) with Rule of Law (RL) as institutional proxy.",
        spec=_spec4_variant(
            label="R1",
            name="Rule of Law (RL)",
            regressors=BASE_CONTROLS + ["real_interest_rate", "rl"],
            note="Institutional robustness: RQ replaced by RL.",
        ),
    ),

    # R2 — Lagged broad money growth (instruct.md §11.2)
    RobustnessCheck(
        label="R2",
        name="Lagged BM Growth",
        description="Replace contemporaneous BM growth with one-period lag (BM_{t-1}).",
        spec=_spec4_variant(
            label="R2",
            name="Lagged BM Growth",
            regressors=(
                ["bm_growth_lag1"]
                + [c for c in BASE_CONTROLS if c != "bm_growth"]
                + ["real_interest_rate", "rq"]
            ),
            note="Monetary robustness: contemporaneous BM replaced by lag.",
        ),
        df_modifier=_add_bm_lag,
    ),

    # R3 — Exclude Singapore (instruct.md §11.3)
    RobustnessCheck(
        label="R3",
        name="Excl. Singapore",
        description="Exclude Singapore (outlier in FDI/GDP and financial development).",
        spec=_spec4_variant(
            label="R3",
            name="Excl. Singapore",
            regressors=BASE_CONTROLS + ["real_interest_rate", "rq"],
            note="Sample robustness: Singapore excluded.",
        ),
        df_filter=lambda df: df[~df["country_code"].isin(EXCLUDE_SINGAPORE)].copy(),
    ),

    # R4 — Exclude crisis periods (instruct.md §11.4)
    RobustnessCheck(
        label="R4",
        name="Excl. Crisis Years",
        description="Exclude GFC (2008–2009) and COVID (2020–2021) periods.",
        spec=_spec4_variant(
            label="R4",
            name="Excl. Crisis Years",
            regressors=BASE_CONTROLS + ["real_interest_rate", "rq"],
            note="Crisis robustness: GFC (2008-09) and COVID (2020-21) excluded.",
        ),
        df_filter=lambda df: df[~df["year"].isin(CRISIS_YEARS)].copy(),
    ),

    # R5 — System GMM dynamic panel (instruct.md §11.5)
    RobustnessCheck(
        label="R5",
        name="System GMM",
        description="Dynamic panel System GMM (Blundell-Bond); lagged FDI endogenous.",
        spec=SPEC_MAP["Spec 5"],
    ),

    # R6 — Composite WGI index (optional, instruct.md §4 / §11)
    RobustnessCheck(
        label="R6",
        name="WGI Composite",
        description="Replace RQ with composite WGI index: (RQ + RL + GE + CC) / 4.",
        spec=_spec4_variant(
            label="R6",
            name="WGI Composite",
            regressors=BASE_CONTROLS + ["real_interest_rate", "wgi_composite"],
            note="Composite institutional index: (RQ + RL + GE + CC) / 4.",
        ),
    ),
]

ROBUSTNESS_CHECK_MAP: dict[str, RobustnessCheck] = {c.label: c for c in ROBUSTNESS_CHECKS}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_robustness_check(
    df: pd.DataFrame,
    check: RobustnessCheck,
    dep: str = DEPENDENT,
) -> dict:
    """Estimate one robustness check and return augmented result dict."""
    # Data modifications
    df_mod = check.df_modifier(df) if check.df_modifier else df
    df_use = check.df_filter(df_mod) if check.df_filter else df_mod

    print(f"  {check.label}: {check.name} "
          f"(N={len(df_use)}, countries={df_use['country_code'].nunique()})")

    if check.spec.estimator == "dynamic":
        result = two_step_sys_gmm(df_use, check.spec, dep=dep)
    else:
        result = estimate_static_spec(df_use, check.spec, dep=dep)
        # Print selection trace
        for line in result.get("selection_log", []):
            print(f"    {line}")

    # Augment with check metadata
    result["check_label"] = check.label
    result["check_name"] = check.name
    result["check_description"] = check.description
    result["n_countries_used"] = int(df_use["country_code"].nunique())

    return result


def run_all_robustness_checks(
    df: pd.DataFrame,
    dep: str = DEPENDENT,
) -> list[dict]:
    """Run all robustness checks and return list of result dicts."""
    results = []
    for check in ROBUSTNESS_CHECKS:
        res = run_robustness_check(df, check, dep=dep)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Comparison table builder
# Focus variable: BM growth (the main explanatory variable)
# ---------------------------------------------------------------------------
def _get_coef(result: dict, var: str) -> tuple[float, float, float]:
    """Return (coef, se, pval) for `var` in a result dict. NaN if absent."""
    res = result.get("preferred_result")
    if res is not None:
        # linearmodels result
        if var in res.params.index:
            return float(res.params[var]), float(res.std_errors[var]), float(res.pvalues[var])
        return np.nan, np.nan, np.nan
    # Dynamic GMM dict
    params = result.get("params", pd.Series(dtype=float))
    ses    = result.get("std_errors", pd.Series(dtype=float))
    pvals  = result.get("p_values", pd.Series(dtype=float))
    if var in params.index:
        return float(params[var]), float(ses[var]), float(pvals[var])
    return np.nan, np.nan, np.nan


def _stars(pval: float) -> str:
    if np.isnan(pval): return ""
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""


def build_comparison_table(
    baseline_result: dict,
    robustness_results: list[dict],
    focus_vars: list[str] | None = None,
) -> pd.DataFrame:
    """Build side-by-side comparison across baseline + all robustness checks.

    Shows coefficient rows for focus_vars (default: key model variables),
    plus model statistics footer.

    Parameters
    ----------
    baseline_result : dict
        Result dict for the Spec 4 (Full) baseline.
    robustness_results : list[dict]
        Result dicts from run_all_robustness_checks().
    focus_vars : list[str] | None
        Variables to display. Defaults to key model variables.

    Returns
    -------
    pd.DataFrame  — rows: (variable, coef/se/stat), cols: one per model.
    """
    if focus_vars is None:
        focus_vars = [
            "bm_growth", "bm_growth_lag1",
            "real_interest_rate",
            "gdp_growth", "ln_gdppc",
            "trade_pct_gdp", "d_ln_exr", "inflation",
            "rq", "rl", "wgi_composite",
            "fdi_pct_gdp_lag1",
        ]

    all_results = [baseline_result] + robustness_results
    col_headers = (
        [f"Baseline\n(Spec 4)"]
        + [f"{r['check_label']}\n{r['check_name']}" for r in robustness_results]
    )

    rows: list[dict] = []

    # Coefficient rows
    for var in focus_vars:
        label = VARIABLE_LABELS.get(var, var)
        row_c = {"row_label": label, "row_type": "coef"}
        row_s = {"row_label": "",    "row_type": "se"}
        for header, rd in zip(col_headers, all_results):
            c, se, pv = _get_coef(rd, var)
            row_c[header] = f"{c:.4f}{_stars(pv)}" if not np.isnan(c) else ""
            row_s[header] = f"({se:.4f})" if not np.isnan(se) else ""
        # Only add row if at least one model has this variable
        if any(row_c[h] for h in col_headers):
            rows.extend([row_c, row_s])

    # Footer stats
    stat_rows: list[tuple[str, Callable]] = [
        ("N obs",       lambda rd: str(rd.get("n_obs", rd.get("obs", "")))),
        ("N countries", lambda rd: str(rd.get("n_countries_used", rd.get("n_entities", "")))),
        ("Estimator",   lambda rd: rd.get("estimator_label", "")),
        ("N instruments (GMM)", lambda rd: str(rd.get("n_instruments", "—"))),
        ("Hansen p (GMM)",      lambda rd: str(rd.get("hansen", {}).get("p_value", "—"))),
        ("AR(1) p (GMM)",       lambda rd: str(rd.get("ar1", {}).get("p_value", "—"))),
        ("AR(2) p (GMM)",       lambda rd: str(rd.get("ar2", {}).get("p_value", "—"))),
    ]

    for stat_label, fn in stat_rows:
        row: dict = {"row_label": stat_label, "row_type": "stat"}
        for header, rd in zip(col_headers, all_results):
            try:
                row[header] = fn(rd)
            except Exception:
                row[header] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def build_bm_summary(
    baseline_result: dict,
    robustness_results: list[dict],
) -> pd.DataFrame:
    """One-row-per-check summary focused on BM growth coefficient."""
    all_results = [
        {"label": "Baseline (Spec 4)", "result": baseline_result}
    ] + [
        {"label": f"{r['check_label']} — {r['check_name']}", "result": r}
        for r in robustness_results
    ]
    rows = []
    for item in all_results:
        rd = item["result"]
        # BM growth could be bm_growth or bm_growth_lag1
        for bm_var in ["bm_growth", "bm_growth_lag1"]:
            c, se, pv = _get_coef(rd, bm_var)
            if not np.isnan(c):
                break
        rows.append({
            "Check":      item["label"],
            "BM variable": "bm_growth_lag1" if bm_var == "bm_growth_lag1" else "bm_growth",
            "Coef":       round(c, 4) if not np.isnan(c) else "",
            "SE":         round(se, 4) if not np.isnan(se) else "",
            "p-value":    round(pv, 4) if not np.isnan(pv) else "",
            "Sig":        _stars(pv),
            "N obs":      rd.get("n_obs", rd.get("obs", "")),
            "Estimator":  rd.get("estimator_label", ""),
            "Decision":   "Reject H0" if (not np.isnan(pv) and pv < 0.05) else "Fail to reject",
        })
    return pd.DataFrame(rows)
