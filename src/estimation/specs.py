"""src/estimation/specs.py

Hardcoded model specifications from instruct.md §10.

Five specifications:
  Spec 1 — Baseline         : base controls only, FE/RE
  Spec 2 — + Monetary       : + real interest rate, FE/RE
  Spec 3 — + Institutional  : + RQ (regulatory quality), FE/RE
  Spec 4 — Full             : + real interest rate + RQ, FE/RE
  Spec 5 — Dynamic GMM      : full model + lagged FDI, System GMM

Base controls (all specs):
  bm_growth, gdp_growth, ln_gdppc, trade_pct_gdp, d_ln_exr, inflation
"""
from __future__ import annotations

from dataclasses import dataclass, field


DEPENDENT = "fdi_pct_gdp"

BASE_CONTROLS: list[str] = [
    "bm_growth",
    "gdp_growth",
    "ln_gdppc",
    "trade_pct_gdp",
    "d_ln_exr",
    "inflation",
]

# Human-readable labels for table output
VARIABLE_LABELS: dict[str, str] = {
    "fdi_pct_gdp":        "FDI (% GDP)",
    "fdi_pct_gdp_lag1":   "Lagged FDI (% GDP)",
    "bm_growth":          "Broad Money Growth",
    "real_interest_rate": "Real Interest Rate",
    "gdp_growth":         "GDP Growth",
    "ln_gdppc":           "ln(GDP per Capita)",
    "trade_pct_gdp":      "Trade Openness (% GDP)",
    "d_ln_exr":           "Δln(Exchange Rate)",
    "inflation":          "Inflation",
    "rq":                 "Regulatory Quality",
    "rl":                 "Rule of Law",
    "wgi_composite":      "WGI Composite",
}

# Expected signs for interpretation (instruct.md §16)
EXPECTED_SIGNS: dict[str, str] = {
    "bm_growth":          "+",
    "real_interest_rate": "−",
    "gdp_growth":         "+",
    "ln_gdppc":           "+",
    "trade_pct_gdp":      "+",
    "d_ln_exr":           "±",
    "inflation":          "−",
    "rq":                 "+",
    "rl":                 "+",
    "wgi_composite":      "+",
    "fdi_pct_gdp_lag1":   "+",
}


@dataclass(frozen=True)
class ModelSpec:
    """Single model specification."""
    label: str               # Short label (e.g. "Spec 1")
    name: str                # Full name (e.g. "Baseline")
    regressors: list[str]    # RHS variables (excluding constant/FE)
    estimator: str           # "static" or "dynamic"
    # Dynamic GMM only
    endogenous: list[str] = field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# The five specifications from instruct.md §10
# ---------------------------------------------------------------------------
SPECS: list[ModelSpec] = [
    ModelSpec(
        label="Spec 1",
        name="Baseline",
        regressors=BASE_CONTROLS,
        estimator="static",
        note="Base controls only; estimator selected via BP-LM → Hausman.",
    ),
    ModelSpec(
        label="Spec 2",
        name="+ Monetary",
        regressors=BASE_CONTROLS + ["real_interest_rate"],
        estimator="static",
        note="Adds real interest rate (monetary tightness proxy).",
    ),
    ModelSpec(
        label="Spec 3",
        name="+ Institutional",
        regressors=BASE_CONTROLS + ["rq"],
        estimator="static",
        note="Adds Regulatory Quality (WGI baseline institutional proxy).",
    ),
    ModelSpec(
        label="Spec 4",
        name="Full",
        regressors=BASE_CONTROLS + ["real_interest_rate", "rq"],
        estimator="static",
        note="Full static model with all controls.",
    ),
    ModelSpec(
        label="Spec 5",
        name="Dynamic GMM",
        regressors=BASE_CONTROLS + ["real_interest_rate", "rq", "fdi_pct_gdp_lag1"],
        endogenous=["fdi_pct_gdp_lag1", "bm_growth"],
        estimator="dynamic",
        note="Two-step System GMM; Windmeijer SE; collapsed instruments lag(2.); "
             "endogenous: lagged FDI, BM growth.",
    ),
]

SPEC_MAP: dict[str, ModelSpec] = {s.label: s for s in SPECS}
