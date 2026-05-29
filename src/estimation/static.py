"""src/estimation/static.py

Static panel estimator selection procedure (instruct.md §8.1–8.2):

  Step 1 — Pooled OLS
  Step 2 — Breusch-Pagan LM test (H0: no random effects)
            → significant: proceed to panel; else use pooled OLS
  Step 3 — Hausman test (H0: RE consistent, i.e., no correlation between
            entity effects and regressors)
            → p < 0.05: Fixed Effects; else: Random Effects
  Step 4 — Cluster-robust standard errors at country level

Returns a structured result dict per spec containing:
  preferred_result, estimator_label, pooled_result, fe_result, re_result,
  bp_lm, hausman, selection_log.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS as LMPooledOLS

from src.estimation.specs import DEPENDENT, ModelSpec


# ---------------------------------------------------------------------------
# Breusch-Pagan LM test for random effects
# H0: σ²_μ = 0  (pooled OLS is consistent)
# Statistic: LM ~ χ²(1) — based on Honda (1985) / Breusch-Pagan (1980)
# ---------------------------------------------------------------------------
def breusch_pagan_lm(
    resid: pd.Series,
    entity_col: pd.Series,
) -> dict:
    """Breusch-Pagan LM test using pooled OLS residuals.

    Parameters
    ----------
    resid : pd.Series
        Residuals from pooled OLS (aligned with entity_col).
    entity_col : pd.Series
        Entity identifier for each observation.

    Returns
    -------
    dict with statistic, p_value, reject.
    """
    df_r = pd.DataFrame({"resid": resid.values, "entity": entity_col.values})
    df_r["resid2"] = df_r["resid"] ** 2
    df_r["resid_sum"] = df_r.groupby("entity")["resid"].transform("sum")
    df_r["T_i"] = df_r.groupby("entity")["resid"].transform("count")

    # Sum of squared group sums
    SSG = float((df_r.groupby("entity")["resid_sum"].first() ** 2).sum())
    SSE = float(df_r["resid2"].sum())
    N = int(df_r["entity"].nunique())
    T_avg = len(df_r) / N

    if SSE == 0:
        return {"statistic": np.nan, "p_value": np.nan, "reject": False}

    lm_stat = (N * T_avg / (2 * (T_avg - 1))) * ((SSG / SSE) - 1) ** 2
    pval = float(1 - chi2.cdf(lm_stat, df=1))

    return {
        "test": "Breusch-Pagan LM",
        "statistic": round(float(lm_stat), 4),
        "p_value": round(pval, 4),
        "reject": pval < 0.05,
        "decision": "Reject H0 → use panel estimator" if pval < 0.05 else "Fail to reject H0 → use Pooled OLS",
    }


# ---------------------------------------------------------------------------
# Hausman test
# H0: RE is consistent (no correlation between entity effects and regressors)
# H = (β_FE − β_RE)' [Cov(β_FE) − Cov(β_RE)]^{-1} (β_FE − β_RE) ~ χ²(k)
# ---------------------------------------------------------------------------
def hausman_test(fe_result, re_result) -> dict:
    """Classical Hausman specification test comparing FE and RE."""
    common = [
        v for v in fe_result.params.index
        if v in re_result.params.index and v != "Intercept"
    ]
    if not common:
        return {"statistic": np.nan, "p_value": np.nan, "reject": False,
                "decision": "Hausman: no common parameters"}

    b_fe = fe_result.params[common].values
    b_re = re_result.params[common].values
    cov_fe = fe_result.cov.loc[common, common].values
    cov_re = re_result.cov.loc[common, common].values

    diff = b_fe - b_re
    cov_diff = cov_fe - cov_re

    try:
        h_stat = float(diff @ np.linalg.pinv(cov_diff) @ diff)
        h_stat = max(h_stat, 0.0)   # numerical clamp
    except np.linalg.LinAlgError:
        return {"statistic": np.nan, "p_value": np.nan, "reject": False,
                "decision": "Hausman: singular covariance matrix"}

    dof = len(common)
    pval = float(1 - chi2.cdf(h_stat, dof))

    return {
        "test": "Hausman",
        "statistic": round(h_stat, 4),
        "p_value": round(pval, 4),
        "degrees_of_freedom": dof,
        "reject": pval < 0.05,
        "decision": "Reject H0 → Fixed Effects preferred" if pval < 0.05 else "Fail to reject H0 → Random Effects preferred",
    }


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------
def _prep_panel(df: pd.DataFrame, dep: str, regressors: list[str]) -> pd.DataFrame:
    """Subset, drop NAs, and set MultiIndex for linearmodels."""
    keep = ["country_code", "year", dep] + [r for r in regressors if r in df.columns]
    sub = df[keep].dropna().copy()
    sub = sub.set_index(["country_code", "year"])
    sub = sub[~sub.index.duplicated()]
    return sub


def estimate_static_spec(
    df: pd.DataFrame,
    spec: ModelSpec,
    dep: str = DEPENDENT,
) -> dict:
    """Run the full estimator-selection procedure for one static spec.

    Returns
    -------
    dict with keys:
      spec_label, spec_name, estimator_label, preferred_result,
      fe_result, re_result, pooled_result,
      bp_lm, hausman, selection_log, obs, n_entities.
    """
    regressors = [r for r in spec.regressors if r in df.columns]
    panel = _prep_panel(df, dep, regressors)

    if panel.empty or len(panel) < len(regressors) + 5:
        return {
            "spec_label": spec.label,
            "spec_name": spec.name,
            "error": "Insufficient observations after listwise deletion.",
        }

    obs = len(panel)
    n_entities = panel.index.get_level_values("country_code").nunique()
    selection_log: list[str] = []

    # ── Step 1: Pooled OLS ────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled_mod = LMPooledOLS(panel[dep], panel[regressors])
        pooled_res = pooled_mod.fit(cov_type="clustered", cluster_entity=True)

    selection_log.append("Step 1: Pooled OLS estimated.")

    # ── Step 2: Breusch-Pagan LM ──────────────────────────────────────────
    entity_idx = panel.index.get_level_values("country_code")
    bp_lm = breusch_pagan_lm(pooled_res.resids, entity_idx)
    selection_log.append(f"Step 2: BP-LM p={bp_lm['p_value']:.4f} → {bp_lm['decision']}")

    if not bp_lm["reject"]:
        # Use pooled OLS
        return {
            "spec_label": spec.label,
            "spec_name": spec.name,
            "estimator_label": "Pooled OLS",
            "preferred_result": pooled_res,
            "pooled_result": pooled_res,
            "fe_result": None,
            "re_result": None,
            "bp_lm": bp_lm,
            "hausman": None,
            "selection_log": selection_log,
            "obs": obs,
            "n_entities": n_entities,
        }

    # ── Step 3: FE and RE, then Hausman ───────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fe_mod = PanelOLS(
            panel[dep], panel[regressors],
            entity_effects=True, time_effects=True,
        )
        fe_res = fe_mod.fit(cov_type="clustered", cluster_entity=True)

        re_mod = RandomEffects(panel[dep], panel[regressors])
        try:
            re_res = re_mod.fit(cov_type="robust")
            re_failed = False
        except Exception as re_exc:
            re_res = None
            re_failed = True
            re_fail_msg = str(re_exc)


    selection_log.append("Step 3: FE (two-way, cluster SE) and RE (robust SE) estimated.")

    if re_failed:
        hausman = {"test": "Hausman", "error": f"RE estimation failed: {re_fail_msg}",
                   "reject": True, "decision": "RE failed → Fixed Effects used by default"}
        selection_log.append(f"        RE failed ({re_fail_msg}) → defaulting to Fixed Effects.")
        preferred_res = fe_res
        estimator_label = "Fixed Effects"
    else:
        hausman = hausman_test(fe_res, re_res)
        selection_log.append(f"        Hausman p={hausman['p_value']:.4f} → {hausman['decision']}")
        if hausman["reject"]:
            preferred_res = fe_res
            estimator_label = "Fixed Effects"
        else:
            preferred_res = re_res
            estimator_label = "Random Effects"


    selection_log.append(f"Step 4: Preferred estimator: {estimator_label} (cluster-robust SE).")

    return {
        "spec_label": spec.label,
        "spec_name": spec.name,
        "estimator_label": estimator_label,
        "preferred_result": preferred_res,
        "pooled_result": pooled_res,
        "fe_result": fe_res,
        "re_result": re_res,
        "bp_lm": bp_lm,
        "hausman": hausman,
        "selection_log": selection_log,
        "obs": obs,
        "n_entities": n_entities,
    }


def run_static_specs(
    df: pd.DataFrame,
    specs: list[ModelSpec],
    dep: str = DEPENDENT,
) -> list[dict]:
    """Estimate all static specs and return list of result dicts."""
    results = []
    for spec in specs:
        if spec.estimator != "static":
            continue
        print(f"  Estimating {spec.label}: {spec.name} ...")
        res = estimate_static_spec(df, spec, dep=dep)
        for line in res.get("selection_log", []):
            print(f"    {line}")
        results.append(res)
    return results
