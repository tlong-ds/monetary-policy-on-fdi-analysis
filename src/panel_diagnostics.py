from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import norm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

R_IMPORT_ERROR: Exception | None = None
_RPY2_HANDLES: tuple | None = None


def _load_rpy2():
    global R_IMPORT_ERROR, _RPY2_HANDLES
    if _RPY2_HANDLES is not None:
        return _RPY2_HANDLES
    R_IMPORT_ERROR = None
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter
        _RPY2_HANDLES = (ro, pandas2ri, importr, localconverter)
        return _RPY2_HANDLES
    except Exception as exc:
        R_IMPORT_ERROR = exc
        return None


def hausman_test(fe_result, re_result, variables: list[str]) -> dict:
    common = [
        variable
        for variable in variables
        if variable in fe_result.params.index and variable in re_result.params.index
    ]
    beta_diff = fe_result.params[common] - re_result.params[common]
    cov_diff = fe_result.cov.loc[common, common] - re_result.cov.loc[common, common]
    raw_statistic = float(beta_diff.T @ np.linalg.pinv(cov_diff.values) @ beta_diff)
    statistic = max(raw_statistic, 0.0)
    dof = len(common)
    p_value = float(1 - chi2.cdf(statistic, dof))
    return {
        "statistic": statistic,
        "raw_statistic": raw_statistic,
        "degrees_of_freedom": dof,
        "p_value": p_value,
        "negative_statistic_flag": bool(raw_statistic < 0),
        "variables": ", ".join(common),
    }


def sample_missingness_breakdown(
    frame: pd.DataFrame,
    specification: str,
    dependent: str,
    regressors: list[str],
) -> pd.DataFrame:
    variables = [dependent, *regressors]
    rows = []
    for variable in variables:
        missing = int(frame[variable].isna().sum())
        rows.append(
            {
                "specification": specification,
                "variable": variable,
                "missing_count": missing,
                "missing_share": float(frame[variable].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def vif_table(model_df: pd.DataFrame, specification: str, regressors: list[str]) -> pd.DataFrame:
    exog = model_df.reset_index()[regressors].copy()
    rows = []
    for index, variable in enumerate(regressors):
        rows.append(
            {
                "specification": specification,
                "variable": variable,
                "vif": float(variance_inflation_factor(exog.values, index)),
            }
        )
    return pd.DataFrame(rows)


def pooled_model_diagnostics(pooled_result, model_df: pd.DataFrame, specification: str, regressors: list[str]) -> dict:
    exog = add_constant(model_df.reset_index()[regressors], has_constant="add")
    bp_lm_stat, bp_lm_pvalue, bp_f_stat, bp_f_pvalue = het_breuschpagan(
        pooled_result.resid,
        exog,
    )
    white_lm_stat, white_lm_pvalue, white_f_stat, white_f_pvalue = het_white(
        pooled_result.resid,
        exog,
    )
    condition_number = float(np.linalg.cond(exog))
    return {
        "specification": specification,
        "test_model": "pooled_ols",
        "condition_number": condition_number,
        "breusch_pagan_lm_stat": float(bp_lm_stat),
        "breusch_pagan_lm_p_value": float(bp_lm_pvalue),
        "breusch_pagan_f_stat": float(bp_f_stat),
        "breusch_pagan_f_p_value": float(bp_f_pvalue),
        "white_lm_stat": float(white_lm_stat),
        "white_lm_p_value": float(white_lm_pvalue),
        "white_f_stat": float(white_f_stat),
        "white_f_p_value": float(white_f_pvalue),
    }


def pesaran_cd_test(residuals: pd.Series, specification: str) -> dict:
    """Pesaran CD test using pairwise residual correlations across countries."""
    residual_frame = residuals.rename("residual").reset_index()
    pivot = residual_frame.pivot(index="year", columns="country", values="residual")
    countries = list(pivot.columns)
    terms = []
    pair_count = 0
    min_pair_years = np.nan
    max_pair_years = np.nan

    for left_index, left_country in enumerate(countries):
        for right_country in countries[left_index + 1 :]:
            pair = pivot[[left_country, right_country]].dropna()
            pair_years = len(pair)
            if pair_years < 3:
                continue
            correlation = pair[left_country].corr(pair[right_country])
            if pd.isna(correlation):
                continue
            terms.append(np.sqrt(pair_years) * float(correlation))
            pair_count += 1
            min_pair_years = pair_years if pd.isna(min_pair_years) else min(min_pair_years, pair_years)
            max_pair_years = pair_years if pd.isna(max_pair_years) else max(max_pair_years, pair_years)

    if pair_count == 0:
        statistic = np.nan
        p_value = np.nan
    else:
        statistic = float(np.sqrt(2 / (pair_count * 2)) * np.sum(terms))
        p_value = float(2 * (1 - norm.cdf(abs(statistic))))

    return {
        "specification": specification,
        "pesaran_cd_stat": statistic,
        "pesaran_cd_p_value": p_value,
        "pesaran_cd_country_pairs": int(pair_count),
        "pesaran_cd_min_pair_years": min_pair_years,
        "pesaran_cd_max_pair_years": max_pair_years,
    }


def pooled_adf_test(frame: pd.DataFrame, variable: str, regression: str = 'c') -> dict:
    """Run Augmented Dickey-Fuller (ADF) unit root test on the pooled (stacked) variable."""
    from statsmodels.tsa.stattools import adfuller
    clean_series = frame[variable].dropna()
    if len(clean_series) < 10:
        return {
            "variable": variable,
            "error": "Insufficient observations after dropping missing values."
        }
    try:
        stat, pval, usedlag, nobs, crit, icbest = adfuller(clean_series, autolag='AIC', regression=regression)
        return {
            "variable": variable,
            "statistic": float(stat),
            "p_value": float(pval),
            "usedlag": int(usedlag),
            "nobs": int(nobs),
            "critical_values": {k: float(v) for k, v in crit.items()},
        }
    except Exception as e:
        return {"variable": variable, "error": str(e).strip()}





def r_wooldridge_serial_correlation_test(frame: pd.DataFrame, dependent: str, regressors: list[str]) -> dict:
    rpy2_handles = _load_rpy2()
    if rpy2_handles is None:
        return {"error": f"R integration unavailable: {R_IMPORT_ERROR}"}
    ro, pandas2ri, importr, localconverter = rpy2_handles
    plm = importr("plm")
    
    keep_cols = ["country", "year", dependent] + regressors
    clean_frame = frame[keep_cols].dropna().copy()
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(clean_frame)
    pdata = plm.pdata_frame(r_df, index=ro.StrVector(["country", "year"]))
    
    formula = f"{dependent} ~ " + " + ".join(regressors)
    try:
        model = plm.plm(ro.Formula(formula), data=pdata, model="pooling")
        test_result = plm.pbgtest(model)
        return {
            "statistic": float(test_result.rx2('statistic')[0]),
            "p_value": float(test_result.rx2('p.value')[0]),
            "method": str(test_result.rx2('method')[0]),
        }
    except Exception as e:
        return {"error": str(e)}


def pooled_engle_granger_test(
    frame: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    exclude_countries: list[str] | None = None,
    trend: str = 'c'
) -> dict:
    """Run Engle-Granger two-step cointegration test on the pooled (stacked) data."""
    from statsmodels.tsa.stattools import coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.adfvalues import mackinnoncrit
    
    keep_cols = [dependent] + regressors
    if "country" in frame.columns:
        keep_cols = ["country"] + keep_cols
        
    clean_frame = frame[keep_cols].copy()
    if exclude_countries and "country" in clean_frame.columns:
        clean_frame = clean_frame[~clean_frame["country"].isin(exclude_countries)]
        
    clean_frame = clean_frame.dropna()
    
    if len(clean_frame) < 10:
        return {
            "error": "Insufficient observations after dropping missing values and filtering countries."
        }
        
    y = clean_frame[dependent]
    X = clean_frame[regressors]
    
    # statsmodels coint fails with IndexError for N > 6 (more than 5 regressors)
    k_vars = len(regressors) + 1
    
    try:
        if k_vars <= 6:
            stat, pval, crit = coint(y, X, trend=trend, autolag='aic')
            p_val_float = float(pval)
        else:
            # Fallback for N > 6
            if trend == 'c':
                xx = add_constant(X)
            else:
                raise ValueError(f"Trend '{trend}' is not supported for N > 6 fallback.")
            
            model = OLS(y, xx).fit()
            # ADF on residuals without constant/trend
            res_adf = adfuller(model.resid, regression='n', autolag='aic')
            stat = res_adf[0]
            p_val_float = np.nan
            crit = mackinnoncrit(N=k_vars, regression=trend, nobs=len(clean_frame) - 1)
            
        return {
            "tested_regressors": regressors,
            "nobs": len(clean_frame),
            "statistic": float(stat),
            "p_value": p_val_float,
            "critical_values": {
                "1%": float(crit[0]),
                "5%": float(crit[1]),
                "10%": float(crit[2])
            }
        }
    except Exception as e:
        return {"error": str(e).strip()}

