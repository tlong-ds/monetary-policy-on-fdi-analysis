from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.panel_diagnostics import (
    hausman_test,
    vif_table,
    pooled_model_diagnostics,
    pesaran_cd_test,
    sample_missingness_breakdown,
)
from src.model_contract import (
    choose_preferred_estimator,
    estimator_label,
    safe_sheet_name,
    build_model_frame,
    parse_regressor_string,
    add_country_lags,
    build_workbook_model_catalog,
    WORKBOOK_VARIABLE_MAP,
)
from src.reporting import (
    stars,
    format_coef_cell,
    hausman_status,
    normalize_expected_sign,
    strip_lag_suffix,
    coefficient_direction,
    significance_label,
    sign_alignment,
    interpret_coefficient_row,
    THEORY_EXPECTED_SIGNS,
    VARIABLE_LABELS,
)
from src.config import OUTPUTS_DIR, FIGURES_DIR, PROCESSED_PANEL_FILE

# Local helpers for estimation
def extract_result_rows(result, model_id: str, estimator: str) -> pd.DataFrame:
    std_errors = result.std_errors if hasattr(result, 'std_errors') else result.bse
    t_stats = result.tstats if hasattr(result, 'tstats') else result.tvalues
    conf_int = result.conf_int()
    summary = pd.DataFrame(
        {
            'term': result.params.index,
            'coef': result.params.values,
            'std_error': std_errors.values,
            't_stat': t_stats.values,
            'p_value': result.pvalues.values,
            'ci_low': conf_int.iloc[:, 0].values,
            'ci_high': conf_int.iloc[:, 1].values,
        }
    )
    summary['model_id'] = model_id
    summary['estimator'] = estimator
    return summary[
        ['model_id', 'estimator', 'term', 'coef', 'std_error', 't_stat', 'p_value', 'ci_low', 'ci_high']
    ]

def placeholder_result_rows(model_id: str, estimator: str, regressors: list[str]) -> pd.DataFrame:
    terms = ['Intercept'] + regressors
    return pd.DataFrame(
        {
            'model_id': model_id,
            'estimator': estimator,
            'term': terms,
            'coef': np.nan,
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
        }
    )

def adjusted_r_squared(result) -> float:
    if hasattr(result, 'rsquared_adj'):
        return float(result.rsquared_adj)
    r_squared = getattr(result, 'rsquared', np.nan)
    nobs = getattr(result, 'nobs', np.nan)
    df_resid = getattr(result, 'df_resid', np.nan)
    if pd.isna(r_squared) or pd.isna(nobs) or pd.isna(df_resid) or float(df_resid) <= 0:
        return np.nan
    return float(1 - (1 - float(r_squared)) * (float(nobs) - 1) / float(df_resid))

def prepared_estimation_frame_and_regressors(model_row, df, dependent='fdi_pct_gdp') -> tuple[pd.DataFrame, list[str]]:
    regressors = parse_regressor_string(model_row.mapped_regressors)
    estimation_frame = df.copy()
    if model_row.lagged_model:
        base_regressors = parse_regressor_string(model_row.base_regressors)
        estimation_frame = add_country_lags(estimation_frame, base_regressors + [dependent], lag=1)
    return estimation_frame, regressors

def complete_case_keys(frame: pd.DataFrame, regressors: list[str], dependent='fdi_pct_gdp') -> pd.DataFrame:
    variables = [dependent, *regressors]
    mask = frame[variables].notna().all(axis=1)
    return frame.loc[mask, ['country', 'year']].drop_duplicates().reset_index(drop=True)

def filter_to_keys(frame: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(keys.assign(_keep_common_sample=True), on=['country', 'year'], how='left').query(
        '_keep_common_sample == True'
    ).drop(columns=['_keep_common_sample'])

def summarize_model_country_windows(keys: pd.DataFrame, model_id: str) -> pd.DataFrame:
    rows = []
    for country, group in keys.groupby('country'):
        years = sorted(group['year'].dropna().astype(int).unique())
        if not years:
            continue
        expected_window = set(range(min(years), max(years) + 1))
        internal_missing = sorted(expected_window - set(years))
        rows.append(
            {
                'model_id': model_id,
                'country': country,
                'usable_years': len(years),
                'first_usable_year': min(years),
                'last_usable_year': max(years),
                'window_length_years': max(years) - min(years) + 1,
                'internal_missing_years': len(internal_missing),
                'internal_missing_year_list': ', '.join(map(str, internal_missing)) if internal_missing else '',
                'is_contiguous_window': len(internal_missing) == 0,
            }
        )
    return pd.DataFrame(rows)

def estimate_model(frame: pd.DataFrame, model_id: str, regressors: list[str], exclude_countries: list[str] | None = None, dependent: str = 'fdi_pct_gdp') -> dict[str, object]:
    model_df = build_model_frame(frame, dependent, regressors, exclude_countries=exclude_countries)
    if model_df.empty:
        raise ValueError(f'No complete-case observations remain for {model_id}.')

    formula = dependent + ' ~ ' + ' + '.join(regressors)
    panel_formula = dependent + ' ~ 1 + ' + ' + '.join(regressors)
    reset_model_df = model_df.reset_index()

    pooled = smf.ols(formula, data=reset_model_df).fit(
        cov_type='cluster',
        cov_kwds={'groups': reset_model_df['country']},
    )
    fe_clustered = PanelOLS.from_formula(
        panel_formula + ' + EntityEffects + TimeEffects',
        data=model_df,
    ).fit(cov_type='clustered', cluster_entity=True)
    fe_dk = PanelOLS.from_formula(
        panel_formula + ' + EntityEffects + TimeEffects',
        data=model_df,
    ).fit(cov_type='kernel', kernel='bartlett')

    re = None
    re_failure_reason = ''
    try:
        re = RandomEffects.from_formula(panel_formula, data=model_df).fit(
            cov_type='clustered',
            cluster_entity=True,
        )
    except (ZeroDivisionError, ValueError) as exc:
        re_failure_reason = str(exc)

    def raw_summary_text(result: object) -> str:
        if result is None:
            return ""
        try:
            summary_obj = getattr(result, "summary")
            if callable(summary_obj):
                summary_obj = summary_obj()
            if hasattr(summary_obj, "as_text"):
                return str(summary_obj.as_text())
            return str(summary_obj)
        except Exception:
            return str(result)

    raw_summaries = {
        "pooled_ols": raw_summary_text(pooled),
        "fixed_effects": raw_summary_text(fe_clustered),
        "fixed_effects_driscoll_kraay": raw_summary_text(fe_dk),
        "random_effects": raw_summary_text(re) if re is not None else f"Random effects failed: {re_failure_reason}",
    }

    coefficient_parts = [
        extract_result_rows(pooled, model_id, 'pooled_ols'),
        extract_result_rows(fe_clustered, model_id, 'fixed_effects'),
        extract_result_rows(fe_dk, model_id, 'fixed_effects_driscoll_kraay'),
    ]
    if re is not None:
        coefficient_parts.append(extract_result_rows(re, model_id, 'random_effects'))
    else:
        coefficient_parts.append(placeholder_result_rows(model_id, 'random_effects', regressors))
    coefficients = pd.concat(coefficient_parts, ignore_index=True)

    hausman_payload = {
        'statistic': np.nan,
        'raw_statistic': np.nan,
        'degrees_of_freedom': len(regressors),
        'p_value': np.nan,
        'negative_statistic_flag': False,
        'variables': ', '.join(regressors),
        'comparison_note': 'random_effects_failed',
    }
    if re is not None:
        hausman_payload = {
            **hausman_test(fe_clustered, re, regressors),
            'comparison_note': 'ok',
        }

    def fit_row(result, estimator: str, covariance_type: str) -> dict:
        r_squared = float(getattr(result, 'rsquared', np.nan))
        within_r2 = float(getattr(result, 'rsquared_within', r_squared))
        nobs = float(getattr(result, 'nobs', len(model_df)))
        slope_count = len(regressors)
        slope_adjusted_within_r2 = np.nan
        if nobs - slope_count - 1 > 0 and not pd.isna(within_r2):
            slope_adjusted_within_r2 = 1 - (1 - within_r2) * (nobs - 1) / (nobs - slope_count - 1)
        adj_r_squared = adjusted_r_squared(result)
        return {
            'model_id': model_id,
            'estimator': estimator,
            'covariance_type': covariance_type,
            'nobs': nobs,
            'r_squared': r_squared,
            'within_r2': within_r2,
            'adj_r_squared': adj_r_squared,
            'slope_adjusted_within_r2': slope_adjusted_within_r2,
            'r2_adj_r2_gap': abs(r_squared - adj_r_squared) if not pd.isna(adj_r_squared) else np.nan,
        }

    fit_rows = [
        fit_row(pooled, 'pooled_ols', 'country-clustered'),
        fit_row(fe_clustered, 'fixed_effects', 'entity-clustered'),
        fit_row(fe_dk, 'fixed_effects_driscoll_kraay', 'Driscoll-Kraay kernel Bartlett'),
    ]
    if re is not None:
        fit_rows.append(fit_row(re, 'random_effects', 'entity-clustered'))
    else:
        fit_rows.append(
            {
                'model_id': model_id,
                'estimator': 'random_effects',
                'covariance_type': 'entity-clustered',
                'nobs': float(len(model_df)),
                'r_squared': np.nan,
                'within_r2': np.nan,
                'adj_r_squared': np.nan,
                'slope_adjusted_within_r2': np.nan,
                'r2_adj_r2_gap': np.nan,
            }
        )
    fit_stats = pd.DataFrame(fit_rows)
    sample_summary = pd.DataFrame(
        [
            {
                'model_id': model_id,
                'rows_used': int(len(model_df)),
                'rows_dropped': int(len(frame) - len(model_df)),
                'countries_used': int(reset_model_df['country'].nunique()),
                'years_used': int(reset_model_df['year'].nunique()),
                'regressors': ', '.join(regressors),
            }
        ]
    )
    hausman = pd.DataFrame([{ 'model_id': model_id, **hausman_payload, 're_failure_reason': re_failure_reason }])
    vif = vif_table(model_df, model_id, regressors).rename(columns={'specification': 'model_id'})
    diagnostics = pd.DataFrame(
        [
            {
                **pooled_model_diagnostics(pooled, model_df, model_id, regressors),
                **pesaran_cd_test(fe_dk.resids, model_id),
            }
        ]
    ).rename(columns={'specification': 'model_id'})
    diagnostics['main_estimator'] = 'fixed_effects_driscoll_kraay'
    diagnostics['main_covariance_type'] = 'Driscoll-Kraay kernel Bartlett'
    diagnostics['random_effects_available'] = re is not None
    diagnostics['re_failure_reason'] = re_failure_reason
    missingness = sample_missingness_breakdown(frame, model_id, dependent, regressors).rename(
        columns={'specification': 'model_id'}
    )

    return {
        'coefficients': coefficients,
        'fit_stats': fit_stats,
        'sample_summary': sample_summary,
        'hausman': hausman,
        'vif': vif,
        'diagnostics': diagnostics,
        'sample_missingness': missingness,
        'raw_summaries': raw_summaries,
    }

# STEPWISE broad money sign decomposition helper
def fit_broad_money_decomposition_model(
    source_frame: pd.DataFrame,
    regressors: list[str],
    estimator: str,
    dependent: str = 'fdi_pct_gdp'
) -> dict[str, object]:
    model_df = build_model_frame(source_frame, dependent, regressors)
    formula = dependent + ' ~ ' + ' + '.join(regressors)
    panel_formula = dependent + ' ~ 1 + ' + ' + '.join(regressors)
    try:
        if estimator == 'pooled_ols':
            reset_frame = model_df.reset_index()
            result = smf.ols(formula, data=reset_frame).fit(
                cov_type='cluster',
                cov_kwds={'groups': reset_frame['country']},
            )
        elif estimator == 'fixed_effects_driscoll_kraay':
            result = PanelOLS.from_formula(
                panel_formula + ' + EntityEffects + TimeEffects',
                data=model_df,
            ).fit(cov_type='kernel', kernel='bartlett')
        elif estimator == 'fixed_effects':
            result = PanelOLS.from_formula(
                panel_formula + ' + EntityEffects + TimeEffects',
                data=model_df,
            ).fit(cov_type='clustered', cluster_entity=True)
        elif estimator == 'random_effects':
            result = RandomEffects.from_formula(panel_formula, data=model_df).fit(
                cov_type='clustered', cluster_entity=True,
            )
        else:
            raise ValueError(f'Unknown estimator: {estimator}')
    except Exception as exc:
        return {
            'nobs': len(model_df),
            'countries': model_df.index.get_level_values('country').nunique(),
            'years': model_df.index.get_level_values('year').nunique(),
            'coef': pd.NA,
            'p_value': pd.NA,
            'r_squared': pd.NA,
            'adj_r_squared': pd.NA,
            'fit_status': f'failed: {type(exc).__name__}: {exc}',
        }

    return {
        'nobs': float(getattr(result, 'nobs', len(model_df))),
        'countries': model_df.index.get_level_values('country').nunique(),
        'years': model_df.index.get_level_values('year').nunique(),
        'coef': float(result.params['broad_money_growth_pct']),
        'p_value': float(result.pvalues['broad_money_growth_pct']),
        'r_squared': float(getattr(result, 'rsquared', pd.NA)),
        'adj_r_squared': adjusted_r_squared(result),
        'fit_status': 'estimated',
    }

def sign_from_coefficient(coef: object) -> str:
    if pd.isna(coef):
        return 'not estimated'
    if float(coef) > 0:
        return 'positive'
    if float(coef) < 0:
        return 'negative'
    return 'zero'

def run_full_estimation_and_export(df: pd.DataFrame, estimated_models: pd.DataFrame, workbook_path: Path | str, dependent: str = 'fdi_pct_gdp'):
    print("--- Starting Full Econometric Estimation & Export Pipeline ---")
    
    # 1. Workbook tables catalog reload
    workbook_catalog_df, workbook_tables = build_workbook_model_catalog(workbook_path, panel_columns=df.columns.tolist())
    workbook_variable_selection = workbook_tables['variable_selection']
    workbook_model_specs = workbook_tables['model_specs']
    workbook_notes = workbook_tables['notes']
    
    # Use the caller-provided estimated_models as the source of truth for which specs to run.
    # This prevents notebook/kernel import drift from causing "no objects to concatenate".
    if estimated_models is None or estimated_models.empty:
        status_counts = workbook_catalog_df['status'].value_counts(dropna=False).to_dict() if 'status' in workbook_catalog_df.columns else {}
        raise ValueError(
            "No estimated models were provided to run_full_estimation_and_export(). "
            f"Workbook catalog status counts: {status_counts}. "
            "Rebuild the processed panel and ensure required columns exist (e.g., broad_money_growth_pct)."
        )

    workbook_catalog_df = workbook_catalog_df.merge(
        estimated_models[['model_id']].drop_duplicates(),
        on='model_id',
        how='left',
        indicator=True,
    )
    workbook_catalog_df['status'] = np.where(
        workbook_catalog_df['_merge'].eq('both'),
        workbook_catalog_df['status'],
        'skipped_not_selected',
    )
    workbook_catalog_df = workbook_catalog_df.drop(columns=['_merge'])

    skipped_models = workbook_catalog_df[workbook_catalog_df['status'].ne('estimated')].copy()
    skipped_models = skipped_models.sort_values('model_order').reset_index(drop=True)

    # Set up collections
    coefficients_tables = []
    fit_stats_tables = []
    sample_summary_tables = []
    hausman_tables = []
    vif_tables = []
    diagnostics_tables = []
    missingness_tables = []
    status_rows = []

    model_row_lookup = {row.model_id: row for row in workbook_catalog_df.itertuples(index=False)}

    def require_model_rows(model_ids: list[str], context: str) -> None:
        missing_model_ids = [model_id for model_id in model_ids if model_id not in model_row_lookup]
        if missing_model_ids:
            raise KeyError(
                f'{context} references model IDs that are not in workbook_catalog_df: {missing_model_ids}'
            )

    # 2. Main Native Samples Estimation Loop
    print("Estimating native specifications...")
    for model_row in workbook_catalog_df.itertuples(index=False):
        status_row = {
            'model_id': model_row.model_id,
            'workbook_code': model_row.workbook_code,
            'workbook_model': model_row.workbook_model,
            'purpose': model_row.purpose,
            'recommendation': model_row.recommendation,
            'model_tier': model_row.model_tier,
            'channel': model_row.channel,
            'thesis_role': model_row.thesis_role,
            'headline_eligible': bool(model_row.headline_eligible),
            'appendix_only': bool(model_row.appendix_only),
            'workbook_variables': model_row.workbook_variables,
            'mapped_regressors': model_row.mapped_regressors,
            'lagged_model': bool(model_row.lagged_model),
            'status': model_row.status,
            'missing_reason': model_row.missing_reason,
        }

        if model_row.status != 'estimated':
            status_rows.append(status_row)
            continue

        exclude_countries = [c.strip() for c in model_row.exclude_countries.split(',') if c.strip()] if hasattr(model_row, 'exclude_countries') and isinstance(model_row.exclude_countries, str) else None
        
        estimation_frame, regressors = prepared_estimation_frame_and_regressors(model_row, df, dependent=dependent)
        model_outputs = estimate_model(estimation_frame, model_row.model_id, regressors, exclude_countries=exclude_countries, dependent=dependent)
        
        coefficients_tables.append(model_outputs['coefficients'])
        fit_stats_tables.append(model_outputs['fit_stats'])
        sample_summary_tables.append(model_outputs['sample_summary'])
        hausman_tables.append(model_outputs['hausman'])
        vif_tables.append(model_outputs['vif'])
        diagnostics_tables.append(model_outputs['diagnostics'])
        missingness_tables.append(model_outputs['sample_missingness'])

        raw_path = OUTPUTS_DIR / f"{model_row.model_id}_regression_raw.txt"
        raw_sections = model_outputs.get("raw_summaries", {}) if isinstance(model_outputs, dict) else {}
        if isinstance(raw_sections, dict) and raw_sections:
            chunks = []
            for key in ["pooled_ols", "fixed_effects", "fixed_effects_driscoll_kraay", "random_effects"]:
                text = str(raw_sections.get(key, "")).strip()
                if not text:
                    continue
                chunks.append(f"===== {key} =====\n{text}\n")
            raw_path.write_text("\n".join(chunks).rstrip() + "\n")

        status_row['status'] = 'estimated'
        status_row['missing_reason'] = ''
        status_rows.append(status_row)

    if not coefficients_tables:
        status_counts = workbook_catalog_df['status'].value_counts(dropna=False).to_dict()
        raise ValueError(
            "No models were estimated (coefficients_tables is empty). "
            f"Status counts: {status_counts}. "
            "This usually means required panel columns are missing or all models were skipped."
        )

    coefficients_df = pd.concat(coefficients_tables, ignore_index=True)
    fit_stats_df = pd.concat(fit_stats_tables, ignore_index=True)
    sample_summary_df = pd.concat(sample_summary_tables, ignore_index=True)
    hausman_df = pd.concat(hausman_tables, ignore_index=True)
    vif_df = pd.concat(vif_tables, ignore_index=True)
    diagnostics_df = pd.concat(diagnostics_tables, ignore_index=True)
    sample_missingness_df = pd.concat(missingness_tables, ignore_index=True)
    model_status_df = pd.DataFrame(status_rows).sort_values('model_id').reset_index(drop=True)

    def _pick_model_id(workbook_code: str, *, contains: str | None = None) -> str | None:
        subset = workbook_catalog_df[workbook_catalog_df["workbook_code"].astype(str).eq(workbook_code)].copy()
        if contains:
            subset = subset[subset["model_id"].astype(str).str.contains(contains, regex=False)]
        subset = subset[subset["status"].eq("estimated")]
        if subset.empty:
            return None
        # Deterministic: prefer non-appendix for baseline codes; otherwise first in sort order.
        subset = subset.sort_values(["appendix_only", "model_order", "model_id"]).reset_index(drop=True)
        return str(subset.at[0, "model_id"])

    # 3. Model Sample Audit Comparisons (workbook-driven)
    print("Running model sample loss audits...")
    sample_audit_specs: list[tuple[str, str, str]] = []

    m1 = _pick_model_id("M1")
    m2 = _pick_model_id("M2")
    m3 = _pick_model_id("M3")
    m4 = _pick_model_id("M4")
    m5 = _pick_model_id("M5")
    m6_from_m2 = _pick_model_id("M6", contains="from_M2")
    m6_from_m4 = _pick_model_id("M6", contains="from_M4")
    m7_from_m2 = _pick_model_id("M7", contains="from_M2")
    m7_from_m4 = _pick_model_id("M7", contains="from_M4")

    if m1 and m2:
        sample_audit_specs.append(("M1_to_M2", m1, m2))
    if m2 and m3:
        sample_audit_specs.append(("M2_to_M3_lagged", m2, m3))
    if m2 and m5:
        sample_audit_specs.append(("M2_to_M5_lending", m2, m5))
    if m2 and m6_from_m2:
        sample_audit_specs.append(("M2_to_M6_tourism", m2, m6_from_m2))
    if m4 and m6_from_m4:
        sample_audit_specs.append(("M4_to_M6_tourism", m4, m6_from_m4))
    if m2 and m7_from_m2:
        sample_audit_specs.append(("M2_to_M7_hc", m2, m7_from_m2))
    if m4 and m7_from_m4:
        sample_audit_specs.append(("M4_to_M7_hc", m4, m7_from_m4))

    require_model_rows(
        sorted({model_id for _, base_id, added_id in sample_audit_specs for model_id in [base_id, added_id]}),
        context="SAMPLE_AUDIT_COMPARISONS",
    )

    sample_audit_rows = []
    sample_loss_driver_rows = []
    for comparison_id, base_model_id, added_model_id in sample_audit_specs:
        base_row = model_row_lookup[base_model_id]
        added_row = model_row_lookup[added_model_id]
        
        base_exclude = [c.strip() for c in base_row.exclude_countries.split(',') if c.strip()] if hasattr(base_row, 'exclude_countries') and isinstance(base_row.exclude_countries, str) else None
        added_exclude = [c.strip() for c in added_row.exclude_countries.split(',') if c.strip()] if hasattr(added_row, 'exclude_countries') and isinstance(added_row.exclude_countries, str) else None

        base_frame, base_regressors = prepared_estimation_frame_and_regressors(base_row, df, dependent=dependent)
        added_frame, added_regressors = prepared_estimation_frame_and_regressors(added_row, df, dependent=dependent)
        
        # Build clean index frames with country exclusions applied
        base_model_df = build_model_frame(base_frame, dependent, base_regressors, exclude_countries=base_exclude)
        added_model_df = build_model_frame(added_frame, dependent, added_regressors, exclude_countries=added_exclude)

        base_keys = base_model_df.reset_index()[['country', 'year']].drop_duplicates().reset_index(drop=True)
        added_keys = added_model_df.reset_index()[['country', 'year']].drop_duplicates().reset_index(drop=True)
        
        lost_keys = base_keys.merge(added_keys, on=['country', 'year'], how='left', indicator=True)
        lost_keys = lost_keys[lost_keys['_merge'].eq('left_only')][['country', 'year']].reset_index(drop=True)
        
        # Track loss drivers
        lost_detail = lost_keys.merge(added_frame, on=['country', 'year'], how='left')
        added_variables = [dependent, *added_regressors]
        driver_counts = lost_detail[added_variables].isna().sum().sort_values(ascending=False)
        driver_counts = driver_counts[driver_counts.gt(0)]
        
        for variable, missing_rows in driver_counts.items():
            affected_countries = sorted(lost_detail.loc[lost_detail[variable].isna(), 'country'].dropna().unique())
            sample_loss_driver_rows.append(
                {
                    'comparison_id': comparison_id,
                    'base_model_id': base_model_id,
                    'added_model_id': added_model_id,
                    'loss_driver_variable': variable,
                    'lost_rows_with_missing_variable': int(missing_rows),
                    'affected_countries': ', '.join(affected_countries),
                }
            )
            
        lost_countries = sorted(lost_keys['country'].dropna().unique())
        sample_audit_rows.append(
            {
                'comparison_id': comparison_id,
                'base_model_id': base_model_id,
                'added_model_id': added_model_id,
                'base_rows': int(len(base_keys)),
                'added_rows': int(len(added_keys)),
                'rows_lost_from_base': int(len(lost_keys)),
                'base_countries': int(base_keys['country'].nunique()),
                'added_countries': int(added_keys['country'].nunique()),
                'countries_lost_from_base': int(len(set(base_keys['country']) - set(added_keys['country']))),
                'lost_countries': ', '.join(lost_countries) if lost_countries else 'None',
                'top_loss_drivers': '; '.join(f'{variable}: {int(count)}' for variable, count in driver_counts.head(5).items()) or 'None',
            }
        )

    model_sample_audit_df = pd.DataFrame(sample_audit_rows)
    model_sample_loss_drivers_df = pd.DataFrame(sample_loss_driver_rows)

    # 4. Common Samples Estimation (mirror the audit comparisons)
    print("Running common sample estimations...")
    common_sample_specs = [(f"{cid}_common_sample", left, right) for cid, left, right in sample_audit_specs]
    require_model_rows(
        sorted({model_id for _, left_id, right_id in common_sample_specs for model_id in [left_id, right_id]}),
        context="COMMON_SAMPLE_COMPARISONS",
    )
    common_coefficients_tables = []
    common_fit_stats_tables = []
    common_sample_summary_tables = []
    common_hausman_tables = []
    common_vif_tables = []
    common_diagnostics_tables = []

    for comparison_id, left_model_id, right_model_id in common_sample_specs:
        prepared = {}
        key_frames = []
        for model_id in [left_model_id, right_model_id]:
            model_row = model_row_lookup[model_id]
            exclude_countries = [c.strip() for c in model_row.exclude_countries.split(',') if c.strip()] if hasattr(model_row, 'exclude_countries') and isinstance(model_row.exclude_countries, str) else None
            
            estimation_frame, regressors = prepared_estimation_frame_and_regressors(model_row, df, dependent=dependent)
            
            # Apply exclusions to keys
            model_df = build_model_frame(estimation_frame, dependent, regressors, exclude_countries=exclude_countries)
            keys = model_df.reset_index()[['country', 'year']].drop_duplicates().reset_index(drop=True)
            
            prepared[model_id] = (model_row, estimation_frame, regressors, keys, exclude_countries)
            key_frames.append(keys)
            
        common_keys = key_frames[0].merge(key_frames[1], on=['country', 'year'], how='inner')
        for model_id, (model_row, estimation_frame, regressors, native_keys, exclude_countries) in prepared.items():
            common_frame = filter_to_keys(estimation_frame, common_keys)
            common_model_id = f'{model_id}__{comparison_id}'
            common_outputs = estimate_model(common_frame, common_model_id, regressors, exclude_countries=exclude_countries, dependent=dependent)
            
            for table_name in ['coefficients', 'fit_stats', 'sample_summary', 'hausman', 'vif', 'diagnostics']:
                table = common_outputs[table_name].copy()
                table['comparison_id'] = comparison_id
                table['native_model_id'] = model_id
                table['sample_mode'] = 'common_sample'
                table['native_rows'] = int(len(native_keys))
                table['common_rows'] = int(len(common_keys))
                if table_name == 'coefficients':
                    common_coefficients_tables.append(table)
                elif table_name == 'fit_stats':
                    common_fit_stats_tables.append(table)
                elif table_name == 'sample_summary':
                    common_sample_summary_tables.append(table)
                elif table_name == 'hausman':
                    common_hausman_tables.append(table)
                elif table_name == 'vif':
                    common_vif_tables.append(table)
                elif table_name == 'diagnostics':
                    common_diagnostics_tables.append(table)

    common_sample_coefficients_df = pd.concat(common_coefficients_tables, ignore_index=True)
    common_sample_fit_stats_df = pd.concat(common_fit_stats_tables, ignore_index=True)
    common_sample_summary_df = pd.concat(common_sample_summary_tables, ignore_index=True)
    common_sample_hausman_df = pd.concat(common_hausman_tables, ignore_index=True)
    common_sample_vif_df = pd.concat(common_vif_tables, ignore_index=True)
    common_sample_diagnostics_df = pd.concat(common_diagnostics_tables, ignore_index=True)
    common_sample_overview = common_sample_summary_df[
        ['comparison_id', 'native_model_id', 'native_rows', 'common_rows', 'rows_used', 'countries_used', 'regressors']
    ].sort_values(['comparison_id', 'native_model_id']).reset_index(drop=True)

    # 5. Panel Balance Metrics
    print("Running model panel balance summaries...")
    model_panel_balance_tables = []
    for model_row in workbook_catalog_df[workbook_catalog_df['status'].eq('estimated')].itertuples(index=False):
        exclude_countries = [c.strip() for c in model_row.exclude_countries.split(',') if c.strip()] if hasattr(model_row, 'exclude_countries') and isinstance(model_row.exclude_countries, str) else None
        estimation_frame, regressors = prepared_estimation_frame_and_regressors(model_row, df, dependent=dependent)
        model_df = build_model_frame(estimation_frame, dependent, regressors, exclude_countries=exclude_countries)
        keys = model_df.reset_index()[['country', 'year']].drop_duplicates().reset_index(drop=True)
        model_panel_balance_tables.append(summarize_model_country_windows(keys, model_row.model_id))

    model_panel_balance_by_country = pd.concat(model_panel_balance_tables, ignore_index=True)
    model_panel_balance_summary = (
        model_panel_balance_by_country.groupby('model_id')
        .agg(
            countries_used=('country', 'nunique'),
            total_rows=('usable_years', 'sum'),
            min_country_years=('usable_years', 'min'),
            median_country_years=('usable_years', 'median'),
            max_country_years=('usable_years', 'max'),
            countries_with_lt_8_years=('usable_years', lambda series: int((series < 8).sum())),
            countries_with_internal_gaps=('internal_missing_years', lambda series: int((series > 0).sum())),
        )
        .reset_index()
    )
    model_panel_balance_summary = model_panel_balance_summary.merge(
        workbook_catalog_df[['model_id', 'workbook_model', 'purpose']],
        on='model_id',
        how='left',
    )
    model_panel_balance_summary['panel_reliability_flag'] = np.select(
        [
            model_panel_balance_summary['countries_used'].lt(8),
            model_panel_balance_summary['countries_with_lt_8_years'].gt(0),
            model_panel_balance_summary['countries_with_internal_gaps'].gt(0),
        ],
        [
            'high_risk_fewer_than_8_countries',
            'review_country_with_short_time_series',
            'review_internal_year_gaps',
        ],
        default='acceptable_unbalanced_panel_with_caveat',
    )
    model_panel_balance_summary = model_panel_balance_summary[
        [
            'model_id',
            'workbook_model',
            'purpose',
            'countries_used',
            'total_rows',
            'min_country_years',
            'median_country_years',
            'max_country_years',
            'countries_with_lt_8_years',
            'countries_with_internal_gaps',
            'panel_reliability_flag',
        ]
    ]

    # Save outputs to csv
    print("Writing main output CSV files...")
    coefficients_df.to_csv(OUTPUTS_DIR / 'model_coefficients.csv', index=False)
    fit_stats_df.to_csv(OUTPUTS_DIR / 'model_fit_stats.csv', index=False)
    sample_summary_df.to_csv(OUTPUTS_DIR / 'model_sample_summary.csv', index=False)
    hausman_df.to_csv(OUTPUTS_DIR / 'hausman_results.csv', index=False)
    vif_df.to_csv(OUTPUTS_DIR / 'model_vif.csv', index=False)
    diagnostics_df.to_csv(OUTPUTS_DIR / 'model_diagnostics.csv', index=False)
    sample_missingness_df.to_csv(OUTPUTS_DIR / 'model_sample_missingness.csv', index=False)
    model_status_df.to_csv(OUTPUTS_DIR / 'model_status.csv', index=False)
    model_sample_audit_df.to_csv(OUTPUTS_DIR / 'model_sample_audit.csv', index=False)
    model_sample_loss_drivers_df.to_csv(OUTPUTS_DIR / 'model_sample_loss_drivers.csv', index=False)
    model_panel_balance_summary.to_csv(OUTPUTS_DIR / 'model_panel_balance_summary.csv', index=False)
    model_panel_balance_by_country.to_csv(OUTPUTS_DIR / 'model_panel_balance_by_country.csv', index=False)
    common_sample_coefficients_df.to_csv(OUTPUTS_DIR / 'common_sample_coefficients.csv', index=False)
    common_sample_fit_stats_df.to_csv(OUTPUTS_DIR / 'common_sample_fit_stats.csv', index=False)
    common_sample_summary_df.to_csv(OUTPUTS_DIR / 'common_sample_summary.csv', index=False)
    common_sample_overview.to_csv(OUTPUTS_DIR / 'common_sample_overview.csv', index=False)
    common_sample_hausman_df.to_csv(OUTPUTS_DIR / 'common_sample_hausman.csv', index=False)
    common_sample_vif_df.to_csv(OUTPUTS_DIR / 'common_sample_vif.csv', index=False)
    common_sample_diagnostics_df.to_csv(OUTPUTS_DIR / 'common_sample_diagnostics.csv', index=False)
    workbook_catalog_df.to_csv(OUTPUTS_DIR / 'workbook_model_catalog.csv', index=False)
    workbook_variable_selection.to_csv(OUTPUTS_DIR / 'workbook_variable_selection.csv', index=False)
    workbook_model_specs.to_csv(OUTPUTS_DIR / 'workbook_model_specs.csv', index=False)
    workbook_notes.to_csv(OUTPUTS_DIR / 'workbook_notes.csv', index=False)
    skipped_models.to_csv(OUTPUTS_DIR / 'skipped_workbook_models.csv', index=False)

    # 6. Load tables generated by 01_data_processing (if present) or handle gracefully
    def load_if_exists(filename, default_df=pd.DataFrame()):
        p = OUTPUTS_DIR / filename
        if p.exists():
            return pd.read_csv(p)
        return default_df

    variable_audit = load_if_exists('variable_audit.csv')
    spec_sample_summary = load_if_exists('spec_sample_summary.csv')
    review_flags = load_if_exists('review_flags.csv')
    spec_panel_balance_summary = load_if_exists('spec_panel_balance_summary.csv')
    spec_panel_balance_by_country = load_if_exists('spec_panel_balance_by_country.csv')
    
    # 7. Descriptive Stats & Main Variables Correlation
    print("Computing descriptive stats and correlations...")
    main_variables = [dependent]
    for regressors_str in estimated_models['base_regressors']:
        main_variables.extend(parse_regressor_string(regressors_str))
    main_variables = list(dict.fromkeys(main_variables))

    descriptive_stats = df[main_variables].describe().T
    descriptive_stats['missing'] = df[main_variables].isna().sum()
    descriptive_stats['missing_pct'] = df[main_variables].isna().mean() * 100

    correlation_matrix = df[main_variables].corr(method='pearson')
    correlation_pairwise_n = pd.DataFrame(
        {
            column: {
                other_column: int(df[[column, other_column]].dropna().shape[0])
                for other_column in main_variables
            }
            for column in main_variables
        }
    ).reindex(index=main_variables, columns=main_variables)

    # Save main variable correlation heatmaps
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson correlation'},
    )
    plt.title('Correlation Matrix Across Workbook-Driven Analysis Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_matrix_main_variables.png', dpi=200)
    plt.close()

    descriptive_stats.to_csv(OUTPUTS_DIR / 'descriptive_stats.csv')
    correlation_matrix.to_csv(OUTPUTS_DIR / 'correlation_matrix_main_variables.csv')
    correlation_pairwise_n.to_csv(OUTPUTS_DIR / 'correlation_pairwise_n.csv')

    # 8. Setup results tables mappings
    fit_lookup = fit_stats_df.set_index(['model_id', 'estimator'])
    sample_lookup = sample_summary_df.set_index('model_id')
    hausman_lookup = hausman_df.set_index('model_id')
    diagnostics_lookup = diagnostics_df.set_index('model_id')

    preferred_estimator_map = {
        row.model_id: 'fixed_effects_driscoll_kraay'
        for row in hausman_df.itertuples(index=False)
    }

    # Setup expected sign match
    expected_sign_lookup = {
        variable: {'expected_sign': sign, 'expected_sign_source': source}
        for variable, (sign, source) in THEORY_EXPECTED_SIGNS.items()
    }
    for row in workbook_variable_selection.itertuples(index=False):
        suggested_name = getattr(row, 'suggested_variable_name', '')
        workbook_sign = normalize_expected_sign(getattr(row, 'expected_sign', pd.NA))
        if workbook_sign == 'not stated':
            continue
        for token in str(suggested_name).replace('/', ' or ').split(' or '):
            workbook_variable = token.strip()
            if not workbook_variable or workbook_variable == 'drop':
                continue
            mapped_variable = WORKBOOK_VARIABLE_MAP.get(workbook_variable, workbook_variable)
            expected_sign_lookup[mapped_variable] = {
                'expected_sign': workbook_sign,
                'expected_sign_source': 'workbook Variable selection expected_sign',
            }

    # 9. Individual Model Results Construction Loop
    inference_tables = {}
    diagnostic_tables = {}
    sample_fit_tables = {}
    regression_detail_notes = {}

    lost_country_note_by_model = {
        row.added_model_id: row.lost_countries
        for row in model_sample_audit_df.itertuples(index=False)
    }

    # Structural proxy coverage note
    structural_coverage_rows = []
    for model_row in estimated_models.itertuples(index=False):
        proxy = getattr(model_row, 'monetary_proxy', '')
        if not isinstance(proxy, str) or not proxy or proxy not in df.columns:
            continue
        coverage = df.groupby('country')[proxy].agg(
            total_years='size',
            nonmissing_years=lambda series: int(series.notna().sum()),
        ).reset_index()
        coverage['missing_years'] = coverage['total_years'] - coverage['nonmissing_years']
        coverage['model_id'] = model_row.model_id
        coverage['workbook_model'] = model_row.workbook_model
        coverage['monetary_proxy'] = proxy
        coverage['structural_gap_flag'] = coverage['nonmissing_years'].le(1)
        structural_coverage_rows.append(coverage)

    structural_proxy_coverage = pd.concat(structural_coverage_rows, ignore_index=True) if structural_coverage_rows else pd.DataFrame()
    structural_coverage_note_by_model = {}
    if not structural_proxy_coverage.empty:
        for model_id, group in structural_proxy_coverage.groupby('model_id'):
            sparse = group[group['structural_gap_flag']].copy()
            if sparse.empty:
                structural_coverage_note_by_model[model_id] = 'No country has <=1 nonmissing monetary-proxy year.'
            else:
                structural_coverage_note_by_model[model_id] = '; '.join(
                    f"{row.country} ({int(row.nonmissing_years)}/{int(row.total_years)} nonmissing {row.monetary_proxy})"
                    for row in sparse.itertuples(index=False)
                )

    print("Building diagnostic tables and regression detail files for individual models...")
    for model_row in estimated_models.itertuples(index=False):
        model_id = model_row.model_id
        preferred_estimator = preferred_estimator_map[model_id]
        
        coefficients = coefficients_df[
            coefficients_df['model_id'].eq(model_id)
            & coefficients_df['estimator'].eq(preferred_estimator)
        ].copy()
        coefficients = coefficients[
            ['term', 'coef', 'std_error', 't_stat', 'p_value', 'ci_low', 'ci_high']
        ].rename(columns={'term': 'variable', 'coef': 'coefficient'})
        inference_tables[model_id] = coefficients

        hausman_row = hausman_lookup.loc[model_id]
        diagnostics_row = diagnostics_lookup.loc[model_id]
        
        model_vif = (
            vif_df[vif_df['model_id'].eq(model_id)][['variable', 'vif']]
            .sort_values('vif', ascending=False)
            .reset_index(drop=True)
        )
        model_missingness = (
            sample_missingness_df[sample_missingness_df['model_id'].eq(model_id)][
                ['variable', 'missing_count', 'missing_share']
            ]
            .sort_values(['missing_count', 'variable'], ascending=[False, True])
            .reset_index(drop=True)
        )
        diagnostics_table = pd.DataFrame(
            {
                'metric': [
                    'Headline estimator',
                    'Hausman diagnostic',
                    'Hausman p-value',
                    'Hausman raw statistic',
                    'Hausman negative-statistic flag',
                    'FE vs RE comparison note',
                    'Diagnostic model note',
                    'Pesaran CD statistic',
                    'Pesaran CD p-value',
                    'Condition number',
                    'Breusch-Pagan LM p-value',
                    'White LM p-value',
                    'Max VIF',
                    'Lost-country note',
                ],
                'value': [
                    estimator_label(preferred_estimator),
                    hausman_status(hausman_row),
                    f"{hausman_row['p_value']:.4f}" if not pd.isna(hausman_row['p_value']) else 'n/a',
                    f"{hausman_row['raw_statistic']:.4f}" if not pd.isna(hausman_row['raw_statistic']) else 'n/a',
                    str(bool(hausman_row['negative_statistic_flag'])),
                    str(hausman_row.get('comparison_note', 'ok')),
                    'BP/White/condition-number diagnostics use pooled OLS residuals/design; Pesaran CD uses FE-DK residuals.',
                    f"{diagnostics_row['pesaran_cd_stat']:.4f}" if not pd.isna(diagnostics_row['pesaran_cd_stat']) else 'n/a',
                    f"{diagnostics_row['pesaran_cd_p_value']:.4f}" if not pd.isna(diagnostics_row['pesaran_cd_p_value']) else 'n/a',
                    f"{diagnostics_row['condition_number']:.4f}",
                    f"{diagnostics_row['breusch_pagan_lm_p_value']:.4f}",
                    f"{diagnostics_row['white_lm_p_value']:.4f}",
                    f"{model_vif['vif'].max():.4f}",
                    lost_country_note_by_model.get(model_id, 'Reference or no adjacent loss audit'),
                ],
            }
        )
        diagnostic_tables[model_id] = {
            'summary': diagnostics_table,
            'vif': model_vif,
            'missingness': model_missingness,
        }

        sample_fit_tables[model_id] = pd.DataFrame(
            {
                'metric': [
                    'Observations',
                    'Countries used',
                    'Years used',
                    'Estimator',
                    'Hausman diagnostic',
                    'Within R-squared',
                    'Slope-adjusted within R-squared',
                    'R2 minus adjusted R2 gap',
                    'Max VIF',
                    'Lost-country note',
                    'Regressors',
                ],
                'value': [
                    int(fit_lookup.at[(model_id, preferred_estimator), 'nobs']),
                    int(sample_lookup.at[model_id, 'countries_used']),
                    int(sample_lookup.at[model_id, 'years_used']),
                    estimator_label(preferred_estimator),
                    hausman_status(hausman_row),
                    f"{fit_lookup.at[(model_id, preferred_estimator), 'within_r2']:.4f}",
                    f"{fit_lookup.at[(model_id, preferred_estimator), 'slope_adjusted_within_r2']:.4f}",
                    f"{fit_lookup.at[(model_id, preferred_estimator), 'r2_adj_r2_gap']:.4f}",
                    f"{model_vif['vif'].max():.4f}",
                    lost_country_note_by_model.get(model_id, 'Reference or no adjacent loss audit'),
                    sample_lookup.at[model_id, 'regressors'],
                ],
            }
        )

        hausman_note = (
            f"Random-effects note: {hausman_row['re_failure_reason']}. "
            if isinstance(hausman_row.get('re_failure_reason', ''), str) and hausman_row.get('re_failure_reason', '')
            else ''
        ) + (
            'Hausman note: covariance-difference inversion became numerically unstable, so the raw statistic turned negative. '
            'The displayed statistic was clipped at zero, which mechanically gives p=1.0000.'
            if bool(hausman_row['negative_statistic_flag'])
            else 'Hausman note: positive test statistic with the standard FE-versus-RE interpretation.'
        )
        regression_detail_notes[model_id] = (
            f"Headline estimator: {estimator_label(preferred_estimator)}. "
            f"Hausman status: {hausman_status(hausman_row)}. "
            f"Recommendation: {model_row.recommendation}. "
            f"Structural coverage note: {structural_coverage_note_by_model.get(model_id, 'No structural monetary-proxy coverage note')}. "
            f"{hausman_note}"
        )

        # Save individual model outputs to files
        inference_tables[model_id].to_csv(OUTPUTS_DIR / f'{model_id}_regression_detail.csv', index=False)
        inference_tables[model_id].to_excel(OUTPUTS_DIR / f'{model_id}_regression_detail.xlsx', index=False)
        (OUTPUTS_DIR / f'{model_id}_regression_detail_note.txt').write_text(regression_detail_notes[model_id])
        diagnostic_tables[model_id]['summary'].to_csv(OUTPUTS_DIR / f'{model_id}_diagnostics.csv', index=False)
        diagnostic_tables[model_id]['vif'].to_csv(OUTPUTS_DIR / f'{model_id}_vif.csv', index=False)
        diagnostic_tables[model_id]['missingness'].to_csv(OUTPUTS_DIR / f'{model_id}_missingness.csv', index=False)
        sample_fit_tables[model_id].to_csv(OUTPUTS_DIR / f'{model_id}_sample_fit_summary.csv', index=False)

    # 10. Main Regression Table Construction
    print("Building main regression table...")
    reporting_models = estimated_models[~estimated_models['appendix_only'].astype(bool)].copy()
    reporting_models['workbook_model_label'] = reporting_models.apply(
        lambda row: f"{row['workbook_model']} (t-1)" if bool(row.get('lagged_model', False)) else str(row['workbook_model']),
        axis=1,
    )
    label_by_model_id = reporting_models.set_index('model_id')['workbook_model_label'].to_dict()
    regression_table_rows = []
    regression_terms = [
        'broad_money_growth_pct',
        'deposit_interest_rate_pct',
        'real_interest_rate_pct',
        'lending_interest_rate_pct',
        'trade_pct_gdp',
        'inflation_gdp_deflator_pct',
        'ln_gdppc',
        'xr_dep_pct',
    ]
    for term in regression_terms:
        row = {'metric': VARIABLE_LABELS.get(term, term)}
        for model_row in reporting_models.itertuples(index=False):
            table = inference_tables[model_row.model_id]
            lookup_term = f"{term}_lag1" if bool(getattr(model_row, 'lagged_model', False)) else term
            if lookup_term not in set(table['variable'].astype(str).unique()):
                lookup_term = term
            match = table[table['variable'].eq(lookup_term)]
            row[label_by_model_id.get(model_row.model_id, model_row.workbook_model)] = (
                format_coef_cell(match.iloc[0]) if not match.empty else ''
            )
        regression_table_rows.append(row)

    summary_metrics = {
        'Observations': lambda model_id, estimator, hausman_row, model_vif, diag_row: int(fit_lookup.at[(model_id, estimator), 'nobs']),
        'Countries used': lambda model_id, estimator, hausman_row, model_vif, diag_row: int(sample_lookup.at[model_id, 'countries_used']),
        'Estimator': lambda model_id, estimator, hausman_row, model_vif, diag_row: estimator_label(estimator),
        'Covariance type': lambda model_id, estimator, hausman_row, model_vif, diag_row: fit_lookup.at[(model_id, estimator), 'covariance_type'],
        'Hausman diagnostic': lambda model_id, estimator, hausman_row, model_vif, diag_row: hausman_status(hausman_row),
        'Within R-squared': lambda model_id, estimator, hausman_row, model_vif, diag_row: f"{fit_lookup.at[(model_id, estimator), 'within_r2']:.4f}",
        'Slope-adjusted within R-squared': lambda model_id, estimator, hausman_row, model_vif, diag_row: f"{fit_lookup.at[(model_id, estimator), 'slope_adjusted_within_r2']:.4f}",
        'R2 minus adjusted R2 gap': lambda model_id, estimator, hausman_row, model_vif, diag_row: f"{fit_lookup.at[(model_id, estimator), 'r2_adj_r2_gap']:.4f}",
        'Max VIF': lambda model_id, estimator, hausman_row, model_vif, diag_row: f"{model_vif['vif'].max():.2f}",
        'Pesaran CD p-value': lambda model_id, estimator, hausman_row, model_vif, diag_row: f"{diag_row['pesaran_cd_p_value']:.4f}" if not pd.isna(diag_row['pesaran_cd_p_value']) else 'n/a',
        'Lost-country note': lambda model_id, estimator, hausman_row, model_vif, diag_row: lost_country_note_by_model.get(model_id, 'Reference/no audited loss'),
        'Structural monetary-proxy coverage note': lambda model_id, estimator, hausman_row, model_vif, diag_row: structural_coverage_note_by_model.get(model_id, 'No structural monetary-proxy coverage note'),
        'Thesis role': lambda model_id, estimator, hausman_row, model_vif, diag_row: estimated_models.set_index('model_id').at[model_id, 'thesis_role'],
    }
    for metric, value_fn in summary_metrics.items():
        row = {'metric': metric}
        for model_row in reporting_models.itertuples(index=False):
            model_id = model_row.model_id
            estimator = preferred_estimator_map[model_id]
            hausman_row = hausman_lookup.loc[model_id]
            diagnostics_row = diagnostics_lookup.loc[model_id]
            model_vif = vif_df[vif_df['model_id'].eq(model_id)][['variable', 'vif']]
            row[label_by_model_id.get(model_id, model_row.workbook_model)] = value_fn(
                model_id, estimator, hausman_row, model_vif, diagnostics_row
            )
        regression_table_rows.append(row)

    regression_table_main = pd.DataFrame(regression_table_rows).set_index('metric')
    regression_table_main.to_csv(OUTPUTS_DIR / 'regression_table_main.csv')
    
    # Expected sign match table
    expected_sign_rows = []
    for model_row in estimated_models.itertuples(index=False):
        model_id = model_row.model_id
        estimator = preferred_estimator_map[model_id]
        model_coefficients = coefficients_df[
            coefficients_df['model_id'].eq(model_id)
            & coefficients_df['estimator'].eq(estimator)
        ]
        for coef_row in model_coefficients.itertuples(index=False):
            if coef_row.term == 'Intercept':
                continue
            base_term = strip_lag_suffix(coef_row.term)
            expectation = expected_sign_lookup.get(
                base_term,
                {'expected_sign': 'not stated', 'expected_sign_source': 'not available'},
            )
            expected_sign_rows.append(
                {
                    'model_id': model_id,
                    'workbook_model': model_row.workbook_model,
                    'model_tier': model_row.model_tier,
                    'headline_eligible': bool(model_row.headline_eligible),
                    'appendix_only': bool(model_row.appendix_only),
                    'estimator': estimator_label(estimator),
                    'covariance_type': fit_lookup.at[(model_id, estimator), 'covariance_type'],
                    'term': coef_row.term,
                    'base_term': base_term,
                    'coefficient': coef_row.coef,
                    'p_value': coef_row.p_value,
                    'coefficient_direction': coefficient_direction(coef_row.coef),
                    'expected_sign': expectation['expected_sign'],
                    'expected_sign_source': expectation['expected_sign_source'],
                    'sign_alignment': sign_alignment(coef_row.coef, expectation['expected_sign']),
                }
            )

    expected_sign_match_table = pd.DataFrame(expected_sign_rows)
    expected_sign_match_table.to_csv(OUTPUTS_DIR / 'expected_sign_match.csv', index=False)

    # Fit gap ranking
    preferred_fit_rows = []
    for model_row in estimated_models.itertuples(index=False):
        model_id = model_row.model_id
        preferred_estimator = preferred_estimator_map[model_id]
        fit_row = fit_lookup.loc[(model_id, preferred_estimator)]
        preferred_fit_rows.append(
            {
                'model_id': model_id,
                'model': model_row.workbook_model,
                'model_tier': model_row.model_tier,
                'headline_eligible': bool(model_row.headline_eligible),
                'appendix_only': bool(model_row.appendix_only),
                'purpose': model_row.purpose,
                'estimator': estimator_label(preferred_estimator),
                'nobs': fit_row['nobs'],
                'within_r2': fit_row['within_r2'],
                'slope_adjusted_within_r2': fit_row['slope_adjusted_within_r2'],
                'r_squared': fit_row['r_squared'],
                'adj_r_squared': fit_row['adj_r_squared'],
                'r2_adj_r2_gap': fit_row['r2_adj_r2_gap'],
            }
        )

    model_fit_gap_ranking = pd.DataFrame(preferred_fit_rows).sort_values(
        ['appendix_only', 'r2_adj_r2_gap', 'slope_adjusted_within_r2', 'within_r2'],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    model_fit_gap_ranking['gap_rank'] = model_fit_gap_ranking.index + 1
    model_fit_gap_ranking['selected_by_low_gap'] = (
        model_fit_gap_ranking['headline_eligible'].astype(bool)
        & model_fit_gap_ranking['gap_rank'].le(3)
    )
    model_fit_gap_ranking['headline_selection_allowed'] = model_fit_gap_ranking['headline_eligible'].astype(bool)

    low_gap_selected_models = model_fit_gap_ranking[model_fit_gap_ranking['selected_by_low_gap']].copy()
    
    coefficient_interpretation_rows = []
    for selected_model in low_gap_selected_models.itertuples(index=False):
        model_id = selected_model.model_id
        preferred_estimator = preferred_estimator_map[model_id]
        selected_coefficients = coefficients_df[
            coefficients_df['model_id'].eq(model_id)
            & coefficients_df['estimator'].eq(preferred_estimator)
            & ~coefficients_df['term'].eq('Intercept')
        ].copy()
        for coef_row in selected_coefficients.itertuples(index=False):
            base_term = strip_lag_suffix(coef_row.term)
            expectation = expected_sign_lookup.get(
                base_term,
                {'expected_sign': 'not stated', 'expected_sign_source': 'not available'},
            )
            row = {
                'gap_rank': selected_model.gap_rank,
                'model_id': model_id,
                'model': selected_model.model,
                'estimator': selected_model.estimator,
                'term': coef_row.term,
                'base_term': base_term,
                'coefficient_label': VARIABLE_LABELS.get(base_term, base_term),
                'coef': coef_row.coef,
                'p_value': coef_row.p_value,
                'coefficient_direction': coefficient_direction(coef_row.coef),
                'significance': significance_label(coef_row.p_value),
                'expected_sign': expectation['expected_sign'],
                'expected_sign_source': expectation['expected_sign_source'],
                'sign_alignment': sign_alignment(coef_row.coef, expectation['expected_sign']),
            }
            row['interpretation'] = interpret_coefficient_row(pd.Series(row))
            coefficient_interpretation_rows.append(row)

    low_gap_coefficient_interpretations = pd.DataFrame(coefficient_interpretation_rows)

    model_fit_gap_ranking.to_csv(OUTPUTS_DIR / 'model_fit_gap_ranking.csv', index=False)
    low_gap_selected_models.to_csv(OUTPUTS_DIR / 'low_gap_selected_models.csv', index=False)
    low_gap_coefficient_interpretations.to_csv(OUTPUTS_DIR / 'low_gap_coefficient_interpretations.csv', index=False)

    # 11. Stepwise Broad Money Sign Decomposition Execution
    print("Running stepwise broad money sign decomposition...")
    broad_money_decomposition_specs = [
        {
            'step': 1,
            'spec_id': 'bm_only',
            'spec_label': 'Broad money only',
            'regressors': ['broad_money_growth_pct'],
        },
        {
            'step': 2,
            'spec_id': 'bm_inflation',
            'spec_label': 'Broad money + inflation',
            'regressors': ['broad_money_growth_pct', 'inflation_gdp_deflator_pct'],
        },
        {
            'step': 3,
            'spec_id': 'bm_trade',
            'spec_label': 'Broad money + trade',
            'regressors': ['broad_money_growth_pct', 'trade_pct_gdp'],
        },
        {
            'step': 4,
            'spec_id': 'bm_gdppc',
            'spec_label': 'Broad money + log GDP per capita',
            'regressors': ['broad_money_growth_pct', 'ln_gdppc'],
        },
        {
            'step': 5,
            'spec_id': 'bm_trade_gdppc',
            'spec_label': 'Broad money + trade + log GDP per capita',
            'regressors': ['broad_money_growth_pct', 'trade_pct_gdp', 'ln_gdppc'],
        },
        {
            'step': 6,
            'spec_id': 'm1_controls',
            'spec_label': 'M1 baseline controls',
            'regressors': ['broad_money_growth_pct', 'inflation_gdp_deflator_pct', 'trade_pct_gdp', 'ln_gdppc', 'xr_dep_pct'],
        },
        {
            'step': 7,
            'spec_id': 'm2_full_controls',
            'spec_label': 'M2 full controls',
            'regressors': [
                'broad_money_growth_pct',
                'deposit_interest_rate_pct',
                'inflation_gdp_deflator_pct',
                'trade_pct_gdp',
                'ln_gdppc',
                'xr_dep_pct',
            ],
        },
    ]

    m2_common_index = build_model_frame(
        df,
        dependent,
        broad_money_decomposition_specs[-1]['regressors'],
        exclude_countries=["Cambodia", "Lao PDR"]
    ).index
    common_m2_frame = (
        df.set_index(['country', 'year'])
        .loc[m2_common_index]
        .reset_index()
    )

    broad_money_decomposition_rows = []
    for sample_type, source_frame in [
        ('native_complete_case', df),
        ('fixed_m2_common_sample', common_m2_frame),
    ]:
        for spec in broad_money_decomposition_specs:
            for estimator in ['pooled_ols', 'fixed_effects_driscoll_kraay', 'fixed_effects', 'random_effects']:
                result_row = fit_broad_money_decomposition_model(source_frame, spec['regressors'], estimator, dependent=dependent)
                result_row.update(
                    {
                        'sample_type': sample_type,
                        'step': spec['step'],
                        'spec_id': spec['spec_id'],
                        'spec_label': spec['spec_label'],
                        'estimator': estimator,
                        'estimator_label': estimator_label(estimator),
                        'regressors': ', '.join(spec['regressors']),
                    }
                )
                broad_money_decomposition_rows.append(result_row)

    broad_money_decomposition_results = pd.DataFrame(broad_money_decomposition_rows)
    broad_money_decomposition_results['coef_sign'] = broad_money_decomposition_results['coef'].apply(sign_from_coefficient)
    
    # Filter output columns
    broad_money_decomposition_results = broad_money_decomposition_results[
        [
            'sample_type',
            'step',
            'spec_id',
            'spec_label',
            'estimator',
            'estimator_label',
            'nobs',
            'countries',
            'years',
            'coef',
            'coef_sign',
            'p_value',
            'r_squared',
            'adj_r_squared',
            'fit_status',
            'regressors',
        ]
    ]

    sign_path_rows = []
    for (sample_type, estimator), group in broad_money_decomposition_results.groupby(['sample_type', 'estimator'], sort=False):
        estimated_group = group[group['fit_status'].eq('estimated')].sort_values('step')
        negative_steps = estimated_group[estimated_group['coef_sign'].eq('negative')]
        positive_steps = estimated_group[estimated_group['coef_sign'].eq('positive')]
        first_negative = negative_steps.iloc[0] if not negative_steps.empty else None
        first_positive = positive_steps.iloc[0] if not positive_steps.empty else None
        sign_path_rows.append(
            {
                'sample_type': sample_type,
                'estimator': estimator,
                'estimator_label': estimator_label(estimator),
                'first_positive_step': pd.NA if first_positive is None else int(first_positive['step']),
                'first_positive_spec': pd.NA if first_positive is None else first_positive['spec_label'],
                'first_negative_step': pd.NA if first_negative is None else int(first_negative['step']),
                'first_negative_spec': pd.NA if first_negative is None else first_negative['spec_label'],
                'sign_path': ' -> '.join(
                    f"{int(row.step)}:{row.coef_sign} ({row.coef:.4f})"
                    for row in estimated_group.itertuples(index=False)
                ),
            }
        )

    broad_money_sign_path_summary = pd.DataFrame(sign_path_rows)
    broad_money_decomposition_pivot = broad_money_decomposition_results.pivot_table(
        index=['sample_type', 'step', 'spec_label', 'nobs'],
        columns='estimator_label',
        values='coef',
        aggfunc='first',
    ).reset_index()

    broad_money_decomposition_results.to_csv(OUTPUTS_DIR / 'broad_money_sign_decomposition.csv', index=False)
    broad_money_decomposition_pivot.to_csv(OUTPUTS_DIR / 'broad_money_sign_decomposition_pivot.csv', index=False)
    broad_money_sign_path_summary.to_csv(OUTPUTS_DIR / 'broad_money_sign_path_summary.csv', index=False)

    # 12. Trade Collinearity Robustness
    print("Running trade collinearity robustness drops...")
    appendix_reference_candidates = [
        'M7_human_capital_robustnessa_from_M2_main_monetary_policy',
        'M7_human_capital_robustnessb_from_M4_real_interest_robustness',
        'M2_main_monetary_policy',
    ]
    estimated_model_ids = set(estimated_models['model_id'])
    appendix_reference_model_id = next(
        candidate for candidate in appendix_reference_candidates
        if candidate in estimated_model_ids
    )
    appendix_reference_regressors = parse_regressor_string(
        estimated_models.set_index('model_id').at[appendix_reference_model_id, 'mapped_regressors']
    )
    
    # Create the regressors-only correlation matrix for reference model
    regressor_correlation_matrix = df[appendix_reference_regressors].corr(method='pearson')
    regressor_pairwise_n = pd.DataFrame(
        {
            column: {
                other_column: int(df[[column, other_column]].dropna().shape[0])
                for other_column in appendix_reference_regressors
            }
            for column in appendix_reference_regressors
        }
    ).reindex(index=appendix_reference_regressors, columns=appendix_reference_regressors)

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        regressor_correlation_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson correlation'},
    )
    plt.title(f'Regressors-Only Correlation Matrix For {appendix_reference_model_id}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'regressor_correlation_matrix_workbook_appendix.png', dpi=200)
    plt.close()

    regressor_correlation_matrix.to_csv(OUTPUTS_DIR / 'regressor_correlation_matrix_workbook_appendix.csv')
    regressor_pairwise_n.to_csv(OUTPUTS_DIR / 'regressor_pairwise_n_workbook_appendix.csv')

    robustness_drop_scenarios = {
        appendix_reference_model_id: appendix_reference_regressors,
        f'{appendix_reference_model_id}_drop_trade_pct_gdp': [
            column for column in appendix_reference_regressors if column != 'trade_pct_gdp'
        ],
    }
    if 'ln_gdppc' in appendix_reference_regressors:
        robustness_drop_scenarios[f'{appendix_reference_model_id}_drop_ln_gdppc'] = [
            column for column in appendix_reference_regressors if column != 'ln_gdppc'
        ]
    if 'hc_human_capital_index' in appendix_reference_regressors:
        robustness_drop_scenarios[f'{appendix_reference_model_id}_drop_hc_human_capital_index'] = [
            column for column in appendix_reference_regressors if column != 'hc_human_capital_index'
        ]

    robustness_summary_rows = []
    robustness_vif_tables = {}

    for scenario_name, scenario_regressors in robustness_drop_scenarios.items():
        # Retrieve exclusion rules if reference is estimated model
        ref_row = model_row_lookup.get(appendix_reference_model_id)
        exclude_countries = [c.strip() for c in ref_row.exclude_countries.split(',') if c.strip()] if ref_row and hasattr(ref_row, 'exclude_countries') and isinstance(ref_row.exclude_countries, str) else None
        
        model_df = build_model_frame(df, dependent, scenario_regressors, exclude_countries=exclude_countries)
        formula = dependent + ' ~ ' + ' + '.join(scenario_regressors)
        panel_formula = dependent + ' ~ 1 + ' + ' + '.join(scenario_regressors)

        pooled = smf.ols(formula, data=model_df.reset_index()).fit(
            cov_type='cluster',
            cov_kwds={'groups': model_df.reset_index()['country']},
        )
        fe = PanelOLS.from_formula(
            panel_formula + ' + EntityEffects + TimeEffects',
            data=model_df,
        ).fit(cov_type='clustered', cluster_entity=True)
        fe_dk = PanelOLS.from_formula(
            panel_formula + ' + EntityEffects + TimeEffects',
            data=model_df,
        ).fit(cov_type='kernel', kernel='bartlett')
        re = None
        re_failure_reason = ''
        try:
            re = RandomEffects.from_formula(panel_formula, data=model_df).fit(
                cov_type='clustered',
                cluster_entity=True,
            )
        except ZeroDivisionError as exc:
            re_failure_reason = str(exc)
            
        hausman_result = {
            'statistic': float('nan'),
            'raw_statistic': float('nan'),
            'p_value': float('nan'),
            'negative_statistic_flag': False,
        }
        if re is not None:
            hausman_result = hausman_test(fe, re, scenario_regressors)
            
        preferred_estimator = choose_preferred_estimator(hausman_result['p_value'])
        if preferred_estimator == 'fixed_effects_driscoll_kraay':
            preferred_result = fe_dk
        elif preferred_estimator == 'fixed_effects':
            preferred_result = fe
        else:
            preferred_result = re if re is not None else fe_dk
            
        diagnostics_result = pooled_model_diagnostics(pooled, model_df, scenario_name, scenario_regressors)
        scenario_vif = (
            vif_table(model_df, scenario_name, scenario_regressors)
            .rename(columns={'specification': 'model_id'})
            .sort_values('vif', ascending=False)
            .reset_index(drop=True)
        )
        robustness_vif_tables[scenario_name] = scenario_vif
        
        # Save appx vif tables
        scenario_vif.to_csv(OUTPUTS_DIR / f'{scenario_name}_robustness_vif.csv', index=False)

        trade_present = 'trade_pct_gdp' in preferred_result.params.index
        std_error_source = preferred_result.std_errors if hasattr(preferred_result, 'std_errors') else preferred_result.bse
        robustness_summary_rows.append(
            {
                'scenario': scenario_name,
                'preferred_estimator': preferred_estimator,
                'hausman_p_value': float(hausman_result['p_value']),
                'hausman_raw_statistic': float(hausman_result['raw_statistic']),
                'negative_hausman_flag': bool(hausman_result['negative_statistic_flag']),
                'comparison_note': 'random_effects_failed' if re is None else 'ok',
                're_failure_reason': re_failure_reason,
                'observations': int(len(model_df)),
                'countries_used': int(model_df.reset_index()['country'].nunique()),
                'years_used': int(model_df.reset_index()['year'].nunique()),
                'trade_coef': float(preferred_result.params['trade_pct_gdp']) if trade_present else float('nan'),
                'trade_std_error': float(std_error_source['trade_pct_gdp']) if trade_present else float('nan'),
                'trade_p_value': float(preferred_result.pvalues['trade_pct_gdp']) if trade_present else float('nan'),
                'max_vif': float(scenario_vif['vif'].max()),
                'condition_number': float(diagnostics_result['condition_number']),
            }
        )

    trade_collinearity_robustness = pd.DataFrame(robustness_summary_rows)
    if appendix_reference_model_id in set(trade_collinearity_robustness['scenario']):
        original_trade_coef = trade_collinearity_robustness.loc[
            trade_collinearity_robustness['scenario'].eq(appendix_reference_model_id),
            'trade_coef',
        ].iloc[0]
        trade_collinearity_robustness['trade_coef_change_vs_reference'] = (
            trade_collinearity_robustness['trade_coef'] - original_trade_coef
        )
    else:
        trade_collinearity_robustness['trade_coef_change_vs_reference'] = float('nan')

    trade_collinearity_robustness.to_csv(OUTPUTS_DIR / 'trade_collinearity_robustness_summary.csv', index=False)

    # 13. WRITE TO BULK EXCEL SHEET 1: model_outputs.xlsx
    print("Writing bulk workbook outputs/model_outputs.xlsx...")
    with pd.ExcelWriter(OUTPUTS_DIR / 'model_outputs.xlsx', engine='openpyxl') as writer:
        coefficients_df.to_excel(writer, sheet_name='coefficients', index=False)
        fit_stats_df.to_excel(writer, sheet_name='fit_stats', index=False)
        sample_summary_df.to_excel(writer, sheet_name='sample_summary', index=False)
        hausman_df.to_excel(writer, sheet_name='hausman', index=False)
        vif_df.to_excel(writer, sheet_name='vif', index=False)
        diagnostics_df.to_excel(writer, sheet_name='diagnostics', index=False)
        sample_missingness_df.to_excel(writer, sheet_name='sample_missingness', index=False)
        model_status_df.to_excel(writer, sheet_name='model_status', index=False)
        model_sample_audit_df.to_excel(writer, sheet_name='sample_audit', index=False)
        model_sample_loss_drivers_df.to_excel(writer, sheet_name='sample_loss_drivers', index=False)
        model_panel_balance_summary.to_excel(writer, sheet_name='panel_balance_summary', index=False)
        model_panel_balance_by_country.to_excel(writer, sheet_name='panel_balance_country', index=False)
        common_sample_coefficients_df.to_excel(writer, sheet_name='common_coefficients', index=False)
        common_sample_fit_stats_df.to_excel(writer, sheet_name='common_fit_stats', index=False)
        common_sample_summary_df.to_excel(writer, sheet_name='common_sample_summary', index=False)
        common_sample_hausman_df.to_excel(writer, sheet_name='common_hausman', index=False)
        common_sample_vif_df.to_excel(writer, sheet_name='common_vif', index=False)
        common_sample_diagnostics_df.to_excel(writer, sheet_name='common_diagnostics', index=False)
        workbook_catalog_df.to_excel(writer, sheet_name='workbook_catalog', index=False)
        workbook_variable_selection.to_excel(writer, sheet_name='workbook_variable_selection', index=False)
        workbook_model_specs.to_excel(writer, sheet_name='workbook_model_specs', index=False)
        workbook_notes.to_excel(writer, sheet_name='workbook_notes', index=False)

    # 14. WRITE TO BULK EXCEL SHEET 2: results_tables.xlsx
    print("Writing bulk workbook outputs/results_tables.xlsx...")
    with pd.ExcelWriter(OUTPUTS_DIR / 'results_tables.xlsx', engine='openpyxl') as writer:
        descriptive_stats.to_excel(writer, sheet_name='descriptive_stats')
        correlation_matrix.to_excel(writer, sheet_name='correlation_matrix')
        correlation_pairwise_n.to_excel(writer, sheet_name='correlation_pairwise_n')
        regressor_correlation_matrix.to_excel(writer, sheet_name='appendix_regressor_corr')
        regressor_pairwise_n.to_excel(writer, sheet_name='appendix_regressor_corr_n')
        regression_table_main.to_excel(writer, sheet_name='regression_table_main')
        expected_sign_match_table.to_excel(writer, sheet_name='expected_sign_match', index=False)
        model_fit_gap_ranking.to_excel(writer, sheet_name='fit_gap_ranking', index=False)
        low_gap_selected_models.to_excel(writer, sheet_name='low_gap_selected_models', index=False)
        low_gap_coefficient_interpretations.to_excel(writer, sheet_name='low_gap_coef_interpret', index=False)
        broad_money_decomposition_results.to_excel(writer, sheet_name='bm_sign_decomp', index=False)
        broad_money_decomposition_pivot.to_excel(writer, sheet_name='bm_sign_decomp_pivot', index=False)
        broad_money_sign_path_summary.to_excel(writer, sheet_name='bm_sign_path_summary', index=False)
        model_sample_audit_df.to_excel(writer, sheet_name='model_sample_audit', index=False)
        model_sample_loss_drivers_df.to_excel(writer, sheet_name='sample_loss_drivers', index=False)
        model_panel_balance_summary.to_excel(writer, sheet_name='model_panel_balance', index=False)
        model_panel_balance_by_country.to_excel(writer, sheet_name='model_panel_country', index=False)
        if not spec_panel_balance_summary.empty:
            spec_panel_balance_summary.to_excel(writer, sheet_name='spec_panel_balance', index=False)
        if not spec_panel_balance_by_country.empty:
            spec_panel_balance_by_country.to_excel(writer, sheet_name='spec_panel_country', index=False)
        common_sample_overview.to_excel(writer, sheet_name='common_sample_overview', index=False)
        common_sample_coefficients_df.to_excel(writer, sheet_name='common_coefficients', index=False)
        common_sample_fit_stats_df.to_excel(writer, sheet_name='common_fit_stats', index=False)
        common_sample_hausman_df.to_excel(writer, sheet_name='common_hausman', index=False)
        common_sample_vif_df.to_excel(writer, sheet_name='common_vif', index=False)
        common_sample_diagnostics_df.to_excel(writer, sheet_name='common_diagnostics', index=False)
        trade_collinearity_robustness.to_excel(writer, sheet_name='trade_collinearity_appx', index=False)
        fit_stats_df.to_excel(writer, sheet_name='fit_stats', index=False)
        sample_summary_df.to_excel(writer, sheet_name='sample_summary', index=False)
        if not structural_proxy_coverage.empty:
            structural_proxy_coverage.to_excel(writer, sheet_name='structural_proxy_coverage', index=False)
        hausman_df.to_excel(writer, sheet_name='hausman', index=False)
        diagnostics_df.to_excel(writer, sheet_name='diagnostics', index=False)
        vif_df.to_excel(writer, sheet_name='vif', index=False)
        sample_missingness_df.to_excel(writer, sheet_name='sample_missingness', index=False)
        model_status_df.to_excel(writer, sheet_name='model_status', index=False)
        workbook_catalog_df.to_excel(writer, sheet_name='workbook_catalog', index=False)
        workbook_variable_selection.to_excel(writer, sheet_name='workbook_variable_selection', index=False)
        workbook_model_specs.to_excel(writer, sheet_name='workbook_model_specs', index=False)
        workbook_notes.to_excel(writer, sheet_name='workbook_notes', index=False)
        if not variable_audit.empty:
            variable_audit.to_excel(writer, sheet_name='variable_audit', index=False)
        if not spec_sample_summary.empty:
            spec_sample_summary.to_excel(writer, sheet_name='spec_sample_summary', index=False)
        if not review_flags.empty:
            review_flags.to_excel(writer, sheet_name='review_flags', index=False)
        skipped_models.to_excel(writer, sheet_name='skipped_models', index=False)
        
        for model_row in estimated_models.itertuples(index=False):
            model_id = model_row.model_id
            inference_tables[model_id].to_excel(
                writer,
                sheet_name=safe_sheet_name(model_id, prefix='coef_'),
                index=False,
            )
            diagnostic_tables[model_id]['summary'].to_excel(
                writer,
                sheet_name=safe_sheet_name(model_id, prefix='diag_'),
                index=False,
            )
            diagnostic_tables[model_id]['vif'].to_excel(
                writer,
                sheet_name=safe_sheet_name(model_id, prefix='vif_'),
                index=False,
            )
            diagnostic_tables[model_id]['missingness'].to_excel(
                writer,
                sheet_name=safe_sheet_name(model_id, prefix='miss_'),
                index=False,
            )
            sample_fit_tables[model_id].to_excel(
                writer,
                sheet_name=safe_sheet_name(model_id, prefix='sum_'),
                index=False,
            )
        for scenario_name, scenario_vif in robustness_vif_tables.items():
            scenario_vif.to_excel(
                writer,
                sheet_name=safe_sheet_name(scenario_name, prefix='appx_vif_'),
                index=False,
            )

    print("--- Econometric Estimation & Export Pipeline Completed Successfully ---")
