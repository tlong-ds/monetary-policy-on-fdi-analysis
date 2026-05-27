from __future__ import annotations

import numpy as np
import pandas as pd


def stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def fmt_num(value: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return "---"
    return f"{value:.{decimals}f}"


def fmt_pval(value: float) -> str:
    if pd.isna(value):
        return "---"
    if value < 0.001:
        return "$<$0.001"
    return f"{value:.4f}"


def fmt_coef_se_pval(coef: float, se: float, pval: float) -> str:
    return f"{fmt_num(coef)}{stars(pval)}\\\\({fmt_num(se)})"


def wrap_table(
    body: str,
    caption: str,
    label: str,
    col_format: str,
    notes: str = "",
    width: str = r"\textwidth",
    fontsize: str = r"\scriptsize",
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{threeparttable}",
        fontsize,
        rf"\resizebox{{{width}}}{{!}}{{%",
        r"\begin{tabular}{" + col_format + "}",
        r"\toprule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
    ]
    if notes:
        lines.extend([
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            rf"\item {notes}",
            r"\end{tablenotes}",
        ])
    lines.extend([
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def wrap_minipage_table(
    body: str,
    caption: str,
    label: str,
    col_format: str,
    notes: str = "",
    width: str = r"\textwidth",
    fontsize: str = r"\scriptsize",
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{minipage}}{{{width}}}",
        fontsize,
        rf"\resizebox{{{width}}}{{!}}{{%",
        r"\begin{tabular}{" + col_format + "}",
        r"\toprule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
    ]
    if notes:
        lines.append(r"\vskip 4pt")
        lines.append(r"\footnotesize")
        lines.append(rf"{notes}")
    lines.extend([
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def build_descriptive_stats(df: pd.DataFrame) -> str:
    """Build descriptive statistics table from outputs/descriptive_stats.csv."""
    rows = []
    for row in df.itertuples():
        var = row.Index if hasattr(row, 'Index') else row[0]
        count = int(row.count) if hasattr(row, 'count') else int(row[1])
        mean = float(row.mean) if hasattr(row, 'mean') else float(row[2])
        std = float(row.std) if hasattr(row, 'std') else float(row[3])
        min_val = float(row[4]) if not hasattr(row, 'min') else float(row.min)
        median = float(row[6]) if not hasattr(row, '50%') else float(getattr(row, '50%'))
        if hasattr(row, '50%'):
            pct50 = float(row['50%'])
        elif hasattr(row, '50%'):
            pct50 = float(row['50%'])
        else:
            pct50 = float(row[6])
        max_val = float(row[7]) if not hasattr(row, 'max') else float(row.max)
        missing_pct = float(row.missing_pct) if hasattr(row, 'missing_pct') else float(row[9])
        rows.append(
            f"{var} & {count} & {fmt_num(mean, 3)} & {fmt_num(std, 3)} "
            f"& {fmt_num(min_val, 3)} & {fmt_num(pct50, 3)} & {fmt_num(max_val, 3)} "
            f"& {fmt_num(missing_pct, 1)}\\\\"
        )
    body = "\n".join(rows)
    col_format = "lrrrrrrr"
    header = (
        r"Variable & Count & Mean & Std.\ dev. & Min. & Median & Max. & Missing (\%) \\"
        r"\midrule"
    )
    return wrap_table(
        header + "\n" + body,
        caption="Descriptive statistics for the processed panel",
        label="tab:descriptive-stats",
        col_format=col_format,
        notes=(
            "The regression dependent variable is FDI net inflows as a percentage of GDP. "
            "This table is descriptive rather than estimated; the regression tables below use "
            "two-way fixed effects with Driscoll--Kraay standard errors. Significance stars are "
            "not applicable. Counts differ because the analytical panel is unbalanced and variables "
            "have different observed coverage."
        ),
    )


def build_main_regression_table(
    coefficients_df: pd.DataFrame,
    fit_stats_df: pd.DataFrame,
    model_catalog: pd.DataFrame,
) -> str:
    """Build main regression results table (FE-DK only)."""
    fe_dk = coefficients_df[
        coefficients_df['estimator'].eq('fixed_effects_driscoll_kraay')
    ].copy()
    estimated_ids = fe_dk['model_id'].unique()

    model_order = [
        m for m in [
            'M1_baseline_liquidity',
            'M2_main_monetary_policy',
            'M3_lagged_main_model',
            'M4_real_interest_robustness',
            'M5_lending_rate_robustness',
        ] if m in estimated_ids
    ]

    labels = {
        'M1_baseline_liquidity': 'M1\nLiquidity',
        'M2_main_monetary_policy': 'M2\nDeposit rate',
        'M3_lagged_main_model': 'M3\nLagged deposit',
        'M4_real_interest_robustness': 'M4\nReal rate',
        'M5_lending_rate_robustness': 'M5\nLending rate',
    }

    terms = [
        'broad_money_growth_pct',
        'deposit_interest_rate_pct',
        'deposit_interest_rate_pct_lag1',
        'real_interest_rate_pct',
        'lending_interest_rate_pct',
        'trade_pct_gdp',
        'inflation_gdp_deflator_pct',
        'ln_gdppc',
        'xr_dep_pct',
    ]

    term_labels = {
        'broad_money_growth_pct': 'Broad money growth (annual \\%)',
        'deposit_interest_rate_pct': 'Deposit interest rate (\\%)',
        'deposit_interest_rate_pct_lag1': 'Deposit interest rate, lag 1 (\\%)',
        'real_interest_rate_pct': 'Real interest rate (\\%)',
        'lending_interest_rate_pct': 'Lending interest rate (\\%)',
        'trade_pct_gdp': 'Trade (\\% GDP)',
        'inflation_gdp_deflator_pct': 'Inflation, GDP deflator (\\%)',
        'ln_gdppc': 'Log GDP per capita',
        'xr_dep_pct': 'Exchange-rate depreciation (\\%)',
    }

    headers = " & " + " & ".join(
        labels.get(mid, mid).replace('\n', ' ')
        for mid in model_order
    ) + r" \\"
    subheaders = " & " + " & ".join(
        labels.get(mid, mid).split('\n')[1] if '\n' in labels.get(mid, mid) else ''
        for mid in model_order
    ) + r" \\"

    body_rows = [headers, subheaders, r"\midrule"]
    for term in terms:
        row_parts = [term_labels.get(term, term)]
        has_any = False
        for mid in model_order:
            row = fe_dk[(fe_dk['model_id'] == mid) & (fe_dk['term'] == term)]
            if row.empty:
                row_parts.append("")
            else:
                has_any = True
                r = row.iloc[0]
                row_parts.append(fmt_coef_se_pval(r['coef'], r['std_error'], r['p_value']))
        if has_any:
            body_rows.append(" & ".join(row_parts) + r" \\")

    body_rows.append(r"\midrule")

    summary_metrics = ['nobs', 'countries_used', 'within_r2', 'slope_adjusted_within_r2']
    summary_labels = {
        'nobs': 'Observations',
        'countries_used': 'Countries used',
        'within_r2': 'Within $R^2$',
        'slope_adjusted_within_r2': 'Slope-adjusted within $R^2$',
    }

    for metric in summary_metrics:
        row_parts = [summary_labels[metric]]
        for mid in model_order:
            fr = fit_stats_df[
                (fit_stats_df['model_id'] == mid)
                & (fit_stats_df['estimator'] == 'fixed_effects_driscoll_kraay')
            ]
            if not fr.empty:
                val = fr.iloc[0][metric]
                if metric in ('nobs', 'countries_used'):
                    row_parts.append(str(int(val)))
                else:
                    row_parts.append(fmt_num(val, 4))
            else:
                row_parts.append("")
        body_rows.append(" & ".join(row_parts) + r" \\")

    body = "\n".join(body_rows)
    ncols = len(model_order) + 1
    return wrap_table(
        body,
        caption="Main fixed-effects Driscoll--Kraay regression results",
        label="tab:main-regression-results",
        col_format="l" + "c" * (ncols - 1),
        notes=(
            "The dependent variable is FDI net inflows as a percentage of GDP. "
            "All columns use two-way fixed effects with country and year effects and "
            "Driscoll--Kraay standard errors using Bartlett kernel covariance. "
            "Standard errors are in parentheses. *, **, and *** denote statistical "
            "significance at the 10, 5, and 1 percent levels. The panel is unbalanced, "
            "and sample sizes vary across columns because monetary-proxy and control "
            "coverage differs by model. In column M3, all regressors (including controls) "
            "are lagged by one period."
        ),
    )


def build_diagnostics_table(diagnostics_df: pd.DataFrame, vif_df: pd.DataFrame) -> str:
    """Build compact diagnostics table."""
    model_ids = [
        'M1_baseline_liquidity',
        'M2_main_monetary_policy',
        'M3_lagged_main_model',
        'M4_real_interest_robustness',
        'M5_lending_rate_robustness',
    ]

    existing = [m for m in model_ids if m in diagnostics_df['model_id'].values]

    body_rows = []
    header = r"Model & Within $R^2$ & Adjusted $R^2$ & Max VIF & White $p$ & Pesaran CD $p$ \\"
    body_rows.append(header)
    body_rows.append(r"\midrule")

    for mid in existing:
        dr = diagnostics_df[diagnostics_df['model_id'] == mid]
        if dr.empty:
            continue
        d = dr.iloc[0]
        fit_row = ... 
        model_code = mid.split('_')[0]
        within_r2 = fmt_num(d.get('within_r2', d.get('r_squared', np.nan)), 3)
        adj_r2 = fmt_num(d.get('adj_r_squared', np.nan), 3)
        white_p = fmt_pval(d['white_lm_p_value'])
        cd_p = fmt_pval(d['pesaran_cd_p_value'])

        vr = vif_df[vif_df['model_id'] == mid]
        max_vif = fmt_num(vr['vif'].max(), 2) if not vr.empty else "---"

        body_rows.append(
            f"{model_code} & {within_r2} & {adj_r2} & {max_vif} & {white_p} & {cd_p} \\\\"
        )

    body = "\n".join(body_rows)
    return wrap_table(
        body,
        caption="Compact diagnostics for the reported models",
        label="tab:compact-diagnostics",
        col_format="lrrrrr",
        width=r"0.8\textwidth",
        notes=(
            "The dependent variable in the corresponding regressions is FDI net inflows "
            "as a percentage of GDP. Reported fit statistics correspond to two-way "
            "fixed-effects specifications with Driscoll--Kraay standard errors in the "
            "regression tables. The Adjusted R-squared reported here represents the "
            "overall adjusted coefficient of determination from OLS on de-meaned data, "
            "which penalizes for entity and time fixed effect degrees of freedom (yielding "
            "negative values due to the small sample size), whereas "
            "Table~\\ref{tab:main-regression-results} reports the slope-adjusted within "
            "R-squared described in Subsection~\\ref{subsec:goodness-of-fit}. "
            "Significance stars are not applicable in this diagnostic table. Diagnostic "
            "samples vary because the panel is unbalanced and each model uses a different "
            "complete-case support."
        ),
    )


def build_correlation_matrix(
    corr_df: pd.DataFrame,
    pairwise_n_df: pd.DataFrame,
) -> tuple[str, str]:
    """Build correlation matrix and pairwise N tables."""
    variables = corr_df.columns.tolist()
    n = len(variables)

    def short_label(v: str) -> str:
        mapping = {
            'fdi_pct_gdp': '(1)',
            'broad_money_growth_pct': '(2)',
            'inflation_gdp_deflator_pct': '(3)',
            'trade_pct_gdp': '(4)',
            'ln_gdppc': '(5)',
            'xr_dep_pct': '(6)',
            'deposit_interest_rate_pct': '(7)',
            'real_interest_rate_pct': '(8)',
            'lending_interest_rate_pct': '(9)',
            'ln_tourism_arrivals': '(10)',
            'hc_human_capital_index': '(11)',
        }
        return mapping.get(v, v)

    var_labels = {v: short_label(v) for v in variables}

    col_format = "l" + "c" * n

    header_parts = ["Variable"] + [var_labels[v] for v in variables]
    header = " & ".join(header_parts) + r" \\"

    body_rows = []
    for i, var_a in enumerate(variables):
        row_parts = [var_labels[var_a]]
        for j, var_b in enumerate(variables):
            if j > i:
                row_parts.append("")
            else:
                val = corr_df.loc[var_a, var_b]
                row_parts.append(fmt_num(val, 3) if not pd.isna(val) else "")
        body_rows.append(" & ".join(row_parts) + r" \\")

    corr_body = header + "\n" + r"\midrule" + "\n" + "\n".join(body_rows)

    corr_notes = (
        "This table displays the Pearson correlation coefficients between all main "
        "analysis variables. The sample is unbalanced and correlation coefficients are "
        "computed based on pairwise-complete observations. Column numbers (1) through "
        f"({n}) correspond to the variables defined in the rows."
    )

    corr_table = wrap_minipage_table(
        corr_body,
        caption="Pearson correlation matrix of main analysis variables",
        label="tab:correlation-matrix",
        col_format=col_format,
        notes=corr_notes,
    )

    n_body_rows = []
    for i, var_a in enumerate(variables):
        row_parts = [var_labels[var_a]]
        for j, var_b in enumerate(variables):
            if j > i:
                row_parts.append("")
            else:
                val = pairwise_n_df.loc[var_a, var_b]
                row_parts.append(str(int(val)) if not pd.isna(val) else "")
        n_body_rows.append(" & ".join(row_parts) + r" \\")

    n_body = header + "\n" + r"\midrule" + "\n" + "\n".join(n_body_rows)

    n_notes = (
        "This table reports the pairwise count of non-missing country-year observations "
        "available for each pair of variables in the analytical dataset. The full panel "
        f"contains a maximum of {int(pairwise_n_df.iloc[0, 0])} country-year observations. "
        f"Column numbers (1) through ({n}) correspond to the variables defined in the rows."
    )

    n_table = wrap_minipage_table(
        n_body,
        caption="Pairwise sample size (N) matrix of variables",
        label="tab:correlation-pairwise-n",
        col_format=col_format,
        notes=n_notes,
    )

    return corr_table, n_table


def build_coverage_by_country(df: pd.DataFrame) -> str:
    """Build country-level data completeness table."""
    var_order = [
        'fdi_pct_gdp', 'broad_money_growth_pct',
        'deposit_interest_rate_pct', 'real_interest_rate_pct',
        'lending_interest_rate_pct', 'trade_pct_gdp',
        'inflation_gdp_deflator_pct', 'ln_gdppc',
        'xr_dep_pct', 'ln_tourism_arrivals', 'ln_population_total',
        'hc_human_capital_index',
    ]
    available = [v for v in var_order if v in df.columns]

    var_labels = {
        'fdi_pct_gdp': '(1)', 'broad_money_growth_pct': '(2)',
        'deposit_interest_rate_pct': '(3)', 'real_interest_rate_pct': '(4)',
        'lending_interest_rate_pct': '(5)', 'trade_pct_gdp': '(6)',
        'inflation_gdp_deflator_pct': '(7)', 'ln_gdppc': '(8)',
        'xr_dep_pct': '(9)', 'ln_tourism_arrivals': '(10)',
        'ln_population_total': '(11)', 'hc_human_capital_index': '(12)',
    }

    ncols = len(available) + 1
    col_format = "l" + "c" * (ncols - 1)

    header_parts = ["Country"] + [var_labels[v] for v in available]
    header = " & ".join(header_parts) + r" \\"

    body_rows = []
    for row in df.itertuples(index=False):
        country = row.country
        row_parts = [country]
        for v in available:
            val = getattr(row, v, np.nan)
            row_parts.append(str(int(val)) if not pd.isna(val) else "0")
        body_rows.append(" & ".join(row_parts) + r" \\")

    body = header + "\n" + r"\midrule" + "\n" + "\n".join(body_rows)

    var_index_entries = [
        f"({i+1}) {lbl}" for i, (v, lbl) in enumerate(var_labels.items()) if v in available
    ]
    notes = (
        "The numbers denote the count of observed years (out of a maximum of "
        + str(int(df[available[0]].max()))
        + " years, 2000--2023) for each country-variable combination. "
        "Variable indices are: " + "; ".join(var_index_entries) + "."
    )

    return wrap_minipage_table(
        body,
        caption="Country-level data completeness and coverage counts",
        label="tab:coverage-by-country",
        col_format=col_format,
        notes=notes,
    )


def build_model_roles_samples(
    panel_balance_df: pd.DataFrame,
    model_catalog: pd.DataFrame,
) -> str:
    """Build model roles and realized sample sizes table."""
    model_id_map = dict(zip(model_catalog['model_id'], model_catalog['workbook_model']))
    purpose_map = dict(zip(model_catalog['model_id'], model_catalog['purpose']))

    model_order = [
        'M1_baseline_liquidity',
        'M2_main_monetary_policy',
        'M3_lagged_main_model',
        'M4_real_interest_robustness',
        'M5_lending_rate_robustness',
        'M6_tourism_robustnessa_from_M2_main_monetary_policy',
        'M6_tourism_robustnessb_from_M4_real_interest_robustness',
        'M7_human_capital_robustnessa_from_M2_main_monetary_policy',
        'M7_human_capital_robustnessb_from_M4_real_interest_robustness',
    ]

    existing = [
        m for m in model_order
        if m in panel_balance_df['model_id'].values
        or m in model_catalog['model_id'].values
    ]

    col_format = r"p{0.07\textwidth}p{0.27\textwidth}rrp{0.38\textwidth}"

    header = (
        r"Model & Thesis role & Observations & Countries "
        r"& Structural coverage note \\"
        r"\midrule"
    )

    body_rows = []
    for mid in existing:
        model_label = model_id_map.get(mid, mid).split(' - ')[0] if ' - ' in model_id_map.get(mid, mid) else mid
        role = purpose_map.get(mid, "")
        pb = panel_balance_df[panel_balance_df['model_id'] == mid]
        if not pb.empty:
            p = pb.iloc[0]
            obs = int(p['total_rows'])
            countries = int(p['countries_used'])
        else:
            obs = "?"
            countries = "?"
        body_rows.append(
            f"{model_label} & {role} & {obs} & {countries} & ---\\\\"
        )

    body = header + "\n" + "\n".join(body_rows)
    return wrap_minipage_table(
        body,
        caption="Model roles and realized FE-DK sample sizes",
        label="tab:model-roles-samples",
        col_format=col_format,
        notes=(
            "The regression dependent variable is FDI net inflows as a percentage of GDP. "
            "FE-DK denotes two-way fixed effects with Driscoll--Kraay standard errors using "
            "Bartlett kernel covariance. Significance stars are not applicable in this "
            "sample-summary table. Sample sizes vary because the panel is unbalanced and each "
            "model requires complete observations for its selected monetary proxy and controls."
        ),
    )


def build_broad_money_decomposition(df: pd.DataFrame) -> str:
    """Build stepwise broad money sign-flip decomposition table."""
    fixed_common = df[
        (df['sample_type'] == 'fixed_m2_common_sample')
    ].copy()

    col_format = r"clccc"
    header = (
        r"Step & Specification & Pooled OLS & Random Effects "
        r"& Fixed Effects (FE/DK-FE) \\"
        r"\midrule"
    )

    body_rows = []
    for step in sorted(fixed_common['step'].unique()):
        step_data = fixed_common[fixed_common['step'] == step]
        spec_label = step_data['spec_label'].iloc[0]
        nobs = int(step_data['nobs'].iloc[0])

        def get_coef_pval(est: str) -> str:
            r = step_data[step_data['estimator'] == est]
            if r.empty or r.iloc[0]['fit_status'] != 'estimated':
                return "---"
            row = r.iloc[0]
            coef = row['coef']
            pval = row['p_value']
            if pd.isna(coef):
                return "---"
            return f"{fmt_num(coef, 4)}\n({fmt_pval(pval)})"

        po = get_coef_pval('pooled_ols')
        re = get_coef_pval('random_effects')
        fe = get_coef_pval('fixed_effects_driscoll_kraay')

        body_rows.append(
            f"{step} & {spec_label} & {po} & {re} & {fe} \\\\"
        )

    body = header + "\n" + "\n".join(body_rows)
    return wrap_table(
        body,
        caption=(
            "Stepwise Broad Money sign-flip decomposition "
            "(on common sample, $N=" + str(int(fixed_common['nobs'].iloc[0])) + "$)"
        ),
        label="tab:broad-money-sign-decomposition",
        col_format=col_format,
        width=r"0.92\textwidth",
        notes=(
            "The table reports the estimated coefficients on broad money growth (annual \\%) "
            "at each step of model expansion. The estimation sample is fixed to the M2 "
            "deposit-rate support to eliminate sample composition effects. "
            "P-values are reported in parentheses. *, **, and *** denote statistical "
            "significance at the 10, 5, and 1 percent levels, respectively."
        ),
    )
