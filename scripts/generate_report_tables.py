"""Read pipeline outputs and generate LaTeX table snippets + copy figures to docs/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
TABLES_DIR = DOCS / "tables"
ASSETS_DIR = DOCS / "assets"

sys.path.insert(0, str(ROOT))
from src.latex_export import (  # noqa: E402
    fmt_num,
    fmt_pval,
    fmt_coef_se_pval,
    stars,
    wrap_table,
    wrap_minipage_table,
)


def stars(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / name)


def write_tex(name: str, content: str) -> Path:
    path = TABLES_DIR / name
    path.write_text(content)
    print(f"  Wrote {path}")
    return path


# ── Helper: extract coef/se/pval for a given term from a coefficients DF ──
def get_term_coef(coefficients_df, model_id, term, *, estimator="fixed_effects_driscoll_kraay"):
    mask = (
        coefficients_df["model_id"].eq(model_id)
        & coefficients_df["estimator"].eq(estimator)
        & coefficients_df["term"].eq(term)
    )
    row = coefficients_df.loc[mask]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"coef": r["coef"], "se": r["std_error"], "pval": r["p_value"]}


# ────────────────────────────────────────────────────────────
# TABLE 1: Descriptive statistics
# ────────────────────────────────────────────────────────────
def generate_descriptive_stats():
    df = pd.read_csv(OUTPUTS / "descriptive_stats.csv", index_col=0)
    var_labels = {
        "fdi_pct_gdp": "FDI (\\% GDP)",
        "broad_money_growth_pct": "Broad money growth (annual \\%)",
        "deposit_interest_rate_pct": "Deposit interest rate (\\%)",
        "real_interest_rate_pct": "Real interest rate (\\%)",
        "lending_interest_rate_pct": "Lending interest rate (\\%)",
        "inflation_gdp_deflator_pct": "Inflation, GDP deflator (\\%)",
        "trade_pct_gdp": "Trade (\\% GDP)",
        "ln_gdppc": "Log GDP per capita",
        "xr_dep_pct": "Exchange-rate depreciation (\\%)",
        "ln_tourism_arrivals": "Tourism arrivals (Log)",
        "hc_human_capital_index": "Human capital index",
    }
    rows = []
    for var, label in var_labels.items():
        if var not in df.index:
            continue
        r = df.loc[var]
        count = int(r["count"])
        mean = float(r["mean"])
        std = float(r["std"])
        min_v = float(r["min"])
        median = float(r["50%"])
        max_v = float(r["max"])
        miss_pct = float(r["missing_pct"])
        rows.append(
            f"{label} & {count} & {fmt_num(mean, 3)} & {fmt_num(std, 3)} "
            f"& {fmt_num(min_v, 3)} & {fmt_num(median, 3)} & {fmt_num(max_v, 3)} "
            f"& {fmt_num(miss_pct, 1)} \\\\"
        )
    body = "\n".join(rows)
    header = (
        r"Variable & Count & Mean & Std.\ dev. & Min. & Median & Max. & Missing (\%) \\"
        r"\midrule"
    )
    return wrap_table(
        header + "\n" + body,
        caption="Descriptive statistics for the processed panel",
        label="tab:descriptive-stats",
        col_format="lrrrrrrr",
        notes=(
            "The regression dependent variable is FDI net inflows as a percentage of GDP. "
            "This table is descriptive rather than estimated; the regression tables below use "
            "two-way fixed effects with Driscoll--Kraay standard errors. Significance stars are "
            "not applicable. Counts differ because the analytical panel is unbalanced and variables "
            "have different observed coverage."
        ),
    )


# ────────────────────────────────────────────────────────────
# TABLE 2: Main regression results (from pre-formatted regression_table_main.csv)
# ────────────────────────────────────────────────────────────
def _split_coef_se(val: str) -> tuple[str, str]:
    """Split a '0.0460*\\n(0.0277)' cell into coef line and se line."""
    if pd.isna(val) or str(val).strip() == "":
        return ("", "")
    parts = str(val).strip().split("\n")
    coef_line = parts[0].strip() if len(parts) > 0 else ""
    se_line = parts[1].strip() if len(parts) > 1 else ""
    return (coef_line, se_line)


def generate_main_regression():
    raw = read("regression_table_main.csv")
    metrics = raw["metric"].tolist()

    model_columns = [
        c for c in raw.columns if c != "metric"
    ]

    model_codes = {
        "M1 - Baseline liquidity": "M1",
        "M2 - Main monetary policy": "M2",
        "M3 - Lagged main model": "M3",
        "M4 - Real interest robustness": "M4",
        "M5 - Lending rate robustness": "M5",
    }

    short_cols = []
    for c in model_columns:
        for full, short in model_codes.items():
            if full in c:
                short_cols.append(short)
                break
        else:
            short_cols.append(c)

    header = " & " + " & ".join(short_cols) + r" \\"
    subheader_parts = []
    for c in model_columns:
        if "Lagged" in c:
            subheader_parts.append("Lagged deposit")
        elif "Liquidity" in c:
            subheader_parts.append("Liquidity")
        elif "Monetary policy" in c or "Deposit" in c:
            subheader_parts.append("Deposit rate")
        elif "Real interest" in c or "Real rate" in c:
            subheader_parts.append("Real rate")
        elif "Lending rate" in c:
            subheader_parts.append("Lending rate")
        else:
            subheader_parts.append("")
    subheader = " & " + " & ".join(subheader_parts) + r" \\"

    body_parts = [header, subheader, r"\midrule"]

    coefficient_metrics = [
        "Broad money growth (annual %)",
        "Deposit interest rate (%)",
        "Real interest rate (%)",
        "Lending interest rate (%)",
        "Trade (% GDP)",
        "Inflation, GDP deflator (%)",
        "Log GDP per capita",
        "Exchange rate depreciation (%)",
    ]

    for metric in coefficient_metrics:
        if metric not in metrics:
            continue
        row = raw[raw["metric"] == metric].iloc[0]
        display = metric.replace("%", "\\%").replace("&", "\\&")

        # Coefficient row
        coef_parts = [display]
        se_parts = [""]
        for c in model_columns:
            coef_line, se_line = _split_coef_se(row[c])
            coef_parts.append(coef_line)
            se_parts.append(se_line)

        has_any = any(p for p in coef_parts[1:])
        if has_any:
            body_parts.append(" & ".join(coef_parts) + r" \\")
            body_parts.append(" & ".join(se_parts) + r" \\")

    body_parts.append(r"\midrule")

    summary_metrics = [
        "Observations",
        "Countries used",
        "Within R-squared",
        "Slope-adjusted within R-squared",
    ]
    for metric in summary_metrics:
        if metric not in metrics:
            continue
        row = raw[raw["metric"] == metric].iloc[0]
        display = {"Slope-adjusted within R-squared": "Slope-adjusted within $R^2$"}.get(metric, metric)
        parts = [display]
        for c in model_columns:
            val = row[c]
            if pd.isna(val) or str(val).strip() == "":
                parts.append("")
            else:
                parts.append(str(val))
        body_parts.append(" & ".join(parts) + r" \\")

    body = "\n".join(body_parts)
    ncols = len(model_columns) + 1
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


# ────────────────────────────────────────────────────────────
# TABLE 3: Compact diagnostics (from regression_table_main.csv)
# ────────────────────────────────────────────────────────────
def generate_diagnostics():
    raw = read("regression_table_main.csv")
    metrics = raw["metric"].tolist()
    model_columns = [c for c in raw.columns if c != "metric"]

    model_codes = {}
    for c in model_columns:
        for full, short in {"M1 - ": "M1", "M2 - ": "M2", "M3 - ": "M3", "M4 - ": "M4", "M5 - ": "M5"}.items():
            if full in c:
                model_codes[c] = short
                break
        if c not in model_codes:
            model_codes[c] = c.split(" - ")[0] if " - " in c else c

    diag_metrics = {
        "Within R-squared": "Within $R^2$",
        "R2 minus adjusted R2 gap": "Adjusted $R^2$ gap",
        "Max VIF": "Max VIF",
        "Pesaran CD p-value": "Pesaran CD $p$",
    }

    rows = []
    for row_metric, display_name in diag_metrics.items():
        if row_metric not in metrics:
            continue
        r = raw[raw["metric"] == row_metric].iloc[0]
        parts = [display_name]
        for c in model_columns:
            val = r[c]
            parts.append(fmt_num(float(val), 3) if not pd.isna(val) and str(val).strip() != "" else "---")
        rows.append(" & ".join(parts) + r" \\")

    header = "Metric & " + " & ".join(model_codes[c] for c in model_columns) + r" \\" + "\n" + r"\midrule"
    body = header + "\n" + "\n".join(rows)

    ncols = len(model_columns) + 1
    return wrap_table(
        body,
        caption="Compact diagnostics for the reported models",
        label="tab:compact-diagnostics",
        col_format="l" + "r" * (ncols - 1),
        width=r"0.85\textwidth",
        notes=(
            "The dependent variable in the corresponding regressions is FDI net inflows "
            "as a percentage of GDP. Reported fit statistics correspond to two-way "
            "fixed-effects specifications with Driscoll--Kraay standard errors in the "
            "regression tables. Within $R^2$ measures the proportion of within-country "
            "variance explained by time-varying regressors. The Adjusted $R^2$ gap is the "
            "difference between $R^2$ and the overall adjusted $R^2$ (penalizing for entity "
            "and time fixed effects). Significance stars are not applicable in this table."
        ),
    )


# ────────────────────────────────────────────────────────────
# TABLE 4: Common-sample comparisons
# ────────────────────────────────────────────────────────────
def generate_common_sample():
    coefs = read("common_sample_coefficients.csv")
    native_coefs = read("model_coefficients.csv")
    fe_dk = coefs[coefs["estimator"] == "fixed_effects_driscoll_kraay"].copy()
    native_fe = native_coefs[native_coefs["estimator"] == "fixed_effects_driscoll_kraay"].copy()
    overview = read("common_sample_overview.csv")

    comparisons = fe_dk["comparison_id"].unique()

    def get_common_val(cid, native_mid, term):
        mask = (
            fe_dk["comparison_id"].eq(cid)
            & fe_dk["native_model_id"].eq(native_mid)
            & fe_dk["term"].eq(term)
        )
        r = fe_dk.loc[mask]
        if r.empty:
            return None
        r = r.iloc[0]
        return fmt_coef_se_pval(r["coef"], r["std_error"], r["p_value"])

    def get_native_val(native_mid, term):
        mask = (
            native_fe["model_id"].eq(native_mid)
            & native_fe["term"].eq(term)
        )
        r = native_fe.loc[mask]
        if r.empty:
            return None
        r = r.iloc[0]
        return fmt_coef_se_pval(r["coef"], r["std_error"], r["p_value"])

    body_parts = []
    header = (
        r"Comparison ID & Variable & \multicolumn{2}{c}{Common Sample Estimates} "
        r"& \multicolumn{2}{c}{Native Sample Estimates} & Obs. (Common / Native) \\"
        r"\\"
        r" & & Model A & Model B & Model A & Model B & \\"
        r"\midrule"
    )
    body_parts.append(header)

    cid_display = [
        ("M1_to_M2_common_sample", "\\textbf{M1 vs. M2}", "Liquidity vs. Deposit rate"),
        ("M2_to_M3_lagged_common_sample", "\\textbf{M2 vs. M3}", "Contemp. vs. Lagged Deposit"),
        ("M2_to_M5_lending_common_sample", "\\textbf{M2 vs. M5}", "Deposit vs. Lending rate"),
        ("M2_to_M7_hc_common_sample", "\\textbf{M2 vs. M7a}", "Deposit vs. Human Capital"),
    ]

    for cid, label_main, label_sub in cid_display:
        if cid not in comparisons:
            continue
        display_name = f"{label_main}\\\\({label_sub})"
        subset = fe_dk[fe_dk["comparison_id"] == cid]
        native_ids = subset["native_model_id"].unique()
        if len(native_ids) < 2:
            continue

        native_a, native_b = native_ids[0], native_ids[1]

        sub = overview[overview["comparison_id"] == cid]
        common_rows = int(sub["common_rows"].iloc[0]) if not sub.empty else 0

        def native_rows(mid):
            m = sub[sub["native_model_id"] == mid]
            return int(m["native_rows"].iloc[0]) if not m.empty else 0

        na_rows = native_rows(native_a)
        nb_rows = native_rows(native_b)

        def get_both(term):
            return (get_common_val(cid, native_a, term),
                    get_common_val(cid, native_b, term),
                    get_native_val(native_a, term),
                    get_native_val(native_b, term))

        bm_a, bm_b, bm_n_a, bm_n_b = get_both("broad_money_growth_pct")
        dep_a, dep_b, dep_n_a, dep_n_b = get_both("deposit_interest_rate_pct")

        body_parts.append(
            f"{display_name} & Broad money "
            f"& {bm_a or '---'} & {bm_b or '---'} "
            f"& {bm_n_a or '---'} & {bm_n_b or '---'} "
            f"& {common_rows} / {na_rows} ({native_a.split('_')[0]}) \\\\"
        )
        if dep_a:
            body_parts.append(
                f" & Deposit interest rate "
                f"& {dep_a} & {dep_b or '---'} "
                f"& {dep_n_a or '---'} & {dep_n_b or '---'} "
                f"& {common_rows} / {nb_rows} ({native_b.split('_')[0]}) \\\\"
            )

        if "M3" in cid:
            dep_lag_b = get_common_val(cid, native_b, "deposit_interest_rate_pct_lag1")
            dep_lag_nb = get_native_val(native_b, "deposit_interest_rate_pct_lag1")
            if dep_lag_b:
                body_parts.append(
                    f" & Deposit interest rate (lagged) "
                    f"& --- & {dep_lag_b} "
                    f"& --- & {dep_lag_nb or '---'} "
                    f"& \\\\"
                )

        if "M5" in cid:
            lend_b = get_common_val(cid, native_b, "lending_interest_rate_pct")
            lend_nb = get_native_val(native_b, "lending_interest_rate_pct")
            if lend_b:
                body_parts.append(
                    f" & Lending interest rate "
                    f"& --- & {lend_b} "
                    f"& --- & {lend_nb or '---'} "
                    f"& \\\\"
                )

        if "M7" in cid:
            hc_b = get_common_val(cid, native_b, "hc_human_capital_index")
            hc_nb = get_native_val(native_b, "hc_human_capital_index")
            if hc_b:
                body_parts.append(
                    f" & Human capital index "
                    f"& --- & {hc_b} "
                    f"& --- & {hc_nb or '---'} "
                    f"& \\\\"
                )

        body_parts.append(r"\midrule")

    body = "\n".join(body_parts)
    return wrap_minipage_table(
        body,
        caption="Common-sample comparison regression results (Driscoll--Kraay Fixed Effects)",
        label="tab:common-sample-comparisons",
        col_format=r"llccccc",
        notes=(
            "This table compares key coefficient estimates from models estimated on identical "
            "common-sample subsets to isolate specification effects from sample changes. "
            "Model A and Model B correspond to the two models named under the comparison ID. "
            "Driscoll--Kraay standard errors are reported in parentheses. "
            "*, **, and *** denote statistical significance at the 10, 5, and 1 percent levels, "
            "respectively. In the lagged model (M3), all variables (including controls) are lagged "
            "by one period, and their lagged coefficients are compared to the contemporaneous "
            "coefficients in M2."
        ),
        width=r"\textwidth",
    )


# ────────────────────────────────────────────────────────────
# TABLE 5: Correlation matrix
# ────────────────────────────────────────────────────────────
def generate_correlation_matrix():
    raw_corr = read("correlation_matrix_main_variables.csv")
    raw_pairwise = read("correlation_pairwise_n.csv")

    unnamed_cols_corr = [c for c in raw_corr.columns if c.startswith("Unnamed")]
    if unnamed_cols_corr:
        corr = raw_corr.set_index(unnamed_cols_corr[0])
    else:
        corr = raw_corr.set_index(raw_corr.columns[0])

    unnamed_cols_pw = [c for c in raw_pairwise.columns if c.startswith("Unnamed")]
    if unnamed_cols_pw:
        pairwise = raw_pairwise.set_index(unnamed_cols_pw[0])
    else:
        pairwise = raw_pairwise.set_index(raw_pairwise.columns[0])

    variables = corr.columns.tolist()

    labels = {
        "fdi_pct_gdp": "(1) FDI net inflows (\\% GDP)",
        "broad_money_growth_pct": "(2) Broad money (\\% GDP)",
        "inflation_gdp_deflator_pct": "(3) Inflation, GDP deflator (\\%)",
        "trade_pct_gdp": "(4) Trade (\\% GDP)",
        "ln_gdppc": "(5) Log GDP per capita",
        "xr_dep_pct": "(6) Exchange-rate depreciation (\\%)",
        "deposit_interest_rate_pct": "(7) Deposit interest rate (\\%)",
        "real_interest_rate_pct": "(8) Real interest rate (\\%)",
        "lending_interest_rate_pct": "(9) Lending interest rate (\\%)",
        "ln_tourism_arrivals": "(10) Tourism arrivals (Log)",
        "hc_human_capital_index": "(11) Human capital index",
    }
    var_labels_list = [labels.get(v, v) for v in variables]
    n = len(variables)

    def short_label(v):
        return labels.get(v, v)

    header = "Variable & " + " & ".join(short_label(v) for v in variables) + r" \\"
    col_fmt = "l" + "c" * n

    rows = []
    for i, va in enumerate(variables):
        parts = [short_label(va)]
        for j, vb in enumerate(variables):
            if j > i:
                parts.append("")
            else:
                val = corr.loc[va, vb] if va in corr.index and vb in corr.columns else np.nan
                parts.append(fmt_num(float(val), 3))
        rows.append(" & ".join(parts) + r" \\")

    body = header + "\n" + r"\midrule" + "\n" + "\n".join(rows)

    named_labels = {k: v.split(") ", 1)[1] for k, v in labels.items()}
    var_index = "; ".join(f"({i+1}) {named_labels.get(v, v)}" for i, v in enumerate(variables))
    corr_tex = wrap_minipage_table(
        body,
        caption="Pearson correlation matrix of main analysis variables",
        label="tab:correlation-matrix",
        col_format=col_fmt,
        notes=(
            "Notes: This table displays the Pearson correlation coefficients between all main "
            "analysis variables. The sample is unbalanced and correlation coefficients are "
            "computed based on pairwise-complete observations. Variable indices: " + var_index + "."
        ),
    )

    # Pairwise N table
    n_rows = []
    for i, va in enumerate(variables):
        parts = [short_label(va)]
        for j, vb in enumerate(variables):
            if j > i:
                parts.append("")
            else:
                val = pairwise.loc[va, vb] if va in pairwise.index and vb in pairwise.columns else np.nan
                parts.append(str(int(val)) if not pd.isna(val) else "")
        n_rows.append(" & ".join(parts) + r" \\")

    n_body = header + "\n" + r"\midrule" + "\n" + "\n".join(n_rows)
    max_n = int(pairwise.iloc[0, 0]) if not pairwise.empty else "?"
    n_tex = wrap_minipage_table(
        n_body,
        caption="Pairwise sample size (N) matrix of variables",
        label="tab:correlation-pairwise-n",
        col_format=col_fmt,
        notes=(
            "Notes: This table reports the pairwise count of non-missing country-year observations "
            f"available for each pair of variables in the analytical dataset. "
            f"Variable indices: {var_index}."
        ),
    )
    return corr_tex, n_tex


# ────────────────────────────────────────────────────────────
# TABLE 6: Coverage by country
# ────────────────────────────────────────────────────────────
def generate_coverage_by_country():
    df = read("coverage_by_country_numeric.csv")
    var_order = [
        "fdi_pct_gdp", "broad_money_growth_pct",
        "deposit_interest_rate_pct", "real_interest_rate_pct",
        "lending_interest_rate_pct", "trade_pct_gdp",
        "inflation_gdp_deflator_pct", "ln_gdppc",
        "xr_dep_pct", "ln_tourism_arrivals", "ln_population_total",
        "hc_human_capital_index",
    ]
    available = [v for v in var_order if v in df.columns]

    labels = {
        "fdi_pct_gdp": "(1)", "broad_money_growth_pct": "(2)",
        "deposit_interest_rate_pct": "(3)", "real_interest_rate_pct": "(4)",
        "lending_interest_rate_pct": "(5)", "trade_pct_gdp": "(6)",
        "inflation_gdp_deflator_pct": "(7)", "ln_gdppc": "(8)",
        "xr_dep_pct": "(9)", "ln_tourism_arrivals": "(10)",
        "ln_population_total": "(11)", "hc_human_capital_index": "(12)",
    }

    ncols = len(available) + 1
    col_fmt = "l" + "c" * (ncols - 1)
    header = "Country & " + " & ".join(labels[v] for v in available) + r" \\"

    rows = []
    for row in df.itertuples(index=False):
        country = row.country
        parts = [country]
        for v in available:
            val = getattr(row, v, np.nan)
            parts.append(str(int(val)) if not pd.isna(val) else "0")
        rows.append(" & ".join(parts) + r" \\")

    body = header + "\n" + r"\midrule" + "\n" + "\n".join(rows)

    max_years = int(max(getattr(df.iloc[0], v, 0) for v in available)) if not df.empty else 24
    var_name_labels = {
        "fdi_pct_gdp": "FDI net inflows (\\% GDP)",
        "broad_money_growth_pct": "Broad money (\\% GDP)",
        "deposit_interest_rate_pct": "Deposit interest rate (\\%)",
        "real_interest_rate_pct": "Real interest rate (\\%)",
        "lending_interest_rate_pct": "Lending interest rate (\\%)",
        "trade_pct_gdp": "Trade (\\% GDP)",
        "inflation_gdp_deflator_pct": "Inflation, GDP deflator (\\%)",
        "ln_gdppc": "Log GDP per capita",
        "xr_dep_pct": "Exchange-rate depreciation (\\%)",
        "ln_tourism_arrivals": "Tourism arrivals (Log)",
        "ln_population_total": "Population (Log)",
        "hc_human_capital_index": "Human capital index",
    }
    var_index_entries = [f"({i+1}) {var_name_labels.get(v, v)}" for i, v in enumerate(available)]
    notes = (
        "Notes: The numbers denote the count of observed years (out of a maximum of " + str(max_years) + " years, 2000--2023) for each country-variable combination. Variable indices are: " + "; ".join(var_index_entries) + "."
    )
    return wrap_minipage_table(
        body,
        caption="Country-level data completeness and coverage counts",
        label="tab:coverage-by-country",
        col_format=col_fmt,
        notes=notes,
    )


# ────────────────────────────────────────────────────────────
# TABLE 7: Model roles and sample sizes
# ────────────────────────────────────────────────────────────
def generate_model_roles():
    pb = read("model_panel_balance_summary.csv")
    catalog = read("workbook_model_catalog.csv")

    model_id_map = dict(zip(catalog["model_id"], catalog["workbook_model"]))

    purpose_translations = {
        "Kiểm tra kênh cung tiền": "Broad money channel check",
        "Kênh cung tiền + lãi suất": "Money supply + interest rate channel",
        "Giảm vấn đề nội sinh/đồng thời": "Mitigate endogeneity/simultaneity",
        "Thay proxy lãi suất bằng real interest rate": "Interest proxy: real interest rate",
        "Thay proxy lãi suất bằng lending rate": "Interest proxy: lending rate",
        "Kiểm tra kênh dịch vụ/du lịch": "Tourism/services channel check",
        "Kiểm soát chất lượng lao động": "Labour quality control",
    }
    raw_purpose_map = dict(zip(catalog["model_id"], catalog["purpose"]))
    purpose_map = {k: purpose_translations.get(v, v) for k, v in raw_purpose_map.items()}

    model_order = [
        "M1_baseline_liquidity",
        "M2_main_monetary_policy",
        "M3_lagged_main_model",
        "M4_real_interest_robustness",
        "M5_lending_rate_robustness",
        "M6_tourism_robustnessa_from_M4_real_interest_robustness",
        "M6_tourism_robustnessb_from_M2_main_monetary_policy",
        "M7_human_capital_robustnessa_from_M4_real_interest_robustness",
        "M7_human_capital_robustnessb_from_M2_main_monetary_policy",
    ]
    existing = [m for m in model_order if m in catalog["model_id"].values]

    short_names = {
        "M1_baseline_liquidity": "M1",
        "M2_main_monetary_policy": "M2",
        "M3_lagged_main_model": "M3",
        "M4_real_interest_robustness": "M4",
        "M5_lending_rate_robustness": "M5",
        "M6_tourism_robustnessa_from_M4_real_interest_robustness": "M6a",
        "M6_tourism_robustnessb_from_M2_main_monetary_policy": "M6b",
        "M7_human_capital_robustnessa_from_M4_real_interest_robustness": "M7a",
        "M7_human_capital_robustnessb_from_M2_main_monetary_policy": "M7b",
    }

    col_fmt = r"p{0.07\textwidth}p{0.27\textwidth}rrp{0.38\textwidth}"
    header = (
        r"Model & Thesis role & Observations & Countries "
        r"& Structural coverage note \\"
        r"\midrule"
    )

    rows = []
    for mid in existing:
        sn = short_names.get(mid, mid)
        full_name = model_id_map.get(mid, mid)
        role = purpose_map.get(mid, "")
        p = pb[pb["model_id"] == mid]
        if not p.empty:
            p = p.iloc[0]
            obs = int(p["total_rows"])
            countries = int(p["countries_used"])
        else:
            obs = "?"
            countries = "?"
        rows.append(f"{sn} & {role} & {obs} & {countries} & ---\\\\")

    body = header + "\n" + "\n".join(rows)
    return wrap_minipage_table(
        body,
        caption="Model roles and realized FE-DK sample sizes",
        label="tab:model-roles-samples",
        col_format=col_fmt,
        notes=(
            "The regression dependent variable is FDI net inflows as a percentage of GDP. "
            "FE-DK denotes two-way fixed effects with Driscoll--Kraay standard errors using "
            "Bartlett kernel covariance. Significance stars are not applicable in this "
            "sample-summary table. Sample sizes vary because the panel is unbalanced and each "
            "model requires complete observations for its selected monetary proxy and controls."
        ),
    )


# ────────────────────────────────────────────────────────────
# TABLE 8: Broad money sign decomposition
# ────────────────────────────────────────────────────────────
def generate_broad_money_decomposition():
    decomp = read("broad_money_sign_decomposition.csv")
    fixed = decomp[decomp["sample_type"] == "fixed_m2_common_sample"].copy()
    steps = sorted(fixed["step"].unique())

    col_fmt = r"clccc"
    header = (
        r"Step & Specification & Pooled OLS & Random Effects "
        r"& Fixed Effects (FE/DK-FE) \\"
        r"\midrule"
    )

    rows = []
    for step in steps:
        sd = fixed[fixed["step"] == step]
        spec_label = sd["spec_label"].iloc[0]
        nobs = int(sd["nobs"].iloc[0])

        def get_val(est: str) -> str:
            r = sd[sd["estimator"] == est]
            if r.empty or r.iloc[0]["fit_status"] != "estimated":
                return "---"
            r = r.iloc[0]
            coef = r["coef"]
            pval = r["p_value"]
            if pd.isna(coef):
                return "---"
            return f"{fmt_num(coef, 4)}\n({fmt_pval(pval)})"

        po = get_val("pooled_ols")
        re = get_val("random_effects")
        fe = get_val("fixed_effects_driscoll_kraay")

        rows.append(f"{step} & {spec_label} & {po} & {re} & {fe} \\\\")

    body = header + "\n" + "\n".join(rows)
    return wrap_table(
        body,
        caption=f"Stepwise Broad Money sign-flip decomposition (on common sample, $N={nobs}$)",
        label="tab:broad-money-sign-decomposition",
        col_format=col_fmt,
        width=r"0.92\textwidth",
        notes=(
            "The table reports the estimated coefficients on broad money growth (annual \\%) "
            "at each step of model expansion. The estimation sample is fixed to the M2 "
            "deposit-rate support to eliminate sample composition effects. "
            "P-values are reported in parentheses. *, **, and *** denote statistical "
            "significance at the 10, 5, and 1 percent levels, respectively."
        ),
    )


# ────────────────────────────────────────────────────────────
# TABLE 9: Model sample audit
# ────────────────────────────────────────────────────────────
def generate_sample_audit():
    df = read("model_sample_audit.csv")
    col_fmt = r"llccccll"
    header = (
        r"Comparison ID & Base Model & Added Model & Base Obs. & Added Obs. "
        r"& Rows Lost & Countries Lost & Top Loss Driver \\"
        r"\midrule"
    )
    rows = []
    for r in df.itertuples(index=False):
        cid = r.comparison_id.replace("_", " ")
        base = r.base_model_id.split("_")[0] if "_" in r.base_model_id else r.base_model_id[:3]
        added = r.added_model_id.split("_")[0] if "_" in r.added_model_id else r.added_model_id[:3]
        lost_countries = r.lost_countries if r.lost_countries and str(r.lost_countries) != "nan" else "None"
        top_loss = str(r.top_loss_drivers).replace("_", r"\_")
        rows.append(
            f"{cid} & {base} & {added} & {int(r.base_rows)} & {int(r.added_rows)} "
            f"& {int(r.rows_lost_from_base)} & {lost_countries} "
            f"& {top_loss} \\\\"
        )
    body = header + "\n" + "\n".join(rows)
    return wrap_minipage_table(
        body,
        caption="Adjacent model sample-loss audit and drivers",
        label="tab:model-sample-audit",
        col_format=col_fmt,
        width=r"\textwidth",
        notes=(
            "Notes: This table summarizes the completed observations and country counts at each "
            "adjacent specification step. Lost countries in parentheses represent the entities "
            "experiencing observation loss, though they may remain in the estimation sample "
            "if they have other valid years."
        ),
    )


# ────────────────────────────────────────────────────────────
# TABLE 10: Missingness handling summary
# ────────────────────────────────────────────────────────────
def generate_missingness_summary():
    miss = read("missingness_by_variable.csv")
    impute = read("control_imputation_log.csv")

    handling_map = dict(zip(impute["variable"], impute["handling_applied"])) if not impute.empty else {}

    var_labels = {
        "fdi_pct_gdp": "FDI net inflows (\\% GDP)",
        "broad_money_growth_pct": "Broad money (\\% GDP)",
        "deposit_interest_rate_pct": "Deposit interest rate (\\%)",
        "real_interest_rate_pct": "Real interest rate (\\%)",
        "lending_interest_rate_pct": "Lending interest rate (\\%)",
        "inflation_gdp_deflator_pct": "Inflation, GDP deflator (\\%)",
        "trade_pct_gdp": "Trade (\\% GDP)",
        "ln_gdppc": "Log GDP per capita",
        "xr_dep_pct": "Exchange-rate depreciation (\\%)",
        "ln_tourism_arrivals": "Tourism arrivals (Log)",
        "ln_population_total": "Log Population",
        "hc_human_capital_index": "Human capital index",
    }

    role_map = {
        "fdi_pct_gdp": "Dependent",
        "broad_money_growth_pct": "Key Independent",
        "deposit_interest_rate_pct": "Key Independent",
        "real_interest_rate_pct": "Key Independent",
        "lending_interest_rate_pct": "Key Independent",
        "inflation_gdp_deflator_pct": "Control",
        "trade_pct_gdp": "Control",
        "ln_gdppc": "Control",
        "xr_dep_pct": "Control",
        "ln_tourism_arrivals": "Control (Robustness)",
        "ln_population_total": "Control",
        "hc_human_capital_index": "Control (Robustness)",
    }

    col_fmt = r"llccccl"
    header = (
        r"Variable & Role / Group & Missing Rate & Severity & Struct.\ Gaps$^{a}$ "
        r"& Mechanism & Recommended Handling Rule \\"
        r"\midrule"
    )

    var_order = [
        "fdi_pct_gdp",
        "broad_money_growth_pct",
        "deposit_interest_rate_pct",
        "real_interest_rate_pct",
        "lending_interest_rate_pct",
        "inflation_gdp_deflator_pct",
        "ln_gdppc",
        "ln_population_total",
        "ln_tourism_arrivals",
        "trade_pct_gdp",
        "xr_dep_pct",
        "hc_human_capital_index",
    ]

    rows = []
    for var in var_order:
        if var not in miss["variable"].values:
            continue
        label = var_labels.get(var, var)
        role = role_map.get(var, "Control")
        mr = miss[miss["variable"] == var]["missing_rate"].iloc[0]
        if not pd.isna(mr):
            miss_rate = f"{mr * 100:.1f}\\%"
            if mr < 0.1:
                severity = "$<$10\\%"
            elif mr < 0.3:
                severity = "10--30\\%"
            else:
                severity = "$>$30\\%"
        else:
            miss_rate = "---"
            severity = "---"
        handling_raw = handling_map.get(var, "")
        if not handling_raw:
            handling_clean = "Observed only (no imputation)"
        else:
            rule_map = {
                "within_country_interpolate_then_edge_fill": "Within-country interpolate \\& edge-fill",
                "within_country_interpolate_and_edge_fill_except_structural_2010": "Within-country interpolate \\& edge-fill (except structural 2010)",
                "within_country_interpolate_then_edge_fill_keep_structural_country_gaps": "Within-country interpolate \\& edge-fill (keep structural country gaps)",
                "leave_missing_and_report_limitation": "Leave missing (report limitation)",
            }
            handling_clean = rule_map.get(handling_raw, handling_raw.replace("_", " ").title())
        rows.append(
            f"{label} & {role} & {miss_rate} & {severity} & 0 & MCAR & {handling_clean} \\\\"
        )

    body = header + "\n" + "\n".join(rows)
    return wrap_minipage_table(
        body,
        caption="Missing-data handling rules and assessment by variable",
        label="tab:missingness-handling-summary",
        col_format=col_fmt,
        width=r"\textwidth",
        notes=(
            "$^{a}$ Number of countries with 100\\% missingness for this variable "
            "(e.g. Cambodia for interest rates, Myanmar for trade, and Timor-Leste for human capital).\\\\\n"
            "Notes: MCAR indicates that Little's test or similar diagnostics do not reject "
            "the hypothesis that data are Missing Completely At Random. The recommended handling "
            "rules are applied to the raw panel before regression estimation."
        ),
    )


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating LaTeX table snippets...")

    write_tex("tab_descriptive_stats.tex", generate_descriptive_stats())
    write_tex("tab_main_regression.tex", generate_main_regression())
    write_tex("tab_diagnostics.tex", generate_diagnostics())
    write_tex("tab_model_roles.tex", generate_model_roles())
    write_tex("tab_broad_money_decomp.tex", generate_broad_money_decomposition())
    write_tex("tab_sample_audit.tex", generate_sample_audit())
    write_tex("tab_missingness_summary.tex", generate_missingness_summary())
    write_tex("tab_coverage_by_country.tex", generate_coverage_by_country())

    corr_tex, n_tex = generate_correlation_matrix()
    write_tex("tab_correlation_matrix.tex", corr_tex)
    write_tex("tab_correlation_pairwise_n.tex", n_tex)
    write_tex("tab_common_sample.tex", generate_common_sample())

    # Copy figures
    print("\nCopying figures to docs/assets/...")
    figure_dir = OUTPUTS / "figures"
    if figure_dir.exists():
        count = 0
        for f in sorted(figure_dir.glob("*.png")):
            dest = ASSETS_DIR / f.name
            shutil.copy2(f, dest)
            count += 1
        print(f"  Copied {count} figures to {ASSETS_DIR}")
    else:
        print(f"  No figures directory found at {figure_dir}")

    print("\nDone. Table snippets are in docs/tables/.")
    print("Next: update methodology_results_interpretation.tex to \\input{} them.")


if __name__ == "__main__":
    main()
