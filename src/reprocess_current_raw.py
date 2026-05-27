from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    CURRENT_ADDITIONAL_SERIES_FILE,
    CURRENT_HC_POP_FILE,
    CURRENT_MACRO_PANEL_WORKBOOK,
    OUTPUTS_DIR,
    PANEL_COUNTRY_ORDER,
    PROCESSED_PANEL_FILE,
    TIME_WINDOW,
    WINSOR_BOUNDS,
    ensure_output_dirs,
)


MACRO_COLUMN_MAP = {
    "Country Name": "country",
    "Country Code": "country_code",
    "Time": "year",
    "Foreign direct investment, net inflows (% of GDP) [BX.KLT.DINV.WD.GD.ZS]": "fdi_pct_gdp",
    "Broad money growth (annual %) [FM.LBL.BMNY.ZG]": "broad_money_growth_pct",
    "Real interest rate (%) [FR.INR.RINR]": "real_interest_rate_pct",
    "Official exchange rate (LCU per US$, period average) [PA.NUS.FCRF]": "official_exchange_rate_lcu_usd",
    "Trade (% of GDP) [NE.TRD.GNFS.ZS]": "trade_pct_gdp",
    "GDP (current US$) [NY.GDP.MKTP.CD]": "gdppc_current_usd",
    "Inflation, GDP deflator (annual %) [NY.GDP.DEFL.KD.ZG]": "inflation_gdp_deflator_pct",
}

ADDITIONAL_SERIES_MAP = {
    "Deposit interest rate (%)": "deposit_interest_rate_pct",
    "Lending interest rate (%)": "lending_interest_rate_pct",
    "International tourism, number of arrivals": "tourism_arrivals",
}

COUNTRY_NAME_NORMALIZATION = {
    "Lao People's DR": "Lao PDR",
}

CLEAN_PANEL_COLUMNS = [
    "country_id",
    "country_code",
    "country",
    "year",
    "fdi_pct_gdp",
    "fdi_pct_gdp_winsorized",
    "broad_money_growth_pct",
    "trade_pct_gdp",
    "inflation_gdp_deflator_pct",
    "deposit_interest_rate_pct",
    "real_interest_rate_pct",
    "lending_interest_rate_pct",
    "hc_human_capital_index",
    "ln_gdppc",
    "xr_dep_pct",
    "xr_dep_pct_winsorized",
    "ln_population_total",
    "ln_tourism_arrivals",
]

CONTROL_VARS = [
    "trade_pct_gdp",
    "inflation_gdp_deflator_pct",
    "ln_gdppc",
    "xr_dep_pct",
    "ln_population_total",
    "ln_tourism_arrivals",
    "hc_human_capital_index",
]


@dataclass(frozen=True)
class InputAuditRow:
    file_name: str
    role: str
    years: str
    countries: int
    rows: int
    notes: str


def ordered_country_frame() -> pd.DataFrame:
    ordered = pd.DataFrame(PANEL_COUNTRY_ORDER, columns=["country_code", "country"])
    ordered["country_id"] = np.arange(1, len(ordered) + 1)
    return ordered


def load_macro_panel() -> pd.DataFrame:
    if not CURRENT_MACRO_PANEL_WORKBOOK.exists():
        raise FileNotFoundError(f"Missing macro workbook: {CURRENT_MACRO_PANEL_WORKBOOK}")

    raw = pd.read_excel(CURRENT_MACRO_PANEL_WORKBOOK, sheet_name="Data")
    missing = [column for column in MACRO_COLUMN_MAP if column not in raw.columns]
    if missing:
        raise ValueError(f"Macro workbook is missing required columns: {missing}")

    df = raw.loc[raw["Country Code"].isin(ordered_country_frame()["country_code"])].copy()
    df = df.rename(columns=MACRO_COLUMN_MAP)
    # Keep only the mapped columns so legacy/unneeded workbook columns never leak into the panel.
    df = df[list(MACRO_COLUMN_MAP.values())].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    start_year, end_year = TIME_WINDOW
    df = df.loc[df["year"].between(start_year, end_year)].copy()
    df["year"] = df["year"].astype(int)
    df["country"] = df["country"].replace(COUNTRY_NAME_NORMALIZATION)

    numeric_columns = [
        "fdi_pct_gdp",
        "broad_money_growth_pct",
        "real_interest_rate_pct",
        "official_exchange_rate_lcu_usd",
        "trade_pct_gdp",
        "gdppc_current_usd",
        "inflation_gdp_deflator_pct",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


def load_hc_pop_panel() -> pd.DataFrame:
    if not CURRENT_HC_POP_FILE.exists():
        raise FileNotFoundError(f"Missing human capital / population file: {CURRENT_HC_POP_FILE}")

    df = pd.read_csv(CURRENT_HC_POP_FILE)
    required_columns = {"ISO code", "Country", "Variable code", "Variable name"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"HC/POP file is missing required columns: {sorted(missing)}")

    year_columns = [column for column in df.columns if str(column).isdigit()]
    if not year_columns:
        raise ValueError("HC/POP file does not contain year columns.")

    long_df = df.melt(
        id_vars=["ISO code", "Country", "Variable code"],
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    ).rename(
        columns={
            "ISO code": "country_code",
            "Country": "country",
            "Variable code": "variable_code",
        }
    )
    long_df["country"] = long_df["country"].replace(COUNTRY_NAME_NORMALIZATION)
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.loc[long_df["country_code"].isin(ordered_country_frame()["country_code"])].copy()
    long_df = long_df.dropna(subset=["year"])
    long_df["year"] = long_df["year"].astype(int)

    start_year, end_year = TIME_WINDOW
    long_df = long_df.loc[long_df["year"].between(start_year, end_year)].copy()
    wide_df = (
        long_df.pivot_table(
            index=["country_code", "country", "year"],
            columns="variable_code",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"hc": "hc_human_capital_index", "pop": "population_millions"})
    )
    wide_df.columns.name = None
    wide_df["population_total"] = wide_df["population_millions"] * 1_000_000
    return wide_df[["country_code", "country", "year", "population_total", "hc_human_capital_index"]].reset_index(
        drop=True
    )


def load_additional_series_panel() -> pd.DataFrame:
    if not CURRENT_ADDITIONAL_SERIES_FILE.exists():
        raise FileNotFoundError(f"Missing additional-series file: {CURRENT_ADDITIONAL_SERIES_FILE}")

    df = pd.read_csv(CURRENT_ADDITIONAL_SERIES_FILE)
    required_columns = {"Country Name", "Country Code", "Series Name"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Additional-series file is missing required columns: {sorted(missing)}")

    year_columns = [column for column in df.columns if "[YR" in str(column)]
    if not year_columns:
        raise ValueError("Additional-series file does not contain year columns.")

    long_df = (
        df.loc[df["Series Name"].isin(ADDITIONAL_SERIES_MAP)]
        .melt(
            id_vars=["Country Name", "Country Code", "Series Name"],
            value_vars=year_columns,
            var_name="year_label",
            value_name="value",
        )
        .rename(columns={"Country Name": "country", "Country Code": "country_code", "Series Name": "series_name"})
    )
    long_df["country"] = long_df["country"].replace(COUNTRY_NAME_NORMALIZATION)
    long_df["year"] = long_df["year_label"].astype(str).str.extract(r"(\d{4})").astype(int)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["variable"] = long_df["series_name"].map(ADDITIONAL_SERIES_MAP)
    long_df = long_df.loc[long_df["country_code"].isin(ordered_country_frame()["country_code"])].copy()
    start_year, end_year = TIME_WINDOW
    long_df = long_df.loc[long_df["year"].between(start_year, end_year)].copy()

    wide_df = (
        long_df.pivot_table(
            index=["country_code", "country", "year"],
            columns="variable",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide_df.columns.name = None
    return wide_df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "deposit_interest_rate_pct" not in enriched.columns:
        enriched["deposit_interest_rate_pct"] = np.nan
    if "lending_interest_rate_pct" not in enriched.columns:
        enriched["lending_interest_rate_pct"] = np.nan
    if "tourism_arrivals" not in enriched.columns:
        enriched["tourism_arrivals"] = np.nan
    enriched["ln_gdppc"] = np.where(enriched["gdppc_current_usd"] > 0, np.log(enriched["gdppc_current_usd"]), np.nan)
    enriched["ln_population_total"] = np.where(
        enriched["population_total"] > 0, np.log(enriched["population_total"]), np.nan
    )
    enriched["ln_tourism_arrivals"] = np.where(
        enriched["tourism_arrivals"] > 0, np.log(enriched["tourism_arrivals"]), np.nan
    )
    enriched["xr_dep_pct"] = (
        enriched.sort_values(["country_id", "year"])
        .groupby("country_code")["official_exchange_rate_lcu_usd"]
        .transform(lambda series: np.log(series).diff() * 100)
    )
    return enriched


def impute_controls(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = df.copy()
    rows: list[dict[str, object]] = []
    for variable in CONTROL_VARS:
        if variable not in analysis_df.columns:
            continue
        before_missing = int(analysis_df[variable].isna().sum())
        if variable == "hc_human_capital_index":
            handling = "leave_missing_and_report_limitation"
        elif variable == "xr_dep_pct":
            handling = "within_country_interpolate_and_edge_fill_except_structural_2010"
            structural_mask = analysis_df["year"].eq(TIME_WINDOW[0])
            imputed = analysis_df.groupby("country_code")[variable].transform(
                lambda series: series.interpolate(method="linear", limit_area="inside").ffill().bfill()
            )
            work = analysis_df[variable].copy()
            work.loc[~structural_mask] = imputed.loc[~structural_mask]
            analysis_df[variable] = work
        elif variable == "ln_tourism_arrivals":
            handling = "within_country_interpolate_then_edge_fill_keep_structural_country_gaps"
            analysis_df[variable] = analysis_df.groupby("country_code")[variable].transform(
                lambda series: series.interpolate(method="linear", limit_area="inside").ffill().bfill()
            )
        else:
            handling = "within_country_interpolate_then_edge_fill"
            analysis_df[variable] = analysis_df.groupby("country_code")[variable].transform(
                lambda series: series.interpolate(method="linear", limit_area="inside").ffill().bfill()
            )
        after_missing = int(analysis_df[variable].isna().sum())
        rows.append(
            {
                "variable": variable,
                "handling_applied": handling,
                "missing_before": before_missing,
                "missing_after": after_missing,
                "filled_values": before_missing - after_missing,
            }
        )
    return analysis_df, pd.DataFrame(rows)


def winsorize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = df.copy()
    rows: list[dict[str, object]] = []
    for source_column in ["fdi_pct_gdp", "xr_dep_pct"]:
        target_column = f"{source_column}_winsorized"
        series = analysis_df[source_column].dropna()
        low, high = series.quantile(WINSOR_BOUNDS).tolist()
        analysis_df[target_column] = analysis_df[source_column].clip(lower=low, upper=high)
        rows.append(
            {
                "source_variable": source_column,
                "sensitivity_variable": target_column,
                "lower_bound_p01": float(low),
                "upper_bound_p99": float(high),
                "values_clipped": int(analysis_df[source_column].ne(analysis_df[target_column]).sum()),
                "use_case": "robustness_only_do_not_replace_main_observed_values",
            }
        )
    return analysis_df, pd.DataFrame(rows)


def build_review_flags(df: pd.DataFrame) -> pd.DataFrame:
    frames = [
        df.loc[df["xr_dep_pct"].abs() > 100, ["country", "year"]].assign(
            flag="xr_dep_pct_extreme_keep_and_review",
            value=lambda frame: df.loc[frame.index, "xr_dep_pct"],
            note="Observed value retained; winsorized version is sensitivity-only.",
        ),
        df.loc[df["inflation_gdp_deflator_pct"].abs() > 30, ["country", "year"]].assign(
            flag="inflation_extreme_keep_and_review",
            value=lambda frame: df.loc[frame.index, "inflation_gdp_deflator_pct"],
            note="Observed value retained for main analysis.",
        ),
        df.loc[df["fdi_pct_gdp"].abs() > 50, ["country", "year"]].assign(
            flag="fdi_truly_extreme_keep_and_review",
            value=lambda frame: df.loc[frame.index, "fdi_pct_gdp"],
            note="Observed value retained; inspect leverage in pooled models.",
        ),
    ]
    review_flags = pd.concat(frames, ignore_index=True)
    if review_flags.empty:
        return pd.DataFrame(columns=["country", "year", "flag", "value", "note"])
    return review_flags.loc[:, ["country", "year", "flag", "value", "note"]].sort_values(
        ["flag", "country", "year"]
    )


def build_transformation_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "source_variable": "gdppc_current_usd",
            "transformed_variable": "ln_gdppc",
            "transformation": "natural_log",
            "status": "created_from_current_macro_workbook",
            "preferred_for_modeling": True,
        },
        {
            "source_variable": "official_exchange_rate_lcu_usd",
            "transformed_variable": "xr_dep_pct",
            "transformation": "country_log_difference_x100",
            "status": "created_from_current_macro_workbook",
            "preferred_for_modeling": True,
        },
        {
            "source_variable": "population_total",
            "transformed_variable": "ln_population_total",
            "transformation": "natural_log",
            "status": "created_from_current_hc_pop_workbook",
            "preferred_for_modeling": True,
        },
        {
            "source_variable": "hc_human_capital_index_raw_digits",
            "transformed_variable": "hc_human_capital_index",
            "transformation": "direct_from_hc_series",
            "status": "created_from_current_hc_pop_workbook",
            "preferred_for_modeling": True,
        },
        {
            "source_variable": CURRENT_ADDITIONAL_SERIES_FILE.name,
            "transformed_variable": "deposit_interest_rate_pct",
            "transformation": "country_series_merge_from_year_columns",
            "status": "created_from_additional_series_file",
            "preferred_for_modeling": True,
        },
        {
            "source_variable": CURRENT_ADDITIONAL_SERIES_FILE.name,
            "transformed_variable": "lending_interest_rate_pct",
            "transformation": "country_series_merge_from_year_columns",
            "status": "created_from_additional_series_file",
            "preferred_for_modeling": False,
        },
        {
            "source_variable": "tourism_arrivals",
            "transformed_variable": "ln_tourism_arrivals",
            "transformation": "natural_log",
            "status": "created_from_additional_series_file",
            "preferred_for_modeling": False,
        },
    ]
    audit = pd.DataFrame(rows)
    audit["non_missing_source"] = audit["source_variable"].map(df.notna().sum())
    audit["non_missing_transformed"] = audit["transformed_variable"].map(df.notna().sum())
    return audit


def build_coverage_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_by_variable = (
        df[CLEAN_PANEL_COLUMNS]
        .notna()
        .sum()
        .rename("non_missing")
        .to_frame()
        .assign(total_rows=len(df))
    )
    coverage_by_variable["missing"] = coverage_by_variable["total_rows"] - coverage_by_variable["non_missing"]
    coverage_by_variable["missing_rate"] = coverage_by_variable["missing"] / coverage_by_variable["total_rows"]

    coverage_columns = [
        "fdi_pct_gdp",
        "broad_money_growth_pct",
        "deposit_interest_rate_pct",
        "real_interest_rate_pct",
        "lending_interest_rate_pct",
        "trade_pct_gdp",
        "inflation_gdp_deflator_pct",
        "ln_gdppc",
        "xr_dep_pct",
        "ln_tourism_arrivals",
        "ln_population_total",
        "hc_human_capital_index",
    ]
    coverage_by_country = (
        df.groupby("country")[coverage_columns].apply(lambda frame: frame.notna().sum()).sort_index()
    )
    return coverage_by_variable, coverage_by_country


def build_variable_audit(coverage_by_variable: pd.DataFrame) -> pd.DataFrame:
    audit = coverage_by_variable.reset_index().rename(columns={"index": "variable"})
    audit["recommended_handling"] = audit["variable"].map(
        {
            "deposit_interest_rate_pct": "processed_from_current_raw_inputs",
            "lending_interest_rate_pct": "processed_from_current_raw_inputs",
            "ln_tourism_arrivals": "processed_from_current_raw_inputs",
            "hc_human_capital_index": "leave_missing_and_report_limitation",
        }
    ).fillna("processed_from_current_raw_inputs")
    return audit


def build_input_audit(macro_df: pd.DataFrame, hc_df: pd.DataFrame, additional_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        InputAuditRow(
            file_name=CURRENT_MACRO_PANEL_WORKBOOK.name,
            role="base_macro_panel",
            years=f"{macro_df['year'].min()}-{macro_df['year'].max()}",
            countries=int(macro_df["country_code"].nunique()),
            rows=len(macro_df),
            notes="Uses WDI-style country-year rows directly from the Data sheet.",
        ),
        InputAuditRow(
            file_name=CURRENT_HC_POP_FILE.name,
            role="human_capital_population_panel",
            years=f"{hc_df['year'].min()}-{hc_df['year'].max()}",
            countries=int(hc_df["country_code"].nunique()),
            rows=len(hc_df),
            notes="Melts row-by-variable yearly columns, uses only hc and pop, and ignores flawed xr entirely.",
        ),
        InputAuditRow(
            file_name=CURRENT_ADDITIONAL_SERIES_FILE.name,
            role="supplementary_interest_tourism_panel",
            years=f"{TIME_WINDOW[0]}-{TIME_WINDOW[1]}",
            countries=int(additional_df["country_code"].nunique()),
            rows=len(additional_df),
            notes="Melts year columns and merges deposit rate, lending rate, and tourism arrivals by country-year.",
        ),
    ]
    return pd.DataFrame(rows)


def save_outputs(
    clean_panel: pd.DataFrame,
    winsorization_thresholds: pd.DataFrame,
    control_imputation_log: pd.DataFrame,
    transformation_audit: pd.DataFrame,
    coverage_by_variable: pd.DataFrame,
    coverage_by_country: pd.DataFrame,
    review_flags: pd.DataFrame,
    variable_audit: pd.DataFrame,
    raw_input_audit: pd.DataFrame,
) -> None:
    clean_panel.to_csv(PROCESSED_PANEL_FILE, index=False)
    winsorization_thresholds.to_csv(OUTPUTS_DIR / "winsorization_thresholds.csv", index=False)
    control_imputation_log.to_csv(OUTPUTS_DIR / "control_imputation_log.csv", index=False)
    transformation_audit.to_csv(OUTPUTS_DIR / "transformation_audit.csv", index=False)
    coverage_by_variable.to_csv(OUTPUTS_DIR / "coverage_by_variable.csv")
    coverage_by_country.to_csv(OUTPUTS_DIR / "coverage_by_country.csv")
    review_flags.to_csv(OUTPUTS_DIR / "review_flags.csv", index=False)
    variable_audit.to_csv(OUTPUTS_DIR / "variable_audit.csv", index=False)
    raw_input_audit.to_csv(OUTPUTS_DIR / "raw_input_audit.csv", index=False)

    workbook_path = OUTPUTS_DIR / "preprocessing_outputs.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        raw_input_audit.to_excel(writer, sheet_name="raw_input_audit", index=False)
        transformation_audit.to_excel(writer, sheet_name="transformations", index=False)
        variable_audit.to_excel(writer, sheet_name="variable_audit", index=False)
        coverage_by_country.to_excel(writer, sheet_name="coverage_by_country")
        coverage_by_variable.to_excel(writer, sheet_name="coverage_by_variable")
        review_flags.to_excel(writer, sheet_name="review_flags", index=False)
        winsorization_thresholds.to_excel(writer, sheet_name="winsor_thresholds", index=False)
        control_imputation_log.to_excel(writer, sheet_name="imputation_log", index=False)


def build_clean_panel() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    macro_df = load_macro_panel()
    hc_df = load_hc_pop_panel()
    additional_df = load_additional_series_panel()
    countries = ordered_country_frame()

    merged = (
        macro_df.merge(hc_df, on=["country_code", "country", "year"], how="left")
        .merge(additional_df, on=["country_code", "country", "year"], how="left")
        .merge(countries, on=["country_code", "country"], how="left")
        .sort_values(["country_id", "year"])
        .reset_index(drop=True)
    )
    analysis_df = add_derived_columns(merged)
    analysis_df, control_imputation_log = impute_controls(analysis_df)
    analysis_df, winsorization_thresholds = winsorize_columns(analysis_df)
    clean_panel = analysis_df[CLEAN_PANEL_COLUMNS].copy()

    transformation_audit = build_transformation_audit(analysis_df)
    coverage_by_variable, coverage_by_country = build_coverage_outputs(clean_panel)
    review_flags = build_review_flags(analysis_df)
    variable_audit = build_variable_audit(coverage_by_variable)
    raw_input_audit = build_input_audit(macro_df, hc_df, additional_df)

    outputs = {
        "winsorization_thresholds": winsorization_thresholds,
        "control_imputation_log": control_imputation_log,
        "transformation_audit": transformation_audit,
        "coverage_by_variable": coverage_by_variable,
        "coverage_by_country": coverage_by_country,
        "review_flags": review_flags,
        "variable_audit": variable_audit,
        "raw_input_audit": raw_input_audit,
    }
    return clean_panel, outputs


def main() -> None:
    ensure_output_dirs()
    clean_panel, outputs = build_clean_panel()
    save_outputs(
        clean_panel=clean_panel,
        winsorization_thresholds=outputs["winsorization_thresholds"],
        control_imputation_log=outputs["control_imputation_log"],
        transformation_audit=outputs["transformation_audit"],
        coverage_by_variable=outputs["coverage_by_variable"],
        coverage_by_country=outputs["coverage_by_country"],
        review_flags=outputs["review_flags"],
        variable_audit=outputs["variable_audit"],
        raw_input_audit=outputs["raw_input_audit"],
    )
    print(f"Saved processed panel to {PROCESSED_PANEL_FILE}")


if __name__ == "__main__":
    main()
