"""src/data/merge.py

Panel assembly: merge macro + WGI + HC/pop → apply transforms → clean → export.

Merge key: (country_code, year)
Final panel columns (instruct.md §4 Required Variables):
  Identifiers : country_code, country, year
  Dependent   : fdi_pct_gdp
  Main explanatory : bm_growth
  Monetary tightness: real_interest_rate
  Controls    : gdp_growth, ln_gdppc, trade_pct_gdp, d_ln_exr, inflation
  Institutional: rq (baseline), rl (robustness), va, pv, ge, cc, wgi_composite
  Auxiliary   : gdppc, gdp_current_usd, population, hc_index, exr_lcu_usd
"""
from __future__ import annotations

import pandas as pd

from src.config import PANEL_FILE, VARIABLE_DICT_FILE, ensure_output_dirs
from src.data.clean import (
    build_coverage_report,
    drop_missing_fdi,
    handle_missing_controls,
    winsorize_panel,
)
from src.data.loader import load_all_wgi, load_hc_pop, load_macro_panel
from src.data.transform import apply_all_transforms

# Ordered final columns in the output panel
PANEL_COLUMNS: list[str] = [
    # Identifiers
    "country_code",
    "country",
    "year",
    # Dependent variable
    "fdi_pct_gdp",
    # Main explanatory
    "bm_growth",
    # Monetary tightness
    "real_interest_rate",
    # Market growth & size
    "gdp_growth",
    "gdppc",
    "ln_gdppc",
    # Openness
    "trade_pct_gdp",
    # Exchange rate
    "d_ln_exr",
    # Inflation
    "inflation",
    # Institutional (WGI)
    "rq",          # Regulatory Quality  ← baseline
    "rl",          # Rule of Law         ← robustness
    "ge",          # Government Effectiveness
    "cc",          # Control of Corruption
    "va",          # Voice & Accountability
    "pv",          # Political Stability
    "wgi_composite",
    # Auxiliary / source series (kept for transparency)
    "gdp_current_usd",
    "population",
    "hc_index",
    "exr_lcu_usd",
]

# Variable dictionary (instruct.md §13 output #2)
VARIABLE_DICT: list[dict] = [
    {"variable": "fdi_pct_gdp",      "label": "FDI net inflows (% of GDP)",          "unit": "% GDP",     "role": "Dependent",          "source": "Macro workbook",  "transformation": "None"},
    {"variable": "bm_growth",        "label": "Broad money growth",                   "unit": "% growth",  "role": "Main explanatory",   "source": "Macro workbook",  "transformation": "None"},
    {"variable": "real_interest_rate","label": "Real interest rate",                   "unit": "Level (%)", "role": "Monetary tightness", "source": "Macro workbook",  "transformation": "None"},
    {"variable": "gdp_growth",       "label": "GDP growth",                           "unit": "% growth",  "role": "Control",            "source": "Macro workbook",  "transformation": "None"},
    {"variable": "gdppc",            "label": "GDP per capita (current USD)",         "unit": "USD",       "role": "Control (source)",   "source": "Derived",         "transformation": "GDP / population"},
    {"variable": "ln_gdppc",         "label": "ln(GDP per capita)",                   "unit": "Log level", "role": "Control",            "source": "Derived",         "transformation": "ln(gdp_current_usd / population)"},
    {"variable": "trade_pct_gdp",    "label": "Trade openness",                       "unit": "% GDP",     "role": "Control",            "source": "Macro workbook",  "transformation": "None"},
    {"variable": "d_ln_exr",         "label": "Δln(Exchange rate)",                   "unit": "% change",  "role": "Control",            "source": "Derived",         "transformation": "[ln(EXR_t) - ln(EXR_t-1)] × 100"},
    {"variable": "inflation",        "label": "Inflation (GDP deflator)",             "unit": "%",         "role": "Control",            "source": "Macro workbook",  "transformation": "None"},
    {"variable": "rq",               "label": "Regulatory Quality (WGI)",             "unit": "Index",     "role": "Institutional",      "source": "WGI workbook",    "transformation": "None — baseline proxy"},
    {"variable": "rl",               "label": "Rule of Law (WGI)",                   "unit": "Index",     "role": "Institutional",      "source": "WGI workbook",    "transformation": "None — robustness proxy"},
    {"variable": "ge",               "label": "Government Effectiveness (WGI)",      "unit": "Index",     "role": "Institutional",      "source": "WGI workbook",    "transformation": "None"},
    {"variable": "cc",               "label": "Control of Corruption (WGI)",         "unit": "Index",     "role": "Institutional",      "source": "WGI workbook",    "transformation": "None"},
    {"variable": "va",               "label": "Voice & Accountability (WGI)",        "unit": "Index",     "role": "Institutional",      "source": "WGI workbook",    "transformation": "None"},
    {"variable": "pv",               "label": "Political Stability (WGI)",           "unit": "Index",     "role": "Institutional",      "source": "WGI workbook",    "transformation": "None"},
    {"variable": "wgi_composite",    "label": "WGI Composite Index",                 "unit": "Index",     "role": "Institutional",      "source": "Derived",         "transformation": "(rq + rl + ge + cc) / 4"},
    {"variable": "hc_index",         "label": "Human Capital Index",                 "unit": "Index",     "role": "Auxiliary",          "source": "HC/pop file",     "transformation": "None"},
]


def build_panel() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Assemble, transform, and clean the full panel.

    Returns
    -------
    panel : pd.DataFrame
        Clean merged panel ready for analysis.
    artifacts : dict
        Audit logs and coverage reports.
    """
    print("[merge] Loading raw data...")
    macro = load_macro_panel()
    wgi = load_all_wgi()
    hc_pop = load_hc_pop()

    print("[merge] Merging datasets on (country_code, year)...")
    panel = (
        macro
        .merge(hc_pop[["country_code", "year", "population", "hc_index"]],
               on=["country_code", "year"], how="left")
        .merge(wgi, on=["country_code", "year"], how="left")
    )

    print("[merge] Applying variable transformations...")
    panel = apply_all_transforms(panel)

    print("[merge] Dropping rows with missing FDI...")
    panel = drop_missing_fdi(panel)

    print("[merge] Handling missing controls...")
    panel, imputation_log = handle_missing_controls(panel)

    print("[merge] Winsorizing continuous variables...")
    panel, winsor_log = winsorize_panel(panel)

    # Select and order final columns (keep only those that exist)
    final_cols = [c for c in PANEL_COLUMNS if c in panel.columns]
    panel = panel[final_cols].copy()

    print("[merge] Building coverage report...")
    coverage = build_coverage_report(panel)

    artifacts = {
        "imputation_log": imputation_log,
        "winsor_log": winsor_log,
        "coverage": coverage,
        "variable_dict": pd.DataFrame(VARIABLE_DICT),
    }

    return panel.sort_values(["country_code", "year"]).reset_index(drop=True), artifacts


def save_panel(panel: pd.DataFrame, artifacts: dict[str, pd.DataFrame]) -> None:
    """Write panel and all audit artifacts to disk."""
    ensure_output_dirs()
    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    panel.to_csv(PANEL_FILE, index=False)
    print(f"[merge] Panel saved → {PANEL_FILE}  ({len(panel)} rows, {panel['country_code'].nunique()} countries)")

    artifacts["variable_dict"].to_csv(VARIABLE_DICT_FILE, index=False)
    print(f"[merge] Variable dictionary → {VARIABLE_DICT_FILE}")

    # Preprocessing report workbook
    from src.config import OUTPUTS_DIR
    report_path = OUTPUTS_DIR / "preprocessing_report.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        artifacts["variable_dict"].to_excel(writer, sheet_name="variable_dict", index=False)
        artifacts["coverage"].to_excel(writer, sheet_name="coverage", index=False)
        artifacts["imputation_log"].to_excel(writer, sheet_name="imputation_log", index=False)
        artifacts["winsor_log"].to_excel(writer, sheet_name="winsor_thresholds", index=False)
    print(f"[merge] Preprocessing report → {report_path}")
