"""scripts/02_run_diagnostics.py

Phase 2 runner: load panel → run all diagnostic tests → export results.

Usage:
    python scripts/02_run_diagnostics.py

Requires:
    data/processed/panel.csv  (produced by scripts/01_process_data.py)

Outputs (all under outputs/diagnostics/):
    descriptive_stats.csv
    correlation_matrix.csv
    high_correlations.csv
    vif_table.csv
    unit_root_tests.csv
    heteroskedasticity_tests.csv
    serial_correlation_tests.csv
    cross_section_tests.csv
    diagnostics_summary.xlsx   ← all sheets combined
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DIAGNOSTICS_DIR, PANEL_FILE, ensure_output_dirs
from src.diagnostics.descriptive import (
    MODEL_VARIABLES,
    correlation_matrix,
    descriptive_stats,
    flag_high_correlations,
)
from src.diagnostics.multicollinearity import vif_for_specs
from src.diagnostics.unit_root import run_unit_root_tests
from src.diagnostics.heteroskedasticity import modified_wald_test
from src.diagnostics.serial_correlation import wooldridge_test
from src.diagnostics.cross_section import pesaran_cd_test

# ---------------------------------------------------------------------------
# Model specification regressors (matching instruct.md §10 Spec 1–4)
# Used to define VIF / diagnostic regressor sets.
# ---------------------------------------------------------------------------
BASE_CONTROLS = [
    "bm_growth",
    "gdp_growth",
    "ln_gdppc",
    "trade_pct_gdp",
    "d_ln_exr",
    "inflation",
]

VIF_SPECS: dict[str, list[str]] = {
    "Spec1_Baseline":      BASE_CONTROLS,
    "Spec2_+Monetary":     BASE_CONTROLS + ["real_interest_rate"],
    "Spec3_+Institutional": BASE_CONTROLS + ["rq"],
    "Spec4_Full":          BASE_CONTROLS + ["real_interest_rate", "rq"],
}

# Variables to test for unit roots (all model variables)
UNIT_ROOT_VARS = [
    "fdi_pct_gdp",
    "bm_growth",
    "real_interest_rate",
    "gdp_growth",
    "ln_gdppc",
    "trade_pct_gdp",
    "d_ln_exr",
    "inflation",
    "rq",
    "rl",
]

DEPENDENT = "fdi_pct_gdp"
FULL_REGRESSORS = BASE_CONTROLS + ["real_interest_rate", "rq"]


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main() -> None:
    ensure_output_dirs()
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 — Diagnostics")
    print("=" * 60)

    if not PANEL_FILE.exists():
        print(f"[ERROR] Panel file not found: {PANEL_FILE}")
        print("        Run scripts/01_process_data.py first.")
        sys.exit(1)

    df = pd.read_csv(PANEL_FILE)
    print(f"[load] Panel: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"       Countries: {sorted(df['country_code'].unique().tolist())}")
    print(f"       Years: {df['year'].min()}–{df['year'].max()}")

    # ------------------------------------------------------------------
    # 2.1  Descriptive statistics
    # ------------------------------------------------------------------
    section("2.1  Descriptive Statistics")
    desc = descriptive_stats(df)
    print(desc.to_string(index=False))
    desc.to_csv(DIAGNOSTICS_DIR / "descriptive_stats.csv", index=False)

    # ------------------------------------------------------------------
    # 2.2  Correlation matrix
    # ------------------------------------------------------------------
    section("2.2  Correlation Matrix")
    corr = correlation_matrix(df)
    print(corr.round(3).to_string())
    corr.to_csv(DIAGNOSTICS_DIR / "correlation_matrix.csv")

    high_corr = flag_high_correlations(corr, threshold=0.7)
    if high_corr.empty:
        print("\n  No pairs with |r| ≥ 0.70.")
    else:
        print(f"\n  High-correlation pairs (|r| ≥ 0.70):")
        print(high_corr.to_string(index=False))
    high_corr.to_csv(DIAGNOSTICS_DIR / "high_correlations.csv", index=False)

    # ------------------------------------------------------------------
    # 2.3  VIF
    # ------------------------------------------------------------------
    section("2.3  Variance Inflation Factors")
    vif = vif_for_specs(df, VIF_SPECS)
    print(vif.to_string(index=False))
    vif.to_csv(DIAGNOSTICS_DIR / "vif_table.csv", index=False)

    # ------------------------------------------------------------------
    # 2.4  Panel unit root tests (LLC + IPS)
    # ------------------------------------------------------------------
    section("2.4  Panel Unit Root Tests (LLC + IPS)")
    unit_root = run_unit_root_tests(df, UNIT_ROOT_VARS)
    # Display compact summary
    display_cols = ["variable", "test", "statistic", "p_value", "decision"]
    display_cols = [c for c in display_cols if c in unit_root.columns]
    print(unit_root[display_cols].to_string(index=False))
    unit_root.to_csv(DIAGNOSTICS_DIR / "unit_root_tests.csv", index=False)

    # ------------------------------------------------------------------
    # 2.5  Modified Wald test (heteroskedasticity)
    # ------------------------------------------------------------------
    section("2.5  Modified Wald Test (Group-Wise Heteroskedasticity)")
    wald = modified_wald_test(df, DEPENDENT, FULL_REGRESSORS)
    for k, v in wald.items():
        print(f"  {k:<30}: {v}")
    wald_df = pd.DataFrame([wald])
    wald_df.to_csv(DIAGNOSTICS_DIR / "heteroskedasticity_tests.csv", index=False)

    # ------------------------------------------------------------------
    # 2.6  Wooldridge serial correlation test
    # ------------------------------------------------------------------
    section("2.6  Wooldridge Test (Serial Correlation)")
    wool = wooldridge_test(df, DEPENDENT, FULL_REGRESSORS)
    for k, v in wool.items():
        print(f"  {k:<30}: {v}")
    wool_df = pd.DataFrame([wool])
    wool_df.to_csv(DIAGNOSTICS_DIR / "serial_correlation_tests.csv", index=False)

    # ------------------------------------------------------------------
    # 2.7  Pesaran CD test (cross-sectional dependence)
    # ------------------------------------------------------------------
    section("2.7  Pesaran CD Test (Cross-Sectional Dependence)")
    cd = pesaran_cd_test(df, DEPENDENT, FULL_REGRESSORS)
    for k, v in cd.items():
        print(f"  {k:<30}: {v}")
    cd_df = pd.DataFrame([cd])
    cd_df.to_csv(DIAGNOSTICS_DIR / "cross_section_tests.csv", index=False)

    # ------------------------------------------------------------------
    # Export combined diagnostics workbook
    # ------------------------------------------------------------------
    section("Exporting diagnostics_summary.xlsx")
    report_path = DIAGNOSTICS_DIR / "diagnostics_summary.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        desc.to_excel(writer, sheet_name="descriptive_stats", index=False)
        corr.to_excel(writer, sheet_name="correlation_matrix")
        high_corr.to_excel(writer, sheet_name="high_correlations", index=False)
        vif.to_excel(writer, sheet_name="vif", index=False)
        unit_root.to_excel(writer, sheet_name="unit_root", index=False)
        wald_df.to_excel(writer, sheet_name="heteroskedasticity", index=False)
        wool_df.to_excel(writer, sheet_name="serial_correlation", index=False)
        cd_df.to_excel(writer, sheet_name="cross_section_dep", index=False)
    print(f"  Saved → {report_path}")

    print("\n" + "=" * 60)
    print("Phase 2 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
