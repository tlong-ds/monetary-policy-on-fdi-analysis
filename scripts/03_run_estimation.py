"""scripts/03_run_estimation.py

Phase 3 runner: load panel → estimate Specs 1–5 → export regression tables.

Usage:
    python scripts/03_run_estimation.py

Requires:
    data/processed/panel.csv  (produced by scripts/01_process_data.py)

Outputs (all under outputs/tables/):
    regression_specs1_4.xlsx / .tex   — static specs 1–4
    regression_spec5_gmm.xlsx / .tex  — dynamic Spec 5
    regression_all_specs.xlsx / .tex  — all five specs combined
    estimator_selection_log.csv       — BP-LM and Hausman decisions
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import PANEL_FILE, TABLES_DIR, ensure_output_dirs
from src.estimation.specs import SPECS, SPEC_MAP
from src.estimation.static import run_static_specs
from src.estimation.dynamic import two_step_sys_gmm
from src.output.tables import (
    build_regression_table,
    build_selection_log,
    export_excel,
    export_latex,
    save_regression_tables,
)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_result_summary(rd: dict) -> None:
    """Print a compact result summary to stdout."""
    if "error" in rd:
        print(f"  [ERROR] {rd['error']}")
        return
    est = rd.get("estimator_label", "")
    obs = rd.get("n_obs", rd.get("obs", ""))
    print(f"  Estimator : {est}")
    print(f"  N obs     : {obs}")

    res = rd.get("preferred_result")
    if res is not None:
        params = res.params
        pvals  = res.pvalues
        print(f"  {'Variable':<25} {'Coef':>10}  {'p-value':>10}")
        for var in params.index:
            stars = "***" if pvals[var] < 0.01 else ("**" if pvals[var] < 0.05 else ("*" if pvals[var] < 0.10 else ""))
            print(f"  {var:<25} {params[var]:>10.4f}{stars}  {pvals[var]:>10.4f}")
    elif "params" in rd:
        print(f"  {'Variable':<25} {'Coef':>10}  {'p-value':>10}")
        for var in rd["params"].index:
            if var.startswith("yr_"):
                continue
            p = rd["p_values"][var]
            c = rd["params"][var]
            stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            print(f"  {var:<25} {c:>10.4f}{stars}  {p:>10.4f}")
        h = rd.get("hansen", {})
        a1 = rd.get("ar1", {}); a2 = rd.get("ar2", {})
        print(f"\n  Hansen J p-value : {h.get('p_value', 'n/a')}")
        print(f"  AR(1) p-value    : {a1.get('p_value', 'n/a')}")
        print(f"  AR(2) p-value    : {a2.get('p_value', 'n/a')}")
        print(f"  N instruments    : {rd.get('n_instruments', 'n/a')}")


def main() -> None:
    ensure_output_dirs()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3 — Estimation")
    print("=" * 60)

    if not PANEL_FILE.exists():
        print(f"[ERROR] Panel not found: {PANEL_FILE}")
        print("        Run scripts/01_process_data.py first.")
        sys.exit(1)

    df = pd.read_csv(PANEL_FILE)
    print(f"[load] {len(df)} rows, {df['country_code'].nunique()} countries, "
          f"{df['year'].min()}–{df['year'].max()}")

    # ── Specs 1–4: Static estimators ────────────────────────────────────
    section("Specs 1–4: Static Panel Estimation")
    static_specs = [s for s in SPECS if s.estimator == "static"]
    static_results = run_static_specs(df, static_specs)

    for rd in static_results:
        print(f"\n── {rd.get('spec_label')} — {rd.get('spec_name')} ──")
        print_result_summary(rd)

    # ── Spec 5: System GMM ───────────────────────────────────────────────
    section("Spec 5: Dynamic Panel — System GMM")
    gmm_spec = SPEC_MAP["Spec 5"]
    print(f"  Estimating {gmm_spec.label}: {gmm_spec.name} ...")
    gmm_result = two_step_sys_gmm(df, gmm_spec)
    print(f"\n── {gmm_result.get('spec_label')} — {gmm_result.get('spec_name')} ──")
    print_result_summary(gmm_result)

    all_results = static_results + [gmm_result]

    # ── Export tables ────────────────────────────────────────────────────
    section("Exporting Regression Tables")

    # Static specs 1–4
    save_regression_tables(
        static_results,
        TABLES_DIR,
        stem="regression_specs1_4",
        caption="FDI Determinants: Static Panel Estimates (Specs 1–4)",
    )

    # Dynamic Spec 5
    save_regression_tables(
        [gmm_result],
        TABLES_DIR,
        stem="regression_spec5_gmm",
        caption="FDI Determinants: System GMM Estimate (Spec 5)",
    )

    # All five specs
    save_regression_tables(
        all_results,
        TABLES_DIR,
        stem="regression_all_specs",
        caption="FDI Determinants: All Specifications",
    )

    # Estimator selection log
    selection_log = build_selection_log(static_results)
    log_path = TABLES_DIR / "estimator_selection_log.csv"
    selection_log.to_csv(log_path, index=False)
    print(f"  → {log_path}")

    print("\n── Estimator Selection Summary ──")
    for _, row in selection_log.iterrows():
        print(f"  {row['Spec']:<8} {row['Name']:<18} → {row['Preferred']}")

    print("\n" + "=" * 60)
    print("Phase 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
