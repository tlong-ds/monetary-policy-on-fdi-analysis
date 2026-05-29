"""scripts/04_run_robustness.py

Phase 4 runner: load panel → run all robustness checks → export tables.

Usage:
    python scripts/04_run_robustness.py

Requires:
    data/processed/panel.csv  (produced by scripts/01_process_data.py)

Outputs (all under outputs/tables/):
    robustness_r1_rl.xlsx / .tex
    robustness_r2_lagged_bm.xlsx / .tex
    robustness_r3_no_sgp.xlsx / .tex
    robustness_r4_crisis.xlsx / .tex
    robustness_r5_gmm.xlsx / .tex         (re-estimated here for comparison)
    robustness_r6_composite_wgi.xlsx / .tex
    robustness_comparison.xlsx / .tex     ← all checks side by side
    robustness_bm_summary.csv             ← BM growth coefficient stability
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import PANEL_FILE, TABLES_DIR, ensure_output_dirs
from src.estimation.specs import BASE_CONTROLS, SPEC_MAP
from src.estimation.static import estimate_static_spec
from src.output.tables import (
    build_regression_table,
    export_excel,
    export_latex,
    save_regression_tables,
)
from src.robustness.checks import (
    ROBUSTNESS_CHECKS,
    ROBUSTNESS_CHECK_MAP,
    build_bm_summary,
    build_comparison_table,
    run_all_robustness_checks,
)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_check_result(rd: dict) -> None:
    """Print compact summary for one robustness check."""
    if "error" in rd:
        print(f"  [ERROR] {rd['error']}")
        return

    est = rd.get("estimator_label", "")
    obs = rd.get("n_obs", rd.get("obs", ""))
    print(f"  Estimator  : {est}")
    print(f"  N obs      : {obs}")

    res = rd.get("preferred_result")
    if res is not None:
        params = res.params; pvals = res.pvalues; ses = res.std_errors
        # Show only main variables, skip year dummies
        for var in params.index:
            if var.startswith("yr_") or var in ("Intercept", "const"):
                continue
            s = "***" if pvals[var] < 0.01 else ("**" if pvals[var] < 0.05 else ("*" if pvals[var] < 0.10 else ""))
            print(f"  {var:<28} {params[var]:>9.4f}{s}  SE={ses[var]:.4f}  p={pvals[var]:.4f}")
    elif "params" in rd:
        params = rd["params"]; pvals = rd["p_values"]; ses = rd["std_errors"]
        for var in params.index:
            if var.startswith("yr_") or var in ("Intercept", "const"):
                continue
            s = "***" if pvals[var] < 0.01 else ("**" if pvals[var] < 0.05 else ("*" if pvals[var] < 0.10 else ""))
            print(f"  {var:<28} {params[var]:>9.4f}{s}  SE={ses[var]:.4f}  p={pvals[var]:.4f}")
        h = rd.get("hansen", {}); a1 = rd.get("ar1", {}); a2 = rd.get("ar2", {})
        print(f"  Hansen J p  : {h.get('p_value', 'n/a')}  |  AR(1) p: {a1.get('p_value', 'n/a')}  |  AR(2) p: {a2.get('p_value', 'n/a')}")
        print(f"  N instruments: {rd.get('n_instruments', 'n/a')}")


def main() -> None:
    ensure_output_dirs()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4 — Robustness Checks")
    print("=" * 60)

    if not PANEL_FILE.exists():
        print(f"[ERROR] Panel not found: {PANEL_FILE}")
        print("        Run scripts/01_process_data.py first.")
        sys.exit(1)

    df = pd.read_csv(PANEL_FILE)
    print(f"[load] {len(df)} rows, {df['country_code'].nunique()} countries, "
          f"{df['year'].min()}–{df['year'].max()}")

    # ── Baseline: Spec 4 (Full model) ───────────────────────────────────
    section("Baseline — Spec 4 (Full Model)")
    spec4 = SPEC_MAP["Spec 4"]
    print(f"  Estimating {spec4.label}: {spec4.name} ...")
    baseline = estimate_static_spec(df, spec4)
    for line in baseline.get("selection_log", []):
        print(f"  {line}")
    print()
    print_check_result(baseline)
    baseline["check_label"] = "Baseline"
    baseline["check_name"] = "Spec 4 (Full)"
    baseline["n_countries_used"] = int(df["country_code"].nunique())

    # ── Run all 6 robustness checks ──────────────────────────────────────
    section("Running Robustness Checks R1–R6")
    robustness_results = run_all_robustness_checks(df)

    # Print each result
    for rd in robustness_results:
        print(f"\n── {rd['check_label']} — {rd['check_name']} ──")
        print(f"  {rd['check_description']}")
        print_check_result(rd)

    # ── BM growth coefficient stability summary ───────────────────────────
    section("BM Growth Coefficient Stability")
    bm_summary = build_bm_summary(baseline, robustness_results)
    print(bm_summary.to_string(index=False))
    bm_path = TABLES_DIR / "robustness_bm_summary.csv"
    bm_summary.to_csv(bm_path, index=False)
    print(f"\n  → {bm_path}")

    # ── Export individual robustness tables ───────────────────────────────
    section("Exporting Individual Robustness Tables")

    stem_map = {
        "R1": ("robustness_r1_rl",            "Robustness R1: Rule of Law (RL) Institutional Proxy"),
        "R2": ("robustness_r2_lagged_bm",      "Robustness R2: Lagged Broad Money Growth"),
        "R3": ("robustness_r3_no_sgp",         "Robustness R3: Excluding Singapore"),
        "R4": ("robustness_r4_crisis",         "Robustness R4: Excluding Crisis Years (2008–09, 2020–21)"),
        "R5": ("robustness_r5_gmm",            "Robustness R5: System GMM (Dynamic Panel)"),
        "R6": ("robustness_r6_composite_wgi",  "Robustness R6: Composite WGI Index"),
    }
    for rd in robustness_results:
        label = rd["check_label"]
        if label in stem_map:
            stem, caption = stem_map[label]
            # Build a two-column table: Baseline vs this check
            save_regression_tables(
                [baseline, rd],
                TABLES_DIR,
                stem=stem,
                caption=caption,
            )

    # ── Export combined comparison table ─────────────────────────────────
    section("Exporting Robustness Comparison Table")
    comp_table = build_comparison_table(baseline, robustness_results)

    comp_xlsx = TABLES_DIR / "robustness_comparison.xlsx"
    comp_tex  = TABLES_DIR / "robustness_comparison.tex"

    export_excel(
        comp_table.rename(columns={"row_label": "Variable"}),
        comp_xlsx,
        sheet_name="Robustness Comparison",
        note="Note: *p<0.10, **p<0.05, ***p<0.01. SE in parentheses. "
             "All static models use cluster-robust SE at country level. "
             "Baseline = Spec 4 (Full model). "
             "R1=RL institutional proxy, R2=Lagged BM, R3=Excl. Singapore, "
             "R4=Excl. Crisis Years, R5=System GMM, R6=Composite WGI.",
    )
    export_latex(
        comp_table.rename(columns={"row_label": "Variable"}),
        comp_tex,
        caption="FDI Determinants: Robustness Checks",
        label="tab:robustness",
        note="\\textit{Note:} $^{*}$p$<$0.10, $^{**}$p$<$0.05, $^{***}$p$<$0.01. "
             "SE in parentheses. Cluster-robust SE at country level (static models). "
             "R1=RL institutional proxy, R2=Lagged BM, R3=Excl. Singapore, "
             "R4=Excl. Crisis Years, R5=System GMM, R6=Composite WGI.",
    )
    print(f"  → {comp_xlsx}")
    print(f"  → {comp_tex}")

    # ── GMM diagnostics note ──────────────────────────────────────────────
    gmm_rd = next((r for r in robustness_results if r["check_label"] == "R5"), None)
    if gmm_rd:
        h = gmm_rd.get("hansen", {}); a2 = gmm_rd.get("ar2", {})
        section("System GMM Diagnostics Note")
        print(f"  Hansen J p-value : {h.get('p_value', 'n/a')}")
        print(f"  AR(2) p-value    : {a2.get('p_value', 'n/a')}")
        print(f"  N instruments    : {gmm_rd.get('n_instruments', 'n/a')}")
        print(f"  N groups         : {gmm_rd.get('n_entities', 'n/a')}")
        n_inst = gmm_rd.get("n_instruments", 0)
        n_grp  = gmm_rd.get("n_entities", 1)
        if isinstance(n_inst, int) and n_inst > n_grp:
            print(f"\n  ⚠  Instrument proliferation detected "
                  f"(N_instruments={n_inst} > N_groups={n_grp}).")
            print(f"     Hansen p≈1.0 and inflated AR stats are expected artefacts.")
            print(f"     Interpretation: GMM coefficient magnitudes should be treated "
                  f"with caution; static FE results (Specs 2–4) are preferred for inference.")

    print("\n" + "=" * 60)
    print("Phase 4 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
