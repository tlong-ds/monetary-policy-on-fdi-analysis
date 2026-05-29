"""scripts/01_process_data.py

Phase 1 runner: load raw data → merge → transform → clean → export.

Usage:
    python scripts/01_process_data.py

Outputs:
    data/processed/panel.csv
    data/processed/variable_dict.csv
    outputs/preprocessing_report.xlsx
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when called from any working directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.merge import build_panel, save_panel


def main() -> None:
    print("=" * 60)
    print("Phase 1 — Data Processing")
    print("=" * 60)

    panel, artifacts = build_panel()

    print("\n[summary] Panel shape:", panel.shape)
    print("[summary] Countries:", sorted(panel["country_code"].unique().tolist()))
    print("[summary] Years:", panel["year"].min(), "–", panel["year"].max())

    print("\n[summary] Coverage (% non-missing):")
    cov = artifacts["coverage"].set_index("variable")
    for var in cov.index:
        share_missing = cov.loc[var, "missing_share"]
        flag = " ← high missing" if share_missing >= 0.05 else ""
        print(f"  {var:<25} {(1 - share_missing)*100:5.1f}% complete{flag}")

    save_panel(panel, artifacts)

    print("\n" + "=" * 60)
    print("Phase 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
