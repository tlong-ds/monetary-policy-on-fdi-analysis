from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root by looking for data/raw/."""
    current = (start or Path(__file__).resolve().parent).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "data" / "raw").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root (expected data/raw/ directory).")


ROOT = find_project_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Raw input files (instruct.md §3)
MACRO_WORKBOOK = RAW_DIR / "2000-2025-bm-rir-xr-gdp-infl.xlsx"
WGI_WORKBOOK = RAW_DIR / "1996-2024-wgi-data.xlsx"
HC_POP_FILE = RAW_DIR / "2000-2023-hc-pop.csv"

# Processed output
PANEL_FILE = PROCESSED_DIR / "panel.csv"
VARIABLE_DICT_FILE = PROCESSED_DIR / "variable_dict.csv"

# WGI sheet names in 1996-2024-wgi-data.xlsx
WGI_SHEETS: dict[str, str] = {
    "va": "va",   # Voice & Accountability
    "pv": "pv",   # Political Stability (note: sheet is 'pv', not 'ps')
    "ge": "ge",   # Government Effectiveness
    "rq": "rq",   # Regulatory Quality  ← baseline institutional proxy
    "rl": "rl",   # Rule of Law         ← robustness institutional proxy
    "cc": "cc",   # Control of Corruption
}

# ASEAN-10 (instruct.md §2 — Timor-Leste excluded)
ASEAN_ISO3: list[str] = [
    "BRN",  # Brunei
    "KHM",  # Cambodia
    "IDN",  # Indonesia
    "LAO",  # Lao PDR
    "MYS",  # Malaysia
    "MMR",  # Myanmar
    "PHL",  # Philippines
    "SGP",  # Singapore
    "THA",  # Thailand
    "VNM",  # Viet Nam
]

ASEAN_NAME_MAP: dict[str, str] = {
    # Normalise country names that differ across datasets
    "Lao People's DR": "Lao PDR",
    "Lao PDR": "Lao PDR",
    "Vietnam": "Viet Nam",
    "Viet Nam": "Viet Nam",
    "Brunei Darussalam": "Brunei Darussalam",
    "Myanmar": "Myanmar",
}

TIME_WINDOW: tuple[int, int] = (2000, 2023)

WINSOR_BOUNDS: tuple[float, float] = (0.01, 0.99)


def ensure_output_dirs() -> None:
    for path in [PROCESSED_DIR, OUTPUTS_DIR, DIAGNOSTICS_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
