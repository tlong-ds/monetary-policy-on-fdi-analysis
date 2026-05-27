from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root from either the repo or notebooks directory."""
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "data" / "raw").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("Could not locate monetary_policy_fdi_analysis project root.")


ROOT = find_project_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"
MODEL_ARTIFACTS_DIR = OUTPUTS_DIR / "model_artifacts"

PRIMARY_WORKBOOK = RAW_DIR / "asean_fdi_monetary_policy_clean_merged.xlsx"
WDI_WORKBOOK = RAW_DIR / "P_Data_Extract_From_World_Development_Indicators.xlsx"
CURRENT_MACRO_PANEL_WORKBOOK = RAW_DIR / "2000-2025-bm-ir-infl.xlsx"
CURRENT_HC_POP_FILE = RAW_DIR / "2000-2023-hc-pop.csv"
CURRENT_ADDITIONAL_SERIES_FILE = RAW_DIR / "2000-2025-lending-deposit-tourism-arrivals.csv"
MODEL_SELECTION_WORKBOOK = ROOT / "model_selection_asean_fdi.xlsx"
PROCESSED_PANEL_FILE = PROCESSED_DIR / "clean_panel.csv"

ASEAN_COUNTRIES = [
    "Brunei Darussalam",
    "Cambodia",
    "Indonesia",
    "Lao PDR",
    "Malaysia",
    "Myanmar",
    "Philippines",
    "Singapore",
    "Thailand",
    "Viet Nam",
]
PANEL_COUNTRY_ORDER = [
    ("BRN", "Brunei Darussalam"),
    ("KHM", "Cambodia"),
    ("IDN", "Indonesia"),
    ("LAO", "Lao PDR"),
    ("MYS", "Malaysia"),
    ("MMR", "Myanmar"),
    ("PHL", "Philippines"),
    ("SGP", "Singapore"),
    ("THA", "Thailand"),
    ("TLS", "Timor-Leste"),
    ("VNM", "Viet Nam"),
]
TIME_WINDOW = (2000, 2023)
WINSOR_BOUNDS = (0.01, 0.99)


def ensure_output_dirs() -> None:
    for path in [PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, DIAGNOSTICS_DIR, MODEL_ARTIFACTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
