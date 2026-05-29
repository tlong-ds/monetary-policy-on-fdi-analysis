"""src/data/loader.py

Raw file readers for:
  - Macro panel:  2000-2025-bm-rir-xr-gdp-infl.xlsx
  - WGI panel:    1996-2024-wgi-data.xlsx (sheets: va, pv, ge, rq, rl, cc)
  - HC/pop panel: 2000-2023-hc-pop.csv   (human capital index + population)

Each loader returns a tidy long-format DataFrame with columns:
  country_code (ISO3), year (int), and the relevant series column(s).
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    ASEAN_ISO3,
    ASEAN_NAME_MAP,
    HC_POP_FILE,
    MACRO_WORKBOOK,
    TIME_WINDOW,
    WGI_SHEETS,
    WGI_WORKBOOK,
)

# ---------------------------------------------------------------------------
# Column map for the macro workbook (instruct.md §3, Dataset 1)
# Only the columns we actually need — Broad money (% of GDP) is intentionally
# excluded; we use growth rate only.
# ---------------------------------------------------------------------------
MACRO_COLUMN_MAP: dict[str, str] = {
    "Country Name": "country",
    "Country Code": "country_code",
    "Time": "year",
    "Foreign direct investment, net inflows (% of GDP) [BX.KLT.DINV.WD.GD.ZS]": "fdi_pct_gdp",
    "Broad money growth (annual %) [FM.LBL.BMNY.ZG]": "bm_growth",
    "Real interest rate (%) [FR.INR.RINR]": "real_interest_rate",
    "Official exchange rate (LCU per US$, period average) [PA.NUS.FCRF]": "exr_lcu_usd",
    "Trade (% of GDP) [NE.TRD.GNFS.ZS]": "trade_pct_gdp",
    "GDP (current US$) [NY.GDP.MKTP.CD]": "gdp_current_usd",
    "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]": "gdp_growth",
    "Inflation, GDP deflator (annual %) [NY.GDP.DEFL.KD.ZG]": "inflation",
}

# WGI column we extract from each sheet
_WGI_ESTIMATE_COL = "Governance estimate (approx. -2.5 to +2.5)"
_WGI_COUNTRY_CODE_COL = "Economy (code)"
_WGI_YEAR_COL = "Year"


def load_macro_panel() -> pd.DataFrame:
    """Load and tidy the macro workbook.

    Returns a DataFrame with one row per (country_code, year) for ASEAN-10,
    filtered to TIME_WINDOW.
    """
    if not MACRO_WORKBOOK.exists():
        raise FileNotFoundError(f"Macro workbook not found: {MACRO_WORKBOOK}")

    raw = pd.read_excel(MACRO_WORKBOOK, sheet_name=0)

    # Validate required columns
    missing = [col for col in MACRO_COLUMN_MAP if col not in raw.columns]
    if missing:
        raise ValueError(
            f"Macro workbook is missing required columns:\n  " + "\n  ".join(missing)
        )

    df = raw[list(MACRO_COLUMN_MAP)].copy()
    df = df.rename(columns=MACRO_COLUMN_MAP)

    # Filter to ASEAN-10
    df = df[df["country_code"].isin(ASEAN_ISO3)].copy()

    # Standardise country names
    df["country"] = df["country"].replace(ASEAN_NAME_MAP)

    # Year as int
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Time window filter
    start, end = TIME_WINDOW
    df = df[df["year"].between(start, end)].copy()

    # Coerce numeric columns
    numeric_cols = [
        "fdi_pct_gdp",
        "bm_growth",
        "real_interest_rate",
        "exr_lcu_usd",
        "trade_pct_gdp",
        "gdp_current_usd",
        "gdp_growth",
        "inflation",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicate country-year rows
    df = df.drop_duplicates(subset=["country_code", "year"])

    return df.sort_values(["country_code", "year"]).reset_index(drop=True)


def load_wgi_sheet(indicator: str) -> pd.DataFrame:
    """Load one WGI sheet and return a tidy (country_code, year, <indicator>) DataFrame.

    Parameters
    ----------
    indicator : str
        One of: 'va', 'pv', 'ge', 'rq', 'rl', 'cc'
    """
    if indicator not in WGI_SHEETS:
        raise ValueError(f"Unknown WGI indicator '{indicator}'. Valid: {list(WGI_SHEETS)}")

    sheet = WGI_SHEETS[indicator]
    raw = pd.read_excel(WGI_WORKBOOK, sheet_name=sheet)

    required = [_WGI_COUNTRY_CODE_COL, _WGI_YEAR_COL, _WGI_ESTIMATE_COL]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"WGI sheet '{sheet}' missing columns: {missing}\n"
            f"Available: {raw.columns.tolist()}"
        )

    df = raw[[_WGI_COUNTRY_CODE_COL, _WGI_YEAR_COL, _WGI_ESTIMATE_COL]].copy()
    df = df.rename(
        columns={
            _WGI_COUNTRY_CODE_COL: "country_code",
            _WGI_YEAR_COL: "year",
            _WGI_ESTIMATE_COL: indicator,
        }
    )

    # Filter to ASEAN-10
    df = df[df["country_code"].isin(ASEAN_ISO3)].copy()

    # Year as int
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Time window filter
    start, end = TIME_WINDOW
    df = df[df["year"].between(start, end)].copy()

    # Coerce estimate
    df[indicator] = pd.to_numeric(df[indicator], errors="coerce")

    # Drop duplicates
    df = df.drop_duplicates(subset=["country_code", "year"])

    return df.sort_values(["country_code", "year"]).reset_index(drop=True)


def load_all_wgi() -> pd.DataFrame:
    """Load all 6 WGI sheets and merge into one wide DataFrame.

    Returns columns: country_code, year, va, pv, ge, rq, rl, cc
    """
    frames: list[pd.DataFrame] = []
    for indicator in WGI_SHEETS:
        frames.append(load_wgi_sheet(indicator))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["country_code", "year"], how="outer")

    return merged.sort_values(["country_code", "year"]).reset_index(drop=True)


def load_hc_pop() -> pd.DataFrame:
    """Load the HC/pop CSV and return a tidy long-format DataFrame.

    Source: 2000-2023-hc-pop.csv
    Wide format (one column per year); Variable code is 'hc' or 'pop'.

    Returns columns:
      country_code, year, population (total persons), hc (human capital index)
    """
    if not HC_POP_FILE.exists():
        raise FileNotFoundError(f"HC/pop file not found: {HC_POP_FILE}")

    raw = pd.read_csv(HC_POP_FILE)

    required = {"ISO code", "Country", "Variable code"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"HC/pop file missing columns: {sorted(missing)}")

    # Year columns are the numeric-named ones
    year_cols = [c for c in raw.columns if str(c).isdigit()]
    if not year_cols:
        raise ValueError("HC/pop file has no year columns.")

    long = raw.melt(
        id_vars=["ISO code", "Country", "Variable code"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    ).rename(columns={"ISO code": "country_code", "Country": "country", "Variable code": "variable"})

    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long = long.dropna(subset=["year"])
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # Filter ASEAN-10 and time window
    long = long[long["country_code"].isin(ASEAN_ISO3)].copy()
    start, end = TIME_WINDOW
    long = long[long["year"].between(start, end)].copy()

    # Pivot to wide: one column per variable code
    wide = (
        long.pivot_table(
            index=["country_code", "country", "year"],
            columns="variable",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    # Population is in millions — convert to total persons
    if "pop" in wide.columns:
        wide["population"] = wide["pop"] * 1_000_000
        wide = wide.drop(columns=["pop"])
    else:
        wide["population"] = float("nan")

    # Rename hc column for clarity
    if "hc" in wide.columns:
        wide = wide.rename(columns={"hc": "hc_index"})
    else:
        wide["hc_index"] = float("nan")

    return wide[["country_code", "country", "year", "population", "hc_index"]].sort_values(
        ["country_code", "year"]
    ).reset_index(drop=True)
