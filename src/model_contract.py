from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd


WORKBOOK_VARIABLE_MAP = {
    "fdi_gdp": "fdi_pct_gdp",
    "bm_gdp": "broad_money_pct_gdp",
    "ir_deposit": "deposit_interest_rate_pct",
    "infl": "inflation_gdp_deflator_pct",
    "trade_gdp": "trade_pct_gdp",
    "ln_gdppc": "ln_gdppc",
    "xr_dep": "xr_dep_pct",
    "ln_xr": "xr_dep_pct",
    "ln_pop": "ln_population_total",
    "ln_tourism": "ln_tourism_arrivals",
    "hc": "hc_human_capital_index",
    "ir_real": "real_interest_rate_pct",
    "ir_lending": "lending_interest_rate_pct",
}

MODEL_ORDER = [
    "M1_baseline_liquidity",
    "M2_main_monetary_policy",
    "M3_lagged_main_model",
    "M4_real_interest_robustness",
    "M5_lending_rate_robustness",
    "M6a_tourism_robustness_from_M2",
    "M6b_tourism_robustness_from_M4",
    "M7a_human_capital_robustness_from_M2",
    "M7b_human_capital_robustness_from_M4",
]


@dataclass(frozen=True)
class ModelBlueprint:
    model_id: str
    workbook_code: str
    workbook_label: str
    workbook_variables: list[str]
    model_tier: str
    channel: str
    thesis_role: str
    monetary_proxy: str | None = None
    lagged_proxy_only: bool = False
    lagged: bool = False
    exclude_countries: list[str] | None = None


MODEL_BLUEPRINTS = [
    ModelBlueprint(
        model_id="M1_baseline_liquidity",
        workbook_code="M1",
        workbook_label="M1 - Baseline liquidity",
        workbook_variables=["bm_gdp", "infl", "trade_gdp", "ln_gdppc", "xr_dep"],
        model_tier="headline",
        channel="liquidity / financial depth",
        thesis_role="Main liquidity-channel model",
        monetary_proxy="bm_gdp",
        lagged_proxy_only=False,
        lagged=False,
    ),
    ModelBlueprint(
        model_id="M2_main_monetary_policy",
        workbook_code="M2",
        workbook_label="M2 - Deposit-rate channel",
        workbook_variables=["bm_gdp", "ir_deposit", "infl", "trade_gdp", "ln_gdppc", "xr_dep"],
        model_tier="headline",
        channel="deposit-rate cost of capital",
        thesis_role="Main deposit-rate-channel model",
        monetary_proxy="ir_deposit",
        lagged_proxy_only=False,
        lagged=False,
        exclude_countries=["Cambodia", "Lao PDR"],
    ),
    ModelBlueprint(
        model_id="M3_lagged_main_model",
        workbook_code="M3",
        workbook_label="M3 - Lagged deposit-rate robustness",
        workbook_variables=["bm_gdp", "ir_deposit", "infl", "trade_gdp", "ln_gdppc", "xr_dep"],
        model_tier="robustness",
        channel="lagged deposit-rate cost of capital",
        thesis_role="Lagged main monetary policy model with lagged controls",
        monetary_proxy="ir_deposit",
        lagged_proxy_only=False,
        lagged=True,
        exclude_countries=["Cambodia", "Lao PDR"],
    ),
    ModelBlueprint(
        model_id="M4_real_interest_robustness",
        workbook_code="M4",
        workbook_label="M4 - Real-rate channel",
        workbook_variables=["bm_gdp", "ir_real", "trade_gdp", "ln_gdppc", "xr_dep"],
        model_tier="headline",
        channel="real-rate cost of capital",
        thesis_role="Main real-rate-channel model; inflation-added variant is appendix only",
        monetary_proxy="ir_real",
        lagged_proxy_only=False,
        lagged=False,
        exclude_countries=["Cambodia", "Lao PDR"],
    ),
    ModelBlueprint(
        model_id="M5_lending_rate_robustness",
        workbook_code="M5",
        workbook_label="M5 - Lending rate robustness",
        workbook_variables=["bm_gdp", "ir_lending", "infl", "trade_gdp", "ln_gdppc", "xr_dep"],
        model_tier="robustness",
        channel="lending-rate cost of capital",
        thesis_role="Robustness only because lending-rate coverage is structurally sparse",
        monetary_proxy="ir_lending",
        lagged_proxy_only=False,
        lagged=False,
        exclude_countries=["Cambodia", "Lao PDR"],
    ),
    ModelBlueprint(
        model_id="M6a_tourism_robustness_from_M2",
        workbook_code="M6",
        workbook_label="M6a - Tourism robustness from M2",
        workbook_variables=["bm_gdp", "ir_deposit", "infl", "trade_gdp", "ln_gdppc", "xr_dep", "ln_tourism"],
        model_tier="appendix",
        channel="tourism sensitivity from deposit-rate channel",
        thesis_role="Appendix sensitivity; not eligible as headline evidence",
        monetary_proxy="ir_deposit",
        lagged_proxy_only=False,
        lagged=False,
        exclude_countries=["Cambodia", "Lao PDR"],
    ),
    ModelBlueprint(
        model_id="M6b_tourism_robustness_from_M4",
        workbook_code="M6",
        workbook_label="M6b - Tourism robustness from M4",
        workbook_variables=["bm_gdp", "ir_real", "trade_gdp", "ln_gdppc", "xr_dep", "ln_tourism"],
        model_tier="appendix",
        channel="tourism sensitivity from real-rate channel",
        thesis_role="Appendix sensitivity; not eligible as headline evidence",
        monetary_proxy="ir_real",
        lagged_proxy_only=False,
        lagged=False,
        exclude_countries=["Cambodia", "Lao PDR"],
    ),
    ModelBlueprint(
        model_id="M7a_human_capital_robustness_from_M2",
        workbook_code="M7",
        workbook_label="M7a - Human capital sensitivity from M2",
        workbook_variables=["bm_gdp", "ir_deposit", "infl", "trade_gdp", "ln_gdppc", "xr_dep", "hc"],
        model_tier="appendix",
        channel="human-capital sensitivity from deposit-rate channel",
        thesis_role="Appendix sensitivity; human capital is not headline evidence",
        monetary_proxy="ir_deposit",
        lagged_proxy_only=False,
        lagged=False,
        exclude_countries=["Cambodia", "Lao PDR"],
    ),
    ModelBlueprint(
        model_id="M7b_human_capital_robustness_from_M4",
        workbook_code="M7",
        workbook_label="M7b - Human capital sensitivity from M4",
        workbook_variables=["bm_gdp", "ir_real", "trade_gdp", "ln_gdppc", "xr_dep", "hc"],
        model_tier="appendix",
        channel="human-capital sensitivity from real-rate channel",
        thesis_role="Appendix sensitivity; human capital is not headline evidence",
        monetary_proxy="ir_real",
        lagged_proxy_only=False,
        lagged=False,
        exclude_countries=["Cambodia", "Lao PDR"],
    ),
]


def _normalize_column_name(column: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    return normalized


def _read_sheet(path: Path | str, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet_name, header=2)
    frame = frame.dropna(how="all").copy()
    frame.columns = [_normalize_column_name(column) for column in frame.columns]
    unnamed_columns = [column for column in frame.columns if column.startswith("unnamed")]
    if unnamed_columns:
        frame = frame.drop(columns=unnamed_columns)
    return frame.reset_index(drop=True)


def load_workbook_tables(path: Path | str) -> dict[str, pd.DataFrame]:
    variable_selection = _read_sheet(path, "Variable selection").rename(
        columns={
            "role": "role",
            "original_variable_in_file": "original_variable_in_file",
            "suggested_variable_name": "suggested_variable_name",
            "decision": "decision",
            "reason": "reason",
            "expected_sign": "expected_sign",
        }
    )
    model_specs = _read_sheet(path, "Model specs").rename(
        columns={
            "model": "model",
            "purpose": "purpose",
            "equation_variables": "equation_variables",
            "estimated_complete_observations": "estimated_complete_observations",
            "recommendation": "recommendation",
        }
    )
    notes = _read_sheet(path, "Notes").rename(columns={"item": "item", "note": "note"})

    if "decision" in variable_selection.columns:
        variable_selection["decision_group"] = (
            variable_selection["decision"]
            .fillna("")
            .str.split("-", n=1)
            .str[0]
            .str.strip()
            .str.lower()
        )

    if "model" in model_specs.columns:
        model_specs["workbook_code"] = model_specs["model"].astype(str).str.extract(r"^(M\d+)")

    return {
        "variable_selection": variable_selection,
        "model_specs": model_specs,
        "notes": notes,
    }


def create_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    derived = frame.copy()
    if "ln_population_total" not in derived.columns and "population_total" in derived.columns:
        derived["ln_population_total"] = np.where(
            derived["population_total"] > 0,
            np.log(derived["population_total"]),
            np.nan,
        )
    if "ln_tourism_arrivals" not in derived.columns and "tourism_arrivals" in derived.columns:
        derived["ln_tourism_arrivals"] = np.where(
            derived["tourism_arrivals"] > 0,
            np.log(derived["tourism_arrivals"]),
            np.nan,
        )
    return derived.sort_values(["country", "year"]).reset_index(drop=True)


def add_country_lags(frame: pd.DataFrame, columns: list[str], lag: int = 1) -> pd.DataFrame:
    lagged = frame.copy()
    for column in columns:
        lagged[f"{column}_lag{lag}"] = lagged.groupby("country")[column].shift(lag)
    return lagged


def map_workbook_variables(workbook_variables: list[str], lagged: bool = False) -> list[str]:
    mapped = [WORKBOOK_VARIABLE_MAP[variable] for variable in workbook_variables]
    if lagged:
        return [f"{column}_lag1" for column in mapped]
    return mapped


def map_blueprint_variables(blueprint: ModelBlueprint, lagged: bool | None = None) -> list[str]:
    use_lagged = blueprint.lagged if lagged is None else lagged
    mapped = [WORKBOOK_VARIABLE_MAP[variable] for variable in blueprint.workbook_variables]
    if not use_lagged:
        return mapped
    if not blueprint.lagged_proxy_only:
        return [f"{column}_lag1" for column in mapped]
    if blueprint.monetary_proxy is None:
        raise ValueError(f"{blueprint.model_id} requested lagged_proxy_only without monetary_proxy.")
    proxy_column = WORKBOOK_VARIABLE_MAP[blueprint.monetary_proxy]
    return [f"{proxy_column}_lag1" if column == proxy_column else column for column in mapped]


def build_workbook_model_catalog(
    workbook_path: Path | str,
    panel_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    workbook_tables = load_workbook_tables(workbook_path)
    model_specs = workbook_tables["model_specs"]
    model_specs_lookup = (
        model_specs.set_index("workbook_code")
        if "workbook_code" in model_specs.columns
        else pd.DataFrame().set_index(pd.Index([]))
    )

    rows = []
    for blueprint in MODEL_BLUEPRINTS:
        base_regressors = map_blueprint_variables(blueprint, lagged=False)
        mapped_regressors = map_blueprint_variables(blueprint, lagged=blueprint.lagged)
        missing_base_regressors = [column for column in base_regressors if column not in panel_columns]
        spec_row = (
            model_specs_lookup.loc[blueprint.workbook_code]
            if blueprint.workbook_code in model_specs_lookup.index
            else pd.Series(dtype=object)
        )
        status = "estimated" if not missing_base_regressors else "skipped_missing_variables"
        reason = ""
        if missing_base_regressors:
            reason = "Missing cleaned-panel columns: " + ", ".join(missing_base_regressors)

        rows.append(
            {
                "model_id": blueprint.model_id,
                "model_order": MODEL_ORDER.index(blueprint.model_id) + 1,
                "workbook_code": blueprint.workbook_code,
                "workbook_model": blueprint.workbook_label,
                "purpose": spec_row.get("purpose", ""),
                "equation_variables": spec_row.get("equation_variables", ""),
                "estimated_complete_observations": spec_row.get(
                    "estimated_complete_observations",
                    np.nan,
                ),
                "recommendation": spec_row.get("recommendation", ""),
                "model_tier": blueprint.model_tier,
                "channel": blueprint.channel,
                "thesis_role": blueprint.thesis_role,
                "monetary_proxy": WORKBOOK_VARIABLE_MAP.get(blueprint.monetary_proxy, blueprint.monetary_proxy),
                "headline_eligible": blueprint.model_tier == "headline",
                "appendix_only": blueprint.model_tier == "appendix",
                "lagged_proxy_only": blueprint.lagged_proxy_only,
                "workbook_variables": ", ".join(blueprint.workbook_variables),
                "base_regressors": ", ".join(base_regressors),
                "mapped_regressors": ", ".join(mapped_regressors),
                "lagged_model": blueprint.lagged,
                "exclude_countries": ", ".join(blueprint.exclude_countries) if blueprint.exclude_countries else "",
                "status": status,
                "missing_reason": reason,
            }
        )

    catalog = pd.DataFrame(rows).sort_values("model_order").reset_index(drop=True)
    return catalog, workbook_tables


def parse_regressor_string(value: str) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_model_frame(frame: pd.DataFrame, dependent: str, regressors: list[str], exclude_countries: list[str] | None = None) -> pd.DataFrame:
    keep_columns = ["country", "year", dependent, *regressors]
    model_frame = frame[keep_columns].copy()
    if exclude_countries:
        model_frame = model_frame[~model_frame["country"].isin(exclude_countries)]
    model_frame = model_frame.dropna()
    return model_frame.sort_values(["country", "year"]).set_index(["country", "year"])


def choose_preferred_estimator(hausman_p_value: float) -> str:
    return "fixed_effects_driscoll_kraay"


def estimator_label(estimator: str) -> str:
    labels = {
        "pooled_ols": "Pooled OLS",
        "fixed_effects": "Fixed effects",
        "fixed_effects_driscoll_kraay": "Two-way FE, Driscoll-Kraay SE",
        "random_effects": "Random effects",
    }
    return labels.get(estimator, estimator)


def safe_sheet_name(name: str, prefix: str = "") -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", f"{prefix}{name}").strip("_")
    if not clean:
        return "sheet"
    if len(clean) <= 31:
        return clean
    suffix = hashlib.md5(clean.encode("utf-8")).hexdigest()[:7]
    head = clean[: 31 - len(suffix) - 1].rstrip("_")
    return f"{head}_{suffix}"
