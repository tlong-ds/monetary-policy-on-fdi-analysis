from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd


WORKBOOK_VARIABLE_MAP: dict[str, str] = {
    "fdi_gdp": "fdi_pct_gdp",
    "bm_gdp": "broad_money_growth_pct",
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

WORKBOOK_CODE_TO_MODEL_ID = {
    # Keep stable model IDs for exported filenames/analysis, but derive regressors from the workbook itself.
    "M1": "M1_baseline_liquidity",
    "M2": "M2_main_monetary_policy",
    "M3": "M3_lagged_main_model",
    "M4": "M4_real_interest_robustness",
    "M5": "M5_lending_rate_robustness",
    "M6": "M6_tourism_robustness",
    "M7": "M7_human_capital_robustness",
}

# Models that use interest-rate proxies structurally missing for Cambodia and Lao PDR.
# The workbook does not carry an explicit exclusion column, so we hard-code the rule here.
WORKBOOK_CODE_EXCLUDE_COUNTRIES: dict[str, list[str]] = {
    "M2": ["Cambodia", "Lao PDR"],
    "M3": ["Cambodia", "Lao PDR"],
    "M4": ["Cambodia", "Lao PDR"],
    "M5": ["Cambodia", "Lao PDR"],
    "M6": ["Cambodia", "Lao PDR"],
    "M7": ["Cambodia", "Lao PDR"],
}


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


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_equation_variables(value: object, *, fallback_controls: list[str] | None = None) -> tuple[list[str], bool]:
    """
    Parse the free-text 'Equation / variables' cell and return (workbook_variable_codes, lagged_model_flag).

    The workbook is the source of truth: we do not inject fixed controls or always-included variables.
    """
    if pd.isna(value):
        return [], False
    text = str(value).strip().lower()
    if not text:
        return [], False

    lagged = "t-1" in text or ",t-1" in text or "i,t-1" in text

    # Prefer explicit 'controls = ...' declarations when present.
    controls: list[str] = []
    match = re.search(r"controls\s*=\s*([^;]+)", text)
    if match:
        controls_blob = match.group(1)
        tokens = re.split(r"[,\n]+", controls_blob)
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            # Handle slash-delimited alternatives like xr_dep/ln_xr
            for sub in token.split("/"):
                sub = sub.strip()
                if sub:
                    controls.append(sub)
    elif "controls" in text and fallback_controls:
        controls = list(fallback_controls)

    # Scan for known workbook variable codes anywhere in the equation text.
    # The workbook sometimes embeds suffixes like bm_gdp_it or ir_real_i,t-1, so we match
    # with alphanumeric boundaries (underscore counts as a separator here).
    found: list[str] = []
    for code in WORKBOOK_VARIABLE_MAP:
        pattern = rf"(?<![a-z0-9]){re.escape(code)}(?![a-z0-9])"
        if re.search(pattern, text):
            found.append(code)

    # Union of explicit controls and scanned vars, excluding the dependent marker if present.
    codes = _dedupe_preserve_order([*found, *controls])
    codes = [code for code in codes if code != "fdi_gdp"]
    return codes, lagged


def build_workbook_model_catalog(
    workbook_path: Path | str,
    panel_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    workbook_tables = load_workbook_tables(workbook_path)
    model_specs = workbook_tables["model_specs"]
    if "workbook_code" not in model_specs.columns:
        raise ValueError("Workbook 'Model specs' sheet is missing a parsable model code (expected 'Model' values like 'M1 - ...').")

    rows: list[dict[str, object]] = []

    # Pass 1: parse explicit models and infer a default controls list from the first row that declares it.
    inferred_controls: list[str] = []
    parsed_by_code: dict[str, dict[str, object]] = {}
    for spec_row in model_specs.itertuples(index=False):
        workbook_code = str(getattr(spec_row, "workbook_code", "")).strip()
        workbook_model = str(getattr(spec_row, "model", "")).strip()
        equation = getattr(spec_row, "equation_variables", "")
        # Extract controls via direct regex (the parse call result is unused in Pass 1)
        controls_match = re.search(r"controls\s*=\s*([^;]+)", str(equation).lower())
        if controls_match and not inferred_controls:
            blob = controls_match.group(1)
            tokens = re.split(r"[,\n]+", blob)
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                for sub in token.split("/"):
                    sub = sub.strip()
                    if sub:
                        inferred_controls.append(sub)

        parsed_by_code[workbook_code] = {
            "workbook_code": workbook_code,
            "workbook_model": workbook_model,
            "purpose": getattr(spec_row, "purpose", ""),
            "equation": equation,
            "estimated_complete_observations": getattr(spec_row, "estimated_complete_observations", np.nan),
            "recommendation": getattr(spec_row, "recommendation", ""),
        }

    # Pass 2: build rows. Handle "Add X to M4/M2" shorthand by expanding into derived model variants.
    def mapped_cols_for_codes(codes: list[str], lagged: bool) -> tuple[list[str], list[str]]:
        mapped_base = _dedupe_preserve_order([WORKBOOK_VARIABLE_MAP.get(code, code) for code in codes])
        mapped_regressors = [f"{col}_lag1" for col in mapped_base] if lagged else mapped_base
        return mapped_base, mapped_regressors

    for workbook_code, payload in parsed_by_code.items():
        equation = payload["equation"]
        equation_text = str(equation).strip().lower()

        if equation_text.startswith("add "):
            # Example: "Add ln_tourism to M4/M2"
            add_match = re.search(r"add\s+([a-z0-9_]+)\s+to\s+([a-z0-9/\s]+)", equation_text)
            if not add_match:
                continue
            add_code = add_match.group(1).strip()
            base_blob = add_match.group(2).strip()
            base_codes = [token.strip().upper() for token in base_blob.split("/") if token.strip()]
            base_codes = [code for code in base_codes if code.startswith("M")]

            for idx, base_code in enumerate(base_codes, start=1):
                base_payload = parsed_by_code.get(base_code)
                if not base_payload:
                    continue
                base_vars, base_lagged = parse_equation_variables(
                    base_payload["equation"],
                    fallback_controls=inferred_controls or None,
                )
                derived_vars = _dedupe_preserve_order([*base_vars, add_code])
                mapped_base, mapped_regressors = mapped_cols_for_codes(derived_vars, lagged=base_lagged)

                model_id = f"{WORKBOOK_CODE_TO_MODEL_ID.get(workbook_code, workbook_code)}{chr(96+idx)}"
                # Keep a readable suffix for derived variants.
                model_id = (
                    f"{model_id}_from_{WORKBOOK_CODE_TO_MODEL_ID.get(base_code, base_code)}"
                    .replace("__", "_")
                    .replace(" ", "_")
                )

                missing_base_regressors = [column for column in mapped_base if column not in panel_columns]
                status = "estimated" if not missing_base_regressors else "skipped_missing_variables"
                reason = ""
                if missing_base_regressors:
                    reason = "Missing cleaned-panel columns: " + ", ".join(missing_base_regressors)

                exclude = WORKBOOK_CODE_EXCLUDE_COUNTRIES.get(base_code, [])
                rows.append(
                    {
                        "model_id": model_id,
                        "model_order": int(re.sub(r"\D", "", workbook_code) or 0) + (0.1 * idx),
                        "workbook_code": workbook_code,
                        "workbook_model": payload["workbook_model"],
                        "purpose": payload["purpose"],
                        "equation_variables": equation,
                        "estimated_complete_observations": payload["estimated_complete_observations"],
                        "recommendation": payload["recommendation"],
                        "model_tier": "workbook_defined",
                        "channel": "",
                        "thesis_role": "",
                        "monetary_proxy": "",
                        "headline_eligible": True,
                        "appendix_only": True,
                        "lagged_proxy_only": False,
                        "workbook_variables": ", ".join(derived_vars),
                        "base_regressors": ", ".join(mapped_base),
                        "mapped_regressors": ", ".join(mapped_regressors),
                        "lagged_model": bool(base_lagged),
                        "exclude_countries": ", ".join(exclude),
                        "status": status,
                        "missing_reason": reason,
                    }
                )
            continue

        workbook_variable_codes, lagged_model = parse_equation_variables(
            equation, fallback_controls=inferred_controls or None
        )
        mapped_base, mapped_regressors = mapped_cols_for_codes(workbook_variable_codes, lagged=lagged_model)

        missing_base_regressors = [column for column in mapped_base if column not in panel_columns]
        status = "estimated" if not missing_base_regressors else "skipped_missing_variables"
        reason = ""
        if missing_base_regressors:
            reason = "Missing cleaned-panel columns: " + ", ".join(missing_base_regressors)

        model_id = WORKBOOK_CODE_TO_MODEL_ID.get(workbook_code, workbook_code)
        exclude = WORKBOOK_CODE_EXCLUDE_COUNTRIES.get(workbook_code, [])

        rows.append(
            {
                "model_id": model_id,
                "model_order": int(re.sub(r"\D", "", workbook_code) or 0),
                "workbook_code": workbook_code,
                "workbook_model": payload["workbook_model"] or workbook_code,
                "purpose": payload["purpose"],
                "equation_variables": equation,
                "estimated_complete_observations": payload["estimated_complete_observations"],
                "recommendation": payload["recommendation"],
                "model_tier": "workbook_defined",
                "channel": "",
                "thesis_role": "",
                "monetary_proxy": "",
                "headline_eligible": True,
                "appendix_only": False,
                "lagged_proxy_only": False,
                "workbook_variables": ", ".join(workbook_variable_codes),
                "base_regressors": ", ".join(mapped_base),
                "mapped_regressors": ", ".join(mapped_regressors),
                "lagged_model": bool(lagged_model),
                "exclude_countries": ", ".join(exclude),
                "status": status,
                "missing_reason": reason,
            }
        )

    catalog = pd.DataFrame(rows).sort_values(["model_order", "workbook_code", "model_id"]).reset_index(drop=True)
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
