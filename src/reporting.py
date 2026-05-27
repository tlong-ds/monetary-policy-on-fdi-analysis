from __future__ import annotations

import pandas as pd
from IPython.display import Markdown, display


VARIABLE_LABELS = {
    "broad_money_growth_pct": "Broad money growth (annual %)",
    "deposit_interest_rate_pct": "Deposit interest rate (%)",
    "deposit_interest_rate_pct_lag1": "Deposit interest rate, lag 1 (%)",
    "real_interest_rate_pct": "Real interest rate (%)",
    "lending_interest_rate_pct": "Lending interest rate (%)",
    "trade_pct_gdp": "Trade (% GDP)",
    "inflation_gdp_deflator_pct": "Inflation, GDP deflator (%)",
    "ln_gdppc": "Log GDP per capita",
    "xr_dep_pct": "Exchange rate depreciation (%)",
    "ln_tourism_arrivals": "Log tourism arrivals",
    "hc_human_capital_index": "Human capital index",
}

THEORY_EXPECTED_SIGNS = {
    "broad_money_growth_pct": ("positive", "theory: faster monetary expansion can support liquidity conditions associated with FDI"),
    "deposit_interest_rate_pct": ("negative / ambiguous", "theory: higher rates can raise capital costs, but deposit-rate effects are ambiguous"),
    "real_interest_rate_pct": ("negative", "theory: higher real borrowing costs can discourage investment"),
    "lending_interest_rate_pct": ("negative", "theory: higher lending costs can discourage investment"),
    "inflation_gdp_deflator_pct": ("negative", "theory: macro instability can reduce FDI attractiveness"),
    "trade_pct_gdp": ("positive", "theory: trade openness is usually associated with stronger FDI integration"),
    "ln_gdppc": ("positive", "theory: higher income can proxy market size and development level"),
    "xr_dep_pct": ("ambiguous", "theory: depreciation may improve cost competitiveness but can signal currency risk"),
    "ln_tourism_arrivals": ("positive", "theory: tourism can proxy service-sector demand and openness"),
    "hc_human_capital_index": ("positive", "theory: stronger human capital can attract higher-quality FDI"),
}


def stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def format_coef_cell(row: pd.Series) -> str:
    if pd.isna(row["coefficient"]):
        return ""
    return f"{row['coefficient']:.4f}{stars(row['p_value'])}\n({row['std_error']:.4f})"


def hausman_status(row: pd.Series) -> str:
    re_failure_reason = row.get("re_failure_reason", "")
    if isinstance(re_failure_reason, str) and re_failure_reason:
        return f"Diagnostic only; RE failed ({re_failure_reason})"
    if pd.isna(row["p_value"]):
        return "Diagnostic only; Hausman unavailable"
    if bool(row["negative_statistic_flag"]):
        return "Diagnostic only; negative statistic clipped"
    if float(row["p_value"]) < 0.05:
        return "Diagnostic only; Hausman favors FE"
    return "Diagnostic only; Hausman does not reject RE"


def normalize_expected_sign(value: object) -> str:
    if pd.isna(value):
        return "not stated"
    text = str(value).strip().lower()
    if not text:
        return "not stated"
    if "ambiguous" in text and "-" in text:
        return "negative / ambiguous"
    if "ambiguous" in text:
        return "ambiguous"
    if "+" in text:
        return "positive"
    if "-" in text:
        return "negative"
    return text


def strip_lag_suffix(term: str) -> str:
    return str(term).removesuffix("_lag1")


def coefficient_direction(value: float) -> str:
    if pd.isna(value):
        return "not available"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def significance_label(p_value: float) -> str:
    if pd.isna(p_value):
        return "p-value unavailable"
    if p_value < 0.01:
        return "statistically significant at 1%"
    if p_value < 0.05:
        return "statistically significant at 5%"
    if p_value < 0.10:
        return "marginally significant at 10%"
    return "not statistically significant at 10%"


def sign_alignment(coef: float, expected_sign: str) -> str:
    direction = coefficient_direction(coef)
    if expected_sign == "positive":
        return "matches expected positive sign" if direction == "positive" else "does not match expected positive sign"
    if expected_sign == "negative":
        return "matches expected negative sign" if direction == "negative" else "does not match expected negative sign"
    if expected_sign == "negative / ambiguous":
        if direction == "negative":
            return "matches the negative part of the expected sign, with ambiguity caveat"
        return "does not match the negative part of the expected sign, with ambiguity caveat"
    if expected_sign == "ambiguous":
        return "no directional sign test because theory/workbook marks this relationship ambiguous"
    return "no expected sign available"


def interpret_coefficient_row(row: pd.Series) -> str:
    label = VARIABLE_LABELS.get(row["base_term"], row["base_term"])
    return (
        f"{label}: coefficient {row['coef']:.4f} is {row['coefficient_direction']} "
        f"with p-value {row['p_value']:.4f}; {row['significance']}; "
        f"expected sign is {row['expected_sign']} from {row['expected_sign_source']}; "
        f"{row['sign_alignment']}."
    )


def display_section_note(title: str, bullets: list[str]) -> None:
    body = "\n".join([f"- {bullet}" for bullet in bullets])
    display(Markdown(f"### {title}\n{body}"))


def compact_display(df: pd.DataFrame, columns: list[str] | None = None, n: int = 12) -> None:
    view = df.copy()
    if columns is not None:
        view = view[[column for column in columns if column in view.columns]]
    display(view.head(n))
