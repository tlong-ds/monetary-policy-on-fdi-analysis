# Analysis of Monetary Policy on FDI in ASEAN Countries

## 1. Research Objective

This study aims to analyze and estimate the impact of monetary policy conditions on Foreign Direct Investment (FDI) inflows across ASEAN countries during the period 2000–2024.

The analysis focuses on how monetary expansion, interest rates, macroeconomic stability, and institutional quality influence FDI attractiveness in ASEAN economies.

The study contributes to the literature by:
- examining monetary policy as an FDI determinant in emerging ASEAN economies,
- integrating institutional quality into the monetary–FDI relationship,
- and applying both static panel estimators and dynamic System GMM estimation.

---

# 2. Sample and Country Coverage

## ASEAN Sample

The panel includes the following ASEAN countries:

- Indonesia
- Malaysia
- Philippines
- Singapore
- Thailand
- Vietnam
- Cambodia
- Laos
- Myanmar
- Brunei

Exclude:
- Timor-Leste

## Sample Structure

- Annual panel data
- Time range: 2000–2024
- Country-year observations
- Unbalanced panel acceptable if missing observations exist

---

# 3. Data Sources

You are provided with the following datasets in `data/raw`:

## Dataset 1 — Main Macroeconomic Dataset

File:
`2000-2025-bm-rir-xr-gdp-infl.xlsx`

Contains:
- FDI inflows
- Broad money growth
- Real interest rate
- GDP growth
- GDP per capita
- Exchange rate
- Inflation
- Trade openness

---

## Dataset 2 — World Governance Indicators

File:
`1996-2024-wgi-data.xlsx`

Contains separate sheets for:
- Voice and Accountability (VA)
- Political Stability and Absence of Violence (PS)
- Government Effectiveness (GE)
- Regulatory Quality (RQ)
- Rule of Law (RL)
- Control of Corruption (CC)

---

# 4. Variable Construction

## Required Variables

| Variable | Unit | Role | Expected Sign | Source | Transformation |
|---|---|---|---|---|---|
| FDI net inflow | % GDP | Dependent Variable |  | Dataset 1 | None |
| Broad money growth | % Growth | Main Explanatory Variable | (+) | Dataset 1 | None |
| Real interest rate | Level | Monetary Tightness | (−) | Dataset 1 | None |
| GDP growth | % Growth | Market Growth Control | (+) | Dataset 1 | None |
| ln(GDP per capita) | Log Level | Market Size Control | (+) | Dataset 1 | Natural log |
| Trade openness | % GDP | Openness Control | (+) | Dataset 1 | None |
| Δln(exchange rate) | % Change | Exchange Rate Control | (+/−) | Dataset 1 | First difference of log |
| Inflation | % | Stability Control | (−) | Dataset 1 | None |
| WGI institutional quality | Level | Institutional Control | (+) | Dataset 2 | RL/RQ |

---

## Transformation Rules

### GDP per Capita

$begin:math:display$
\\ln\(GDPpc\_\{it\}\)
$end:math:display$

Apply natural logarithm transformation.

---

### Exchange Rate Change

Use official nominal exchange rate:
- local currency units per USD.

Construct:

$begin:math:display$
\\Delta \\ln\(EXR\_\{it\}\) \=
\\ln\(EXR\_\{it\}\) \- \\ln\(EXR\_\{it\-1\}\)
$end:math:display$

Interpretation:
- positive value = currency depreciation.

---

## Institutional Quality Specification

### Baseline Institutional Proxy
- Regulatory Quality (RQ)

### Robustness Institutional Proxy
- Rule of Law (RL)

### Optional Composite Index

Construct:

$begin:math:display$
WGI\_\{composite\} \=
\\frac\{RQ \+ RL \+ GE \+ CC\}\{4\}
$end:math:display$

if required for robustness analysis.

---

# 5. Data Processing Instructions

## Data Cleaning

Perform the following steps:

1. Standardize country names across datasets
2. Convert country identifiers into ISO3 codes
3. Convert year variable into integer format
4. Keep annual frequency only
5. Remove duplicate country-year observations

---

## Missing Data Handling

### Dependent Variable
- Drop observations with missing FDI values

### Independent Variables
- If missing share < 5%:
  - use listwise deletion
- Otherwise:
  - interpolate macroeconomic controls only

### Institutional Variables
- Do NOT interpolate WGI indicators

---

## Outlier Treatment

Winsorize continuous variables at:
- 1st percentile
- 99th percentile

Apply after variable transformations.

---

## Merge Procedure

Merge datasets using:
- ISO3 country code
- year

Final output:
- one merged panel dataset for 2000–2024.

---

# 6. Theoretical Framework

## 6.1 Monetary Conditions Channel

Underlying theory:
- Financial deepening theory (McKinnon, 1973; Shaw, 1973)

Mechanism:
- higher broad money growth increases credit availability,
- reduces financing costs,
- improves investment liquidity,
- increases FDI attractiveness.

---

## 6.2 Market Size Channel

Underlying theory:
- OLI paradigm (Dunning, 1977)
- Gravity model

Mechanism:
- larger and faster-growing markets attract multinational firms.

Variables:
- GDP growth
- GDP per capita

---

## 6.3 Trade Openness Channel

Underlying theory:
- OLI openness advantage
- New Trade Theory

Mechanism:
- openness facilitates export-platform FDI.

Variable:
- Trade (% GDP)

---

## 6.4 Macroeconomic Stability Channel

Underlying theory:
- Macroeconomic stability hypothesis
- Mundell–Fleming framework

Mechanism:
- inflation and exchange rate volatility affect investment risk.

Variables:
- Inflation
- Δln(exchange rate)

---

## 6.5 Institutional Quality Channel

Underlying theory:
- Institutional theory (North, 1990)
- Transaction cost economics

Mechanism:
- stronger institutions reduce uncertainty and transaction costs,
- improve property rights protection,
- increase investor confidence.

Variables:
- Regulatory Quality
- Rule of Law

---

# 7. Research Hypotheses

| Hypothesis | Expectation |
|---|---|
| H1 | Broad money growth positively affects FDI inflows |
| H2 | Higher real interest rates reduce FDI inflows |
| H3 | GDP growth positively affects FDI |
| H4 | Trade openness positively affects FDI |
| H5 | Inflation negatively affects FDI |
| H6 | Better institutional quality increases FDI |
| H7 | Exchange rate depreciation has ambiguous effects on FDI |

---

# 8. Econometric Methodology

## 8.1 Baseline Panel Equation

The baseline empirical model is:

$begin:math:display$
FDI\_\{it\} \=
\\alpha \+
\\beta\_1 BM\_\{it\} \+
\\beta\_2 GDPG\_\{it\} \+
\\beta\_3 \\ln\(GDPpc\_\{it\}\) \+
\\beta\_4 Trade\_\{it\} \+
\\beta\_5 \\Delta \\ln\(EXR\_\{it\}\) \+
\\beta\_6 Inflation\_\{it\} \+
\\mu\_i \+
\\lambda\_t \+
\\varepsilon\_\{it\}
$end:math:display$

Where:
- $begin:math:text$i$end:math:text$ = country
- $begin:math:text$t$end:math:text$ = year
- $begin:math:text$\\mu\_i$end:math:text$ = country fixed effects
- $begin:math:text$\\lambda\_t$end:math:text$ = year fixed effects

---

## 8.2 Estimator Selection Procedure

Follow these steps:

### Step 1 — Pooled OLS
Estimate pooled OLS model.

### Step 2 — Breusch–Pagan LM Test
- insignificant:
  - use pooled OLS
- significant:
  - proceed to panel estimator

### Step 3 — Hausman Test
- $begin:math:text$p \< 0\.05$end:math:text$:
  - Fixed Effects preferred
- otherwise:
  - Random Effects preferred

### Step 4 — Robust Standard Errors
Use:
- cluster-robust standard errors at country level.

---

## 8.3 Dynamic Panel and Endogeneity

Potential endogeneity sources:
- reverse causality between FDI and monetary expansion,
- omitted macroeconomic shocks,
- institutional persistence.

Therefore:
- estimate System GMM as robustness specification.

---

## 8.4 System GMM Setup

Use:
- two-step System GMM,
- Windmeijer corrected standard errors,
- collapsed instruments,
- lag(2 .) GMM-style instruments.

Treat as endogenous:
- lagged FDI,
- broad money growth.

Diagnostics:
- Hansen test
- Arellano-Bond AR(1)
- Arellano-Bond AR(2)

---

# 9. Diagnostic Tests

Run the following diagnostics before estimation.

---

## 9.1 Descriptive Statistics

Generate:
- mean
- median
- standard deviation
- minimum
- maximum

---

## 9.2 Correlation Matrix

Generate pairwise correlations among explanatory variables.

---

## 9.3 Multicollinearity

Run Variance Inflation Factor (VIF).

Interpretation:
- VIF < 5:
  - acceptable
- 5–10:
  - moderate concern
- >10:
  - problematic

Special attention:
- inflation and real interest rate may exhibit multicollinearity.

---

## 9.4 Unit Root Tests

Run:
- Levin-Lin-Chu (LLC)
OR
- Im-Pesaran-Shin (IPS)

---

## 9.5 Heteroskedasticity

Run:
- Modified Wald test.

---

## 9.6 Serial Correlation

Run:
- Wooldridge test for autocorrelation.

---

## 9.7 Cross-Sectional Dependence

Run:
- Pesaran CD test.

---

# 10. Model Specifications

| Model Specifications | Spec 1 | Spec 2 | Spec 3 | Spec 4 | Spec 5 |
|---|---|---|---|---|---|
| Label | Baseline | + Monetary | + Institutional | Full | Robustness |
| Estimator | FE/RE | FE/RE | FE/RE | FE/RE | System GMM |
| FDI/GDP | ✓ | ✓ | ✓ | ✓ | ✓ |
| Broad money growth | ✓ | ✓ | ✓ | ✓ | ✓ |
| GDP growth | ✓ | ✓ | ✓ | ✓ | ✓ |
| ln(GDPpc) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Trade (%GDP) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Δln(exchange rate) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Inflation | ✓ | ✓ | ✓ | ✓ | ✓ |
| Real interest rate |  | ✓ |  | ✓ | ✓ |
| WGI |  |  | ✓ | ✓ | ✓ |
| Lagged FDI |  |  |  |  | ✓ |
| Country FE | ✓ | ✓ | ✓ | ✓ | — |
| Year FE | ✓ | ✓ | ✓ | ✓ | ✓ |

---

# 11. Robustness Checks

## 11.1 Institutional Robustness
- Replace RQ with RL

---

## 11.2 Monetary Robustness
- Use lagged broad money growth

---

## 11.3 Sample Robustness
- Exclude Singapore

---

## 11.4 Crisis Robustness
Exclude:
- Global Financial Crisis:
  - 2008–2009
- COVID period:
  - 2020–2021

---

## 11.5 Dynamic Robustness
- Estimate System GMM specification

---

# 12. Interpretation Guidelines

Interpret results using:
- coefficient sign,
- statistical significance,
- economic magnitude,
- ASEAN-specific institutional context,
- consistency with prior literature,
- policy implications.

Discuss:
- liquidity transmission mechanism,
- institutional moderation effects,
- exchange rate uncertainty,
- macroeconomic stability.

---

# 13. Required Outputs

Generate:

1. Clean merged panel dataset
2. Variable dictionary
3. Descriptive statistics table
4. Correlation matrix
5. VIF table
6. Diagnostic test results
7. Regression tables
8. Robustness tables
9. Interpretation of each specification
10. Policy implications for ASEAN economies

---

# 14. Programming Requirements

Provide:
- Python code only

Recommended libraries:
- pandas
- numpy
- statsmodels
- linearmodels

The code should:
- clean data,
- merge datasets,
- generate transformed variables,
- run diagnostics,
- estimate panel models,
- estimate System GMM,
- export regression tables.

---

# 15. Literature Support

## 15.1 Broad Money Growth

### Theory
- McKinnon (1973)
- Shaw (1973)

### Key References
- Agbloyor et al. (2014)
- Asongu & Nwachukwu (2016)
- Alfaro et al. (2004)

---

## 15.2 GDP Growth and GDP per Capita

### Theory
- Dunning (1977, 1988)
- Gravity model

### Key References
- Blonigen & Piger (2014)
- Chakrabarti (2001)

---

## 15.3 Trade Openness

### Theory
- OLI framework
- New Trade Theory

### Key References
- Asiedu (2002)
- Mottaleb & Kalirajan (2010)

---

## 15.4 Exchange Rate

### Theory
- UIP
- Mundell–Fleming model

### Key References
- Froot & Stein (1991)
- Blonigen (1997)
- Goldberg & Kolstad (1995)

---

## 15.5 Inflation

### Theory
- Macroeconomic stability hypothesis

### Key References
- Asiedu (2002)
- Demirhan & Masca (2008)

---

## 15.6 Real Interest Rate

### Theory
- Cost of capital theory
- McKinnon–Shaw hypothesis

### Key References
- Desai et al. (2006)
- IMF (2013)

---

## 15.7 Institutional Quality

### Theory
- North (1990)
- Williamson (1985)

### Key References
- Buchanan et al. (2012)
- Globerman & Shapiro (2002)

---

# 16. Summary of Expected Signs

| Variable | Expected Sign | Literature Consensus |
|---|---|---|
| Broad money growth | (+) | Moderate |
| GDP growth | (+) | Strong |
| ln(GDP per capita) | (+) | Strong |
| Trade openness | (+) | Strong |
| Δln(exchange rate) | (±) | Ambiguous |
| Inflation | (−) | Moderate |
| Real interest rate | (−) | Moderate |
| Institutional quality | (+) | Strong |

---

# 17. Optional Extensions

Optional advanced analysis:

## Interaction Effect

Test:

$begin:math:display$
BroadMoney \\times InstitutionalQuality
$end:math:display$

Hypothesis:
- monetary expansion is more effective in attracting FDI under stronger institutional quality.

---

## Regional Comparison

Compare:
- ASEAN-5
vs
- CLMV countries

(Cambodia, Laos, Myanmar, Vietnam)

---

# 18. Study Limitations

Potential limitations:
- relatively small ASEAN sample,
- measurement issues in governance indicators,
- omitted global financial variables,
- persistence in institutional quality,
- possible remaining endogeneity concerns.