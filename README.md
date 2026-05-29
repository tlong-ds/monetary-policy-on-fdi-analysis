# Monetary Policy & FDI Analysis — ASEAN Panel Study

Reproducible analysis pipeline examining the effect of monetary policy
(broad money growth, real interest rate) on FDI net inflows across 10
ASEAN economies, 2000–2023.

## Requirements

```
pip install -r requirements.txt
```

## Reproduce in order

```bash
python scripts/01_process_data.py      # data loading, merging, cleaning
python scripts/02_run_diagnostics.py   # panel diagnostic tests
python scripts/03_run_estimation.py    # Specs 1–5 regression
python scripts/04_run_robustness.py    # robustness checks R1–R6
```

## Project structure

```
data/
  raw/
    2000-2025-bm-rir-xr-gdp-infl.xlsx   macro panel (WB format)
    1996-2024-wgi-data.xlsx              WGI indicators (sheets: va pv ge rq rl cc)
    2000-2023-hc-pop.csv                 human capital index & population
  processed/
    panel.csv                            final estimation panel
    variable_dict.csv                    variable dictionary

src/
  config.py                              project-wide paths and constants
  data/
    loader.py                            raw data readers
    transform.py                         ln_gdppc, d_ln_exr, wgi_composite
    clean.py                             missing data handling, winsorization
    merge.py                             pipeline orchestrator
  diagnostics/
    descriptive.py                       summary statistics, correlation matrix
    multicollinearity.py                 VIF
    unit_root.py                         LLC and IPS panel unit root tests
    heteroskedasticity.py                Modified Wald test
    serial_correlation.py                Wooldridge (2002) test
    cross_section.py                     Pesaran CD test
  estimation/
    specs.py                             hardcoded ModelSpec dataclasses (Specs 1–5)
    static.py                            BP-LM → Hausman → FE/RE selection
    dynamic.py                           two-step System GMM (Blundell-Bond 1998)
  robustness/
    checks.py                            R1–R6 robustness variants + comparison tables
  output/
    tables.py                            regression table formatter (.xlsx + .tex)

scripts/
  01_process_data.py
  02_run_diagnostics.py
  03_run_estimation.py
  04_run_robustness.py

outputs/
  preprocessing_report.xlsx             data coverage, imputation log, winsor thresholds
  diagnostics/
    diagnostics_summary.xlsx            all 7 diagnostic tests
  tables/
    regression_specs1_4.{xlsx,tex}      static estimation results
    regression_spec5_gmm.{xlsx,tex}     System GMM results
    regression_all_specs.{xlsx,tex}     all 5 specs combined
    robustness_comparison.{xlsx,tex}    all robustness checks side-by-side
    robustness_bm_summary.csv           BM coefficient stability table
    robustness_r1_rl.{xlsx,tex}         R1: Rule of Law
    robustness_r2_lagged_bm.{xlsx,tex}  R2: Lagged BM Growth
    robustness_r3_no_sgp.{xlsx,tex}     R3: Excluding Singapore
    robustness_r4_crisis.{xlsx,tex}     R4: Excluding Crisis Years
    robustness_r5_gmm.{xlsx,tex}        R5: System GMM
    robustness_r6_composite_wgi.{xlsx,tex} R6: Composite WGI
    estimator_selection_log.csv         BP-LM and Hausman decisions per spec
```

## Key model specifications

| Spec | Name | Estimator | Key addition |
|---|---|---|---|
| 1 | Baseline | RE | Base controls only |
| 2 | + Monetary | FE | + Real interest rate |
| 3 | + Institutional | FE | + Regulatory Quality (RQ) |
| 4 | Full | FE | + RIR + RQ |
| 5 | Dynamic GMM | SysGMM | + Lagged FDI, endogenous BM |

## Configuration

All paths and constants are centralised in [`src/config.py`](src/config.py).
Time window: 2000–2023. Winsorization: p1/p99.
