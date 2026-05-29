# Copilot Instructions

## Project shape

This repository is a Python panel-data analysis pipeline for ASEAN FDI and monetary policy. The workflow is stage-based:

1. `scripts/01_process_data.py` builds `data/processed/panel.csv` and `data/processed/variable_dict.csv`.
2. `scripts/02_run_diagnostics.py` runs panel diagnostics and writes `outputs/diagnostics/`.
3. `scripts/03_run_estimation.py` estimates Specs 1–5 and exports regression tables to `outputs/tables/`.
4. `scripts/04_run_robustness.py` runs robustness checks anchored to Spec 4 and exports comparison tables.

`src/config.py` is the central source of truth for paths, ASEAN country codes, WGI sheet names, time window, and output directories. It discovers the repo root by locating `data/raw/`, so keep that directory layout intact.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline stages in order:

```bash
python scripts/01_process_data.py
python scripts/02_run_diagnostics.py
python scripts/03_run_estimation.py
python scripts/04_run_robustness.py
```

When changing one stage, rerun that stage and any downstream stages it feeds.

## Architecture

- `src/data/` handles raw loading, variable construction, cleaning, merging, and export artifacts.
- `src/diagnostics/` contains descriptive stats, multicollinearity, unit root, heteroskedasticity, serial correlation, and cross-section dependence checks.
- `src/estimation/` defines the model specs plus static FE/RE/pool selection and dynamic System GMM estimation.
- `src/robustness/` derives all robustness variants from Spec 4.
- `src/output/tables.py` formats results into Excel and LaTeX tables.

The merged panel uses `(country_code, year)` as the join key throughout. `scripts/*` add the repo root to `sys.path` before importing from `src/`.

## Conventions

- Keep country coverage to ASEAN-10 ISO3 codes; Timor-Leste is excluded.
- Preserve the WGI sheet mapping in `src/config.py` (`va`, `pv`, `ge`, `rq`, `rl`, `cc`), including the `pv` sheet name for political stability.
- Use the established variable transformations: `ln_gdppc`, `d_ln_exr` (first difference of log exchange rate × 100), and `wgi_composite = (rq + rl + ge + cc) / 4`.
- Missing FDI is dropped; macro controls may be interpolated only when missingness is at least 5%; WGI variables are never interpolated.
- Winsorization happens after transformations, at p1/p99.
- Specs 1–4 are static panel models; Spec 5 is dynamic System GMM with lagged FDI and broad money growth treated as endogenous.
- Static estimation follows pooled OLS → Breusch-Pagan LM → Hausman, and uses country-level clustered standard errors.
- Table output is label-driven; reuse the variable labels and expected-sign mappings in `src/estimation/specs.py` rather than hardcoding names in new report code.
