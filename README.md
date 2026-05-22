# Monetary Policy and FDI Analysis

Notebook-first workflow for the ASEAN monetary-policy and FDI panel. The active modeling contract is `model_selection_asean_fdi.xlsx`; notebooks should use the workbook-driven model IDs and the helper code in `src/` rather than hardcoded legacy specification names.

## Pipeline

1. `notebooks/00_exploratory_analysis.ipynb` - exploratory audit tables, currently the broad-money sign-flip decomposition.
2. `notebooks/01_data_processing.ipynb` - raw workbook ingestion, WDI interest-rate merge, role-based missingness handling, and clean panel export.
3. `notebooks/02_econometric_models.ipynb` - workbook-driven M1-M7b estimation, Hausman diagnostics, native-sample and common-sample outputs.
4. `notebooks/03_results_and_figures.ipynb` - thesis tables, figures, coefficient interpretation, and appendix exports.
5. `notebooks/05_limitations_missing_data.ipynb` - missing-data limitations, imputation audit, and sample-loss reporting.

## Structure

- `src/config.py` centralizes paths and stable constants.
- `src/model_contract.py` contains the workbook model catalog and M1-M7b branch split.
- `src/panel_diagnostics.py` contains Hausman, VIF, pooled diagnostics, and residual checks.
- `src/plot_helpers.py` centralizes plotting style and figure saving.
- Root-level `workbook_model_contract.py` and `panel_model_diagnostics.py` are compatibility shims for older notebooks or ad-hoc scripts.
- `notebooks/archive/` stores the pre-refactor notebook copies.

Existing generated artifacts remain in `outputs/` so the current workbook/reporting notebooks keep working. The new `outputs/tables/`, `outputs/diagnostics/`, and `outputs/model_artifacts/` directories are available for a later output-migration pass once table consumers are updated.
