from __future__ import annotations

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def lines(text: str) -> list[str]:
    return [f"{line}\n" for line in textwrap.dedent(text).strip("\n").splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def load_metadata(notebook_name: str) -> dict:
    notebook_path = NOTEBOOKS_DIR / notebook_name
    return json.loads(notebook_path.read_text())["metadata"]


def write_notebook(notebook_name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": load_metadata(notebook_name),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS_DIR / notebook_name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def rewrite_notebook_01() -> None:
    cells = [
        markdown_cell(
            """
            # 01 Data Processing

            This notebook rebuilds the panel from the current raw inputs in `data/raw`:

            - `2000-2025-bm-ir-infl.xlsx` for the base macro country-year panel and exchange rate
            - `2000-2023-hc-pop.csv` for human capital and population
            - `2000-2025-lending-deposit-tourism-arrivals.csv` for deposit rate, lending rate, and tourism arrivals

            The active processing window is `2000-2023`, and the notebook now delegates the ingestion logic to `src.reprocess_current_raw` so the notebook and exported CSV stay on the same contract.
            """
        ),
        markdown_cell(
            """
            ## Setup

            Load the shared preprocessing functions and inspect the source files before rebuilding the panel.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import importlib

            import pandas as pd

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.config import (
                CURRENT_ADDITIONAL_SERIES_FILE,
                CURRENT_HC_POP_FILE,
                CURRENT_MACRO_PANEL_WORKBOOK,
                OUTPUTS_DIR,
                PROCESSED_PANEL_FILE,
                TIME_WINDOW,
                ensure_output_dirs,
            )
            import src.reprocess_current_raw as reprocess_current_raw
            importlib.reload(reprocess_current_raw)

            build_clean_panel = reprocess_current_raw.build_clean_panel
            load_additional_series_panel = reprocess_current_raw.load_additional_series_panel
            load_hc_pop_panel = reprocess_current_raw.load_hc_pop_panel
            load_macro_panel = reprocess_current_raw.load_macro_panel
            save_outputs = reprocess_current_raw.save_outputs

            ensure_output_dirs()
            pd.set_option("display.max_columns", None)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
            """
        ),
        markdown_cell(
            """
            ## Source Audit

            Confirm the three active sources and their effective coverage in the processing window.
            """
        ),
        code_cell(
            """
            macro_df = load_macro_panel()
            hc_df = load_hc_pop_panel()
            additional_df = load_additional_series_panel()

            source_audit = pd.DataFrame(
                [
                    {
                        "file": CURRENT_MACRO_PANEL_WORKBOOK.name,
                        "rows": len(macro_df),
                        "countries": macro_df["country_code"].nunique(),
                        "first_year": int(macro_df["year"].min()),
                        "last_year": int(macro_df["year"].max()),
                    },
                    {
                        "file": CURRENT_HC_POP_FILE.name,
                        "rows": len(hc_df),
                        "countries": hc_df["country_code"].nunique(),
                        "first_year": int(hc_df["year"].min()),
                        "last_year": int(hc_df["year"].max()),
                    },
                    {
                        "file": CURRENT_ADDITIONAL_SERIES_FILE.name,
                        "rows": len(additional_df),
                        "countries": additional_df["country_code"].nunique(),
                        "first_year": int(additional_df["year"].min()),
                        "last_year": int(additional_df["year"].max()),
                    },
                ]
            )

            print(f"Configured TIME_WINDOW: {TIME_WINDOW}")
            source_audit
            """
        ),
        code_cell(
            """
            display_columns = ["country_code", "country", "year"]

            print("Macro panel sample")
            display(
                macro_df[
                    display_columns + ["fdi_pct_gdp", "broad_money_growth_pct", "real_interest_rate_pct"]
                ].head()
            )

            print("HC / population sample")
            display(hc_df[display_columns + ["hc_human_capital_index", "population_total"]].head())

            print("Supplementary series sample")
            display(
                additional_df[
                    display_columns + ["deposit_interest_rate_pct", "lending_interest_rate_pct", "tourism_arrivals"]
                ].head()
            )
            """
        ),
        markdown_cell(
            """
            ## Build Clean Panel

            Run the shared reprocessor, inspect the resulting panel, and then write the refreshed outputs.
            """
        ),
        code_cell(
            """
            clean_panel, outputs = build_clean_panel()

            panel_overview = pd.DataFrame(
                {
                    "metric": ["rows", "countries", "first_year", "last_year"],
                    "value": [
                        len(clean_panel),
                        clean_panel["country"].nunique(),
                        int(clean_panel["year"].min()),
                        int(clean_panel["year"].max()),
                    ],
                }
            )

            panel_overview
            """
        ),
        code_cell(
            """
            clean_panel.head()
            """
        ),
        code_cell(
            """
            clean_panel.notna().sum().rename("non_missing").to_frame()
            """
        ),
        code_cell(
            """
            save_outputs(
                clean_panel=clean_panel,
                winsorization_thresholds=outputs["winsorization_thresholds"],
                control_imputation_log=outputs["control_imputation_log"],
                transformation_audit=outputs["transformation_audit"],
                coverage_by_variable=outputs["coverage_by_variable"],
                coverage_by_country=outputs["coverage_by_country"],
                review_flags=outputs["review_flags"],
                variable_audit=outputs["variable_audit"],
                raw_input_audit=outputs["raw_input_audit"],
            )

            print(f"Saved processed panel to {PROCESSED_PANEL_FILE}")
            print(f"Saved preprocessing workbook to {OUTPUTS_DIR / 'preprocessing_outputs.xlsx'}")
            """
        ),
        markdown_cell(
            """
            ## Processing Diagnostics

            Review the export-side audit tables that downstream notebooks rely on.
            """
        ),
        code_cell(
            """
            outputs["raw_input_audit"]
            """
        ),
        code_cell(
            """
            outputs["variable_audit"].sort_values(["missing_rate", "variable"], ascending=[False, True]).reset_index(drop=True)
            """
        ),
        code_cell(
            """
            outputs["control_imputation_log"]
            """
        ),
        code_cell(
            """
            outputs["review_flags"]
            """
        ),
    ]
    write_notebook("01_data_processing.ipynb", cells)


def rewrite_notebook_02() -> None:
    cells = [
        markdown_cell(
            """
            # 02 Econometric Models (Workbook-Driven)

            This notebook estimates the workbook-driven specifications (M1--M7b) using the refreshed
            processed panel produced by `01_data_processing.ipynb`.

            Broad money is measured as broad money growth (annual %) (`broad_money_growth_pct`).
            """
        ),
        markdown_cell("## Setup"),
        code_cell(
            """
            from pathlib import Path
            import sys
            import importlib

            import pandas as pd

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.config import MODEL_SELECTION_WORKBOOK, OUTPUTS_DIR, PROCESSED_PANEL_FILE, ensure_output_dirs
            import src.model_contract as model_contract
            import src.estimate_and_export as estimate_and_export

            importlib.reload(model_contract)
            importlib.reload(estimate_and_export)

            build_workbook_model_catalog = model_contract.build_workbook_model_catalog
            run_full_estimation_and_export = estimate_and_export.run_full_estimation_and_export

            ensure_output_dirs()
            pd.set_option("display.max_columns", None)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
            """
        ),
        markdown_cell("## Load Processed Panel"),
        code_cell(
            """
            panel = pd.read_csv(PROCESSED_PANEL_FILE)
            panel.shape, int(panel['year'].min()), int(panel['year'].max())
            """
        ),
        code_cell(
            """
            required = [
                'country', 'year', 'fdi_pct_gdp', 'broad_money_growth_pct',
                'inflation_gdp_deflator_pct', 'trade_pct_gdp', 'ln_gdppc', 'xr_dep_pct',
            ]
            missing = [c for c in required if c not in panel.columns]
            if missing:
                raise ValueError(
                    f"Processed panel is missing required columns: {missing}. "
                    "Re-run 01_data_processing.ipynb from the top to rebuild clean_panel.csv."
                )
            """
        ),
        markdown_cell("## Build Workbook Catalog"),
        code_cell(
            """
            catalog, workbook_tables = build_workbook_model_catalog(
                MODEL_SELECTION_WORKBOOK,
                panel_columns=panel.columns.tolist(),
            )
            catalog[['model_id','workbook_model','status','missing_reason','base_regressors','mapped_regressors']]
            """
        ),
        markdown_cell("## Estimate And Export"),
        code_cell(
            """
            estimated_models = catalog[catalog['status'].eq('estimated')].copy()
            run_full_estimation_and_export(panel, estimated_models, MODEL_SELECTION_WORKBOOK, dependent='fdi_pct_gdp')
            """
        ),
        markdown_cell("## Quick Output Check"),
        code_cell(
            """
            for name in [
                'model_coefficients.csv',
                'model_fit_stats.csv',
                'regression_table_main.csv',
                'correlation_matrix_main_variables.csv',
            ]:
                path = OUTPUTS_DIR / name
                print(name, 'exists' if path.exists() else 'MISSING')
            """
        ),
    ]
    write_notebook("02_panel_diagnostics.ipynb", cells)


def rewrite_notebook_03() -> None:
    cells = [
        markdown_cell(
            """
            # 03 Descriptive Statistics (Learn-The-Data)

            Learn-the-data version: this notebook teaches the processed panel from three angles:

            1) Data overview (coverage + missingness)
            2) Univariate structure (per-variable distributions + time trends)
            3) Bivariate + multivariate structure (correlations, scatter sweeps, collinearity signals)

            The notebook automatically detects **all numeric columns** in `data/processed/clean_panel.csv` (excluding IDs)
            and saves key figures + CSV tables under `outputs/`.
            """
        ),
        markdown_cell("## Setup"),
        code_cell(
            """
            from pathlib import Path
            import sys
            import importlib

            import numpy as np
            import pandas as pd
            from scipy.stats import gaussian_kde

            import matplotlib.pyplot as plt

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            import src.config as config
            importlib.reload(config)

            OUTPUTS_DIR = config.OUTPUTS_DIR
            FIGURES_DIR = config.FIGURES_DIR
            PROCESSED_PANEL_FILE = config.PROCESSED_PANEL_FILE
            TIME_WINDOW = config.TIME_WINDOW
            ensure_output_dirs = config.ensure_output_dirs

            ensure_output_dirs()
            pd.set_option("display.max_columns", None)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            """
        ),
        markdown_cell(
            """
            ## Load Panel (Processed Contract)

            Source of truth: `data/processed/clean_panel.csv` produced by `01_data_processing.ipynb`.
            """
        ),
        code_cell(
            """
            df = pd.read_csv(PROCESSED_PANEL_FILE)
            df.shape, int(df['year'].min()), int(df['year'].max())
            """
        ),
        code_cell(
            """
            start_year, end_year = TIME_WINDOW
            year_min, year_max = int(df['year'].min()), int(df['year'].max())
            if (year_min, year_max) != (start_year, end_year):
                raise ValueError(f'Expected year window {TIME_WINDOW}, got {(year_min, year_max)}. Re-run 01_data_processing.ipynb.')

            exclude = {'country_id', 'year', 'country', 'country_code'}
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
            if not numeric_cols:
                raise ValueError('No numeric columns detected after exclusions; check processed panel schema.')

            display(df[['country', 'country_code', 'year'] + [c for c in ['fdi_pct_gdp', 'broad_money_growth_pct'] if c in df.columns]].head())
            numeric_cols[:10], len(numeric_cols)
            """
        ),
        markdown_cell(
            """
            ## 1) Data Overview (Sanity + Coverage)

            Deliverables:
            - `outputs/descriptive_numeric_columns.csv`
            - `outputs/missingness_by_variable.csv`
            - `outputs/coverage_by_country_numeric.csv`
            """
        ),
        code_cell(
            """
            numeric_columns_df = pd.DataFrame({'numeric_column': numeric_cols})
            numeric_columns_path = OUTPUTS_DIR / 'descriptive_numeric_columns.csv'
            numeric_columns_df.to_csv(numeric_columns_path, index=False)
            print('Saved', numeric_columns_path)

            missingness = pd.DataFrame({
                'variable': numeric_cols,
                'missing_count': [int(df[c].isna().sum()) for c in numeric_cols],
            })
            missingness['missing_rate'] = missingness['missing_count'] / len(df)
            missingness = missingness.sort_values(['missing_rate', 'variable'], ascending=[False, True]).reset_index(drop=True)
            missingness_path = OUTPUTS_DIR / 'missingness_by_variable.csv'
            missingness.to_csv(missingness_path, index=False)
            print('Saved', missingness_path)

            coverage_by_country = (
                df.groupby('country')[numeric_cols]
                  .apply(lambda frame: frame.notna().sum())
                  .reset_index()
            )
            coverage_by_country_path = OUTPUTS_DIR / 'coverage_by_country_numeric.csv'
            coverage_by_country.to_csv(coverage_by_country_path, index=False)
            print('Saved', coverage_by_country_path)

            missingness.head(20)
            """
        ),
        markdown_cell(
            """
            ## 2) Univariate Analysis (Per Variable)

            For each numeric column `x`, produce:
            - Summary stats in `outputs/univariate_summary.csv`
            - Distribution plots in `outputs/figures/univariate_dist__{col}.png`
            - Time-trend plots in `outputs/figures/univariate_trend__{col}.png`
            """
        ),
        code_cell(
            """
            def safe_slug(name: str) -> str:
                return ''.join(ch if (ch.isalnum() or ch in {'_', '-'}) else '_' for ch in name).strip('_')

            def compute_univariate_summary(series: pd.Series) -> dict:
                values = series.dropna().astype(float)
                missing = int(series.isna().sum())
                missing_rate = missing / len(series)
                if values.empty:
                    return {
                        'count': 0,
                        'mean': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'p25': np.nan,
                        'median': np.nan,
                        'p75': np.nan,
                        'max': np.nan,
                        'missing': missing,
                        'missing_pct': 100 * missing_rate,
                    }
                q = values.quantile([0.25, 0.5, 0.75]).to_dict()
                return {
                    'count': int(values.shape[0]),
                    'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)),
                    'min': float(values.min()),
                    'p25': float(q.get(0.25, np.nan)),
                    'median': float(q.get(0.5, np.nan)),
                    'p75': float(q.get(0.75, np.nan)),
                    'max': float(values.max()),
                    'missing': missing,
                    'missing_pct': 100 * missing_rate,
                }

            def plot_distribution(series: pd.Series, col: str) -> None:
                values = series.dropna().astype(float).values
                slug = safe_slug(col)
                out_path = FIGURES_DIR / f'univariate_dist__{slug}.png'

                fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
                ax_hist, ax_box = axes

                if values.size == 0:
                    ax_hist.text(0.5, 0.5, 'No non-missing values', ha='center', va='center')
                    ax_box.text(0.5, 0.5, 'No non-missing values', ha='center', va='center')
                else:
                    ax_hist.hist(values, bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='white')
                    if np.unique(values).size >= 5:
                        xs = np.linspace(np.nanmin(values), np.nanmax(values), 200)
                        kde = gaussian_kde(values)
                        ax_hist.plot(xs, kde(xs), color='black', linewidth=1.2, label='KDE')
                        ax_hist.legend(loc='best', frameon=False)
                    ax_hist.set_title(f'Histogram + KDE: {col}')
                    ax_hist.set_xlabel(col)
                    ax_hist.set_ylabel('Density')

                    ax_box.boxplot(values, vert=False, showfliers=True)
                    ax_box.set_title('Boxplot (outliers)')
                    ax_box.set_xlabel(col)

                fig.tight_layout()
                fig.savefig(out_path, dpi=200)
                plt.close(fig)

            def plot_time_trend(frame: pd.DataFrame, col: str) -> None:
                slug = safe_slug(col)
                out_path = FIGURES_DIR / f'univariate_trend__{slug}.png'

                tmp = frame[['year', col]].dropna()
                if tmp.empty:
                    fig, ax = plt.subplots(figsize=(10, 3.6))
                    ax.text(0.5, 0.5, f'No non-missing values for {col}', ha='center', va='center')
                    ax.set_axis_off()
                    fig.tight_layout()
                    fig.savefig(out_path, dpi=200)
                    plt.close(fig)
                    return

                grouped = tmp.groupby('year')[col]
                summary = pd.DataFrame({
                    'mean': grouped.mean(),
                    'median': grouped.median(),
                    'n': grouped.size(),
                    'p25': grouped.quantile(0.25),
                    'p75': grouped.quantile(0.75),
                }).reset_index()

                fig, ax = plt.subplots(figsize=(10, 3.8))
                ax.plot(summary['year'], summary['mean'], label='Mean', color='steelblue')
                ax.plot(summary['year'], summary['median'], label='Median', color='darkorange', linewidth=1.2)

                ok_band = summary['n'] >= 10
                if ok_band.any():
                    ax.fill_between(
                        summary.loc[ok_band, 'year'],
                        summary.loc[ok_band, 'p25'],
                        summary.loc[ok_band, 'p75'],
                        color='gray',
                        alpha=0.15,
                        label='IQR band (n>=10)',
                    )

                ax.set_title(f'Panel-average time trend: {col}')
                ax.set_xlabel('Year')
                ax.legend(loc='best', frameon=False)
                fig.tight_layout()
                fig.savefig(out_path, dpi=200)
                plt.close(fig)

            rows = []
            for col in numeric_cols:
                summary = compute_univariate_summary(df[col])
                rows.append({'variable': col, **summary})
                plot_distribution(df[col], col)
                plot_time_trend(df, col)

            univariate_summary = pd.DataFrame(rows).sort_values('variable').reset_index(drop=True)
            univariate_summary_path = OUTPUTS_DIR / 'univariate_summary.csv'
            univariate_summary.to_csv(univariate_summary_path, index=False)
            print('Saved', univariate_summary_path)
            univariate_summary.head(15)
            """
        ),
        markdown_cell(
            """
            ### Robustness Variant Checks (Winsorized vs Raw)

            When both raw and winsorized variants exist for the same base variable, compare their distributions.
            """
        ),
        code_cell(
            """
            def find_winsor_pairs(columns: list[str]) -> list[tuple[str, str]]:
                pairs = []
                for col in columns:
                    if col.endswith('_winsorized'):
                        base = col[:-len('_winsorized')]
                        if base in columns:
                            pairs.append((base, col))
                return sorted(pairs)

            winsor_pairs = find_winsor_pairs(numeric_cols)
            winsor_pairs[:10], len(winsor_pairs)
            """
        ),
        code_cell(
            """
            def compare_winsor_pair(raw_col: str, wins_col: str) -> None:
                raw = df[raw_col].dropna().astype(float)
                win = df[wins_col].dropna().astype(float)
                fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
                for ax, series, title in [
                    (axes[0], raw, f'Raw: {raw_col}'),
                    (axes[1], win, f'Winsorized: {wins_col}'),
                ]:
                    if series.empty:
                        ax.text(0.5, 0.5, 'No values', ha='center', va='center')
                        continue
                    ax.hist(series.values, bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='white')
                    if np.unique(series.values).size >= 5:
                        xs = np.linspace(series.min(), series.max(), 200)
                        kde = gaussian_kde(series.values)
                        ax.plot(xs, kde(xs), color='black', linewidth=1.2)
                    ax.set_title(title)
                fig.tight_layout()
                out_path = FIGURES_DIR / f'univariate_dist_compare__{safe_slug(raw_col)}.png'
                fig.savefig(out_path, dpi=200)
                plt.close(fig)

            for raw_col, wins_col in winsor_pairs:
                compare_winsor_pair(raw_col, wins_col)

            print(f'Generated {len(winsor_pairs)} winsorized vs raw comparison plots (if any).')
            """
        ),
        markdown_cell(
            """
            ## 3) Bivariate Analysis (Pairs)

            Deliverables:
            - `outputs/correlation_matrix_numeric.csv`
            - `outputs/correlation_pairwise_n_numeric.csv`
            - `outputs/figures/bivariate_fdi_vs__{col}.png`
            - `outputs/figures/bivariate_bm_growth_vs__{col}.png`
            - `outputs/top_correlations_with_fdi.csv`
            """
        ),
        code_cell(
            """
            numeric_df = df[numeric_cols].copy()

            corr = numeric_df.corr()
            corr_path = OUTPUTS_DIR / 'correlation_matrix_numeric.csv'
            corr.to_csv(corr_path)
            print('Saved', corr_path)

            pairwise_n = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype=int)
            for i, a in enumerate(numeric_cols):
                a_vals = numeric_df[a]
                for b in numeric_cols[i:]:
                    n = int(pd.concat([a_vals, numeric_df[b]], axis=1).dropna().shape[0])
                    pairwise_n.loc[a, b] = n
                    pairwise_n.loc[b, a] = n

            pairwise_n_path = OUTPUTS_DIR / 'correlation_pairwise_n_numeric.csv'
            pairwise_n.to_csv(pairwise_n_path)
            print('Saved', pairwise_n_path)

            corr.iloc[:8, :8]
            """
        ),
        code_cell(
            """
            def choose_fdi_column(columns: list[str]) -> str:
                if 'fdi_pct_gdp_winsorized' in columns:
                    return 'fdi_pct_gdp_winsorized'
                if 'fdi_pct_gdp' in columns:
                    return 'fdi_pct_gdp'
                raise ValueError('Could not find FDI column: expected fdi_pct_gdp(_winsorized).')

            fdi_col = choose_fdi_column(df.columns.tolist())
            fdi_col
            """
        ),
        code_cell(
            """
            def scatter_with_fit(x: pd.Series, y: pd.Series, x_name: str, y_name: str, out_path: Path) -> None:
                tmp = pd.concat([x.rename(x_name), y.rename(y_name)], axis=1).dropna()
                n = int(tmp.shape[0])
                corr_xy = float(tmp[x_name].corr(tmp[y_name])) if n >= 2 else np.nan

                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                ax.scatter(tmp[x_name], tmp[y_name], alpha=0.35, s=14, color='steelblue')

                if n >= 2 and np.isfinite(tmp[x_name]).all() and np.isfinite(tmp[y_name]).all():
                    slope, intercept = np.polyfit(tmp[x_name], tmp[y_name], 1)
                    xs = np.linspace(tmp[x_name].min(), tmp[x_name].max(), 100)
                    ax.plot(xs, intercept + slope * xs, color='darkorange', linewidth=1.8, label='OLS fit')
                    ax.legend(loc='best', frameon=False)

                ax.set_xlabel(x_name)
                ax.set_ylabel(y_name)
                ax.set_title(f'{y_name} vs {x_name}')
                ax.text(
                    0.02,
                    0.98,
                    f'corr={corr_xy:,.3f}\\nN={n}',
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
                )
                fig.tight_layout()
                fig.savefig(out_path, dpi=200)
                plt.close(fig)

            # FDI vs X sweep
            for col in numeric_cols:
                if col == fdi_col or col.startswith('fdi'):
                    continue
                out_path = FIGURES_DIR / f'bivariate_fdi_vs__{safe_slug(col)}.png'
                scatter_with_fit(df[fdi_col], df[col], fdi_col, col, out_path)

            # Broad money growth vs X sweep (if present)
            if 'broad_money_growth_pct' in df.columns:
                bm_col = 'broad_money_growth_pct'
                for col in numeric_cols:
                    if col == bm_col:
                        continue
                    out_path = FIGURES_DIR / f'bivariate_bm_growth_vs__{safe_slug(col)}.png'
                    scatter_with_fit(df[bm_col], df[col], bm_col, col, out_path)
            else:
                print('broad_money_growth_pct not found; skipping BM-growth sweep.')

            print('Bivariate sweeps complete.')
            """
        ),
        code_cell(
            """
            # Top correlations with FDI (exclude mechanically related FDI columns)
            fdi_corr = corr[fdi_col].drop(index=[c for c in corr.index if c.startswith('fdi')], errors='ignore')
            fdi_corr = fdi_corr.dropna()
            top_k = 12
            top_pos = fdi_corr.sort_values(ascending=False).head(top_k)
            top_neg = fdi_corr.sort_values(ascending=True).head(top_k)
            top_table = (
                pd.concat(
                    [
                        top_pos.rename('corr').to_frame().assign(direction='positive'),
                        top_neg.rename('corr').to_frame().assign(direction='negative'),
                    ]
                )
                .reset_index()
                .rename(columns={'index': 'variable'})
            )
            top_table['pairwise_n'] = top_table['variable'].map(lambda v: int(pairwise_n.loc[fdi_col, v]) if v in pairwise_n.columns else np.nan)
            top_out_path = OUTPUTS_DIR / 'top_correlations_with_fdi.csv'
            top_table.to_csv(top_out_path, index=False)
            print('Saved', top_out_path)
            top_table
            """
        ),
        markdown_cell(
            """
            ## 4) Multivariate Analysis (Correlation Structure)

            Deliverables:
            - `outputs/figures/correlation_heatmap_numeric.png`
            - `outputs/high_correlation_pairs.csv` (|corr| >= 0.7 by default)
            - `outputs/vif_learning_check.csv` (if columns available)
            """
        ),
        code_cell(
            """
            # Heatmap
            n = len(numeric_cols)
            fig_w = max(10, min(24, 0.55 * n))
            fig_h = max(8, min(22, 0.55 * n))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
            ax.set_yticklabels(numeric_cols, fontsize=7)
            ax.set_title('Correlation heatmap (numeric columns)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            heatmap_path = FIGURES_DIR / 'correlation_heatmap_numeric.png'
            fig.savefig(heatmap_path, dpi=220)
            plt.close(fig)
            print('Saved', heatmap_path)
            """
        ),
        code_cell(
            """
            # High-correlation edges table
            threshold = 0.7
            pairs = []
            for i, a in enumerate(numeric_cols):
                for j in range(i + 1, len(numeric_cols)):
                    b = numeric_cols[j]
                    value = corr.loc[a, b]
                    if pd.isna(value):
                        continue
                    if abs(value) >= threshold:
                        pairs.append(
                            {
                                'var_a': a,
                                'var_b': b,
                                'corr': float(value),
                                'abs_corr': float(abs(value)),
                                'pairwise_n': int(pairwise_n.loc[a, b]),
                            }
                        )
            high_corr_pairs = pd.DataFrame(pairs).sort_values(['abs_corr', 'pairwise_n'], ascending=[False, False]).reset_index(drop=True)
            high_corr_path = OUTPUTS_DIR / 'high_correlation_pairs.csv'
            high_corr_pairs.to_csv(high_corr_path, index=False)
            print('Saved', high_corr_path, f'(threshold={threshold})')
            high_corr_pairs.head(25)
            """
        ),
        code_cell(
            """
            # Collinearity quick check: VIF on workbook-core controls (learning-oriented)
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                from statsmodels.tools.tools import add_constant
            except Exception as exc:
                variance_inflation_factor = None
                print('statsmodels not available; cannot compute VIF:', exc)

            core_controls = [
                'broad_money_growth_pct',
                'inflation_gdp_deflator_pct',
                'trade_pct_gdp',
                'ln_gdppc',
                'xr_dep_pct',
            ]
            rate_proxy_candidates = ['real_interest_rate_pct', 'deposit_interest_rate_pct', 'lending_interest_rate_pct']
            rate_proxy = next((c for c in rate_proxy_candidates if c in df.columns), None)
            if rate_proxy is not None and rate_proxy not in core_controls:
                core_controls = core_controls + [rate_proxy]

            missing = [c for c in core_controls if c not in df.columns]
            if missing:
                print('Skipping VIF: missing required columns:', missing)
            elif variance_inflation_factor is None:
                print('Skipping VIF: statsmodels missing.')
            else:
                X = df[core_controls].dropna().astype(float)
                if X.shape[0] < 10:
                    print(f'Skipping VIF: too few complete rows for VIF (n={X.shape[0]}).')
                else:
                    Xc = add_constant(X, has_constant='add')
                    vif_rows = []
                    for i, col in enumerate(Xc.columns):
                        if col == 'const':
                            continue
                        vif_rows.append({'variable': col, 'vif': float(variance_inflation_factor(Xc.values, i)), 'n': int(X.shape[0])})
                    vif_df = pd.DataFrame(vif_rows).sort_values('vif', ascending=False).reset_index(drop=True)
                    vif_path = OUTPUTS_DIR / 'vif_learning_check.csv'
                    vif_df.to_csv(vif_path, index=False)
                    print('Saved', vif_path)
                    vif_df
            """
        ),
    ]
    write_notebook("03_descriptive_statistics.ipynb", cells)


def rewrite_notebook_04() -> None:
    cells = [
        markdown_cell(
            """
            # 04 Methodology, Results, and Diagnostics

            This notebook matches the "current pipeline + workbook contract" style used in notebooks 01–03:

            - Load the processed panel contract (`data/processed/clean_panel.csv`)
            - Assert the active window (`2000–2023`)
            - Build the **workbook-driven model catalog** (source of truth for which models exist / are estimated)
            - Load exported estimation artifacts produced by `02_panel_diagnostics.ipynb` (if present)
            - Save a small set of deterministic **CSV + PNG** artifacts under `outputs/`
            """
        ),
        markdown_cell("## Setup"),
        code_cell(
            """
            from pathlib import Path
            import sys
            import importlib

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from IPython.display import Markdown, display

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            import src.config as config
            import src.model_contract as model_contract
            import src.reporting as reporting

            importlib.reload(config)  # stale-kernel guard
            importlib.reload(model_contract)
            importlib.reload(reporting)

            OUTPUTS_DIR = config.OUTPUTS_DIR
            FIGURES_DIR = config.FIGURES_DIR
            PROCESSED_PANEL_FILE = config.PROCESSED_PANEL_FILE
            TIME_WINDOW = config.TIME_WINDOW
            ensure_output_dirs = config.ensure_output_dirs
            compact_display = reporting.compact_display
            VARIABLE_LABELS = reporting.VARIABLE_LABELS

            ensure_output_dirs()
            pd.set_option("display.max_columns", None)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            sns.set_theme(style="whitegrid", context="talk")
            """
        ),
        markdown_cell(
            """
            ## Load Processed Panel (New Data Contract)

            Source of truth: `data/processed/clean_panel.csv` produced by `01_data_processing.ipynb`.
            """
        ),
        code_cell(
            """
            df = pd.read_csv(PROCESSED_PANEL_FILE)
            df.shape, int(df['year'].min()), int(df['year'].max())
            """
        ),
        code_cell(
            """
            start_year, end_year = TIME_WINDOW
            year_min, year_max = int(df['year'].min()), int(df['year'].max())
            if (year_min, year_max) != (start_year, end_year):
                raise ValueError(f'Expected year window {TIME_WINDOW}, got {(year_min, year_max)}. Re-run 01_data_processing.ipynb.')

            snapshot_vars = [
                'fdi_pct_gdp',
                'broad_money_growth_pct',
                'inflation_gdp_deflator_pct',
                'trade_pct_gdp',
                'ln_gdppc',
                'xr_dep_pct',
                'real_interest_rate_pct',
                'deposit_interest_rate_pct',
                'lending_interest_rate_pct',
                'ln_tourism_arrivals',
                'hc_human_capital_index',
            ]
            snapshot_vars = [c for c in snapshot_vars if c in df.columns]
            coverage = (
                df[snapshot_vars]
                .notna()
                .sum()
                .rename('non_missing')
                .to_frame()
                .assign(non_missing_pct=lambda frame: 100 * frame['non_missing'] / len(df))
                .reset_index()
                .rename(columns={'index': 'variable'})
                .sort_values(['non_missing', 'variable'], ascending=[True, True])
                .reset_index(drop=True)
            )
            coverage
            """
        ),
        markdown_cell(
            """
            ## Workbook Model Contract (Source of Truth)

            The workbook (`model_selection_asean_fdi.xlsx`) defines which specifications exist and which ones can be estimated
            given the processed panel columns.
            """
        ),
        code_cell(
            """
            catalog, workbook_tables = model_contract.build_workbook_model_catalog(
                config.MODEL_SELECTION_WORKBOOK,
                panel_columns=df.columns.tolist(),
            )

            view_cols = [
                'model_id',
                'workbook_code',
                'workbook_model',
                'status',
                'missing_reason',
                'lagged_model',
                'mapped_regressors',
            ]
            compact_display(catalog, columns=view_cols, n=30)

            estimated_catalog = catalog[catalog['status'].eq('estimated')].copy()
            estimated_model_ids = estimated_catalog['model_id'].tolist()
            print('Estimated models:', len(estimated_model_ids))
            estimated_catalog[['model_id', 'workbook_code', 'workbook_model', 'mapped_regressors']].head(10)
            """
        ),
        code_cell(
            """
            # Prefer module outputs written by notebook 02 (source of truth on disk).
            workbook_catalog_path = OUTPUTS_DIR / 'workbook_model_catalog.csv'
            if workbook_catalog_path.exists():
                workbook_catalog_disk = pd.read_csv(workbook_catalog_path)
                compact_display(workbook_catalog_disk, columns=view_cols, n=30)
            else:
                print('MISSING:', workbook_catalog_path.name, '(run 02_panel_diagnostics.ipynb)')
            """
        ),
        markdown_cell(
            """
            ## Load Exported Estimation Artifacts (Produced by Notebook 02)

            If these files are missing, run `02_panel_diagnostics.ipynb` first.
            """
        ),
        code_cell(
            """
            def read_csv_if_exists(path: Path, **kwargs):
                if path.exists():
                    return pd.read_csv(path, **kwargs)
                print('MISSING:', path.name, '(run 02_panel_diagnostics.ipynb)')
                return None

            regression_table_main = read_csv_if_exists(OUTPUTS_DIR / 'regression_table_main.csv', index_col=0)
            model_coefficients = read_csv_if_exists(OUTPUTS_DIR / 'model_coefficients.csv')
            model_fit_stats = read_csv_if_exists(OUTPUTS_DIR / 'model_fit_stats.csv')
            model_sample_audit = read_csv_if_exists(OUTPUTS_DIR / 'model_sample_audit.csv')
            low_gap_interpretations = read_csv_if_exists(OUTPUTS_DIR / 'low_gap_coefficient_interpretations.csv')
            """
        ),
        code_cell(
            """
            def filter_to_estimated(frame):
                if frame is None:
                    return None
                if 'model_id' not in frame.columns:
                    return frame
                return frame[frame['model_id'].isin(estimated_model_ids)].copy()

            model_coefficients_est = filter_to_estimated(model_coefficients)
            model_fit_stats_est = filter_to_estimated(model_fit_stats)
            model_sample_audit_est = filter_to_estimated(model_sample_audit)

            if model_fit_stats_est is not None:
                compact_display(model_fit_stats_est, n=25)
            """
        ),
        markdown_cell(
            """
            ## Per-Model Regression Tables (Module Exports)

            Notebook 02 exports raw, text-based regression output and a short text note for each workbook-estimated model:

            - `{model_id}_regression_raw.txt`
            - `{model_id}_regression_detail_note.txt`

            Display these per-model outputs first (as raw module text), then show the consolidated summary table.
            """
        ),
        code_cell(
            """
            workbook_name_lookup = estimated_catalog.set_index('model_id')['workbook_model'].to_dict()

            any_missing = False
            for model_id in estimated_model_ids:
                raw_path = OUTPUTS_DIR / f'{model_id}_regression_raw.txt'
                note_path = OUTPUTS_DIR / f'{model_id}_regression_detail_note.txt'

                if not raw_path.exists() and not note_path.exists():
                    any_missing = True
                    continue

                workbook_name = workbook_name_lookup.get(model_id, model_id)
                display(Markdown(f\"### {workbook_name}  \\\\n`{model_id}`\"))

                if note_path.exists():
                    note = note_path.read_text().strip()
                    if note:
                        display(Markdown('**Model note**'))
                        print(note)

                if raw_path.exists():
                    raw_text = raw_path.read_text().rstrip()
                    if raw_text:
                        print(raw_text)
                    else:
                        print('(empty)', raw_path.name)
                else:
                    print('MISSING:', raw_path.name)

            if any_missing:
                print('Some per-model regression raw outputs are missing. Re-run 02_panel_diagnostics.ipynb if needed.')
            """
        ),
        markdown_cell(
            """
            ## Summary Table (Module Export)

            Consolidated view exported by notebook 02: `outputs/regression_table_main.csv`.
            """
        ),
        code_cell(
            """
            regression_table_main if regression_table_main is not None else 'Run notebook 02 first.'
            """
        ),
        markdown_cell(
            """
            ## Note On Outputs

            This notebook does **not** synthesize new result tables; it reads and visualizes the module exports
            produced by notebook 02 under `outputs/`.
            """
        ),
        markdown_cell(
            """
            ## Diagnostics (Module Outputs)

            Use the consolidated exports produced by `src.estimate_and_export` (not per-model file globs).
            """
        ),
        code_cell(
            """
            model_diagnostics = read_csv_if_exists(OUTPUTS_DIR / 'model_diagnostics.csv')
            model_vif = read_csv_if_exists(OUTPUTS_DIR / 'model_vif.csv')
            model_panel_balance = read_csv_if_exists(OUTPUTS_DIR / 'model_panel_balance_summary.csv')

            model_diagnostics_est = filter_to_estimated(model_diagnostics)
            model_vif_est = filter_to_estimated(model_vif)
            model_panel_balance_est = filter_to_estimated(model_panel_balance)

            compact_display(model_diagnostics_est, n=12) if model_diagnostics_est is not None else 'Run notebook 02 first.'
            """
        ),
        markdown_cell(
            """
            ## Summary Figures (PNG Only)

            Produce two compact figures:

            - Sample sizes by model
            - Key coefficients across models (with CI when available)
            """
        ),
        code_cell(
            """
            def preferred_estimator_per_model(fit_stats):
                if fit_stats is None or fit_stats.empty:
                    return {}
                preferred = {}
                for model_id, grp in fit_stats.groupby('model_id'):
                    if grp['estimator'].astype(str).str.contains('random_effects').any():
                        preferred[model_id] = 'random_effects'
                    elif grp['estimator'].astype(str).str.contains('fixed_effects').any():
                        preferred[model_id] = 'fixed_effects'
                    else:
                        preferred[model_id] = grp['estimator'].astype(str).iloc[0]
                return preferred

            def load_sample_fit_summary_fallback() -> pd.DataFrame:
                rows = []
                for path in sorted(OUTPUTS_DIR.glob('*_sample_fit_summary.csv')):
                    tmp = pd.read_csv(path)
                    if 'model_id' not in tmp.columns:
                        tmp = tmp.assign(model_id=path.name.replace('_sample_fit_summary.csv', ''))
                    rows.append(tmp)
                if not rows:
                    return pd.DataFrame()
                return pd.concat(rows, ignore_index=True)

            def sample_sizes_table(fit_stats, estimated_ids, model_order):
                if fit_stats is not None and not fit_stats.empty and {'model_id', 'nobs'}.issubset(fit_stats.columns):
                    preferred_map = preferred_estimator_per_model(fit_stats)
                    tmp = fit_stats.copy()
                    tmp = tmp[tmp['model_id'].isin(estimated_ids)].copy()
                    if 'estimator' in tmp.columns and preferred_map:
                        tmp = tmp[tmp.apply(lambda row: str(row['estimator']) == preferred_map.get(row['model_id'], str(row['estimator'])), axis=1)].copy()
                    out = (
                        tmp.groupby('model_id', as_index=False)['nobs']
                        .max()
                        .rename(columns={'nobs': 'nobs'})
                    )
                    out['nobs'] = out['nobs'].astype(float).round().astype(int)
                else:
                    fallback = load_sample_fit_summary_fallback()
                    if fallback.empty or 'nobs' not in fallback.columns:
                        return pd.DataFrame(columns=['model_id', 'nobs'])
                    out = fallback[['model_id', 'nobs']].copy()
                    out = out[out['model_id'].isin(estimated_ids)].copy()
                    out['nobs'] = out['nobs'].astype(float).round().astype(int)

                out['model_id'] = out['model_id'].astype(str)
                order_index = {m: i for i, m in enumerate(model_order)}
                out['order'] = out['model_id'].map(order_index).fillna(1e9)
                out = out.sort_values(['order', 'model_id']).drop(columns=['order']).reset_index(drop=True)
                return out

            model_order = estimated_catalog['model_id'].tolist()
            nobs_table = sample_sizes_table(model_fit_stats_est, estimated_model_ids, model_order)
            nobs_table
            """
        ),
        code_cell(
            """
            if not nobs_table.empty:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.bar(nobs_table['model_id'], nobs_table['nobs'], color='steelblue')
                ax.set_title('Sample sizes (nobs) by model')
                ax.set_xlabel('Model')
                ax.set_ylabel('Observations (nobs)')
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')
                fig.tight_layout()
                out_path = FIGURES_DIR / 'methodology__sample_sizes_by_model.png'
                fig.savefig(out_path, dpi=220)
                plt.close(fig)
                print('Saved', out_path)
            else:
                print('No sample-size information available (missing model_fit_stats + *_sample_fit_summary.csv).')
            """
        ),
        code_cell(
            """
            def key_terms_present(coeff_df: pd.DataFrame, requested: list[str]) -> list[str]:
                present = set(coeff_df['term'].astype(str).unique())
                return [term for term in requested if term in present]

            requested_terms = [
                'broad_money_growth_pct',
                'inflation_gdp_deflator_pct',
                'trade_pct_gdp',
                'ln_gdppc',
                'xr_dep_pct',
                # optional / robustness terms
                'real_interest_rate_pct',
                'deposit_interest_rate_pct',
                'lending_interest_rate_pct',
                'ln_tourism_arrivals',
                'hc_human_capital_index',
            ]

            if model_coefficients_est is None or model_coefficients_est.empty:
                print('No model_coefficients available (run 02_panel_diagnostics.ipynb).')
            else:
                preferred_fit = preferred_estimator_per_model(model_fit_stats_est) if model_fit_stats_est is not None else {}
                coeff = model_coefficients_est.copy()
                if preferred_fit and 'estimator' in coeff.columns:
                    coeff = coeff[coeff.apply(lambda row: str(row['estimator']) == preferred_fit.get(row['model_id'], str(row['estimator'])), axis=1)].copy()

                terms = key_terms_present(coeff, requested_terms)
                if not terms:
                    raise ValueError('None of the requested key terms are present in model_coefficients.csv.')

                coeff = coeff[coeff['term'].isin(terms)].copy()
                coeff['term_label'] = coeff['term'].map(lambda t: VARIABLE_LABELS.get(t, t))
                order_index = {m: i for i, m in enumerate(model_order)}
                coeff['model_order'] = coeff['model_id'].map(order_index).fillna(1e9)
                coeff = coeff.sort_values(['term', 'model_order', 'model_id']).reset_index(drop=True)
                coeff.head(10)
            """
        ),
        code_cell(
            """
            if model_coefficients_est is not None and not model_coefficients_est.empty:
                fig, ax = plt.subplots(figsize=(13, 5))

                # Scatter + CI by term (small number of terms and models, so a single axis is readable).
                x_positions = {m: i for i, m in enumerate(model_order)}
                x = coeff['model_id'].map(x_positions)

                term_labels = list(dict.fromkeys(coeff['term_label'].tolist()))
                palette = sns.color_palette('tab10', n_colors=max(3, len(term_labels)))
                color_map = {label: palette[i % len(palette)] for i, label in enumerate(term_labels)}

                for label in term_labels:
                    tmp = coeff[coeff['term_label'].eq(label)]
                    xs = tmp['model_id'].map(x_positions).astype(float).values
                    ys = tmp['coef'].astype(float).values
                    ax.scatter(xs, ys, label=label, color=color_map[label], s=45)
                    if {'ci_low', 'ci_high'}.issubset(tmp.columns):
                        yerr_low = ys - tmp['ci_low'].astype(float).values
                        yerr_high = tmp['ci_high'].astype(float).values - ys
                        ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], fmt='none', ecolor=color_map[label], alpha=0.55, linewidth=1)

                ax.axhline(0, color='black', linewidth=1, alpha=0.6)
                ax.set_xticks(list(x_positions.values()))
                ax.set_xticklabels(list(x_positions.keys()), rotation=45, ha='right')
                ax.set_title('Key coefficients across workbook-estimated models (preferred estimator)')
                ax.set_xlabel('Model')
                ax.set_ylabel('Coefficient (with 95% CI when available)')
                ax.legend(loc='best', frameon=True, fontsize=9)

                fig.tight_layout()
                out_path = FIGURES_DIR / 'methodology__key_coefficients_across_models.png'
                fig.savefig(out_path, dpi=220)
                plt.close(fig)
                print('Saved', out_path)
            """
        ),
        markdown_cell("## Optional: Interpretation Tables (If Present)"),
        code_cell(
            """
            compact_display(model_sample_audit_est, n=25) if model_sample_audit_est is not None else 'Run notebook 02 first.'
            """
        ),
        code_cell(
            """
            compact_display(low_gap_interpretations, n=25) if low_gap_interpretations is not None else 'Run notebook 02 first.'
            """
        ),
    ]
    write_notebook("04_methodology_results_and_diagnostics_refactored.ipynb", cells)


def rewrite_notebook_05() -> None:
    cells = [
        markdown_cell(
            """
            # 05 Limitations: Missing Data And Sample Reliability

            This notebook now reads the current preprocessing artifacts produced by `src.reprocess_current_raw` and focuses on the remaining sample limitations after rebuilding the `2000-2023` panel from the new raw files.
            """
        ),
        markdown_cell(
            """
            ## Setup

            Load the clean panel and the export-side audit tables produced during preprocessing.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            import pandas as pd

            ROOT = Path.cwd().resolve()
            if ROOT.name == "notebooks":
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.config import OUTPUTS_DIR, PROCESSED_PANEL_FILE

            pd.set_option("display.max_columns", None)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

            clean_panel = pd.read_csv(PROCESSED_PANEL_FILE)
            raw_input_audit = pd.read_csv(OUTPUTS_DIR / "raw_input_audit.csv")
            variable_audit = pd.read_csv(OUTPUTS_DIR / "variable_audit.csv")
            control_imputation_log = pd.read_csv(OUTPUTS_DIR / "control_imputation_log.csv")
            coverage_by_country = pd.read_csv(OUTPUTS_DIR / "coverage_by_country.csv")
            transformation_audit = pd.read_csv(OUTPUTS_DIR / "transformation_audit.csv")
            review_flags = pd.read_csv(OUTPUTS_DIR / "review_flags.csv")
            """
        ),
        markdown_cell(
            """
            ## Source Coverage

            Start from the file-level audit so it is clear which source drives each remaining structural gap.
            """
        ),
        code_cell(
            """
            raw_input_audit
            """
        ),
        markdown_cell(
            """
            ## Variable-Level Missingness

            Rank variables by remaining missing share after the current preprocessing rules.
            """
        ),
        code_cell(
            """
            variable_audit.sort_values(["missing_rate", "variable"], ascending=[False, True]).reset_index(drop=True)
            """
        ),
        markdown_cell(
            """
            ## Imputation Log

            Separate variables that were actually filled from variables that remain structurally incomplete.
            """
        ),
        code_cell(
            """
            control_imputation_log
            """
        ),
        markdown_cell(
            """
            ## Country Coverage

            Show how much usable information remains by country across the main variables carried into later notebooks.
            """
        ),
        code_cell(
            """
            coverage_by_country
            """
        ),
        markdown_cell(
            """
            ## Remaining Gaps

            Summarize which countries still have zero coverage for the variables that matter most for the expanded panel.
            """
        ),
        code_cell(
            """
            key_variables = [
                "deposit_interest_rate_pct",
                "real_interest_rate_pct",
                "lending_interest_rate_pct",
                "ln_tourism_arrivals",
                "hc_human_capital_index",
                "ln_population_total",
            ]

            zero_coverage_rows = []
            for variable in key_variables:
                zero_countries = coverage_by_country.loc[coverage_by_country[variable].eq(0), "country"].tolist()
                zero_coverage_rows.append(
                    {
                        "variable": variable,
                        "countries_with_zero_coverage": ", ".join(zero_countries) if zero_countries else "",
                        "zero_coverage_country_count": len(zero_countries),
                    }
                )

            pd.DataFrame(zero_coverage_rows)
            """
        ),
        markdown_cell(
            """
            ## Transformation And Review Audit

            Keep the processing decisions visible for later writeup and diagnostics.
            """
        ),
        code_cell(
            """
            transformation_audit
            """
        ),
        code_cell(
            """
            review_flags
            """
        ),
        code_cell(
            """
            pd.DataFrame(
                {
                    "metric": ["rows", "countries", "first_year", "last_year"],
                    "value": [
                        len(clean_panel),
                        clean_panel["country"].nunique(),
                        int(clean_panel["year"].min()),
                        int(clean_panel["year"].max()),
                    ],
                }
            )
            """
        ),
    ]
    write_notebook("05_limitations_missing_data.ipynb", cells)


def main() -> None:
    rewrite_notebook_01()
    rewrite_notebook_02()
    rewrite_notebook_03()
    rewrite_notebook_04()
    rewrite_notebook_05()


if __name__ == "__main__":
    main()
