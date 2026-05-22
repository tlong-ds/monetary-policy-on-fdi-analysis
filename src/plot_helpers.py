from __future__ import annotations

from pathlib import Path
import os

from .config import ROOT

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib_cache'))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR


def setup_matplotlib_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig, name: str, output_dir: Path | None = None, *, dpi: int = 200, bbox_inches: str = "tight") -> Path:
    output_path = (output_dir or FIGURES_DIR) / name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    return output_path


def correlation_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "coolwarm",
):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig, ax


def coefficient_forest_plot(
    frame: pd.DataFrame,
    *,
    estimate_col: str = "coef",
    lower_col: str = "ci_lower",
    upper_col: str = "ci_upper",
    label_col: str = "coefficient_label",
    title: str = "Coefficient Estimates",
    figsize: tuple[int, int] = (9, 6),
):
    plot_frame = frame.dropna(subset=[estimate_col]).copy()
    plot_frame = plot_frame.iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(plot_frame))
    xerr = None
    if lower_col in plot_frame and upper_col in plot_frame:
        xerr = [
            plot_frame[estimate_col] - plot_frame[lower_col],
            plot_frame[upper_col] - plot_frame[estimate_col],
        ]
    ax.errorbar(plot_frame[estimate_col], y_pos, xerr=xerr, fmt="o", color="#1f77b4", ecolor="#4c78a8")
    ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_frame[label_col] if label_col in plot_frame else plot_frame.index.astype(str))
    ax.set_title(title)
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    return fig, ax
