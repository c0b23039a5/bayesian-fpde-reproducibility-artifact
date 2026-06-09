from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import ensure_dirs


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_metric_boxplot(df: pd.DataFrame, *, metric: str, path: str | Path, title: str) -> None:
    plt = _plt()
    ensure_dirs(Path(path).parent)
    fig, ax = plt.subplots(figsize=(8, 4))
    if df.empty or metric not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        df.boxplot(column=metric, by="method", ax=ax, rot=30)
        fig.suptitle("")
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_line_plot(df: pd.DataFrame, *, x: str, y: str, path: str | Path, title: str, group: Optional[str] = None) -> None:
    plt = _plt()
    ensure_dirs(Path(path).parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    if df.empty or x not in df.columns or y not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    elif group and group in df.columns:
        for label, sub in df.groupby(group):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[y], marker="o", label=str(label))
        ax.legend(fontsize=8)
    else:
        sub = df.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mean_ci_summary(df: pd.DataFrame, *, x: str, y: str, group: Optional[str] = None, ci: float = 1.96) -> pd.DataFrame:
    required = {x, y}
    if group:
        required.add(group)
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=([group] if group else []) + [x, "mean", "std", "count", "sem", "ci"])
    plot_df = df[list(required)].copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])
    if plot_df.empty:
        return pd.DataFrame(columns=([group] if group else []) + [x, "mean", "std", "count", "sem", "ci"])
    group_cols = [x] if not group else [group, x]
    summary = (
        plot_df.groupby(group_cols, dropna=False)[y]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].astype(float))
    summary["ci"] = (float(ci) * summary["sem"]).fillna(0.0)
    return summary


def save_mean_ci_line_plot(df: pd.DataFrame, *, x: str, y: str, path: str | Path, title: str, group: Optional[str] = None, ci: float = 1.96) -> None:
    plt = _plt()
    ensure_dirs(Path(path).parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    summary = _mean_ci_summary(df, x=x, y=y, group=group, ci=ci)
    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        if group:
            for label, sub in summary.groupby(group, dropna=False):
                sub = sub.sort_values(x)
                ax.errorbar(sub[x], sub["mean"], yerr=sub["ci"], marker="o", capsize=3, linewidth=1.5, label=str(label))
            ax.legend(fontsize=8)
        else:
            sub = summary.sort_values(x)
            ax.errorbar(sub[x], sub["mean"], yerr=sub["ci"], marker="o", capsize=3, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
