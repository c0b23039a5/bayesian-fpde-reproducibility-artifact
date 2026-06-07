from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from bayesian_fpde.plotting import save_line_plot
from bayesian_fpde.utils import read_csv_preserve_metadata


def _status_counts(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    counts = df.groupby(list(group_cols), dropna=False)["status"].value_counts().unstack(fill_value=0).reset_index()
    for status in ["ok", "skipped", "error"]:
        if status not in counts.columns:
            counts[status] = 0
    return counts.rename(columns={"ok": "status_ok", "skipped": "status_skipped", "error": "status_error"})


def _mean_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [col for col in group_cols if col in df.columns]
    if not group_cols:
        group_cols = ["method"] if "method" in df.columns else []
    exclude = {"seed", "task_id", "fold", "sample", "repeat", "explained_index", "explained_order", "feature_index"}
    numeric_cols = [col for col in df.select_dtypes(include="number").columns if col not in exclude and col not in group_cols]
    grouped = df.groupby(group_cols, dropna=False) if group_cols else df.groupby(lambda _: 0)
    first_col = df.columns[0]
    summary = grouped.agg(n_rows=(first_col, "size")).reset_index()
    if numeric_cols:
        means = grouped[numeric_cols].mean().reset_index().rename(columns={col: f"mean_{col}" for col in numeric_cols})
        summary = summary.merge(means, on=group_cols, how="left") if group_cols else summary.join(means.drop(columns=["index"], errors="ignore"))
    if "dataset_name" in df.columns and "dataset_name" not in group_cols:
        n_datasets = grouped["dataset_name"].nunique().reset_index(name="n_datasets")
        summary = summary.merge(n_datasets, on=group_cols, how="left") if group_cols else summary.join(n_datasets.drop(columns=["index"], errors="ignore"))
    if "seed" in df.columns and "seed" not in group_cols:
        n_seeds = grouped["seed"].nunique().reset_index(name="n_seeds")
        summary = summary.merge(n_seeds, on=group_cols, how="left") if group_cols else summary.join(n_seeds.drop(columns=["index"], errors="ignore"))
    status = _status_counts(df, group_cols)
    if not status.empty:
        summary = summary.merge(status, on=group_cols, how="left")
    if "metric_direction" not in summary.columns:
        summary["metric_direction"] = "mixed"
    return summary


def write_public_experiment_summaries(results_dir: str | Path, figures_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    specs = [
        ("public_uncertainty_validation.csv", "public_uncertainty_validation", [["dataset_name", "task_id", "method"], ["method"]]),
        ("stability_metrics.csv", "stability", [["dataset_name", "task_id", "method"], ["method"]]),
        ("faithfulness_metrics.csv", "faithfulness", [["dataset_name", "task_id", "method"], ["method"]]),
        ("training_size_uncertainty.csv", "training_size_uncertainty", [["dataset_name", "task_id", "method", "train_fraction"], ["method", "train_fraction"]]),
    ]
    for csv_name, stem, groupings in specs:
        path = results_dir / csv_name
        if not path.exists():
            continue
        df = read_csv_preserve_metadata(path)
        for group_cols in groupings:
            suffix = "method_summary" if group_cols == ["method"] or group_cols == ["method", "train_fraction"] else "summary"
            _mean_summary(df, group_cols).to_csv(results_dir / f"{stem}_{suffix}.csv", index=False, lineterminator="\n")

    method_summary_path = results_dir / "training_size_uncertainty_method_summary.csv"
    if method_summary_path.exists():
        summary = read_csv_preserve_metadata(method_summary_path)
        save_line_plot(
            summary,
            x="train_fraction",
            y="mean_mean_ci_width",
            group="method",
            path=figures_dir / "training_size_uncertainty_summary.png",
            title="Mean CI width vs training fraction",
        )
