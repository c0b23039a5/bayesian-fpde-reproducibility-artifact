from __future__ import annotations

import math

import pandas as pd

from bayesian_fpde.plotting import _mean_ci_summary, save_mean_ci_line_plot


def test_mean_ci_summary_aggregates_by_x_and_group():
    df = pd.DataFrame(
        {
            "method": ["a", "a", "a", "b"],
            "train_fraction": [0.5, 0.5, 1.0, 0.5],
            "mean_ci_width": [1.0, 3.0, 5.0, 10.0],
        }
    )

    summary = _mean_ci_summary(df, x="train_fraction", y="mean_ci_width", group="method", ci=1.96)
    row = summary[(summary["method"] == "a") & (summary["train_fraction"] == 0.5)].iloc[0]

    assert row["mean"] == 2.0
    assert row["count"] == 2
    assert math.isclose(row["sem"], 1.0)
    assert math.isclose(row["ci"], 1.96)


def test_save_mean_ci_line_plot_writes_png(tmp_path):
    df = pd.DataFrame(
        {
            "method": ["a", "a", "b", "b"],
            "x": [1, 2, 1, 2],
            "y": [0.5, 0.4, 0.7, 0.6],
        }
    )
    path = tmp_path / "mean_ci.png"

    save_mean_ci_line_plot(df, x="x", y="y", group="method", path=path, title="Mean CI")

    assert path.exists()
    assert path.stat().st_size > 0
