"""Figures for the real-data study."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from bpk.study.run import Built

INK, MUTED, GRID, SURFACE, BLUE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb", "#2a78d6"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_figures(b: Built, daily: pd.DataFrame, gaps: dict[str, list[float]], out: Path) -> None:
    _style()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. the pricing kernel: median and interquartile range across dates
    logk = np.log(b.kernels)
    logk = logk - logk[:, logk.shape[1] // 2 : logk.shape[1] // 2 + 1]
    u = np.linspace(0.0, 1.0, logk.shape[1])
    q25, q50, q75 = np.percentile(logk, [25, 50, 75], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(u, q25, q75, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(u, q50, color=BLUE, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["5th percentile", "midpoint", "95th percentile"])
    ax.set_xlabel("6 month return, from the 5th to the 95th percentile of the physical density")
    ax.set_ylabel("Log pricing kernel, relative to the midpoint")
    ax.set_title(
        "Pricing kernel across dates: median and interquartile range", loc="left", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "pricing_kernel.png", dpi=150)
    plt.close(fig)

    # 2. the estimated curvature through time
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(daily.index, daily["curvature"], color=BLUE, linewidth=1.3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_title(
        "Probability weighting curvature (1 - alpha) by date; zero means no distortion",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "curvature.png", dpi=150)
    plt.close(fig)

    # 3. where each model misses, out of sample
    labels = ["P(fall of 20%+)", "10th pct", "median", "90th pct", "P(below +20%)"]
    names = {
        "none": "No risk adjustment",
        "crra": "CRRA",
        "habit": "Habit-style",
        "weight": "Probability weighting",
    }
    colors = {"none": "#8a8980", "crra": BLUE, "habit": "#eb6834", "weight": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.2
    for k, (m, lab) in enumerate(names.items()):
        ax.bar(
            np.arange(5) + (k - 1.5) * width,
            100 * np.array(gaps[m]),
            width * 0.9,
            color=colors[m],
            label=lab,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Percentage points")
    ax.set_title(
        "Model-implied minus published risk-neutral probability, out of sample mean",
        loc="left",
        fontsize=11,
    )
    ax.legend(frameon=False, ncol=4, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(fig_dir / "model_gaps.png", dpi=150)
    plt.close(fig)
