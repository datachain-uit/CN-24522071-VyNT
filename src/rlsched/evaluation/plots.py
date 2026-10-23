"""Biểu đồ chuẩn cho báo cáo: CDF của JCT, cột mean với khoảng tin cậy 95%, learning curve."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_jct_cdf(jobs: pd.DataFrame, out: Path, by: str = "method", title: str = "CDF of job completion time") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in jobs.groupby(by, sort=False):
        x = np.sort(g["jct_s"].dropna().to_numpy())
        if len(x):
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=str(name), linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Job completion time (s, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_mean_ci(summary: pd.DataFrame, out: Path, by: str = "method", ylabel: str = "Average JCT (s)") -> None:
    s = summary.sort_values("mean")
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(s)), 4))
    x = np.arange(len(s))
    yerr = np.vstack([s["mean"] - s["ci95_low"], s["ci95_high"] - s["mean"]])
    ax.bar(x, s["mean"], yerr=yerr, capsize=4, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(s[by], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title("Mean over test episodes (error bar: 95% bootstrap CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_learning_curves(curves: pd.DataFrame, out: Path, x: str = "iteration", y: str = "value",
                         by: str = "run") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in curves.groupby(by, sort=False):
        ax.plot(g[x], g[y], label=str(name), linewidth=1.2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
