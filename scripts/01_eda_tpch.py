"""Step 1 - Explore the 154 TPC-H job templates and calibrate the offered load.

Output (results/eda/):
  tpch_templates.csv        : one row per (query, size) with stage count, task count, total work, critical path and depth
  tpch_by_size.csv          : statistics by data size
  arrival_rate_for_load.csv : job_arrival_rate (1/ms) for the target load with N executors
  *.png                     : distribution plots
Offered load rho = E[work]/(N*E[IAT]) is an approximation. Actual utilization also depends on
moving delay, warm-up, wave count and scheduling policy. Compare it with executor_utilization in baseline runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import bootstrap, require_data

bootstrap()
require_data()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rlsched.data import QUERY_SIZES, template_table  # noqa: E402
from rlsched.paths import RESULTS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--executors", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--loads", type=float, nargs="+", default=[0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--out", default=str(RESULTS_DIR / "eda"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = template_table()
    df.to_csv(out / "tpch_templates.csv", index=False)
    by_size = df.groupby("size").agg(num_stages=("num_stages", "mean"), num_tasks=("num_tasks", "mean"),
                                     total_work_s=("total_work_s", "mean"), cp_work_s=("cp_work_s", "mean"),
                                     depth=("depth", "mean")).reindex(QUERY_SIZES)
    by_size.to_csv(out / "tpch_by_size.csv")

    # phân phối job theo bộ sinh của spark-sched-sim: query ~ Uniform{1..22}, size ~ Uniform{7 mức}
    mean_work = float(df["total_work_s"].mean())
    share = df.sort_values("total_work_s", ascending=False)["total_work_s"].cumsum() / df["total_work_s"].sum()
    top_frac = float((share <= 0.8).mean())
    rows = []
    for n in args.executors:
        for rho in args.loads:
            iat_s = mean_work / (n * rho)
            rows.append({"num_executors": n, "target_load": rho, "mean_interarrival_s": iat_s,
                         "job_arrival_rate_per_ms": 1.0 / (iat_s * 1e3)})
    pd.DataFrame(rows).to_csv(out / "arrival_rate_for_load.csv", index=False)

    print(f"[eda] {len(df)} templates | stages: {df.num_stages.min()}-{df.num_stages.max()} "
          f"(mean {df.num_stages.mean():.1f}) | mean total work {mean_work:.0f} executor-seconds")
    print(f"[eda] The largest {top_frac:.0%} of templates account for 80% of total work (long tail)")
    print(by_size.round(1).to_string())
    print(pd.DataFrame(rows).round(6).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(df["num_stages"], bins=range(1, int(df.num_stages.max()) + 2), color="#4C78A8")
    axes[0].set_xlabel("Stages per job")
    axes[0].set_ylabel("Number of templates")
    axes[1].boxplot([df.loc[df["size"] == s, "total_work_s"] for s in QUERY_SIZES], labels=QUERY_SIZES)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Total work (executor-s, log)")
    axes[2].scatter(df["total_work_s"], df["cp_work_s"], s=10, color="#F58518")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Total work (s)")
    axes[2].set_ylabel("Critical path work (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tpch_distributions.png", dpi=200)
    print(f"[eda] wrote results to {out}")


if __name__ == "__main__":
    main()
