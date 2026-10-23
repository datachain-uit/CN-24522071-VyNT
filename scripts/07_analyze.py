"""Bước 7 — Phân tích thống kê kết quả (bảng mean ± std, CI 95%, Wilcoxon + Holm, CDF).

Ví dụ:
    python scripts/07_analyze.py --in results/final/tpch_small_test --reference Decima
Phương pháp có nhiều seed huấn luyện (cột train_seed >= 0) được lấy trung bình theo seed huấn luyện cho
từng episode test trước khi kiểm định; độ phân tán giữa các seed huấn luyện được báo cáo riêng.
Đầu ra: <in>/analysis/{summary_*.csv, paired_*.csv, train_seed_spread.csv, report.md, *.png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import bootstrap

bootstrap()

import pandas as pd  # noqa: E402

from rlsched.evaluation.plots import plot_jct_cdf, plot_mean_ci  # noqa: E402
from rlsched.evaluation.stats import paired_compare, summarize  # noqa: E402
from rlsched.utils import df_to_markdown  # noqa: E402

METRICS = ["avg_jct_s", "p95_jct_s", "makespan_s", "executor_utilization", "mean_decision_ms"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--reference", default=None, help="phương pháp đem so với các phương pháp còn lại "
                                                       "(mặc định: phương pháp có avg JCT thấp nhất)")
    args = ap.parse_args()
    inp = Path(args.inp)
    out = inp / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    eps = pd.read_csv(inp / "episodes.csv")
    jobs = pd.read_csv(inp / "jobs.csv")
    if "algo" not in eps:
        eps["algo"] = eps["method"]
    if "train_seed" not in eps:
        eps["train_seed"] = -1

    # 1) độ phân tán giữa các seed huấn luyện (chỉ với phương pháp học)
    learned = eps[eps["train_seed"] >= 0]
    if len(learned):
        spread = (learned.groupby(["algo", "train_seed"])["avg_jct_s"].mean().groupby("algo")
                  .agg(["mean", "std", "count"]).rename(columns={"count": "n_train_seeds"}))
        spread.to_csv(out / "train_seed_spread.csv")
    # 2) gộp theo (algo, seed test): trung bình qua các seed huấn luyện
    agg = eps.groupby(["algo", "seed"])[METRICS].mean().reset_index().rename(columns={"algo": "method"})

    lines = ["# Kết quả trên tập đánh giá", "", f"Nguồn: `{inp}`", ""]
    for m in METRICS:
        s = summarize(agg, m)
        s.to_csv(out / f"summary_{m}.csv", index=False)
        lines += [f"## {m}", "", df_to_markdown(s), ""]
    s_main = summarize(agg, "avg_jct_s")
    ref = args.reference or s_main.iloc[0]["method"]
    paired = paired_compare(agg, ref, [m for m in s_main["method"] if m != ref], metric="avg_jct_s")
    paired.to_csv(out / f"paired_{ref.replace('/', '_')}.csv", index=False)
    lines += [f"## So sánh ghép cặp: {ref} với từng baseline (avg JCT, Wilcoxon + Holm)", "",
              df_to_markdown(paired, ".4f") if len(paired) else "(không có cặp nào)", ""]
    if len(learned):
        lines += ["## Độ phân tán giữa các seed huấn luyện (mean ± std của avg JCT)", "",
                  df_to_markdown(pd.read_csv(out / "train_seed_spread.csv"), ".2f"), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    jobs_plot = jobs.copy()
    if "algo" in jobs_plot:
        jobs_plot["method"] = jobs_plot["algo"].fillna(jobs_plot["method"])
    plot_jct_cdf(jobs_plot, out / "jct_cdf.png")
    plot_mean_ci(s_main, out / "avg_jct_mean_ci.png")
    print(s_main.round(2).to_string(index=False))
    if len(paired):
        print(paired.round(4).to_string(index=False))
    print(f"[analyze] đã ghi {out}")


if __name__ == "__main__":
    main()
