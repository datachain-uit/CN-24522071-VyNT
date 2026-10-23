"""Bước 6 — Đánh giá CUỐI CÙNG trên tập TEST (chạy một lần cho bảng kết quả).

Ví dụ:
    python scripts/06_evaluate.py --env configs/env/tpch_small.yaml \
        --methods random fifo fair sjf_cp srpt_cp \
                  "weighted_fair:alpha=@results/tuning/tpch_small/best.json" \
        --decima-runs results/train/decima_small/seed0 results/train/decima_small/seed1 results/train/decima_small/seed2
Đầu ra: results/final/<env>_test/{episodes.csv, jobs.csv, meta.json}
Thư mục đích đã có kết quả thì script dừng (tránh "nhìn trộm" test nhiều lần); dùng --force nếu thật sự cần.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import bootstrap, require_data
from _methods import parse_method

bootstrap()
require_data()

import pandas as pd  # noqa: E402

from rlsched.envs import load_yaml  # noqa: E402
from rlsched.evaluation import load_splits, run_many  # noqa: E402
from rlsched.paths import RESULTS_DIR  # noqa: E402
from rlsched.schedulers import make_scheduler  # noqa: E402
from rlsched.utils import run_metadata, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--methods", nargs="*", default=["random", "fifo", "fair", "sjf_cp", "srpt_cp"])
    ap.add_argument("--decima-runs", nargs="*", default=[], help="các thư mục run đã có best.json (bước 5)")
    ap.add_argument("--decima-label", default="Decima")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--num-seeds", type=int, default=None)
    ap.add_argument("--pyg-fallback", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    env_cfg = load_yaml(args.env)
    seeds = load_splits().get(args.split).seeds()[: args.num_seeds]
    out = Path(args.out or RESULTS_DIR / "final" / f"{env_cfg.get('name', Path(args.env).stem)}_{args.split}")
    if (out / "episodes.csv").exists() and not args.force:
        raise SystemExit(f"{out} đã có kết quả. Tập test chỉ nên dùng một lần; thêm --force nếu có lý do chính đáng "
                         "(và ghi lại lý do trong nhật ký thực nghiệm).")
    out.mkdir(parents=True, exist_ok=True)

    eps_all, jobs_all = [], []
    for spec in args.methods:
        name, kw = parse_method(spec)
        sched = make_scheduler(name, env_cfg["num_executors"], **kw)
        eps, jobs = run_many(env_cfg, sched, seeds, label=sched.name)
        for e in eps:
            e.update({"algo": sched.name, "train_seed": -1})
        eps_all += eps
        jobs_all += jobs

    for run in map(Path, args.decima_runs):
        best = json.loads((run / "best.json").read_text(encoding="utf-8"))
        cfg = load_yaml(str((run / "config.yaml").resolve()))
        train_seed = json.loads((run / "meta.json").read_text(encoding="utf-8"))["train_seed_index"]
        sched = make_scheduler("decima", env_cfg["num_executors"], state_dict_path=best["path"],
                               agent_cfg=cfg["agent"], force_pyg_fallback=args.pyg_fallback)
        label = f"{args.decima_label}/s{train_seed}"
        eps, jobs = run_many(env_cfg, sched, seeds, label=label)
        for e in eps:
            e.update({"algo": args.decima_label, "train_seed": int(train_seed), "checkpoint": best["path"]})
        for j in jobs:
            j["algo"] = args.decima_label
        eps_all += eps
        jobs_all += jobs

    eps_df = pd.DataFrame(eps_all)
    eps_df["algo"] = eps_df["algo"].fillna(eps_df["method"])
    eps_df.to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(jobs_all).to_csv(out / "jobs.csv", index=False)
    save_json(run_metadata(env=env_cfg, split=args.split, seeds=seeds, methods=args.methods,
                           decima_runs=args.decima_runs), out / "meta.json")
    print(eps_df.groupby("method")["avg_jct_s"].agg(["mean", "std", "count"]).round(1).to_string())
    print(f"[evaluate] đã ghi {out}. Bước tiếp theo: python scripts/07_analyze.py --in {out}")


if __name__ == "__main__":
    main()
