"""Step 2 - Run heuristic baselines on a set of seeds (default: validation).

Examples:
    python scripts/02_run_baselines.py --env configs/env/tpch_small.yaml --split val
    python scripts/02_run_baselines.py --env configs/env/tpch_small.yaml --split val \
        --methods random fifo fair sjf_cp srpt_cp weighted_fair:alpha=-1
Output: results/baselines/<env>_<split>/{episodes.csv, jobs.csv, summary.csv, meta.json}
Reserve the test split for a single run of scripts/06_evaluate.py. Do not use --split test here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import bootstrap, require_data
from _methods import parse_method

bootstrap()
require_data()

import pandas as pd  # noqa: E402

from rlsched.envs import load_yaml  # noqa: E402
from rlsched.evaluation import load_splits, run_many  # noqa: E402
from rlsched.evaluation.stats import summarize  # noqa: E402
from rlsched.paths import RESULTS_DIR  # noqa: E402
from rlsched.schedulers import make_scheduler  # noqa: E402
from rlsched.utils import run_metadata, save_json  # noqa: E402

DEFAULT_METHODS = ["random", "fifo", "fair", "sjf_cp", "srpt_cp", "weighted_fair:alpha=-1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--num-seeds", type=int, default=None, help="use the first n seeds in the split for a quick run")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env_cfg = load_yaml(args.env)
    seeds = load_splits().get(args.split).seeds()[: args.num_seeds]
    out = Path(args.out or RESULTS_DIR / "baselines" / f"{env_cfg.get('name', Path(args.env).stem)}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    all_eps, all_jobs = [], []
    for spec in args.methods:
        name, kw = parse_method(spec)
        sched = make_scheduler(name, env_cfg["num_executors"], **kw)
        eps, jobs = run_many(env_cfg, sched, seeds, label=sched.name)
        all_eps += eps
        all_jobs += jobs
        m = pd.DataFrame(eps)["avg_jct_s"]
        print(f"{sched.name:<32} avg JCT = {m.mean():9.1f} +/- {m.std(ddof=1):7.1f} s  (n={len(m)})")

    eps_df, jobs_df = pd.DataFrame(all_eps), pd.DataFrame(all_jobs)
    eps_df.to_csv(out / "episodes.csv", index=False)
    jobs_df.to_csv(out / "jobs.csv", index=False)
    summ = summarize(eps_df, "avg_jct_s")
    summ.to_csv(out / "summary.csv", index=False)
    save_json(run_metadata(env=env_cfg, split=args.split, seeds=seeds, methods=args.methods), out / "meta.json")
    print(summ.round(2).to_string(index=False))
    print(f"[baselines] wrote results to {out}")


if __name__ == "__main__":
    main()
