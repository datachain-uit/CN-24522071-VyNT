"""Step 3 - Tune Weighted-Fair alpha on the validation split (tier 2 baseline, "opt. weighted fair").

Decima searches alpha values in {-2, -1.9, ..., 2}. The default step here is 0.25 for quick laptop runs.
For publication, use a step of 0.1 and report the tuning budget (number of values x number of seeds).
Example:
    python scripts/03_tune_weighted_fair.py --env configs/env/tpch_small.yaml
Output: results/tuning/<env>/weighted_fair_val.csv and best.json  ({"alpha": ..., ...})
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import bootstrap, require_data

bootstrap()
require_data()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlsched.envs import load_yaml  # noqa: E402
from rlsched.evaluation import load_splits, run_many  # noqa: E402
from rlsched.paths import RESULTS_DIR  # noqa: E402
from rlsched.schedulers import WeightedFairPolicy  # noqa: E402
from rlsched.utils import run_metadata, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--alpha-min", type=float, default=-2.0)
    ap.add_argument("--alpha-max", type=float, default=2.0)
    ap.add_argument("--alpha-step", type=float, default=0.25)
    ap.add_argument("--weight", choices=["total", "remaining"], default="total")
    ap.add_argument("--num-seeds", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env_cfg = load_yaml(args.env)
    seeds = load_splits().get("val").seeds()[: args.num_seeds]
    out = Path(args.out or RESULTS_DIR / "tuning" / env_cfg.get("name", Path(args.env).stem))
    out.mkdir(parents=True, exist_ok=True)

    alphas = np.round(np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step), 4)
    rows = []
    for a in alphas:
        pol = WeightedFairPolicy(env_cfg["num_executors"], alpha=float(a), weight=args.weight)
        eps, _ = run_many(env_cfg, pol, seeds, label=pol.name)
        x = pd.DataFrame(eps)["avg_jct_s"]
        rows.append({"alpha": float(a), "mean_avg_jct_s": x.mean(), "std": x.std(ddof=1), "n": len(x)})
        print(f"alpha={a:+.2f}  avg JCT = {x.mean():9.1f} +/- {x.std(ddof=1):7.1f} s")
    df = pd.DataFrame(rows)
    df.to_csv(out / "weighted_fair_val.csv", index=False)
    best = df.loc[df["mean_avg_jct_s"].idxmin()].to_dict()
    best.update({"weight": args.weight, "split": "val", "num_values_tried": len(alphas), "seeds": seeds})
    save_json(best, out / "best.json")
    save_json(run_metadata(env=env_cfg, alphas=alphas.tolist(), weight=args.weight), out / "meta.json")
    print(f"[tune] best alpha on validation = {best['alpha']:+.2f} -> {out / 'best.json'}")


if __name__ == "__main__":
    main()
