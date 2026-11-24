"""Step 5 - Select the best Decima checkpoint on the validation split and export the learning curve.

Example:
    python scripts/05_select_checkpoint.py --run results/train/decima_small/seed0 --env configs/env/tpch_small.yaml
Output in the run directory: val_checkpoints.csv, best.json, learning_curve.csv, learning_curve.png
Never select checkpoints on the test split. The upstream trainer saves the best model from the last n
iterations by average job count during training rollouts. This script selects independently on validation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from _common import bootstrap, require_data

bootstrap()
require_data()

import pandas as pd  # noqa: E402

from rlsched.envs import load_yaml  # noqa: E402
from rlsched.evaluation import load_splits, run_many  # noqa: E402
from rlsched.evaluation.plots import plot_learning_curves  # noqa: E402
from rlsched.schedulers.decima import make_decima  # noqa: E402
from rlsched.utils import save_json  # noqa: E402

LINE = re.compile(r"Iteration (\d+) complete\. Avg\. # jobs: ([0-9.]+)")


def learning_curve(run: Path) -> pd.DataFrame:
    log = run / "stdout" / "main.out"
    rows = []
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = LINE.search(line)
            if m:
                rows.append({"iteration": int(m.group(1)), "value": float(m.group(2)), "run": run.name})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--env", required=True, help="evaluation environment config (same num_executors as training)")
    ap.add_argument("--num-seeds", type=int, default=None)
    ap.add_argument("--pyg-fallback", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_yaml(str((run / "config.yaml").resolve()))
    env_cfg = load_yaml(args.env)
    if env_cfg["num_executors"] != cfg["env"]["num_executors"]:
        raise SystemExit("evaluation and training environments have different num_executors")
    seeds = load_splits().get("val").seeds()[: args.num_seeds]

    lc = learning_curve(run)
    if len(lc):
        lc.to_csv(run / "learning_curve.csv", index=False)
        plot_learning_curves(lc.rename(columns={"value": "avg_num_jobs_train"}), run / "learning_curve.png",
                             y="avg_num_jobs_train")

    ckpts = sorted((run / "checkpoints").glob("*/model.pt"), key=lambda p: int(p.parent.name))
    if not ckpts:
        raise SystemExit(f"no checkpoints found in {run / 'checkpoints'}")
    rows = []
    for ck in ckpts:
        sched = make_decima(env_cfg["num_executors"], str(ck), agent_cfg=cfg["agent"], force_pyg_fallback=args.pyg_fallback)
        eps, _ = run_many(env_cfg, sched, seeds, label=f"ckpt{ck.parent.name}")
        x = pd.DataFrame(eps)["avg_jct_s"]
        rows.append({"iteration": int(ck.parent.name), "path": str(ck), "val_mean_avg_jct_s": x.mean(),
                     "val_std": x.std(ddof=1), "n": len(x)})
        print(f"checkpoint {ck.parent.name:>5}: val avg JCT = {x.mean():9.1f} +/- {x.std(ddof=1):7.1f} s")
    df = pd.DataFrame(rows)
    df.to_csv(run / "val_checkpoints.csv", index=False)
    best = df.loc[df["val_mean_avg_jct_s"].idxmin()].to_dict()
    save_json(best, run / "best.json")
    print(f"[select] best checkpoint (validation): iteration {best['iteration']} -> {run / 'best.json'}")


if __name__ == "__main__":
    main()
