"""Bước 4 — Huấn luyện lại Decima (PPO, spark-sched-sim) với một seed huấn luyện.

Ví dụ:
    python scripts/04_train_decima.py --config configs/train/decima_smoke.yaml --seed 0       # chạy thử 1-2 phút
    python scripts/04_train_decima.py --config configs/train/decima_small.yaml --seed 0       # MVP
    python scripts/04_train_decima.py --config configs/train/decima_small.yaml --seed 1 --pyg-fallback
Mỗi seed huấn luyện s dùng trainer.seed = s * 100000 (dải seed chuỗi job riêng, nằm trong dải train của
configs/splits.yaml). Kết quả: results/train/<config>/seed<s>/{checkpoints/, stdout/, tb/, config.yaml, meta.json}
Nhật ký huấn luyện được ghi vào stdout/main.out (trainer của upstream chuyển hướng stdout).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from _common import bootstrap, require_data

bootstrap()
require_data()

import yaml  # noqa: E402

from rlsched.envs import load_yaml, normalize_env_cfg  # noqa: E402
from rlsched.evaluation import load_splits  # noqa: E402
from rlsched.paths import RESULTS_DIR, enable_pyg_fallback  # noqa: E402
from rlsched.utils import run_metadata, save_json  # noqa: E402

SEED_STRIDE = 100_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0, help="chỉ số seed huấn luyện (0, 1, 2, ...)")
    ap.add_argument("--iterations", type=int, default=None, help="ghi đè trainer.num_iterations")
    ap.add_argument("--pyg-fallback", action="store_true", help="dùng bản thay thế torch_scatter/torch_sparse")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = load_yaml(args.config)
    cfg = copy.deepcopy(raw)
    env_ref = cfg.pop("env_config", None)
    env_cfg = load_yaml(env_ref) if env_ref else {}
    env_cfg.update(cfg.get("env", {}))
    cfg["env"] = normalize_env_cfg(env_cfg)

    tr = cfg["trainer"]
    if args.iterations is not None:
        tr["num_iterations"] = args.iterations
    tr["seed"] = args.seed * SEED_STRIDE
    if tr["num_sequences"] * (tr["num_iterations"] + 1) >= SEED_STRIDE:
        raise SystemExit("num_sequences*(num_iterations+1) vượt SEED_STRIDE — tăng SEED_STRIDE")
    load_splits().check_trainer_cfg(tr["seed"], tr["num_iterations"], tr["num_sequences"])

    name = Path(args.config).stem
    run_dir = Path(args.out or RESULTS_DIR / "train" / name / f"seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tr["artifacts_dir"] = str(run_dir)

    used_fallback = enable_pyg_fallback(force=args.pyg_fallback)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    save_json(run_metadata(config_file=args.config, train_seed_index=args.seed, trainer_seed=tr["seed"],
                           pyg_fallback=used_fallback), run_dir / "meta.json")
    print(f"[train] bắt đầu: {run_dir}  (log: {run_dir / 'stdout' / 'main.out'})", flush=True)

    from trainers import make_trainer  # import trễ: cần torch + torch_geometric

    make_trainer(cfg).train()


if __name__ == "__main__":
    main()
