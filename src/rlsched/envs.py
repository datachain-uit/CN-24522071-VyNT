"""Tạo môi trường mô phỏng (Gymnasium) từ file cấu hình YAML.

Ghi chú về spark-sched-sim (commit 2ba8d80):
  * Khóa bắt buộc: num_executors, job_arrival_cap, job_arrival_rate, moving_delay,
    warmup_delay, data_sampler_cls. File ``config/decima_tpch.yaml`` của upstream dùng
    khóa ``dataset`` nên gây KeyError 'data_sampler_cls' -> hàm ``normalize_env_cfg`` sửa lỗi này.
  * job_arrival_rate là tham số phân phối mũ theo ms: khoảng cách đến trung bình = 1/rate (ms).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml

from .paths import REPO_ROOT, ensure_sim_on_path

ensure_sim_on_path()
import spark_sched_sim  # noqa: E402,F401  (đăng ký env id với Gymnasium)

ENV_ID = "spark_sched_sim:SparkSchedSimEnv-v0"
REQUIRED_KEYS = ("num_executors", "job_arrival_cap", "job_arrival_rate", "moving_delay", "warmup_delay")


def load_yaml(path) -> dict[str, Any]:
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def normalize_env_cfg(env_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(env_cfg)
    cfg.pop("dataset", None)
    cfg.pop("name", None)
    cfg.pop("description", None)
    cfg.setdefault("data_sampler_cls", "TPCHDataSampler")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"missing environment configuration keys: {missing}")
    if not cfg.get("job_arrival_cap"):
        raise ValueError("evaluation requires job_arrival_cap > 0 so every episode runs until all jobs finish "
                         "(to avoid censored JCTs).")
    return cfg


def make_env(env_cfg: dict[str, Any], wrapper_cls=None) -> gym.Env:
    env = gym.make(ENV_ID, env_cfg=normalize_env_cfg(env_cfg), disable_env_checker=True)
    if wrapper_cls is not None:
        env = wrapper_cls(env)
    return env


def offered_load(mean_job_work_s: float, num_executors: int, job_arrival_rate_per_ms: float) -> float:
    """Tải đề nghị rho = E[work] / (N * E[inter-arrival]). rho -> 1: cluster bão hòa."""
    mean_iat_s = (1.0 / job_arrival_rate_per_ms) / 1e3
    return mean_job_work_s / (num_executors * mean_iat_s)
