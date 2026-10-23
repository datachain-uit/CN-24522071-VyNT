"""Chạy một episode với một scheduler và thu thập số đo ở mức job và mức episode.

Bất biến được kiểm tra trong tests/: khi beta = 0 và mọi job đều xong,
    -(tổng reward)  ==  tổng JCT (ms)      (reward của env = -tích phân số job trong hệ thống theo thời gian)
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data import critical_path
from ..envs import make_env
from .metrics import episode_summary


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # phần heuristic không cần torch
        pass


class _DurationRecorder:
    """Bọc data_sampler.task_duration để cộng dồn thời gian executor thực sự chạy task (gồm warm-up)."""

    def __init__(self, sampler) -> None:
        self.total_ms = 0.0
        self.num_tasks = 0
        self._orig = sampler.task_duration

        def wrapped(*args, **kwargs):
            d = self._orig(*args, **kwargs)
            self.total_ms += float(d)
            self.num_tasks += 1
            return d

        sampler.task_duration = wrapped


@dataclass
class EpisodeResult:
    summary: dict[str, Any]
    jobs: list[dict[str, Any]] = field(default_factory=list)


def _job_record(job, rough_fn) -> dict[str, Any]:
    stages = job.stages
    work = np.array([s.num_tasks * rough_fn(s) for s in stages], dtype=float)
    children = [list(job.dag.successors(s.id_)) for s in stages]
    cp = critical_path(children, work) if len(stages) else np.zeros(1)
    t_c = float(job.t_completed)
    return {
        "job_id": int(job.id_),
        "query": int(getattr(job, "query_num", -1)),
        "size": str(getattr(job, "query_size", "")),
        "num_stages": len(stages),
        "num_tasks": int(sum(s.num_tasks for s in stages)),
        "total_work_s": float(work.sum()) / 1e3,
        "cp_work_s": float(cp.max()) / 1e3,
        "t_arrival_s": float(job.t_arrival) / 1e3,
        "t_completed_s": t_c / 1e3 if np.isfinite(t_c) else np.nan,
        "jct_s": (t_c - float(job.t_arrival)) / 1e3 if np.isfinite(t_c) else np.nan,
    }


def run_episode(env_cfg: dict[str, Any], scheduler, seed: int, max_steps: int | None = None) -> EpisodeResult:
    env = make_env(env_cfg, wrapper_cls=getattr(scheduler, "env_wrapper_cls", None))
    seed_everything(seed)
    if hasattr(scheduler, "bind_env"):
        scheduler.bind_env(env)
    if hasattr(scheduler, "reset"):
        scheduler.reset(seed)
    base = env.unwrapped
    recorder = _DurationRecorder(base.data_sampler)

    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    total_reward, steps, decision_s = 0.0, 0, []
    t_start = time.perf_counter()
    while not (terminated or truncated):
        t0 = time.perf_counter()
        action, _ = scheduler.schedule(obs)
        decision_s.append(time.perf_counter() - t0)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    wall_clock = time.perf_counter() - t_start

    from spark_sched_sim.data_samplers.tpch import TPCHDataSampler

    def rough(stage) -> float:
        data = getattr(stage, "task_duration_data", None)
        return float(TPCHDataSampler._rough_task_duration(data)) if data is not None else float(stage.most_recent_duration)

    jobs = [_job_record(job, rough) for job in base.jobs.values()]
    summary = episode_summary(
        jobs=jobs,
        num_executors=base.num_executors,
        busy_time_s=recorder.total_ms / 1e3,
        makespan_s=float(base.wall_time) / 1e3,
        total_reward=total_reward,
        decision_times_s=decision_s,
        wall_clock_s=wall_clock,
    )
    summary.update({"seed": int(seed), "scheduler": getattr(scheduler, "name", type(scheduler).__name__),
                    "num_steps": steps, "terminated": bool(terminated), "truncated": bool(truncated)})
    for j in jobs:
        j["seed"] = int(seed)
        j["scheduler"] = summary["scheduler"]
    env.close()
    return EpisodeResult(summary=summary, jobs=jobs)


def run_many(env_cfg: dict[str, Any], scheduler, seeds: list[int], label: str | None = None,
             progress: bool = True) -> tuple[list[dict], list[dict]]:
    episodes, jobs = [], []
    it = seeds
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(seeds, desc=label or getattr(scheduler, "name", "scheduler"), leave=False)
        except ImportError:
            pass
    for s in it:
        res = run_episode(env_cfg, scheduler, s)
        if label:
            res.summary["method"] = label
            for j in res.jobs:
                j["method"] = label
        episodes.append(res.summary)
        jobs.extend(res.jobs)
    return episodes, jobs
