"""Kiểm thử môi trường + heuristic: tất định theo seed, hoàn thành mọi job, bất biến reward = -tổng JCT."""
import numpy as np
import pytest

from rlsched.envs import make_env
from rlsched.evaluation import run_episode
from rlsched.schedulers import make_scheduler

METHODS = [("random", {}), ("fifo", {}), ("fair", {}), ("sjf_cp", {}), ("srpt_cp", {}),
           ("weighted_fair", {"alpha": -1.0}), ("weighted_fair", {"alpha": 0.5, "weight": "remaining"})]


def _job_sequence(env_cfg, seed):
    env = make_env(env_cfg)
    env.reset(seed=seed)
    jobs = env.unwrapped.jobs
    seq = [(j.t_arrival, j.query_num, j.query_size) for j in jobs.values()]
    env.close()
    return seq


def test_same_seed_same_job_sequence(debug_env_cfg):
    assert _job_sequence(debug_env_cfg, 1234) == _job_sequence(debug_env_cfg, 1234)
    assert _job_sequence(debug_env_cfg, 1234) != _job_sequence(debug_env_cfg, 1235)


@pytest.mark.parametrize("name,kw", METHODS)
def test_heuristic_completes_all_jobs(debug_env_cfg, name, kw):
    sched = make_scheduler(name, debug_env_cfg["num_executors"], **kw)
    res = run_episode(debug_env_cfg, sched, seed=7)
    s = res.summary
    assert s["terminated"] and s["incomplete_jobs"] == 0
    assert s["num_jobs"] == debug_env_cfg["job_arrival_cap"]
    jct_sum_ms = sum(j["jct_s"] for j in res.jobs) * 1e3
    # reward của env = -tích phân (số job trong hệ thống) theo thời gian => tổng = -tổng JCT
    assert np.isclose(-s["total_reward"], jct_sum_ms, rtol=1e-9)
    assert 0.0 < s["executor_utilization"] <= 1.0 + 1e-9
    assert all(j["jct_s"] > 0 for j in res.jobs)


@pytest.mark.parametrize("name,kw", METHODS)
def test_heuristic_reproducible(debug_env_cfg, name, kw):
    a = run_episode(debug_env_cfg, make_scheduler(name, debug_env_cfg["num_executors"], **kw), seed=11).summary
    b = run_episode(debug_env_cfg, make_scheduler(name, debug_env_cfg["num_executors"], **kw), seed=11).summary
    assert a["avg_jct_s"] == b["avg_jct_s"] and a["makespan_s"] == b["makespan_s"]


def test_fair_matches_upstream_round_robin(debug_env_cfg):
    """FairPolicy phải cho kết quả GIỐNG HỆT RoundRobinScheduler(dynamic_partition=True) của upstream."""
    torch = pytest.importorskip("torch")  # noqa: F841 - upstream schedulers import torch
    from rlsched.paths import enable_pyg_fallback

    enable_pyg_fallback()
    from schedulers import RoundRobinScheduler

    ours = run_episode(debug_env_cfg, make_scheduler("fair", debug_env_cfg["num_executors"]), seed=3).summary
    theirs = run_episode(debug_env_cfg, RoundRobinScheduler(debug_env_cfg["num_executors"], True), seed=3).summary
    assert ours["avg_jct_s"] == pytest.approx(theirs["avg_jct_s"], rel=1e-12)
