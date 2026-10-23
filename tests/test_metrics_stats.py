import numpy as np
import pandas as pd

from rlsched.evaluation.metrics import episode_summary
from rlsched.evaluation.stats import bootstrap_ci, holm, paired_compare, summarize


def test_episode_summary_littles_law():
    jobs = [{"jct_s": 10.0}, {"jct_s": 30.0}]
    s = episode_summary(jobs, num_executors=2, busy_time_s=40.0, makespan_s=40.0, total_reward=-40e3,
                        decision_times_s=[0.001, 0.003], wall_clock_s=1.0)
    assert s["avg_jct_s"] == 20.0 and s["avg_num_jobs"] == 1.0
    assert s["executor_utilization"] == 0.5 and s["mean_decision_ms"] == 2.0


def test_holm_known_values():
    # p sắp xếp: 0.01*3=0.03 ; 0.03*2=0.06 ; 0.04*1=0.04 -> max dồn = 0.06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert all(0 <= p <= 1 for p in holm([0.5, 0.9, 0.7]))


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(10, 1, size=200)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_paired_compare_detects_consistent_gain():
    seeds = np.arange(30)
    base = 100 + np.random.default_rng(1).normal(0, 5, size=30)
    df = pd.DataFrame({"seed": np.r_[seeds, seeds], "method": ["A"] * 30 + ["B"] * 30,
                       "avg_jct_s": np.r_[base * 0.9, base]})
    out = paired_compare(df, "A", ["B"])
    assert out.loc[0, "rel_improvement_pct"] > 9.0 and out.loc[0, "p_wilcoxon"] < 1e-5
    assert summarize(df).iloc[0]["method"] == "A"
