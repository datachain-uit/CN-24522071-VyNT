"""Số đo chất lượng lập lịch ở mức episode (tính trên TOÀN BỘ job của episode, không lấy trung bình theo batch).

  avg/median/p90/p95/p99/max JCT   : thời gian hoàn thành job (giây) — độ đo chính là avg JCT
  makespan_s                       : thời điểm job cuối cùng hoàn thành (job đầu tiên đến ở t=0)
  avg_num_jobs                     : số job trung bình trong hệ thống = sum(JCT) / makespan (định luật Little)
  executor_utilization             : tổng thời gian chạy task / (số executor x makespan)
  throughput_jobs_per_h            : số job hoàn thành / giờ mô phỏng
  decision latency                 : thời gian (ms) để scheduler ra một quyết định
Nếu còn job chưa xong (episode bị cắt theo thời gian) thì incomplete_jobs > 0 và các số đo JCT
chỉ tính trên job đã xong — KHÔNG được dùng để so sánh; bộ chạy đánh giá luôn đặt job_arrival_cap.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def episode_summary(jobs: list[dict[str, Any]], num_executors: int, busy_time_s: float, makespan_s: float,
                    total_reward: float, decision_times_s: list[float], wall_clock_s: float) -> dict[str, Any]:
    jct = np.array([j["jct_s"] for j in jobs], dtype=float)
    done = np.isfinite(jct)
    jct_done = jct[done]
    dec_ms = np.asarray(decision_times_s, dtype=float) * 1e3
    out: dict[str, Any] = {
        "num_jobs": int(len(jobs)),
        "completed_jobs": int(done.sum()),
        "incomplete_jobs": int((~done).sum()),
        "avg_jct_s": float(jct_done.mean()) if done.any() else np.nan,
        "median_jct_s": float(np.median(jct_done)) if done.any() else np.nan,
        "p90_jct_s": float(np.percentile(jct_done, 90)) if done.any() else np.nan,
        "p95_jct_s": float(np.percentile(jct_done, 95)) if done.any() else np.nan,
        "p99_jct_s": float(np.percentile(jct_done, 99)) if done.any() else np.nan,
        "max_jct_s": float(jct_done.max()) if done.any() else np.nan,
        "makespan_s": float(makespan_s),
        "avg_num_jobs": float(jct_done.sum() / makespan_s) if makespan_s > 0 and done.all() else np.nan,
        "executor_utilization": float(busy_time_s / (num_executors * makespan_s)) if makespan_s > 0 else np.nan,
        "throughput_jobs_per_h": float(done.sum() / (makespan_s / 3600.0)) if makespan_s > 0 else np.nan,
        "total_reward": float(total_reward),
        "num_decisions": int(len(dec_ms)),
        "mean_decision_ms": float(dec_ms.mean()) if len(dec_ms) else np.nan,
        "p99_decision_ms": float(np.percentile(dec_ms, 99)) if len(dec_ms) else np.nan,
        "wall_clock_s": float(wall_clock_s),
    }
    return out


PRIMARY_METRIC = "avg_jct_s"
LOWER_IS_BETTER = {
    "avg_jct_s": True, "median_jct_s": True, "p90_jct_s": True, "p95_jct_s": True, "p99_jct_s": True,
    "max_jct_s": True, "makespan_s": True, "avg_num_jobs": True, "executor_utilization": False,
    "throughput_jobs_per_h": False, "mean_decision_ms": True,
}
