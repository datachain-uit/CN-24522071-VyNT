"""Baseline heuristic (tầng 1–2 trong kế hoạch baseline), cài lại theo mô tả của Decima (§7.1).

Mọi scheduler có cùng giao diện với spark-sched-sim:
    name, env_wrapper_cls=None, bind_env(env), reset(seed), schedule(obs) -> (action: dict, info: dict)

  - RandomPolicy       : chọn ngẫu nhiên job/stage/số executor (mốc sàn).
  - FIFOPolicy         : job đến trước được dùng tối đa executor (giống RoundRobin(dynamic_partition=False)).
  - FairPolicy         : chia đều executor cho các job đang có mặt (giống RoundRobin(dynamic_partition=True)).
  - WeightedFairPolicy : job j được tỷ lệ W_j^alpha / sum W^alpha executor (alpha=0 -> Fair); tune alpha
                         TRÊN VALIDATION (Decima quét alpha trong [-2, 2] bước 0.1).
  - SJFCPPolicy        : ưu tiên job có W_j nhỏ nhất; trong job chọn stage nằm trên đường găng.
                         priority="total"     -> SJF-CP như mô tả trong Decima (W = tổng work của job)
                         priority="remaining" -> SRPT-CP (W = work còn lại) — biến thể mạnh hơn cho avg JCT.

Thông tin dùng: (a) quan sát của môi trường (số task còn lại, thời lượng task gần nhất, cấu trúc DAG);
(b) với weight/priority="total": tổng work của job lấy từ profile TPC-H (số task x thời lượng task trung bình),
cùng loại thông tin profile mà các heuristic trong Decima được phép dùng. Phải ghi rõ điều này khi báo cáo.
"""
from __future__ import annotations

import math

import numpy as np

from .obs_view import ObsView, parse_obs


def _noop(view: ObsView) -> dict:
    return {"stage_idx": -1, "num_exec": max(1, view.num_committable)}


class _Base:
    name = "base"
    env_wrapper_cls = None

    def __init__(self, num_executors: int, **kwargs) -> None:
        if kwargs:
            raise TypeError(f"{type(self).__name__}: tham số không hợp lệ {sorted(kwargs)}")
        self.num_executors = int(num_executors)
        self._env = None
        self._total_work_cache: dict[int, float] = {}

    # ------------------------------------------------------------------ vòng đời
    def bind_env(self, env) -> None:
        """Bộ chạy đánh giá gọi trước reset(); cần cho các biến thể dùng tổng work của job."""
        self._env = env.unwrapped
        self._total_work_cache = {}

    def reset(self, seed: int | None = None) -> None:
        self._total_work_cache = {}

    def schedule(self, obs: dict) -> tuple[dict, dict]:
        view = parse_obs(obs)
        if view.num_committable <= 0 or not view.action_of_node:
            return _noop(view), {}
        return self._schedule(view), {}

    def _schedule(self, view: ObsView) -> dict:  # pragma: no cover - lớp con cài đặt
        raise NotImplementedError

    # ------------------------------------------------------------------ tiện ích
    def _job_total_work(self, view: ObsView) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("biến thể dùng tổng work cần gọi bind_env(env) trước (run_episode tự làm việc này)")
        from spark_sched_sim.data_samplers.tpch import TPCHDataSampler

        out = np.empty(view.num_jobs)
        ids = self._env.active_job_ids
        if len(ids) != view.num_jobs:
            raise RuntimeError("số job trong quan sát không khớp với môi trường")
        for j, job_id in enumerate(ids):
            if job_id not in self._total_work_cache:
                job = self._env.jobs[job_id]
                self._total_work_cache[job_id] = float(sum(
                    s.num_tasks * TPCHDataSampler._rough_task_duration(s.task_duration_data) for s in job.stages))
            out[j] = self._total_work_cache[job_id]
        return out


class RandomPolicy(_Base):
    name = "Random"

    def __init__(self, num_executors: int, seed: int = 0) -> None:
        super().__init__(num_executors)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _schedule(self, view: ObsView) -> dict:
        jobs = [j for j in range(view.num_jobs) if view.job_has_schedulable(j)]
        j = int(self.rng.choice(jobs))
        v = int(self.rng.choice(view.schedulable_nodes_of_job(j)))
        num_exec = int(self.rng.integers(1, view.num_committable + 1))
        return {"stage_idx": view.action_of_node[v], "num_exec": num_exec}


class _PartitionPolicy(_Base):
    """Khung chung FIFO / Fair / Weighted-Fair: mỗi job có trần (cap) số executor.

    1) Nếu job đang nhả executor còn stage schedulable -> giữ executor ở job đó (tránh moving delay).
    2) Duyệt job theo thứ tự đến; job chưa đạt cap và có stage schedulable nhận min(committable, cap - supply).
    3) (tùy chọn work_conserving) nếu không job nào dưới cap còn việc, giao executor thừa cho job đầu tiên
       còn stage schedulable thay vì để rảnh.
    """

    work_conserving = False

    def caps(self, view: ObsView) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _schedule(self, view: ObsView) -> dict:
        if view.has_source_job:
            a = view.pick_stage(view.source_job_idx, "frontier")
            if a != -1:
                return {"stage_idx": a, "num_exec": view.num_committable}
        caps = self.caps(view)
        for j in range(view.num_jobs):
            if j == view.source_job_idx or view.exec_supplies[j] >= caps[j]:
                continue
            a = view.pick_stage(j, "frontier")
            if a != -1:
                return {"stage_idx": a, "num_exec": max(1, int(min(view.num_committable, caps[j] - view.exec_supplies[j])))}
        if self.work_conserving:
            for j in range(view.num_jobs):
                a = view.pick_stage(j, "frontier")
                if a != -1:
                    return {"stage_idx": a, "num_exec": view.num_committable}
        return _noop(view)


class FIFOPolicy(_PartitionPolicy):
    name = "FIFO"

    def caps(self, view: ObsView) -> np.ndarray:
        return np.full(view.num_jobs, self.num_executors)


class FairPolicy(_PartitionPolicy):
    name = "Fair"

    def caps(self, view: ObsView) -> np.ndarray:
        return np.full(view.num_jobs, int(math.ceil(self.num_executors / max(1, view.num_jobs))))


class WeightedFairPolicy(_PartitionPolicy):
    def __init__(self, num_executors: int, alpha: float = -1.0, weight: str = "total",
                 work_conserving: bool = True) -> None:
        super().__init__(num_executors)
        if weight not in ("total", "remaining"):
            raise ValueError("weight phải là 'total' hoặc 'remaining'")
        self.alpha = float(alpha)
        self.weight = weight
        self.work_conserving = bool(work_conserving)
        tag = "" if weight == "total" else ",remaining"
        tag += "" if self.work_conserving else ",non-wc"
        self.name = f"WeightedFair(a={self.alpha:g}{tag})"

    def caps(self, view: ObsView) -> np.ndarray:
        w_raw = self._job_total_work(view) if self.weight == "total" else view.job_work
        w = np.maximum(w_raw, 1e-9) ** self.alpha
        share = w / w.sum()
        return np.maximum(1, np.ceil(self.num_executors * share)).astype(int)


class SJFCPPolicy(_Base):
    """SJF-CP (priority='total') hoặc SRPT-CP (priority='remaining')."""

    def __init__(self, num_executors: int, priority: str = "total", keep_local: bool = False) -> None:
        super().__init__(num_executors)
        if priority not in ("total", "remaining"):
            raise ValueError("priority phải là 'total' hoặc 'remaining'")
        self.priority = priority
        self.keep_local = bool(keep_local)
        self.name = ("SJF-CP" if priority == "total" else "SRPT-CP") + ("(keep-local)" if keep_local else "")

    def _schedule(self, view: ObsView) -> dict:
        if self.keep_local and view.has_source_job:
            a = view.pick_stage(view.source_job_idx, "critical_path")
            if a != -1:
                return {"stage_idx": a, "num_exec": view.num_committable}
        w = self._job_total_work(view) if self.priority == "total" else view.job_work
        for j in np.argsort(w, kind="stable"):
            j = int(j)
            if j != view.source_job_idx and view.exec_supplies[j] >= self.num_executors:
                continue
            a = view.pick_stage(j, "critical_path")
            if a != -1:
                # xin toàn bộ executor đang chờ gán; môi trường tự cắt về đúng nhu cầu của stage
                return {"stage_idx": a, "num_exec": view.num_committable}
        return _noop(view)


def _srpt_cp(num_executors: int, **kw) -> SJFCPPolicy:
    return SJFCPPolicy(num_executors, priority="remaining", **kw)


REGISTRY = {
    "random": RandomPolicy,
    "fifo": FIFOPolicy,
    "fair": FairPolicy,
    "weighted_fair": WeightedFairPolicy,
    "sjf_cp": SJFCPPolicy,
    "srpt_cp": _srpt_cp,
}


def make_heuristic(name: str, num_executors: int, **kwargs):
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"heuristic '{name}' không tồn tại; chọn một trong {sorted(REGISTRY)}")
    return REGISTRY[key](num_executors=num_executors, **kwargs)
