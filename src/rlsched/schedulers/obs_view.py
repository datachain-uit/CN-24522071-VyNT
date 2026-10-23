"""Đọc quan sát (observation) THÔ của SparkSchedSimEnv thành cấu trúc dễ dùng cho heuristic.

Quan sát gốc (xem spark_sched_sim.py, hàm _observe):
  dag_batch.nodes      [N, 3]: (số task còn lại, thời lượng task gần nhất (ms), có schedulable không)
  dag_batch.edge_links [E, 2]: cạnh cha -> con giữa các stage CÒN HOẠT ĐỘNG (chỉ số toàn cục 0..N-1)
  dag_ptr              [J+1] : stage của job j nằm trong [dag_ptr[j], dag_ptr[j+1])
  num_committable_execs      : số executor cần được gán ở vòng quyết định hiện tại
  source_job_idx             : job đang nhả executor (== J nếu nguồn là common pool)
  exec_supplies        [J]   : số executor đang gắn với từng job (kể cả đang di chuyển/đã cam kết)
Hành động: {"stage_idx": chỉ số trong danh sách stage schedulable (hoặc -1), "num_exec": 1..committable}
Thứ tự các stage schedulable trùng với thứ tự xuất hiện của chúng trong ``nodes``.
Heuristic ở đây CHỈ dùng thông tin có trong quan sát — cùng loại thông tin mà Decima nhận được.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObsView:
    nodes: np.ndarray
    edge_links: np.ndarray
    dag_ptr: np.ndarray
    num_committable: int
    source_job_idx: int
    exec_supplies: np.ndarray
    node_job: np.ndarray = field(init=False)
    action_of_node: dict = field(init=False)
    frontier: np.ndarray = field(init=False)
    work: np.ndarray = field(init=False)
    job_work: np.ndarray = field(init=False)
    _children: list | None = field(init=False, default=None)
    _cp: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        n = self.nodes.shape[0]
        counts = np.diff(self.dag_ptr)
        self.node_job = np.repeat(np.arange(len(counts)), counts)
        sched = np.nonzero(self.nodes[:, 2] > 0.5)[0] if n else np.zeros(0, dtype=int)
        self.action_of_node = {int(v): i for i, v in enumerate(sched)}
        self.frontier = np.ones(n, dtype=bool)
        if self.edge_links.size:
            self.frontier[self.edge_links[:, 1]] = False
        self.work = self.nodes[:, 0] * self.nodes[:, 1] if n else np.zeros(0)
        self.job_work = np.bincount(self.node_job, weights=self.work, minlength=self.num_jobs) if n \
            else np.zeros(self.num_jobs)

    # ------------------------------------------------------------------ thuộc tính
    @property
    def num_jobs(self) -> int:
        return len(self.dag_ptr) - 1

    @property
    def has_source_job(self) -> bool:
        return self.source_job_idx < self.num_jobs

    def nodes_of_job(self, j: int) -> range:
        return range(int(self.dag_ptr[j]), int(self.dag_ptr[j + 1]))

    def schedulable_nodes_of_job(self, j: int) -> list[int]:
        return [v for v in self.nodes_of_job(j) if v in self.action_of_node]

    def job_has_schedulable(self, j: int) -> bool:
        return any(v in self.action_of_node for v in self.nodes_of_job(j))

    # ------------------------------------------------------------------ đường găng
    @property
    def children(self) -> list:
        if self._children is None:
            ch = [[] for _ in range(self.nodes.shape[0])]
            for p, c in self.edge_links:
                ch[int(p)].append(int(c))
            self._children = ch
        return self._children

    @property
    def critical_path_work(self) -> np.ndarray:
        """cp(v) = work(v) + max cp(con) trên đồ thị các stage còn hoạt động (ms·executor)."""
        if self._cp is None:
            from ..data import critical_path

            self._cp = critical_path(self.children, self.work) if self.nodes.shape[0] else np.zeros(0)
        return self._cp

    # ------------------------------------------------------------------ chọn stage trong một job
    def pick_stage(self, j: int, rule: str = "frontier") -> int:
        """Trả về chỉ số HÀNH ĐỘNG (không phải chỉ số node) hoặc -1 nếu job không có stage schedulable.

        rule="frontier": ưu tiên stage sẵn sàng ngay (mọi stage cha đã xong) — giống heuristic Fair upstream.
        rule="critical_path": chọn stage có cp lớn nhất (nằm trên đường găng) — dùng cho SJF-CP.
        """
        cands = self.schedulable_nodes_of_job(j)
        if not cands:
            return -1
        if rule == "critical_path":
            cp = self.critical_path_work
            ready = [v for v in cands if self.frontier[v]] or cands
            best = max(ready, key=lambda v: (cp[v], -v))
            return self.action_of_node[best]
        for v in cands:
            if self.frontier[v]:
                return self.action_of_node[v]
        return self.action_of_node[cands[0]]


def parse_obs(obs: dict) -> ObsView:
    dag = obs["dag_batch"]
    edge_links = np.asarray(dag.edge_links, dtype=int).reshape(-1, 2)
    return ObsView(
        nodes=np.asarray(dag.nodes, dtype=float).reshape(-1, 3),
        edge_links=edge_links,
        dag_ptr=np.asarray(obs["dag_ptr"], dtype=int),
        num_committable=int(obs["num_committable_execs"]),
        source_job_idx=int(obs["source_job_idx"]),
        exec_supplies=np.asarray(obs["exec_supplies"], dtype=int).reshape(-1),
    )
