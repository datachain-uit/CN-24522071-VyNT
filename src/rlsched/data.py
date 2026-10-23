"""Tải, kiểm tra và thống kê dữ liệu TPC-H DAG (job profile dùng trong Decima / spark-sched-sim).

Cấu trúc thư mục kỳ vọng: data/tpch/<size>/{adj_mat_<q>.npy, task_duration_<q>.npy}
với size ∈ {2g, 5g, 10g, 20g, 50g, 80g, 100g} và q ∈ {1..22}  (7 x 22 = 154 job template).
Nguồn dữ liệu: thư mục spark_env/tpch của repo gốc Decima (github.com/hongzimao/decima-sim).
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .paths import DATA_DIR

QUERY_SIZES = ["2g", "5g", "10g", "20g", "50g", "80g", "100g"]
NUM_QUERIES = 22
DECIMA_GIT_URL = "https://github.com/hongzimao/decima-sim.git"
DECIMA_ZIP_URL = "https://codeload.github.com/hongzimao/decima-sim/zip/refs/heads/master"
UPSTREAM_TPCH_URL = "https://bit.ly/3F1Go8t"  # link mà spark-sched-sim dùng mặc định


# ----------------------------------------------------------------------------- kiểm tra
def missing_files(data_dir: Path = DATA_DIR) -> list[str]:
    missing = []
    for size in QUERY_SIZES:
        for q in range(1, NUM_QUERIES + 1):
            for stem in (f"adj_mat_{q}.npy", f"task_duration_{q}.npy"):
                if not (data_dir / size / stem).is_file():
                    missing.append(f"{size}/{stem}")
    return missing


def tpch_ready(data_dir: Path = DATA_DIR) -> bool:
    return not missing_files(data_dir)


# ----------------------------------------------------------------------------- tải dữ liệu
def _copy_tpch_tree(src: Path, dst: Path = DATA_DIR) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for size in QUERY_SIZES:
        shutil.copytree(src / size, dst / size, dirs_exist_ok=True)


def fetch_from_local_clone(decima_repo: Path, dst: Path = DATA_DIR) -> None:
    _copy_tpch_tree(Path(decima_repo) / "spark_env" / "tpch", dst)


def fetch_with_git(dst: Path = DATA_DIR) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", DECIMA_GIT_URL, tmp + "/decima-sim"], check=True)
        fetch_from_local_clone(Path(tmp) / "decima-sim", dst)


def fetch_with_zip(dst: Path = DATA_DIR) -> None:
    with urllib.request.urlopen(DECIMA_ZIP_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf, tempfile.TemporaryDirectory() as tmp:
        members = [m for m in zf.namelist() if "/spark_env/tpch/" in m and m.endswith(".npy")]
        zf.extractall(tmp, members=members)
        root = next(Path(tmp).iterdir())
        fetch_from_local_clone(root, dst)


def fetch_upstream(dst: Path = DATA_DIR) -> None:
    """Cách mặc định của spark-sched-sim (bit.ly -> zip chứa thư mục tpch/)."""
    with urllib.request.urlopen(UPSTREAM_TPCH_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf, tempfile.TemporaryDirectory() as tmp:
        zf.extractall(tmp)
        src = next(p for p in Path(tmp).rglob("2g")).parent
        _copy_tpch_tree(src, dst)


def fetch_tpch(method: str = "auto", local_clone: str | None = None, dst: Path = DATA_DIR) -> str:
    if tpch_ready(dst):
        return "already-present"
    if local_clone:
        fetch_from_local_clone(Path(local_clone), dst)
        return "local"
    order = {"auto": ["git", "zip", "upstream"], "git": ["git"], "zip": ["zip"], "upstream": ["upstream"]}[method]
    errors = []
    for m in order:
        try:
            {"git": fetch_with_git, "zip": fetch_with_zip, "upstream": fetch_upstream}[m](dst)
            if tpch_ready(dst):
                return m
            errors.append(f"{m}: thiếu {len(missing_files(dst))} file sau khi tải")
        except Exception as exc:  # noqa: BLE001 - báo lỗi gộp ở cuối
            errors.append(f"{m}: {exc!r}")
    raise RuntimeError("Không tải được dữ liệu TPC-H:\n  " + "\n  ".join(errors))


# ----------------------------------------------------------------------------- thống kê job template
def _rough_duration(td: dict) -> float:
    """Đúng như TPCHDataSampler: trung bình mọi mẫu thời lượng task (ms) sau tiền xử lý."""
    from spark_sched_sim.data_samplers.tpch import TPCHDataSampler

    return float(TPCHDataSampler._rough_task_duration(td))


def _preprocess(td: dict) -> None:
    # dùng lại đúng hàm tiền xử lý của spark-sched-sim để thống kê khớp với mô phỏng
    from spark_sched_sim.data_samplers.tpch import TPCHDataSampler

    TPCHDataSampler._pre_process_task_duration(td)


def load_template(query: int, size: str, data_dir: Path = DATA_DIR) -> dict:
    """Đọc một job template và trả về cấu trúc DAG + ước lượng khối lượng công việc."""
    qdir = Path(data_dir) / size
    adj = np.load(qdir / f"adj_mat_{query}.npy", allow_pickle=True)
    tds = np.load(qdir / f"task_duration_{query}.npy", allow_pickle=True).item()
    n = adj.shape[0]
    num_tasks, rough = np.zeros(n, dtype=int), np.zeros(n)
    for s in range(n):
        td = tds[s]
        e = next(iter(td["first_wave"]))
        num_tasks[s] = len(td["first_wave"][e]) + len(td["rest_wave"][e])
        _preprocess(td)
        rough[s] = _rough_duration(td)
    work = num_tasks * rough  # ms, ước lượng tổng thời gian-executor của mỗi stage
    children = [np.nonzero(adj[i])[0].tolist() for i in range(n)]
    cp_work = critical_path(children, work)
    cp_time = critical_path(children, rough)  # độ dài đường găng nếu song song vô hạn (mỗi stage ~ 1 task)
    depth = critical_path(children, np.ones(n))
    return {
        "query": query, "size": size, "num_stages": n, "num_edges": int(adj.sum()),
        "num_tasks": int(num_tasks.sum()), "total_work_s": float(work.sum()) / 1e3,
        "cp_work_s": float(cp_work.max()) / 1e3, "cp_time_s": float(cp_time.max()) / 1e3,
        "depth": int(depth.max()), "max_tasks_per_stage": int(num_tasks.max()),
        "mean_task_duration_s": float(rough.mean()) / 1e3,
    }


def critical_path(children: list[list[int]], weight: np.ndarray) -> np.ndarray:
    """cp(v) = weight(v) + max_{u ∈ children(v)} cp(u)   (định nghĩa trong Decima, §5.1, chú thích 5)."""
    n = len(children)
    indeg = np.zeros(n, dtype=int)
    for v in range(n):
        for u in children[v]:
            indeg[u] += 1
    order, stack = [], [v for v in range(n) if indeg[v] == 0]
    while stack:
        v = stack.pop()
        order.append(v)
        for u in children[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                stack.append(u)
    if len(order) != n:
        raise ValueError("đồ thị có chu trình — không phải DAG")
    cp = np.zeros(n)
    for v in reversed(order):
        cp[v] = weight[v] + max((cp[u] for u in children[v]), default=0.0)
    return cp


def template_table(data_dir: Path = DATA_DIR):
    import pandas as pd

    rows = [load_template(q, s, data_dir) for s in QUERY_SIZES for q in range(1, NUM_QUERIES + 1)]
    return pd.DataFrame(rows)
