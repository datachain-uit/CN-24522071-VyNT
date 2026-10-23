"""Đường dẫn dùng chung và các hàm khởi tạo môi trường chạy.

spark-sched-sim đọc dữ liệu TPC-H theo đường dẫn TƯƠNG ĐỐI ``data/tpch``
(tính từ thư mục hiện hành). Vì vậy mọi script trong ``scripts/`` đều gọi
``enter_repo_root()`` trước khi tạo môi trường.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO_ROOT / "third_party" / "spark-sched-sim"
PYG_FALLBACK_ROOT = REPO_ROOT / "third_party" / "pyg_ext_fallback"
DATA_DIR = REPO_ROOT / "data" / "tpch"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR = REPO_ROOT / "configs"


def ensure_sim_on_path() -> None:
    """Cho phép ``import spark_sched_sim``, ``schedulers``, ``trainers`` từ bản vendored."""
    p = str(SIM_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def enter_repo_root() -> Path:
    """Chuyển thư mục hiện hành về gốc repo (bắt buộc vì đường dẫn dữ liệu là tương đối)."""
    os.chdir(REPO_ROOT)
    return REPO_ROOT


def enable_pyg_fallback(force: bool = False) -> bool:
    """Bật bản thay thế thuần PyTorch cho ``torch_scatter``/``torch_sparse`` khi cần.

    - Mặc định chỉ bật khi KHÔNG cài được hai gói này (ví dụ trên một số máy macOS).
    - ``force=True`` hoặc biến môi trường ``RLSCHED_PYG_FALLBACK=1`` buộc dùng bản thay thế.
    Bản thay thế chỉ cài đặt đúng các hàm mà Decima trong spark-sched-sim sử dụng
    (``segment_csr`` dạng tổng, ``SparseTensor``, ``matmul``) và có test đối chiếu
    với phép tính dày (dense) trong ``tests/test_pyg_fallback.py``.
    """
    force = force or os.environ.get("RLSCHED_PYG_FALLBACK") == "1"
    missing = (
        importlib.util.find_spec("torch_scatter") is None
        or importlib.util.find_spec("torch_sparse") is None
    )
    if not (force or missing):
        return False
    p = str(PYG_FALLBACK_ROOT)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    # các tiến trình con (multiprocessing 'spawn') kế thừa sys.path của tiến trình cha,
    # còn biến môi trường giúp script con biết đang dùng bản thay thế
    os.environ["RLSCHED_PYG_FALLBACK"] = "1"
    warnings.warn(
        "Đang dùng bản thay thế thuần PyTorch cho torch_scatter/torch_sparse "
        "(third_party/pyg_ext_fallback). Kết quả tương đương về toán học nhưng chậm hơn.",
        stacklevel=2,
    )
    return True
