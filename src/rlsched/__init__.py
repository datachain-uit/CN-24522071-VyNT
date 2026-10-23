"""rlsched — khung thực nghiệm RL cho bài toán lập lịch DAG job trên cluster.

Lõi mô phỏng: spark-sched-sim (vendored tại ``third_party/spark-sched-sim``, MIT License),
bản cài đặt lại Decima (Mao et al., SIGCOMM 2019) bằng PyTorch Geometric + Gymnasium.
Gói này bổ sung: baseline heuristic, chia seed train/val/test, đo đạc, thống kê và
các script chạy thực nghiệm có thể tái lập.
"""
from .paths import (  # noqa: F401
    CONFIG_DIR,
    DATA_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    SIM_ROOT,
    enable_pyg_fallback,
    ensure_sim_on_path,
    enter_repo_root,
)

ensure_sim_on_path()

__version__ = "0.1.0"
