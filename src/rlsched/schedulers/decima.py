"""Bọc DecimaScheduler của spark-sched-sim (cần torch + torch_geometric).

Lưu ý tái lập: hàm lấy mẫu hành động của Decima (schedulers/decima/utils.py::sample) dùng
module ``random`` toàn cục của Python, nên bộ chạy đánh giá gọi ``random.seed(seed)`` trước mỗi episode.
"""
from __future__ import annotations

from typing import Any

from ..paths import enable_pyg_fallback, ensure_sim_on_path

# cấu hình kiến trúc mặc định của upstream (config/decima_tpch.yaml)
DEFAULT_AGENT_CFG: dict[str, Any] = {
    "agent_cls": "DecimaScheduler",
    "embed_dim": 16,
    "gnn_mlp_kwargs": {"hid_dims": [32, 16], "act_cls": "LeakyReLU",
                       "act_kwargs": {"inplace": True, "negative_slope": 0.2}},
    "policy_mlp_kwargs": {"hid_dims": [64, 64], "act_cls": "Tanh"},
}


def make_decima(num_executors: int, state_dict_path: str | None, agent_cfg: dict | None = None,
                force_pyg_fallback: bool = False):
    ensure_sim_on_path()
    enable_pyg_fallback(force=force_pyg_fallback)
    from schedulers import DecimaScheduler  # noqa: WPS433 (import trễ để phần heuristic không cần torch)

    cfg = {**DEFAULT_AGENT_CFG, **(agent_cfg or {})}
    cfg.update({"num_executors": int(num_executors), "state_dict_path": state_dict_path})
    cfg.pop("agent_cls", None)
    sched = DecimaScheduler(**cfg)
    sched.eval()
    sched.name = "Decima" if state_dict_path is None else f"Decima[{state_dict_path}]"
    return sched
