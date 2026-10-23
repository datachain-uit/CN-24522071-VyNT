"""Bản thay thế TỐI THIỂU, thuần PyTorch, cho ``torch_sparse`` (chỉ dùng khi không cài được gói gốc).

Chỉ cài đặt những gì Decima trong spark-sched-sim dùng:
  - ``SparseTensor(row=..., col=..., sparse_sizes=..., value=None, ...)`` và ``.t()``
  - ``matmul(sparse, dense)``: out[r] = sum_{(r,c)} value_rc * dense[c]  (reduce='sum')
Kiểm chứng với phép nhân ma trận dày trong tests/test_pyg_fallback.py.
"""
from __future__ import annotations

import torch

__all__ = ["SparseTensor", "matmul"]
__version__ = "0.0-fallback"


class SparseTensor:
    def __init__(self, row: torch.Tensor, col: torch.Tensor, sparse_sizes, value: torch.Tensor | None = None,
                 is_sorted: bool = False, trust_data: bool = False, **kwargs) -> None:
        self.row = row.long()
        self.col = col.long()
        self.sparse_sizes_ = (int(sparse_sizes[0]), int(sparse_sizes[1]))
        self.value = value

    def sizes(self):
        return list(self.sparse_sizes_)

    def t(self) -> "SparseTensor":
        return SparseTensor(self.col, self.row, (self.sparse_sizes_[1], self.sparse_sizes_[0]), self.value)

    def to_dense(self) -> torch.Tensor:
        dense = torch.zeros(self.sparse_sizes_)
        vals = torch.ones(self.row.numel()) if self.value is None else self.value
        dense.index_put_((self.row, self.col), vals, accumulate=True)
        return dense


def matmul(src: SparseTensor, other: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    if reduce not in ("sum", "add"):
        raise NotImplementedError(f"fallback matmul chưa hỗ trợ reduce='{reduce}'")
    row = src.row.to(other.device)
    col = src.col.to(other.device)
    msgs = other[col]
    if src.value is not None:
        v = src.value.to(other.device, other.dtype)
        msgs = msgs * v.view((-1,) + (1,) * (other.dim() - 1))
    out = torch.zeros((src.sparse_sizes_[0],) + tuple(other.shape[1:]), dtype=other.dtype, device=other.device)
    return out.index_add(0, row, msgs)
