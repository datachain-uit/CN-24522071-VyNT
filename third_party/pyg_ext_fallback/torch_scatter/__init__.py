"""Bản thay thế TỐI THIỂU, thuần PyTorch, cho ``torch_scatter`` (chỉ dùng khi không cài được gói gốc).

Chỉ cài đặt ``segment_csr`` — hàm mà Decima (spark-sched-sim) và ``torch_geometric.utils.softmax``
(PyG 2.3, gọi với reduce='max' và 'sum') sử dụng.
Ngữ nghĩa giống gói gốc: gom (reduce) các hàng của ``src`` theo các đoạn liên tiếp được mô tả
bởi con trỏ CSR ``indptr`` (độ dài B+1), trả về tensor kích thước [B, ...]; đoạn rỗng cho giá trị 0.
Kiểm chứng: tests/test_pyg_fallback.py.
"""
from __future__ import annotations

import torch

__all__ = ["segment_csr"]
__version__ = "0.0-fallback"


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, out=None, reduce: str = "sum") -> torch.Tensor:
    if out is not None:
        raise NotImplementedError("fallback segment_csr không hỗ trợ tham số out")
    if indptr.dim() != 1:
        raise NotImplementedError("fallback segment_csr chỉ hỗ trợ indptr 1 chiều")
    indptr = indptr.to(device=src.device, dtype=torch.long)
    counts = indptr[1:] - indptr[:-1]
    num_segments = counts.numel()
    seg_index = torch.repeat_interleave(torch.arange(num_segments, device=src.device), counts)
    start, end = int(indptr[0]), int(indptr[-1])
    values = src[start:end]
    shape = (num_segments,) + tuple(src.shape[1:])
    result = torch.zeros(shape, dtype=src.dtype, device=src.device)

    if reduce in ("sum", "add", "mean"):
        result = result.index_add(0, seg_index, values)
        if reduce == "mean":
            denom = counts.clamp(min=1).to(result.dtype)
            result = result / denom.view((-1,) + (1,) * (result.dim() - 1))
        return result

    if reduce in ("max", "min"):
        idx = seg_index.view((-1,) + (1,) * (values.dim() - 1)).expand_as(values)
        return result.scatter_reduce(0, idx, values, reduce="a" + reduce, include_self=False)

    raise NotImplementedError(f"fallback segment_csr chưa hỗ trợ reduce='{reduce}'")
