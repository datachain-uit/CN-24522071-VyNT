"""Bản thay thế torch_scatter / torch_sparse phải khớp phép tính dày (dense) tham chiếu."""
import importlib.util

import pytest

torch = pytest.importorskip("torch")
ROOT_FALLBACK = str(__import__("rlsched").paths.PYG_FALLBACK_ROOT)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"fb_{name}", f"{ROOT_FALLBACK}/{name}/__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_segment_csr_matches_reference():
    ts = _load("torch_scatter")
    src = torch.randn(10, 3)
    ptr = torch.tensor([0, 3, 3, 7, 10])
    for red, fn in (("sum", torch.sum), ("mean", torch.mean), ("max", torch.amax)):
        out = ts.segment_csr(src, ptr, reduce=red)
        for k in range(len(ptr) - 1):
            a, b = int(ptr[k]), int(ptr[k + 1])
            ref = fn(src[a:b], dim=0) if b > a else torch.zeros(3)
            assert torch.allclose(out[k], ref, atol=1e-6), (red, k)
    ints = ts.segment_csr(torch.tensor([1, 2, 3, 4]), torch.tensor([0, 1, 4]))
    assert ints.tolist() == [1, 9]


def test_sparse_matmul_matches_dense():
    tsp = _load("torch_sparse")
    row = torch.tensor([0, 0, 1, 3])
    col = torch.tensor([1, 2, 2, 0])
    adj = tsp.SparseTensor(row=row, col=col, sparse_sizes=(4, 4))
    x = torch.randn(4, 5, requires_grad=True)
    dense = adj.to_dense()
    assert torch.allclose(tsp.matmul(adj, x), dense @ x, atol=1e-6)
    assert torch.allclose(tsp.matmul(adj.t(), x), dense.t() @ x, atol=1e-6)
    tsp.matmul(adj, x).sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(importlib.util.find_spec("torch_scatter") is None or "pyg_ext_fallback" in str(
    getattr(importlib.util.find_spec("torch_scatter"), "origin", "")), reason="native torch_scatter unavailable for comparison")
def test_fallback_matches_real_torch_scatter():
    import torch_scatter as real

    fb = _load("torch_scatter")
    src = torch.randn(20, 4)
    ptr = torch.tensor([0, 5, 5, 12, 20])
    for red in ("sum", "mean", "max"):
        assert torch.allclose(fb.segment_csr(src, ptr, reduce=red), real.segment_csr(src, ptr, reduce=red), atol=1e-6)
