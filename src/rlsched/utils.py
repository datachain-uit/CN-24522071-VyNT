"""Ghi metadata để tái lập: commit git, phiên bản thư viện, cấu hình đã dùng."""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown (chưa phải git repo)"


def versions() -> dict[str, str]:
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "pandas", "gymnasium", "networkx", "torch", "torch_geometric",
                "torch_scatter", "torch_sparse"):
        try:
            v[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception:  # noqa: BLE001
            v[mod] = "not installed"
    return v


def run_metadata(**extra: Any) -> dict[str, Any]:
    return {"time": _dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
            "argv": sys.argv, "versions": versions(), **extra}


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def df_to_markdown(df, floatfmt: str = ".3f") -> str:
    """Bảng Markdown đơn giản (không cần gói tabulate)."""
    import math

    cols = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)
