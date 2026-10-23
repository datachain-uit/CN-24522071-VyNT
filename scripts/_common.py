"""Khởi tạo chung cho mọi script: thêm src/ vào sys.path và chuyển về thư mục gốc repo."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bootstrap() -> Path:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from rlsched.paths import enter_repo_root

    return enter_repo_root()


def require_data() -> None:
    from rlsched.data import missing_files

    miss = missing_files()
    if miss:
        sys.exit(f"Thiếu {len(miss)} file dữ liệu TPC-H (vd. {miss[0]}). Chạy: python scripts/00_setup_data.py")
