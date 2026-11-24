"""Set up each script by adding src/ to sys.path and switching to the repository root."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bootstrap() -> Path:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from rlsched.paths import enter_repo_root

    return enter_repo_root()


def require_data() -> None:
    from rlsched.data import missing_files

    miss = missing_files()
    if miss:
        sys.exit(f"Missing {len(miss)} TPC-H data files (e.g. {miss[0]}). Run: python scripts/00_setup_data.py")
