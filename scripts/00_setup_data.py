"""Bước 0 — Tải dữ liệu TPC-H DAG vào data/tpch và kiểm tra đủ 154 job template.

Ví dụ:
    python scripts/00_setup_data.py                  # tự thử git -> zip -> link bit.ly của upstream
    python scripts/00_setup_data.py --method zip
    python scripts/00_setup_data.py --local-clone ../decima-sim
"""
from __future__ import annotations

import argparse

from _common import bootstrap

bootstrap()

from rlsched.data import DATA_DIR, fetch_tpch, missing_files  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["auto", "git", "zip", "upstream"], default="auto")
    ap.add_argument("--local-clone", default=None, help="đường dẫn tới bản clone decima-sim có sẵn")
    args = ap.parse_args()
    how = fetch_tpch(method=args.method, local_clone=args.local_clone)
    miss = missing_files()
    print(f"[data] nguồn: {how}; thư mục: {DATA_DIR}; số file thiếu: {len(miss)}")
    if miss:
        raise SystemExit(1)
    print("[data] OK: đủ 7 kích thước x 22 query = 154 job template.")


if __name__ == "__main__":
    main()
