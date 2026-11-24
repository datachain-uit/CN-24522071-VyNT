"""Step 0 - Download the TPC-H DAG data to data/tpch and check all 154 job templates.

Examples:
    python scripts/00_setup_data.py                  # try git, then zip, then the upstream bit.ly link
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
    ap.add_argument("--local-clone", default=None, help="path to an existing decima-sim clone")
    args = ap.parse_args()
    how = fetch_tpch(method=args.method, local_clone=args.local_clone)
    miss = missing_files()
    print(f"[data] source: {how}; directory: {DATA_DIR}; missing files: {len(miss)}")
    if miss:
        raise SystemExit(1)
    print("[data] OK: 7 sizes x 22 queries = 154 job templates.")


if __name__ == "__main__":
    main()
