import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # spark-sched-sim đọc dữ liệu theo đường dẫn tương đối data/tpch


@pytest.fixture(scope="session")
def debug_env_cfg():
    from rlsched.data import tpch_ready
    from rlsched.envs import load_yaml

    if not tpch_ready():
        pytest.skip("TPC-H data missing: run python scripts/00_setup_data.py")
    return load_yaml("configs/env/tpch_debug.yaml")
