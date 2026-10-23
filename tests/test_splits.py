"""Tập train/val/test phải rời nhau; seed huấn luyện không được chạm tới val/test."""
import pytest

from rlsched.envs import load_yaml
from rlsched.evaluation.splits import SeedRange, Splits, load_splits


def test_config_splits_disjoint():
    s = load_splits()
    s.check_disjoint()
    assert set(s.val.seeds()).isdisjoint(s.test.seeds())


def test_overlap_detected():
    bad = Splits(train=SeedRange(0, 100), val=SeedRange(50, 10), test=SeedRange(200, 10))
    with pytest.raises(ValueError):
        bad.check_disjoint()


@pytest.mark.parametrize("cfg_file", ["configs/train/decima_smoke.yaml", "configs/train/decima_small.yaml"])
@pytest.mark.parametrize("train_seed_index", [0, 1, 2, 4])
def test_trainer_seeds_inside_train_range(cfg_file, train_seed_index):
    tr = load_yaml(cfg_file)["trainer"]
    load_splits().check_trainer_cfg(train_seed_index * 100_000, tr["num_iterations"], tr["num_sequences"])
