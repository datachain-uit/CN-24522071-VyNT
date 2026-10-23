"""Chia tập theo SEED (Cách 3 trong báo cáo yêu cầu môi trường): mỗi seed sinh một chuỗi job
(thời điểm đến + loại query + kích thước) khác nhau. Train / validation / test dùng các dải seed rời nhau.

Trainer của spark-sched-sim dùng seed = trainer.seed + k (k = 0, 1, 2, ... tăng theo số lần reset),
nên dải train được khai báo rộng và dải validation/test đặt rất xa để không bao giờ trùng.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..envs import load_yaml


@dataclass(frozen=True)
class SeedRange:
    start: int
    count: int

    @property
    def stop(self) -> int:
        return self.start + self.count

    def seeds(self) -> list[int]:
        return list(range(self.start, self.stop))

    def overlaps(self, other: "SeedRange") -> bool:
        return self.start < other.stop and other.start < self.stop


@dataclass(frozen=True)
class Splits:
    train: SeedRange
    val: SeedRange
    test: SeedRange

    def get(self, name: str) -> SeedRange:
        return {"train": self.train, "val": self.val, "test": self.test}[name]

    def check_disjoint(self) -> None:
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        for a, b in pairs:
            if self.get(a).overlaps(self.get(b)):
                raise ValueError(f"dải seed '{a}' và '{b}' bị chồng lấn — vi phạm nguyên tắc tách tập")

    def check_trainer_cfg(self, trainer_seed: int, num_iterations: int, num_sequences: int) -> None:
        """Seed lớn nhất mà trainer có thể dùng: seed + num_sequences * (num_iterations + 1)."""
        max_seed = trainer_seed + num_sequences * (num_iterations + 1)
        if trainer_seed < self.train.start or max_seed >= self.train.stop:
            raise ValueError(f"seed huấn luyện [{trainer_seed}, {max_seed}] vượt ra ngoài dải train "
                             f"[{self.train.start}, {self.train.stop}) — có nguy cơ trùng validation/test")


def load_splits(path: str = "configs/splits.yaml") -> Splits:
    raw = load_yaml(path)
    s = Splits(**{k: SeedRange(int(v["start"]), int(v["count"])) for k, v in raw.items() if k in ("train", "val", "test")})
    s.check_disjoint()
    return s
