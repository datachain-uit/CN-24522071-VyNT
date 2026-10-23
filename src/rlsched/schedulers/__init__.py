from .heuristics import (  # noqa: F401
    REGISTRY,
    FairPolicy,
    FIFOPolicy,
    RandomPolicy,
    SJFCPPolicy,
    WeightedFairPolicy,
    make_heuristic,
)
from .obs_view import ObsView, parse_obs  # noqa: F401


def make_scheduler(name: str, num_executors: int, **kwargs):
    """Tạo scheduler theo tên: random | fifo | fair | weighted_fair | sjf_cp | decima."""
    if name.lower() == "decima":
        from .decima import make_decima

        return make_decima(num_executors=num_executors, **kwargs)
    return make_heuristic(name, num_executors=num_executors, **kwargs)
