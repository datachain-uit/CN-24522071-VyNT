"""Parse command-line method specifications: ``name`` or ``name:key=value,key2=value2``.

Examples: fair | sjf_cp | srpt_cp | weighted_fair:alpha=-1 | weighted_fair:alpha=@results/tuning/tpch_small/best.json
For values starting with '@', read the matching key (e.g. "alpha") from the specified JSON file.
"""
from __future__ import annotations

import json


def _convert(key: str, raw: str):
    if raw.startswith("@"):
        with open(raw[1:], encoding="utf-8") as fh:
            return json.load(fh)[key]
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def parse_method(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    kwargs = {}
    if rest:
        for item in rest.split(","):
            k, _, v = item.partition("=")
            kwargs[k.strip()] = _convert(k.strip(), v.strip())
    return name.strip(), kwargs
