"""Forward paper log. Observations can be appended. Orders cannot."""

from __future__ import annotations

import json
from pathlib import Path

from aegis.safety import LiveExecutionForbidden, assert_research_only


PAPER_FIELDS = (
    "timestamp",
    "instrument",
    "expected_entry",
    "actual_simulated_entry",
    "expected_spread",
    "actual_spread",
    "model_probability",
    "regime",
    "analogue_count",
    "realized_outcome",
    "notes",
)


def append_observation(path: Path, observation: dict) -> None:
    assert_research_only()
    if observation.get("live_order") or observation.get("send_order"):
        raise LiveExecutionForbidden("Paper log refuses an observation that requests an order.")
    unknown = set(observation) - set(PAPER_FIELDS)
    if unknown:
        raise ValueError(f"Unknown paper-log fields: {sorted(unknown)}")
    missing = [field for field in PAPER_FIELDS if field not in observation]
    if missing:
        raise ValueError(f"Paper observation is missing fields: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(observation, sort_keys=True) + "\n")
