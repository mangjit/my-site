"""Leakage and integrity checks. A failed check invalidates the experiment."""

from __future__ import annotations

from aegis.config import FEATURE_NAMES
from aegis.features import compute_features
from aegis.table import CandleTable


def feature_causality_errors(
    table: CandleTable,
    features: dict,
    indices: list[int],
    vol_window: int,
    slow_vol_bars: int,
) -> list[str]:
    """Recompute features on a strict prefix and compare them with the full-series values."""
    errors = []
    for index in indices:
        prefix = compute_features(
            table.prefix(index + 1),
            vol_window=vol_window,
            slow_vol_bars=slow_vol_bars,
        )
        for name in FEATURE_NAMES:
            left = features[name][index]
            right = prefix[name][index]
            if left is None and right is None:
                continue
            if left is None or right is None or abs(float(left) - float(right)) > 1e-9:
                errors.append(f"feature {name} at {index} changed when the future was removed")
                break
    return errors


def future_perturbation_errors(
    table: CandleTable,
    features: dict,
    index: int,
    vol_window: int,
    slow_vol_bars: int,
) -> list[str]:
    """Changing a later bar must not change an earlier feature row."""
    if index + 2 >= len(table):
        return ["perturbation_index_too_close_to_end"]
    mutated = table.copy()
    mutated.bid_c[-1] = mutated.bid_c[-1] + 0.01
    mutated.ask_c[-1] = mutated.ask_c[-1] + 0.01
    mutated.bid_h[-1] = max(mutated.bid_h[-1], mutated.bid_c[-1])
    mutated.ask_h[-1] = max(mutated.ask_h[-1], mutated.ask_c[-1])
    recomputed = compute_features(mutated, vol_window=vol_window, slow_vol_bars=slow_vol_bars)
    errors = []
    for name in FEATURE_NAMES:
        left = features[name][index]
        right = recomputed[name][index]
        if left is None and right is None:
            continue
        if left is None or right is None or abs(float(left) - float(right)) > 1e-9:
            errors.append(f"feature {name} at {index} changed when a later bar was perturbed")
    return errors
