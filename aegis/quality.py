"""Data-quality measurements. A failed check blocks research use rather than being averaged away."""

from __future__ import annotations

from datetime import timedelta

from aegis.table import COLUMNS, CandleTable
from aegis.timeutil import granularity_delta


def _ohlc_consistent(o: float, h: float, l: float, c: float) -> bool:
    if min(o, h, l, c) <= 0:
        return False
    return l <= o <= h and l <= c <= h and l <= h


def assess_quality(table: CandleTable) -> dict:
    rows = len(table)
    duplicate_timestamps = 0
    non_monotonic = 0
    crossed = 0
    bad_ohlc = 0
    non_positive = 0
    incomplete = 0
    large_gaps = 0
    missing_volume = 0
    gap_limit = granularity_delta(table.granularity) * 3 + timedelta(hours=49)

    previous = None
    for i in range(rows):
        ts = table.timestamp[i]
        if previous is not None:
            if ts == previous:
                duplicate_timestamps += 1
            elif ts < previous:
                non_monotonic += 1
            elif ts - previous > gap_limit:
                large_gaps += 1
        previous = ts
        if not table.complete[i]:
            incomplete += 1
        if table.volume[i] is None:
            missing_volume += 1
        for name in COLUMNS:
            value = getattr(table, name)[i]
            if value <= 0:
                non_positive += 1
                break
        if (
            table.bid_o[i] > table.ask_o[i]
            or table.bid_h[i] > table.ask_h[i]
            or table.bid_l[i] > table.ask_l[i]
            or table.bid_c[i] > table.ask_c[i]
        ):
            crossed += 1
        if not _ohlc_consistent(table.bid_o[i], table.bid_h[i], table.bid_l[i], table.bid_c[i]):
            bad_ohlc += 1
        elif not _ohlc_consistent(table.ask_o[i], table.ask_h[i], table.ask_l[i], table.ask_c[i]):
            bad_ohlc += 1

    hard_failures = {
        "duplicate_timestamps": duplicate_timestamps,
        "non_monotonic": non_monotonic,
        "crossed_markets": crossed,
        "bad_ohlc": bad_ohlc,
        "non_positive_prices": non_positive,
        "incomplete": incomplete,
    }
    unsafe = rows == 0 or any(hard_failures.values())
    warnings = []
    if large_gaps:
        warnings.append("large_gaps")
    if missing_volume:
        warnings.append("missing_volume")
    if unsafe:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARNING"
    else:
        verdict = "PASS"
    return {
        "rows": rows,
        "duplicate_timestamps": duplicate_timestamps,
        "non_monotonic": non_monotonic,
        "crossed_markets": crossed,
        "bad_ohlc": bad_ohlc,
        "non_positive_prices": non_positive,
        "incomplete": incomplete,
        "large_gaps": large_gaps,
        "missing_volume": missing_volume,
        "warnings": warnings,
        "verdict": verdict,
        "unsafe_for_research": unsafe,
    }
