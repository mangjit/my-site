"""Causal features. A value at bar i is a function of bars 0..i only."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from collections import deque

from aegis.config import FEATURE_NAMES
from aegis.table import CandleTable
from aegis.timeutil import session_flags


def _rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    total = 0.0
    count = 0
    for i, value in enumerate(values):
        if value is not None:
            total += value
            count += 1
        if i >= window:
            old = values[i - window]
            if old is not None:
                total -= old
                count -= 1
        if i >= window - 1 and count == window:
            out[i] = total / window
    return out


def _rolling_std(values: list[float | None], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    total = 0.0
    total_sq = 0.0
    count = 0
    for i, value in enumerate(values):
        if value is not None:
            total += value
            total_sq += value * value
            count += 1
        if i >= window:
            old = values[i - window]
            if old is not None:
                total -= old
                total_sq -= old * old
                count -= 1
        if i >= window - 1 and count == window and window >= 2:
            variance = (total_sq - (total * total) / window) / (window - 1)
            out[i] = math.sqrt(variance) if variance > 0 else 0.0
    return out


def _rolling_extreme(values: list[float], window: int, maximum: bool) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    dq: deque[int] = deque()
    for i, value in enumerate(values):
        while dq and dq[0] <= i - window:
            dq.popleft()
        if maximum:
            while dq and values[dq[-1]] <= value:
                dq.pop()
        else:
            while dq and values[dq[-1]] >= value:
                dq.pop()
        dq.append(i)
        if i >= window - 1:
            out[i] = values[dq[0]]
    return out


def _wilder_atr(true_range: list[float], length: int) -> list[float | None]:
    out: list[float | None] = [None] * len(true_range)
    if len(true_range) < length:
        return out
    seed = sum(true_range[:length]) / length
    out[length - 1] = seed
    previous = seed
    for i in range(length, len(true_range)):
        previous = ((previous * (length - 1)) + true_range[i]) / length
        out[i] = previous
    return out


def _log_return(closes: list[float], lag: int) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    for i in range(lag, len(closes)):
        previous = closes[i - lag]
        current = closes[i]
        if previous > 0 and current > 0:
            out[i] = math.log(current / previous)
    return out


def _percentile_at(history: list, end: int, window: int) -> float | None:
    start = end - window + 1
    if start < 0:
        return None
    current = history[end]
    if current is None:
        return None
    less = 0
    equal = 0
    for value in history[start : end + 1]:
        if value is None:
            return None
        if value < current:
            less += 1
        elif value == current:
            equal += 1
    return (less + 0.5 * equal) / window


def _rolling_percentile(values: list[float | None], window: int) -> list[float | None]:
    """Causal percentile rank of the current value inside the trailing window."""
    out: list[float | None] = [None] * len(values)
    ordered: list[float] = []
    for i, value in enumerate(values):
        if value is not None:
            insort(ordered, value)
        if i >= window:
            old = values[i - window]
            if old is not None:
                del ordered[bisect_left(ordered, old)]
        if (
            value is not None
            and i >= window - 1
            and len(ordered) == window
        ):
            less = bisect_left(ordered, value)
            equal = bisect_right(ordered, value) - less
            out[i] = (less + 0.5 * equal) / window
    return out


def compute_features(table: CandleTable, vol_window: int = 24 * 63, slow_vol_bars: int = 120) -> dict[str, list]:
    n = len(table)
    mid_c = [(table.bid_c[i] + table.ask_c[i]) / 2.0 for i in range(n)]
    mid_h = [(table.bid_h[i] + table.ask_h[i]) / 2.0 for i in range(n)]
    mid_l = [(table.bid_l[i] + table.ask_l[i]) / 2.0 for i in range(n)]
    spread = [table.ask_c[i] - table.bid_c[i] for i in range(n)]
    ret_1 = _log_return(mid_c, 1)
    ret_3 = _log_return(mid_c, 3)
    ret_6 = _log_return(mid_c, 6)
    ret_12 = _log_return(mid_c, 12)
    ret_24 = _log_return(mid_c, 24)
    true_range = []
    for i in range(n):
        if i == 0:
            true_range.append(mid_h[i] - mid_l[i])
        else:
            true_range.append(
                max(
                    mid_h[i] - mid_l[i],
                    abs(mid_h[i] - mid_c[i - 1]),
                    abs(mid_l[i] - mid_c[i - 1]),
                )
            )
    atr = _wilder_atr(true_range, 14)
    rv_24 = _rolling_std(ret_1, 24)
    rv_120 = _rolling_std(ret_1, slow_vol_bars)
    ma_20 = _rolling_mean(mid_c, 20)
    ma_50 = _rolling_mean(mid_c, 50)
    high_24 = _rolling_extreme(mid_h, 24, True)
    low_24 = _rolling_extreme(mid_l, 24, False)
    positive = [None if value is None else (1.0 if value > 0 else 0.0) for value in ret_1]
    persist = _rolling_mean(positive, 12)

    columns: dict[str, list] = {name: [None] * n for name in FEATURE_NAMES}
    vol_pct_all = _rolling_percentile(rv_24, vol_window)
    spread_pct_all = _rolling_percentile(spread, vol_window)
    sessions = [session_flags(table.timestamp[i], table.granularity) for i in range(n)]

    for i in range(n):
        atr_i = atr[i]
        rv = rv_24[i]
        if atr_i is None or atr_i <= 0 or rv is None or rv <= 0 or rv_120[i] in (None, 0):
            continue
        if ma_20[i] is None or ma_50[i] is None or persist[i] is None:
            continue
        if high_24[i] is None or low_24[i] is None:
            continue
        width = high_24[i] - low_24[i]
        if width < 0:
            continue
        slope_base = ma_20[i - 5] if i >= 5 else None
        if slope_base is None:
            continue
        vol_pct = vol_pct_all[i]
        spread_pct = spread_pct_all[i]
        if vol_pct is None or spread_pct is None:
            continue
        hour = table.timestamp[i].hour + table.timestamp[i].minute / 60.0
        weekday = table.timestamp[i].weekday()
        z = lambda ret, lag: None if ret[i] is None else ret[i] / (rv * math.sqrt(lag))
        values = {
            "ret_1_z": z(ret_1, 1),
            "ret_3_z": z(ret_3, 3),
            "ret_6_z": z(ret_6, 6),
            "ret_12_z": z(ret_12, 12),
            "ret_24_z": z(ret_24, 24),
            "rv_ratio": rv / rv_120[i],
            "vol_pct": vol_pct,
            "ma_dist_20_atr": (mid_c[i] - ma_20[i]) / atr_i,
            "ma_dist_50_atr": (mid_c[i] - ma_50[i]) / atr_i,
            "ma_slope_20_atr": (ma_20[i] - slope_base) / atr_i,
            "persist_12": persist[i],
            "range_pos_24": 0.5 if width == 0 else (mid_c[i] - low_24[i]) / width,
            "dist_high_24_atr": (mid_c[i] - high_24[i]) / atr_i,
            "compression_24": width / atr_i,
            "spread_over_atr": spread[i] / atr_i,
            "spread_pct": spread_pct,
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            "weekday_sin": math.sin(2 * math.pi * weekday / 7.0),
            "weekday_cos": math.cos(2 * math.pi * weekday / 7.0),
            "overlap_lon_ny": sessions[i]["overlap_lon_ny"],
            "rollover_bar": sessions[i]["rollover_bar"],
        }
        if any(values[name] is None for name in FEATURE_NAMES):
            continue
        for name in FEATURE_NAMES:
            columns[name][i] = values[name]

    columns["atr"] = atr
    columns["rv_24"] = rv_24
    columns["spread"] = spread
    columns["mid_c"] = mid_c
    columns["session"] = sessions
    return columns


def row_ready(features: dict, index: int) -> bool:
    return all(features[name][index] is not None for name in FEATURE_NAMES)


def feature_vector(features: dict, index: int) -> list[float] | None:
    if not row_ready(features, index):
        return None
    return [float(features[name][index]) for name in FEATURE_NAMES]
