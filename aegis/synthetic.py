"""Deterministic laboratory price process.

This is not a model of EUR/USD. It exists so the engine, the auditor, and the
report can be run without inventing a historical market series. Zero drift is
intentional. A pattern found here is a fact about this generator, not about a market.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta, timezone

from aegis.table import CandleTable


DATASET_ID = "SYNTHETIC_LAB_EURUSD_H1_V1"
GENERATOR_VERSION = "synthetic_zero_drift_v1"


def iter_weekday_hours(start: date, end: date):
    day = start
    while day <= end:
        if day.weekday() < 5:
            for hour in range(24):
                yield datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
        day += timedelta(days=1)


def generate_laboratory_eurusd(seed: int = 20260922) -> tuple[CandleTable, dict]:
    rng = random.Random(seed)
    opens = list(iter_weekday_hours(date(2010, 1, 4), date(2018, 12, 31)))
    table = CandleTable(
        instrument="EUR_USD",
        granularity="H1",
        timestamp=opens,
        bid_o=[],
        bid_h=[],
        bid_l=[],
        bid_c=[],
        ask_o=[],
        ask_h=[],
        ask_l=[],
        ask_c=[],
        volume=[],
        complete=[True] * len(opens),
        timestamp_meaning="bar_open_utc",
    )
    mid = 1.1500
    log_sigma = math.log(0.00035)
    vol_mu = log_sigma
    phi = 0.985
    for _ in opens:
        log_sigma = vol_mu + phi * (log_sigma - vol_mu) + rng.gauss(0.0, 0.035)
        if rng.random() < 0.001:
            log_sigma += 0.7
        sigma = min(max(math.exp(log_sigma), 1e-6), 0.008)
        ret = rng.gauss(0.0, sigma)
        bar_open = mid
        bar_close = max(0.5, mid * math.exp(ret))
        wick = abs(rng.gauss(0.0, sigma)) * bar_open
        bar_high = max(bar_open, bar_close) + wick * rng.random()
        bar_low = min(bar_open, bar_close) - wick * rng.random()
        bar_low = max(0.4, bar_low)
        if bar_low > min(bar_open, bar_close):
            bar_low = min(bar_open, bar_close)
        spread = max(0.00004, 0.00005 + 0.15 * sigma + abs(rng.gauss(0.0, 0.000008)))
        half = spread / 2.0
        table.bid_o.append(bar_open - half)
        table.bid_h.append(bar_high - half)
        table.bid_l.append(bar_low - half)
        table.bid_c.append(bar_close - half)
        table.ask_o.append(bar_open + half)
        table.ask_h.append(bar_high + half)
        table.ask_l.append(bar_low + half)
        table.ask_c.append(bar_close + half)
        # Volume is a liquidity-shaped proxy, not exchange volume.
        table.volume.append(1000.0 / spread * (0.5 + rng.random()))
        mid = bar_close
    meta = {
        "source": "SYNTHETIC",
        "synthetic": True,
        "not_market_data": True,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "process": "zero_drift_log_price_with_ar1_volatility",
        "drift": 0.0,
        "volume_is_proxy": True,
        "event_join_allowed": False,
        "notes": (
            "Laboratory clock uses 2010-2018 weekday labels so walk-forward year "
            "splits can be tested. These timestamps are not historical market dates. "
            "Do not join this file to the real event catalog. Do not describe a "
            "result from this file as evidence about EUR/USD."
        ),
    }
    return table, meta
