"""UTC time helpers. Bar open time and the time a bar becomes knowable are not the same."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


GRANULARITY_SECONDS = {
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D": 24 * 60 * 60,
}

NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
TOKYO = ZoneInfo("Asia/Tokyo")


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Naive timestamps are rejected. Pass timezone-aware UTC.")
    return value.astimezone(timezone.utc)


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp. Naive values are rejected.

    OANDA v20 emits nanosecond fractions. Python accepts microseconds, so extra
    digits are truncated rather than rounded into a neighboring microsecond by
    an implicit float conversion.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" not in text:
        raise ValueError(f"Timestamp is not ISO-8601 with a time component: {value}")
    date_part, time_part = text.split("T", 1)
    tz = ""
    clock = time_part
    if "+" in time_part:
        clock, tzrest = time_part.split("+", 1)
        tz = "+" + tzrest
    else:
        # A negative offset is the only remaining hyphenated timezone form after the date.
        minus = time_part.find("-")
        if minus != -1:
            clock = time_part[:minus]
            tz = time_part[minus:]
    if not tz:
        raise ValueError(f"Naive timestamp rejected: {value}")
    if "." in clock:
        hms, frac = clock.split(".", 1)
        digits = "".join(ch for ch in frac if ch.isdigit())[:6].ljust(6, "0")
        clock = f"{hms}.{digits}"
    parsed = datetime.fromisoformat(f"{date_part}T{clock}{tz}")
    return ensure_utc(parsed)


def iso_z(value: datetime) -> str:
    utc = ensure_utc(value).replace(microsecond=0)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def granularity_delta(granularity: str) -> timedelta:
    try:
        return timedelta(seconds=GRANULARITY_SECONDS[granularity])
    except KeyError as exc:
        raise KeyError(f"Unsupported granularity {granularity}") from exc


def bar_available_time(open_time: datetime, granularity: str) -> datetime:
    """When a completed bar's OHLC could first be known: its close, not its open."""
    return ensure_utc(open_time) + granularity_delta(granularity)


def session_flags(open_time: datetime, granularity: str) -> dict[str, float]:
    """Session flags known at the bar close.

    Windows are a research convention (local 08:00–17:00), not an exchange session
    definition. The New York 17:00 rollover flag marks an H1-or-finer bar whose
    interval contains that local time, and a daily bar is not given a fake
    intraday rollover flag.
    """
    close_time = bar_available_time(open_time, granularity)
    start = ensure_utc(open_time)
    ny_start = start.astimezone(NY)
    lon_start = start.astimezone(LONDON)
    ty_start = start.astimezone(TOKYO)
    ny_close = close_time.astimezone(NY)

    def in_desk(local: datetime) -> float:
        return 1.0 if 8 <= local.hour < 17 else 0.0

    ny_on = in_desk(ny_start)
    lon_on = in_desk(lon_start)
    # Overlap uses the bar open's local hour. A bar is not marked overlap merely
    # because the desks overlap later, after this bar would already have been entered.
    overlap = 1.0 if ny_on == 1.0 and lon_on == 1.0 else 0.0
    rollover = 0.0
    if granularity != "D":
        # Interval is [start, close). Contains 17:00 New York if local start is 16:00
        # on an hourly grid, or more generally if 17:00 NY falls inside the interval.
        ny_hour_17 = ny_start.replace(hour=17, minute=0, second=0, microsecond=0)
        if ny_start <= ny_hour_17 < ny_close:
            rollover = 1.0
        elif ny_close.hour == 17 and ny_close.minute == 0 and ny_start < ny_close:
            # close exactly 17:00 means the interval ends at rollover and does not
            # contain the print. Leave the flag at 0 unless start <= 17:00 < close.
            rollover = 0.0
    return {
        "tokyo_desk": in_desk(ty_start),
        "london_desk": lon_on,
        "new_york_desk": ny_on,
        "overlap_lon_ny": overlap,
        "rollover_bar": rollover,
    }
