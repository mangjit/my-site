"""Columnar candle table. Timestamps are UTC bar-open times."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from aegis.timeutil import ensure_utc


COLUMNS = (
    "bid_o",
    "bid_h",
    "bid_l",
    "bid_c",
    "ask_o",
    "ask_h",
    "ask_l",
    "ask_c",
)


@dataclass
class CandleTable:
    instrument: str
    granularity: str
    timestamp: list[datetime]
    bid_o: list[float]
    bid_h: list[float]
    bid_l: list[float]
    bid_c: list[float]
    ask_o: list[float]
    ask_h: list[float]
    ask_l: list[float]
    ask_c: list[float]
    volume: list[float | None]
    complete: list[bool]
    timestamp_meaning: str = "bar_open_utc"

    def __len__(self) -> int:
        return len(self.timestamp)

    def prefix(self, n: int) -> CandleTable:
        """Return the first n bars. Used to prove a feature does not read the future."""
        if n < 0 or n > len(self):
            raise IndexError(n)
        return self._slice(0, n)

    def copy(self) -> CandleTable:
        return self._slice(0, len(self))

    def _slice(self, start: int, end: int) -> CandleTable:
        return CandleTable(
            instrument=self.instrument,
            granularity=self.granularity,
            timestamp=list(self.timestamp[start:end]),
            bid_o=list(self.bid_o[start:end]),
            bid_h=list(self.bid_h[start:end]),
            bid_l=list(self.bid_l[start:end]),
            bid_c=list(self.bid_c[start:end]),
            ask_o=list(self.ask_o[start:end]),
            ask_h=list(self.ask_h[start:end]),
            ask_l=list(self.ask_l[start:end]),
            ask_c=list(self.ask_c[start:end]),
            volume=list(self.volume[start:end]),
            complete=list(self.complete[start:end]),
            timestamp_meaning=self.timestamp_meaning,
        )

    def validate_clock(self) -> None:
        if self.timestamp_meaning != "bar_open_utc":
            raise ValueError(
                f"Unsupported timestamp_meaning {self.timestamp_meaning!r}. "
                "Refusing to guess when a bar became knowable."
            )
        previous: datetime | None = None
        for ts in self.timestamp:
            utc = ensure_utc(ts)
            if previous is not None and utc <= previous:
                raise ValueError("Timestamps must be strictly increasing UTC bar-open times.")
            previous = utc


def empty_table(instrument: str, granularity: str) -> CandleTable:
    return CandleTable(
        instrument=instrument,
        granularity=granularity,
        timestamp=[],
        bid_o=[],
        bid_h=[],
        bid_l=[],
        bid_c=[],
        ask_o=[],
        ask_h=[],
        ask_l=[],
        ask_c=[],
        volume=[],
        complete=[],
    )
