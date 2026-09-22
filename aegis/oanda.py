"""Read-only OANDA v20 historical candle client.

Implemented endpoint, and the only endpoint this module will request:

    GET https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles

The live host is rejected. Order, trade, position, and close paths are rejected.
The token is read from the environment and is never written to disk by this module.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from aegis.config import GRANULARITIES, INSTRUMENT_UNIVERSE
from aegis.safety import AegisError, PRACTICE_HOST, LiveExecutionForbidden, assert_read_only_request, redact
from aegis.table import CandleTable
from aegis.timeutil import iso_z, parse_timestamp


class OandaHistoricalClient:
    def __init__(self, token: str | None = None, host: str = PRACTICE_HOST, opener=None):
        self.host = host
        self._token = token if token is not None else os.environ.get("OANDA_API_TOKEN", "")
        self._opener = opener or urllib.request.urlopen
        if not self._token:
            raise LiveExecutionForbidden(
                "OANDA_API_TOKEN is not set. Refusing to invent a credential or a price series."
            )
        if host != PRACTICE_HOST:
            raise LiveExecutionForbidden(
                f"Host {host} is not the practice historical host. Live access is forbidden."
            )

    def __repr__(self) -> str:
        return f"OandaHistoricalClient(host={self.host!r}, token='[REDACTED]')"

    def _get(self, instrument: str, params: dict) -> dict:
        if instrument not in INSTRUMENT_UNIVERSE:
            raise LiveExecutionForbidden(
                f"{instrument} is outside the research universe. Refusing the request."
            )
        granularity = params.get("granularity")
        if granularity not in GRANULARITIES:
            raise LiveExecutionForbidden(
                f"Granularity {granularity} is outside the research allowlist."
            )
        if str(params.get("smooth", "false")).lower() == "true":
            raise LiveExecutionForbidden(
                "Smoothed candles rewrite the open from the previous close. Refusing."
            )
        path = f"/v3/instruments/{instrument}/candles"
        query = urllib.parse.urlencode(params)
        url = f"https://{self.host}{path}?{query}"
        assert_read_only_request(url)
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept-Datetime-Format": "RFC3339",
                "User-Agent": "aegis-research-readonly",
            },
        )
        try:
            with self._opener(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            detail = redact(detail, self._token)
            raise AegisError(
                f"Historical candle request failed with HTTP {exc.code}. Body not logged raw."
            ) from exc
        if self._token in payload:
            raise LiveExecutionForbidden("Response contained the API token. Refusing to parse or store it.")
        return json.loads(payload)

    def fetch_candles(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> tuple[CandleTable, dict]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware UTC.")
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        if end <= start:
            raise ValueError("end must be after start.")
        table = CandleTable(
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
            timestamp_meaning="bar_open_utc",
        )
        cursor = start
        pages = 0
        dropped_incomplete = 0
        dropped_missing_side = 0
        seen: set[str] = set()
        while cursor < end and pages < 10000:
            params = {
                "price": "BA",
                "granularity": granularity,
                "from": iso_z(cursor),
                "to": iso_z(end),
                "includeFirst": "false" if pages else "true",
                "smooth": "false",
                "dailyAlignment": "17",
                "alignmentTimezone": "America/New_York",
            }
            payload = self._get(instrument, params)
            candles = payload.get("candles") or []
            if not candles:
                break
            last_open = None
            added = 0
            for candle in candles:
                parsed = _parse_candle(candle)
                if parsed is None:
                    if candle.get("complete") is False:
                        dropped_incomplete += 1
                    else:
                        dropped_missing_side += 1
                    continue
                open_time, row = parsed
                if open_time >= end:
                    continue
                key = iso_z(open_time)
                if key in seen:
                    continue
                if last_open is not None and open_time <= last_open:
                    raise LiveExecutionForbidden("Candle page was not strictly increasing. Refusing to stitch it.")
                seen.add(key)
                table.timestamp.append(open_time)
                for name, value in row.items():
                    getattr(table, name).append(value)
                last_open = open_time
                added += 1
            pages += 1
            if last_open is None:
                break
            next_cursor = last_open + timedelta(seconds=1)
            if next_cursor <= cursor:
                raise LiveExecutionForbidden("Pagination cursor did not advance. Refusing to loop.")
            cursor = next_cursor
            if added == 0:
                break
        meta = {
            "source": "OANDA_V20_PRACTICE_CANDLES",
            "synthetic": False,
            "not_market_data": False,
            "host": self.host,
            "price": "BA",
            "smooth": False,
            "daily_alignment": 17,
            "alignment_timezone": "America/New_York",
            "pages": pages,
            "dropped_incomplete": dropped_incomplete,
            "dropped_missing_side": dropped_missing_side,
            "token_stored": False,
            "live_host_used": False,
            "notes": (
                "Practice-host historical candles. OANDA's candle endpoint is a base-price "
                "history and can differ from an account's live pricing stream. "
                "This extract does not authorize trading."
            ),
        }
        return table, meta


def _parse_candle(candle: dict) -> tuple[datetime, dict] | None:
    if candle.get("complete") is not True:
        return None
    bid = candle.get("bid")
    ask = candle.get("ask")
    if not isinstance(bid, dict) or not isinstance(ask, dict):
        return None
    needed = ("o", "h", "l", "c")
    if any(side.get(key) in (None, "") for side in (bid, ask) for key in needed):
        return None
    open_time = parse_timestamp(candle["time"])
    volume = candle.get("volume")
    row = {
        "bid_o": float(bid["o"]),
        "bid_h": float(bid["h"]),
        "bid_l": float(bid["l"]),
        "bid_c": float(bid["c"]),
        "ask_o": float(ask["o"]),
        "ask_h": float(ask["h"]),
        "ask_l": float(ask["l"]),
        "ask_c": float(ask["c"]),
        "volume": None if volume is None else float(volume),
        "complete": True,
    }
    return open_time, row
