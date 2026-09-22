"""Execution boundary. Research code may read historical candles. It cannot trade."""

from __future__ import annotations

from urllib.parse import urlparse


class AegisError(Exception):
    """Base error for the research engine."""


class LiveExecutionForbidden(AegisError):
    """Raised if any path would reach a live order or a live trading host."""


class DataIntegrityError(AegisError):
    """Raised when stored data does not match its manifest."""


class LeakageError(AegisError):
    """Raised when a check finds future information in a past decision."""


class HoldoutLockError(AegisError):
    """Raised when a locked holdout would be examined more than once."""


RESEARCH_ONLY = True
LIVE_EXECUTION = False
OANDA_LIVE_FORBIDDEN = True

PRACTICE_HOST = "api-fxpractice.oanda.com"
LIVE_HOST = "api-fxtrade.oanda.com"

ALLOWED_READ_HOSTS = frozenset({PRACTICE_HOST})

# Any request path containing one of these fragments is refused, even on the practice host.
FORBIDDEN_PATH_FRAGMENTS = (
    "/orders",
    "/trades",
    "/positions",
    "/close",
    "/replaces",
    "/cancel",
    "/configuration",
    "/changes",
    "/pricing/stream",
)

ALLOWED_PATH_PREFIX = "/v3/instruments/"
ALLOWED_PATH_SUFFIX = "/candles"


def assert_research_only() -> None:
    if not RESEARCH_ONLY or LIVE_EXECUTION or not OANDA_LIVE_FORBIDDEN:
        raise LiveExecutionForbidden(
            "Safety flags were mutated. This process must stay research-only."
        )


def assert_read_only_request(url: str) -> None:
    """Refuse every request that is not a practice-host historical candle GET."""
    assert_research_only()
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise LiveExecutionForbidden(f"Refusing non-HTTPS request: {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if host == LIVE_HOST or "fxtrade" in host:
        raise LiveExecutionForbidden(
            "OANDA live host is forbidden. Historical research must not use api-fxtrade.oanda.com."
        )
    if host not in ALLOWED_READ_HOSTS:
        raise LiveExecutionForbidden(f"Host is not on the historical-read allowlist: {host}")
    path = parsed.path or ""
    lowered = path.lower()
    for fragment in FORBIDDEN_PATH_FRAGMENTS:
        if fragment in lowered:
            raise LiveExecutionForbidden(f"Refusing forbidden path fragment {fragment!r} in {path}")
    if not (lowered.startswith(ALLOWED_PATH_PREFIX) and lowered.endswith(ALLOWED_PATH_SUFFIX)):
        raise LiveExecutionForbidden(
            "Only GET /v3/instruments/{instrument}/candles is implemented. "
            f"Refusing path {path}"
        )
    if parsed.username or parsed.password:
        raise LiveExecutionForbidden("Refusing a URL that embeds credentials.")


def redact(value: str, secret: str | None) -> str:
    if secret and secret in value:
        return value.replace(secret, "[REDACTED]")
    return value
