"""As-of alignment on availability times. A later print is never a feature of an earlier decision."""

from __future__ import annotations

from datetime import datetime


def asof_indices(decision_times: list[datetime], available_times: list[datetime]) -> list[int | None]:
    """For each decision time, the last available_time that is already known.

    `available_times` must be sorted ascending. An equal timestamp is treated as
    known at that instant. A later timestamp is never returned.
    """
    if any(available_times[i] > available_times[i + 1] for i in range(len(available_times) - 1)):
        raise ValueError("available_times must be sorted ascending.")
    out: list[int | None] = []
    cursor = -1
    n = len(available_times)
    k = 0
    for decision in decision_times:
        while k < n and available_times[k] <= decision:
            cursor = k
            k += 1
        out.append(cursor if cursor >= 0 else None)
    return out
