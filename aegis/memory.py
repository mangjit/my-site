"""Market-memory retrieval.

Neighbors must already have been knowable, and their outcomes must already have
been knowable, at the decision being explained. Similarity never sees the outcome.
"""

from __future__ import annotations

import math

from aegis.config import LABELS


def _distance(left: list[float], right: list[float]) -> tuple[float, float]:
    dist_sq = 0.0
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for a, b in zip(left, right):
        delta = a - b
        dist_sq += delta * delta
        dot += a * b
        left_sq += a * a
        right_sq += b * b
    euclidean = math.sqrt(dist_sq)
    denom = math.sqrt(left_sq) * math.sqrt(right_sq)
    cosine = dot / denom if denom > 0 else 0.0
    return euclidean, cosine


def retrieve(
    memory_rows: list[list[float]],
    memory_meta: list[dict],
    query: list[float],
    query_decision_time: str,
    k: int,
) -> dict:
    """Return two top-k lists. Euclidean is principal. Cosine is reported, not selected on.

    `memory_meta` entries must include `decision_time` and `label_available_time`,
    both ISO strings comparable lexicographically, plus `label` and `forward_mid_return`.
    A neighbor is illegal if its decision time is not strictly before the query, or
    if its label was not yet available.
    """
    if len(memory_rows) != len(memory_meta):
        raise ValueError("Memory rows and metadata are misaligned.")
    euclidean_hits = []
    cosine_hits = []
    violations = {"future_neighbor": 0, "outcome_not_yet_known": 0}
    for row, meta in zip(memory_rows, memory_meta):
        if meta["decision_time"] >= query_decision_time:
            violations["future_neighbor"] += 1
            continue
        if meta["label_available_time"] > query_decision_time:
            violations["outcome_not_yet_known"] += 1
            continue
        euclidean, cosine = _distance(query, row)
        item = {
            "decision_time": meta["decision_time"],
            "label": meta["label"],
            "forward_mid_return": meta["forward_mid_return"],
            "euclidean": euclidean,
            "cosine": cosine,
            "regime": meta.get("regime"),
            "instrument": meta.get("instrument"),
        }
        euclidean_hits.append(item)
        cosine_hits.append(item)
    euclidean_hits.sort(key=lambda item: item["euclidean"])
    cosine_hits.sort(key=lambda item: item["cosine"], reverse=True)
    principal = euclidean_hits[:k]
    secondary = cosine_hits[:k]
    return {
        "matches": len(principal),
        "principal": principal,
        "cosine_secondary": secondary,
        "violations": violations,
        "historical_cutoff_verified": violations["future_neighbor"] == 0 and violations["outcome_not_yet_known"] == 0,
        "distribution": _distribution(principal, weight="euclidean"),
        "cosine_distribution": _distribution(secondary, weight="cosine"),
    }


def _distribution(hits: list[dict], weight: str) -> dict:
    if not hits:
        return {
            "labels": {label: None for label in LABELS},
            "positive_outcome_probability": None,
            "negative_outcome_probability": None,
            "median_forward_outcome": None,
        }
    weights = {label: 0.0 for label in LABELS}
    total = 0.0
    returns = []
    for item in hits:
        if weight == "euclidean":
            w = 1.0 / (1.0 + item["euclidean"])
        else:
            w = max(item["cosine"], 0.0) + 1e-9
        weights[item["label"]] += w
        total += w
        if item["forward_mid_return"] is not None:
            returns.append(item["forward_mid_return"])
    shares = {label: (weights[label] / total if total else None) for label in LABELS}
    positives = [value for value in returns if value > 0]
    negatives = [value for value in returns if value < 0]
    return {
        "labels": shares,
        "positive_outcome_probability": (len(positives) / len(returns)) if returns else None,
        "negative_outcome_probability": (len(negatives) / len(returns)) if returns else None,
        "median_forward_outcome": _median(returns),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def side_from_distribution(distribution: dict) -> tuple[str | None, dict[str, float]]:
    shares = distribution["labels"]
    if any(shares[label] is None for label in LABELS):
        return None, {label: 0.0 for label in LABELS}
    probs = {label: float(shares[label]) for label in LABELS}
    return None, probs
