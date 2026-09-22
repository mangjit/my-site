"""Quantitative baselines. Probabilities here are model estimates, not frequencies observed in the future."""

from __future__ import annotations

import math

from aegis.config import LABELS


def fit_scaler(rows: list[list[float]]) -> dict:
    if not rows:
        raise ValueError("Cannot fit a scaler on an empty training window.")
    width = len(rows[0])
    means = [0.0] * width
    for row in rows:
        for j, value in enumerate(row):
            means[j] += value
    n = float(len(rows))
    means = [value / n for value in means]
    variances = [0.0] * width
    for row in rows:
        for j, value in enumerate(row):
            delta = value - means[j]
            variances[j] += delta * delta
    denom = max(len(rows) - 1, 1)
    stds = []
    active = []
    for j, variance in enumerate(variances):
        std = math.sqrt(variance / denom)
        if std < 1e-12:
            stds.append(1.0)
            active.append(False)
        else:
            stds.append(std)
            active.append(True)
    return {"mean": means, "std": stds, "active": active, "n_rows": len(rows), "fit_on": "training_rows_only"}


def transform(row: list[float], scaler: dict) -> list[float]:
    out = []
    for j, value in enumerate(row):
        if not scaler["active"][j]:
            out.append(0.0)
        else:
            out.append((value - scaler["mean"][j]) / scaler["std"][j])
    return out


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def fit_ovr_logit(
    rows: list[list[float]],
    labels: list[str],
    l2: float,
    steps: int,
    lr: float,
) -> dict:
    """L2-regularized one-versus-rest logistic regression, fit on the rows given.

    The caller must pass training rows only. This function does not look at a
    test year, and it does not tune itself on one.
    """
    if len(rows) != len(labels) or not rows:
        raise ValueError("Logit training rows and labels must be non-empty and aligned.")
    width = len(rows[0])
    classes = [label for label in LABELS if label in set(labels)]
    models = {}
    for label in classes:
        y = [1.0 if item == label else 0.0 for item in labels]
        positives = sum(y)
        negatives = len(y) - positives
        if positives == 0 or negatives == 0:
            models[label] = {"weights": [0.0] * (width + 1), "degenerate": True}
            continue
        pos_w = 0.5 * len(y) / positives
        neg_w = 0.5 * len(y) / negatives
        weights = [0.0] * (width + 1)
        for _ in range(steps):
            grad = [0.0] * (width + 1)
            for row, target in zip(rows, y):
                score = weights[0]
                for j, value in enumerate(row):
                    score += weights[j + 1] * value
                error = _sigmoid(score) - target
                sample_w = pos_w if target == 1.0 else neg_w
                weighted = error * sample_w
                grad[0] += weighted
                for j, value in enumerate(row):
                    grad[j + 1] += weighted * value
            scale = 1.0 / len(rows)
            weights[0] -= lr * grad[0] * scale
            for j in range(width):
                penalty = l2 * weights[j + 1]
                weights[j + 1] -= lr * (grad[j + 1] * scale + penalty)
        models[label] = {"weights": weights, "degenerate": False}
    return {
        "models": models,
        "classes": classes,
        "n_rows": len(rows),
        "probability_kind": "ovr_normalized_estimate",
    }


def predict_ovr(model: dict, row: list[float]) -> dict[str, float]:
    raw = {}
    for label in LABELS:
        spec = model["models"].get(label)
        if spec is None:
            raw[label] = 0.0
            continue
        weights = spec["weights"]
        score = weights[0]
        for j, value in enumerate(row):
            score += weights[j + 1] * value
        raw[label] = _sigmoid(score)
    total = sum(raw.values())
    if total <= 0:
        uniform = 1.0 / len(LABELS)
        return {label: uniform for label in LABELS}
    return {label: raw[label] / total for label in LABELS}


def decide_side(
    logit: dict[str, float],
    knn: dict[str, float],
    train_base: dict[str, float],
    min_probability: float,
    margin_over_train_base: float,
    min_margin_vs_opposite: float,
    neighbor_count: int,
    min_neighbors: int,
) -> str | None:
    """Pre-registered agreement rule. Returns LONG, SHORT, or None. Never both."""
    if neighbor_count < min_neighbors:
        return None
    eligible = []
    for side, label, opposite in (
        ("LONG", "LONG_SUCCESS", "SHORT_SUCCESS"),
        ("SHORT", "SHORT_SUCCESS", "LONG_SUCCESS"),
    ):
        threshold = max(min_probability, train_base.get(label, 0.0) + margin_over_train_base)
        logit_ok = (
            logit[label] >= threshold
            and logit[label] >= logit[opposite] + min_margin_vs_opposite
        )
        knn_ok = (
            knn[label] >= threshold
            and knn[label] >= knn[opposite] + min_margin_vs_opposite
        )
        if logit_ok and knn_ok:
            eligible.append(side)
    if len(eligible) == 1:
        return eligible[0]
    return None


def base_rates(labels: list[str]) -> dict[str, float]:
    if not labels:
        return {label: 0.0 for label in LABELS}
    n = float(len(labels))
    return {label: labels.count(label) / n for label in LABELS}
