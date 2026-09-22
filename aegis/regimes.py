"""Regime labels from measurable state at T. Names like 'COVID' are not a regime."""

from __future__ import annotations


def classify_regime(features: dict, index: int) -> dict:
    required = (
        "vol_pct",
        "rv_ratio",
        "spread_pct",
        "ret_12_z",
        "ma_dist_20_atr",
        "ma_dist_50_atr",
        "persist_12",
    )
    if any(features.get(name, [None])[index] is None for name in required):
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "confidence_kind": "heuristic_not_probability",
            "inputs_known": False,
        }
    vol_pct = features["vol_pct"][index]
    rv_ratio = features["rv_ratio"][index]
    spread_pct = features["spread_pct"][index]
    z_12 = features["ret_12_z"][index]
    ma_20 = features["ma_dist_20_atr"][index]
    ma_50 = features["ma_dist_50_atr"][index]
    persist = features["persist_12"][index]
    expansion = rv_ratio >= 1.30
    contraction = rv_ratio <= 0.75

    if vol_pct >= 0.90 and spread_pct >= 0.85 and abs(z_12) >= 2.0:
        name = "CRISIS"
        distance = min(vol_pct - 0.90, spread_pct - 0.85, abs(z_12) - 2.0)
    elif vol_pct >= 0.80 and expansion:
        name = "VOLATILITY_EXPANSION"
        distance = min(vol_pct - 0.80, rv_ratio - 1.30)
    elif vol_pct >= 0.80:
        name = "HIGH_VOLATILITY"
        distance = vol_pct - 0.80
    elif vol_pct <= 0.20 and contraction:
        name = "VOLATILITY_CONTRACTION"
        distance = min(0.20 - vol_pct, 0.75 - rv_ratio)
    elif vol_pct <= 0.25 and abs(ma_20) < 0.50:
        name = "LOW_VOLATILITY_RANGE"
        distance = min(0.25 - vol_pct, 0.50 - abs(ma_20))
    elif abs(ma_50) >= 1.0 and persist >= 0.67:
        name = "TRENDING"
        distance = min(abs(ma_50) - 1.0, persist - 0.67)
    else:
        name = "NORMAL"
        distance = 0.05
    confidence = max(0.0, min(0.99, 0.35 + distance))
    return {
        "classification": name,
        "confidence": round(confidence, 4),
        "confidence_kind": "heuristic_not_probability",
        "inputs_known": True,
        "risk_on_off": "NOT_CLASSIFIED_SINGLE_INSTRUMENT",
    }
