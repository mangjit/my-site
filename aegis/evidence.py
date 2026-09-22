"""Point-in-time evidence package.

The narrative is a template over measured fields. It is not allowed to add a price,
a cause, or a trade instruction.
"""

from __future__ import annotations


def build_point_in_time(
    instrument: str,
    timeframe: str,
    decision_time: str,
    dataset: dict,
    regime: dict,
    retrieval: dict,
    events: dict,
    logit: dict[str, float],
    quality: dict,
    cost: dict,
    audit_ok: bool,
    agreement: bool,
    agreed_side: str | None,
) -> dict:
    synthetic = bool(dataset.get("synthetic"))
    flags = ["RESEARCH_ONLY", "NOT_A_TRADE_INSTRUCTION", "LIVE_EXECUTION_DISABLED"]
    if synthetic:
        flags.append("SYNTHETIC_DATA")
    if not dataset.get("financing_measured", False):
        flags.append("FINANCING_NOT_MEASURED")
    if not cost.get("slippage_measured", False):
        flags.append("SLIPPAGE_IS_ASSUMPTION")
    if quality.get("unsafe_for_research"):
        flags.append("DATA_QUALITY_FAIL")
    if not audit_ok:
        flags.append("AUDIT_FAILED")
    if retrieval.get("matches", 0) < 20:
        flags.append("THIN_MEMORY")

    blocked = (
        synthetic
        or not audit_ok
        or quality.get("unsafe_for_research")
        or not retrieval.get("historical_cutoff_verified", False)
        or events.get("future_information_detected", False)
        or events.get("unsafe_used", False)
    )
    if blocked or not agreement or agreed_side is None:
        classification = "INSUFFICIENT_DATA" if blocked or retrieval.get("matches", 0) < 20 else "NO_EDGE"
        if blocked:
            classification = "INSUFFICIENT_DATA"
        elif not agreement:
            classification = "NO_EDGE"
        confidence = 0.0
    else:
        classification = "LONG_BIAS" if agreed_side == "LONG" else "SHORT_BIAS"
        confidence = min(logit.get("LONG_SUCCESS", 0.0), logit.get("SHORT_SUCCESS", 0.0))
        # Confidence is the weaker of the two model estimates on the agreed label, not certainty.
        label = "LONG_SUCCESS" if agreed_side == "LONG" else "SHORT_SUCCESS"
        knn_share = (retrieval.get("distribution") or {}).get("labels", {}).get(label) or 0.0
        confidence = min(float(logit.get(label, 0.0)), float(knn_share))
        flags.append("SINGLE_TIMESTAMP_NOT_A_VALIDATION")
        flags.append("HOLDOUT_STATUS_MUST_BE_READ_SEPARATELY")

    dist = retrieval.get("distribution") or {}
    summary = (
        f"At {decision_time} the measurable regime label is {regime.get('classification')} "
        f"(heuristic confidence {regime.get('confidence')}, not a probability). "
        f"Logistic estimates — { _fmt_probs(logit) } — are model output, not frequencies. "
        f"Euclidean memory returned {retrieval.get('matches')} prior states with cutoff verified "
        f"{retrieval.get('historical_cutoff_verified')}. "
        f"Median analogue forward midpoint return: {_fmt(dist.get('median_forward_outcome'))}. "
        f"Event memory contributed {len(events.get('historical_analogues') or [])} usable events. "
        f"Classification {classification} is a research label. It is not an order."
    )
    uncertainty = (
        "OHLC bars do not reveal which barrier printed first inside a single bar. "
        "Probabilities are not calibrated. Nearest-neighbor outcomes are a sample, not a destiny. "
        "Slippage and financing in this package follow the frozen declaration, not a measured swap curve, "
        "unless financing_measured is true. A bias at one timestamp is not walk-forward evidence."
    )
    if synthetic:
        uncertainty = (
            "This timestamp belongs to a synthetic laboratory clock. Nothing in the package is evidence "
            "about a traded market. " + uncertainty
        )
    return {
        "instrument": instrument,
        "timestamp": decision_time,
        "timeframe": timeframe,
        "decision_time_definition": "bar_close_utc",
        "regime": {
            "classification": regime.get("classification"),
            "confidence": regime.get("confidence"),
            "confidence_kind": regime.get("confidence_kind"),
        },
        "market_memory": {
            "matches": retrieval.get("matches"),
            "historical_cutoff_verified": retrieval.get("historical_cutoff_verified"),
            "median_forward_outcome": dist.get("median_forward_outcome"),
            "positive_outcome_probability": dist.get("positive_outcome_probability"),
            "negative_outcome_probability": dist.get("negative_outcome_probability"),
            "retrieval": "euclidean",
            "cosine_used_for_classification": False,
        },
        "event_memory": {
            "active_event": events.get("active_event"),
            "historical_analogues": events.get("historical_analogues") or [],
            "future_information_detected": events.get("future_information_detected", False),
        },
        "quant_model": {
            "long_probability": logit.get("LONG_SUCCESS"),
            "short_probability": logit.get("SHORT_SUCCESS"),
            "no_trade_probability": logit.get("NO_RESOLUTION"),
            "ambiguous_path_probability": logit.get("AMBIGUOUS_PATH"),
            "probability_kind": "ovr_normalized_estimate",
        },
        "agreement": {
            "quant_vs_rag": agreement,
            "event_vs_market": None,
        },
        "research_classification": classification,
        "confidence": round(confidence, 4),
        "risk_flags": flags,
        "evidence_summary": summary,
        "uncertainty": uncertainty,
        "research_only": True,
        "live_execution": False,
    }


def _fmt(value) -> str:
    if value is None:
        return "not available"
    return f"{value:.6f}"


def _fmt_probs(probs: dict) -> str:
    if not probs:
        return "not available"
    return ", ".join(f"{key} {probs[key]:.3f}" for key in probs)
