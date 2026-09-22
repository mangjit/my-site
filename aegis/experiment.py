"""Walk-forward experiment runner.

In-sample predictions are not computed. The locked holdout is not scored here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from aegis import PROTOCOL_VERSION, __version__
from aegis.audit import feature_causality_errors, future_perturbation_errors
from aegis.config import FEATURE_NAMES, ResearchSettings, price_increment
from aegis.costs import cost_model_from_settings
from aegis.events import catalog, retrieve_events
from aegis.evidence import build_point_in_time
from aegis.features import compute_features, feature_vector, row_ready
from aegis.memory import retrieve
from aegis.metrics import daily_returns_from_trades, summarize_trades
from aegis.models import base_rates, decide_side, fit_ovr_logit, fit_scaler, predict_ovr, transform
from aegis.quality import assess_quality
from aegis.regimes import classify_regime
from aegis.safety import HoldoutLockError, assert_research_only
from aegis.table import CandleTable
from aegis.targets import label_end_index, path_label, simulate_side
from aegis.timeutil import bar_available_time, iso_z
from aegis.walkforward import build_folds, purge_training_indices


LIMITATIONS = [
    "One instrument and one timeframe. A result does not transfer to the rest of the universe.",
    "In-sample fit is not reported, because it is not evidence.",
    "The locked final year is not scored by the research command.",
    "Same-bar barrier order is unidentified. Those paths are ambiguous, not wins.",
    "Financing in the laboratory settings is an adverse declaration, not a measured swap.",
    "Slippage is an assumption. The quote spread is taken from the bar, where the bar has a bid and an ask.",
    "Logistic outputs are regularized one-versus-rest scores normalized to sum to one. They are not calibrated probabilities.",
    "Euclidean retrieval is the pre-registered memory method. Cosine is reported and is not used to pick a result.",
    "Sharpe-type and Sortino-type ratios are descriptive. Overlapping volatility means the returns are not independent.",
    "Session flags use a local 08:00-17:00 research convention, not an exchange session definition.",
    "No cross-asset feature is used. Risk-on and risk-off are therefore not classified.",
    "A language model is not allowed to replace this calculation or to invent a price.",
]


def canonical_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_research(table: CandleTable, meta: dict, settings: ResearchSettings) -> dict:
    assert_research_only()
    if settings.instrument != table.instrument or settings.granularity != table.granularity:
        raise ValueError("Settings instrument/timeframe do not match the dataset.")
    quality = assess_quality(table)
    # Fold years follow the bar open. A bar that merely closes after midnight must not
    # create a one-bar "year" and thereby unlock the previous year.
    open_years = [ts.year for ts in table.timestamp]
    folds, holdout_years = build_folds(open_years, settings.min_train_years, settings.holdout_years)
    holdout_start = _research_cutoff(table, set(holdout_years))
    research = table.prefix(holdout_start)
    research_years = open_years[:holdout_start]
    features = compute_features(
        research,
        vol_window=settings.vol_percentile_window,
        slow_vol_bars=settings.slow_vol_bars,
    )
    decisions = _decision_indices(research, features, settings)
    cost = cost_model_from_settings(settings, price_increment(settings.instrument), 1.0)
    fold_reports = []
    predictions = []
    trades = []
    for fold in folds:
        print(
            f"walk-forward {fold.procedure} test year {fold.test_year} "
            f"train {fold.train_years[0]}-{fold.train_years[-1]}",
            flush=True,
        )
        report, fold_predictions, fold_trades = _run_fold(
            research, features, decisions, research_years, fold, settings, cost, meta
        )
        fold_reports.append(report)
        predictions.extend(fold_predictions)
        trades.extend(fold_trades)
    expanding_trades = [trade for trade in trades if trade["procedure"] == "expanding"]
    rolling_trades = [trade for trade in trades if trade["procedure"] == "rolling"]
    expanding_metrics = _metrics_for(research, expanding_trades, research_years, [fold.test_year for fold in folds if fold.procedure == "expanding"])
    rolling_metrics = _metrics_for(research, rolling_trades, research_years, [fold.test_year for fold in folds if fold.procedure == "rolling"])
    robustness = _robustness(research, predictions, expanding_trades, settings, cost, expanding_metrics)
    audit = _audit(
        research,
        features,
        decisions,
        fold_reports,
        trades,
        quality,
        settings,
        holdout_years,
        meta,
        holdout_bars=len(table) - holdout_start,
    )
    point = _last_point(research, features, predictions, quality, meta, settings, audit["valid"], cost)
    verdict = _verdict(expanding_metrics, rolling_metrics, robustness, audit, settings, meta)
    return {
        "experiment_id": _experiment_id(settings, meta),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aegis_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "research_only": True,
        "live_execution": False,
        "oanda_live": False,
        "hypothesis": settings.to_dict(),
        "settings_sha256": canonical_hash(settings.to_dict()),
        "dataset": _dataset_block(meta, table),
        "quality": quality,
        "holdout": {
            "years": holdout_years,
            "examined": False,
            "bars_excluded": len(table) - holdout_start,
            "reason": "Locked until features, retrieval, model family, costs, and the decision rule are frozen, then run once.",
        },
        "in_sample_performance_computed": False,
        "decision_count": len(decisions),
        "model_estimates": _estimate_span(predictions),
        "folds": fold_reports,
        "oos": {"expanding": expanding_metrics, "rolling": rolling_metrics},
        "robustness": robustness,
        "audit": audit,
        "verdict": verdict,
        "point_in_time": point,
        "limitations": LIMITATIONS,
        "predictions": predictions,
        "trades": trades,
    }


def run_holdout_once(table: CandleTable, meta: dict, settings: ResearchSettings, lock_dir: Path) -> dict:
    """Score the locked year once. A second call for the same settings hash is refused.

    The lock is written before scoring so a failed run cannot be iterated on.
    """
    assert_research_only()
    digest = canonical_hash(settings.to_dict())
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{digest}.json"
    if lock_path.exists():
        raise HoldoutLockError(
            f"Holdout for settings {digest} was already opened. Refusing a second look. "
            "A new question requires a new pre-registered settings hash, not another pass."
        )
    lock_path.write_text(
        json.dumps(
            {
                "settings_sha256": digest,
                "status": "OPENED_ONCE",
                "opened_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "research_only": True,
            },
            indent=2,
        )
        + "\n"
    )
    open_years = [ts.year for ts in table.timestamp]
    _, holdout_years = build_folds(open_years, settings.min_train_years, settings.holdout_years)
    if len(holdout_years) != 1:
        raise HoldoutLockError("This runner scores one locked year. Split a longer holdout explicitly.")
    from aegis.walkforward import Fold

    train_years = tuple(year for year in calendar_years(open_years) if year not in set(holdout_years))
    # The fold's holdout tuple is empty on purpose. The test year IS the locked year, so
    # embargoing labels that touch "holdout" would drop every holdout label. Training is
    # still purged where a pre-holdout label reaches into this year.
    fold = Fold("holdout_once", train_years, holdout_years[0], ())
    features = compute_features(
        table,
        vol_window=settings.vol_percentile_window,
        slow_vol_bars=settings.slow_vol_bars,
    )
    decisions = _decision_indices(table, features, settings)
    cost = cost_model_from_settings(settings, price_increment(settings.instrument), 1.0)
    report, predictions, trades = _run_fold(
        table, features, decisions, open_years, fold, settings, cost, meta
    )
    return {
        "settings_sha256": digest,
        "holdout_years": holdout_years,
        "examined_once": True,
        "research_only": True,
        "live_execution_authorized": False,
        "fold": report,
        "trades": len(trades),
        "net_pips": sum(trade["net_pips"] for trade in trades),
        "note": (
            "This file is a one-time historical examination. It is not part of the "
            "walk-forward equity curve and it does not authorize live execution."
        ),
        "predictions": predictions,
        "trade_rows": trades,
    }


def calendar_years(years: list[int]) -> list[int]:
    return sorted(set(years))


def write_experiment(directory: Path, result: dict) -> Path:
    experiment_id = result["experiment_id"]
    destination = directory / experiment_id
    if destination.exists():
        raise FileExistsError(f"Experiment {experiment_id} already exists and will not be overwritten.")
    destination.mkdir(parents=True)
    predictions = result.pop("predictions")
    trades = result.pop("trades")
    (destination / "experiment.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv(destination / "predictions.csv", predictions)
    _write_csv(destination / "trades.csv", trades)
    _write_equity(destination / "equity.csv", trades)
    registry = directory / "registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_id": experiment_id,
        "created_at_utc": result["created_at_utc"],
        "settings_sha256": result["settings_sha256"],
        "dataset_id": result["dataset"]["dataset_id"],
        "laboratory_process_verdict": result["verdict"]["laboratory_process_verdict"],
        "market_classification": result["verdict"]["market_classification"],
        "audit_valid": result["audit"]["valid"],
        "research_only": True,
    }
    with registry.open("a") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
    result["predictions"] = predictions
    result["trades"] = trades
    return destination


def _experiment_id(settings: ResearchSettings, meta: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = canonical_hash({"settings": settings.to_dict(), "dataset": meta.get("dataset_id"), "stamp": stamp})
    return f"exp_{stamp}_{digest[:12]}"


def _dataset_block(meta: dict, table: CandleTable) -> dict:
    return {
        "dataset_id": meta.get("dataset_id"),
        "source": meta.get("source"),
        "synthetic": bool(meta.get("synthetic")),
        "not_market_data": bool(meta.get("not_market_data")),
        "sha256": meta.get("sha256"),
        "rows": len(table),
        "start": iso_z(table.timestamp[0]) if table.timestamp else None,
        "end": iso_z(table.timestamp[-1]) if table.timestamp else None,
        "notes": meta.get("notes"),
        "financing_measured": False,
        "event_join_allowed": bool(meta.get("event_join_allowed", False)),
    }


def _estimate_span(predictions: list[dict]) -> dict | None:
    rows = [row for row in predictions if row.get("procedure") == "expanding"]
    if not rows:
        return None

    def span(key: str) -> dict:
        values = [float(row[key]) for row in rows]
        return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}

    return {
        "expanding_decisions": len(rows),
        "logit_long": span("logit_long"),
        "logit_short": span("logit_short"),
        "knn_long": span("knn_long"),
        "knn_short": span("knn_short"),
        "note": "Out-of-sample model estimates. They are not a license to retune the frozen threshold.",
    }


def _research_cutoff(table: CandleTable, holdout_years: set[int]) -> int:
    """First bar that must stay out of research.

    A bar is excluded if it opens in the holdout year or its close falls in the
    holdout year. The close rule stops the last hour of the prior year from
    carrying holdout prices into a feature or a label.
    """
    for index, opened in enumerate(table.timestamp):
        closed = bar_available_time(opened, table.granularity)
        if opened.year in holdout_years or closed.year in holdout_years:
            return index
    return len(table)


def _first_index(years: list[int], wanted: set[int]) -> int:
    for index, year in enumerate(years):
        if year in wanted:
            return index
    return len(years)


def _decision_indices(table: CandleTable, features: dict, settings: ResearchSettings) -> list[int]:
    ready = []
    for index in range(len(table)):
        if not row_ready(features, index):
            continue
        if features["atr"][index] is None or features["atr"][index] <= 0:
            continue
        if label_end_index(index, settings.horizon_bars, len(table)) is None:
            continue
        ready.append(index)
    if not ready:
        return []
    origin = ready[0]
    return [index for index in ready if (index - origin) % settings.stride_bars == 0]


def _labels_from_features(table, decisions, features, settings) -> dict[int, dict]:
    labels = {}
    for index in decisions:
        atr = features["atr"][index]
        if atr is None or atr <= 0:
            continue
        label = path_label(
            table, index, atr, settings.barrier_atr, settings.stop_atr, settings.horizon_bars
        )
        if label is not None:
            labels[index] = label
    return labels


def _run_fold(table, features, decisions, years, fold, settings, cost, meta):
    labels = _labels_from_features(table, decisions, features, settings)
    test_start = _first_index(years, {fold.test_year})
    holdout_start = _first_index(years, set(fold.holdout_years))
    candidates = [
        index
        for index in decisions
        if index in labels and years[index] in fold.train_years
    ]
    label_ends = {index: labels[index]["label_end_index"] for index in candidates}
    train_idx, embargoed = purge_training_indices(candidates, label_ends, test_start)
    test_idx = [
        index
        for index in decisions
        if index in labels and years[index] == fold.test_year and labels[index]["label_end_index"] < holdout_start
    ]
    train_rows = [feature_vector(features, index) for index in train_idx]
    train_y = [labels[index]["label"] for index in train_idx]
    scaler = fit_scaler(train_rows) if train_rows else None
    model = None
    rates = base_rates(train_y)
    if scaler is not None:
        scaled = [transform(row, scaler) for row in train_rows]
        model = fit_ovr_logit(scaled, train_y, settings.logit_l2, settings.logit_steps, settings.logit_lr)
    memory_rows = []
    memory_meta = []
    if scaler is not None:
        for index in train_idx:
            decision_time = iso_z(bar_available_time(table.timestamp[index], table.granularity))
            available = iso_z(
                bar_available_time(table.timestamp[labels[index]["label_end_index"]], table.granularity)
            )
            memory_rows.append(transform(feature_vector(features, index), scaler))
            memory_meta.append(
                {
                    "decision_time": decision_time,
                    "label_available_time": available,
                    "label": labels[index]["label"],
                    "forward_mid_return": labels[index]["forward_mid_return"],
                    "regime": classify_regime(features, index)["classification"],
                    "instrument": table.instrument,
                }
            )
    fold_predictions = []
    fold_trades = []
    violations = {"future_neighbor": 0, "outcome_not_yet_known": 0}
    cosine_disagreements = 0
    for index in test_idx:
        vector = feature_vector(features, index)
        if vector is None or scaler is None or model is None:
            continue
        scaled = transform(vector, scaler)
        logit = predict_ovr(model, scaled)
        query_time = iso_z(bar_available_time(table.timestamp[index], table.granularity))
        memory = retrieve(memory_rows, memory_meta, scaled, query_time, settings.knn_k)
        for key in violations:
            violations[key] += memory["violations"][key]
        knn = memory["distribution"]["labels"]
        if any(knn[label] is None for label in knn):
            knn = {label: 0.0 for label in knn}
        side = decide_side(
            logit,
            knn,
            rates,
            settings.min_probability,
            settings.margin_over_train_base,
            settings.min_margin_vs_opposite,
            memory["matches"],
            settings.min_neighbors,
        )
        cosine = memory["cosine_distribution"]["labels"]
        if any(cosine[label] is None for label in cosine):
            cosine = {label: 0.0 for label in cosine}
        cosine_side = decide_side(
            logit,
            cosine,
            rates,
            settings.min_probability,
            settings.margin_over_train_base,
            settings.min_margin_vs_opposite,
            len(memory["cosine_secondary"]),
            settings.min_neighbors,
        )
        if cosine_side != side:
            cosine_disagreements += 1
        regime = classify_regime(features, index)
        prediction = {
            "procedure": fold.procedure,
            "test_year": fold.test_year,
            "decision_time": query_time,
            "bar_index": index,
            "regime": regime["classification"],
            "logit_long": logit["LONG_SUCCESS"],
            "logit_short": logit["SHORT_SUCCESS"],
            "logit_none": logit["NO_RESOLUTION"],
            "knn_long": knn["LONG_SUCCESS"],
            "knn_short": knn["SHORT_SUCCESS"],
            "knn_none": knn["NO_RESOLUTION"],
            "neighbors": memory["matches"],
            "cutoff_verified": memory["historical_cutoff_verified"],
            "agreed_side": side or "",
            "cosine_side": cosine_side or "",
            "path_label": labels[index]["label"],
            "traded": bool(side),
            "median_forward_outcome": memory["distribution"]["median_forward_outcome"],
            "positive_outcome_probability": memory["distribution"]["positive_outcome_probability"],
            "negative_outcome_probability": memory["distribution"]["negative_outcome_probability"],
        }
        fold_predictions.append(prediction)
        if not side:
            continue
        trade = simulate_side(
            table,
            index,
            side,
            features["atr"][index],
            settings.barrier_atr,
            settings.stop_atr,
            settings.horizon_bars,
            cost,
        )
        if trade is None:
            continue
        trade.update(
            {
                "procedure": fold.procedure,
                "test_year": fold.test_year,
                "year": fold.test_year,
                "decision_time": query_time,
                "regime": regime["classification"],
                "instrument": table.instrument,
                "path_label": labels[index]["label"],
                "logit_long": logit["LONG_SUCCESS"],
                "knn_long": knn["LONG_SUCCESS"],
                "atr": features["atr"][index],
            }
        )
        fold_trades.append(trade)
    report = {
        "procedure": fold.procedure,
        "train_years": list(fold.train_years),
        "test_year": fold.test_year,
        "holdout_years": list(fold.holdout_years),
        "train_rows": len(train_idx),
        "embargoed_rows": len(embargoed),
        "test_rows": len(test_idx),
        "trades": len(fold_trades),
        "base_rates": rates,
        "scaler_rows": None if scaler is None else scaler["n_rows"],
        "retrieval_violations": violations,
        "cosine_disagreements": cosine_disagreements,
        "cosine_used_for_trades": False,
    }
    return report, fold_predictions, fold_trades


def _metrics_for(table, trades, years, test_years):
    if not trades:
        empty = summarize_trades([], [], 0.0)
        empty["oos_days"] = 0
        return empty
    test_year_set = set(test_years)
    oos_days = []
    seen = set()
    oos_bars = 0
    for index, year in enumerate(years):
        if year not in test_year_set:
            continue
        oos_bars += 1
        day = table.timestamp[index].date().isoformat()
        if day not in seen:
            seen.add(day)
            oos_days.append(day)
    paths = [_mark_path(table, trade) for trade in trades]
    daily = daily_returns_from_trades(trades, oos_days, paths)
    occupied = 0
    for trade in trades:
        occupied += trade["exit_index"] - trade["entry_index"] + 1
    exposure = occupied / oos_bars if oos_bars else 0.0
    summary = summarize_trades(trades, daily, exposure)
    summary["oos_days"] = len(oos_days)
    summary["oos_bars"] = oos_bars
    return summary


def _mark_path(table: CandleTable, trade: dict) -> list[tuple[str, float]]:
    entry = trade["entry_fill"]
    previous = 0.0
    path = []
    for index in range(trade["entry_index"], trade["exit_index"] + 1):
        if trade["side"] == "LONG":
            marked = (table.bid_c[index] - entry) / entry
        else:
            marked = (entry - table.ask_c[index]) / entry
        if index == trade["exit_index"]:
            marked = trade["net_return"]
        day = table.timestamp[index].date().isoformat()
        path.append((day, marked - previous))
        previous = marked
    return path


def _robustness(table, predictions, trades, settings, cost, expanding_metrics) -> dict:
    stressed = []
    for trade in trades:
        if trade["procedure"] != "expanding":
            continue
        index = trade["decision_index"]
        # ATR is recovered from the barrier distance implicit in the original trade only
        # if we re-simulate. The original call site has the feature ATR. Re-simulation
        # requires that ATR. Store it on the trade in _run_fold — added below if present.
        atr = trade.get("atr")
        if atr is None:
            continue
        again = simulate_side(
            table,
            index,
            trade["side"],
            atr,
            settings.barrier_atr,
            settings.stop_atr,
            settings.horizon_bars,
            cost,
            slippage_multiplier=2.0,
        )
        if again is not None:
            stressed.append(again["net_pips"])
    drop_five = expanding_metrics.get("net_without_best_five_trades")
    principal = expanding_metrics.get("net_pips") or 0.0
    return {
        "diagnostic_not_for_selection": True,
        "slippage_2x_net_pips": sum(stressed) if stressed else None,
        "slippage_2x_expectancy_pips": (sum(stressed) / len(stressed)) if stressed else None,
        "principal_net_pips": principal,
        "net_without_best_five": drop_five,
        "sign_flips_without_best_five": bool(principal > 0 and drop_five is not None and drop_five <= 0),
        "best_year_share_of_net": expanding_metrics.get("best_year_share_of_net"),
        "note": "These stresses do not choose a new parameter. The frozen rule remains the principal result.",
    }


def _audit(
    table,
    features,
    decisions,
    fold_reports,
    trades,
    quality,
    settings,
    holdout_years,
    meta,
    holdout_bars: int = 0,
) -> dict:
    findings = []
    invalid = False
    if quality["unsafe_for_research"]:
        findings.append("DATA_QUALITY_FAIL")
        invalid = True
    if not cost_model_from_settings(settings, price_increment(settings.instrument), 1.0).complete_declaration:
        findings.append("TRANSACTION_COSTS_MISSING")
        invalid = True
    for report in fold_reports:
        overlap = set(report["train_years"]) & {report["test_year"]}
        if overlap or report["test_year"] in holdout_years or set(report["train_years"]) & set(holdout_years):
            findings.append("FOLD_YEAR_LEAKAGE")
            invalid = True
        if report["scaler_rows"] not in (None, report["train_rows"]):
            findings.append("SCALER_ROW_MISMATCH")
            invalid = True
        if report["retrieval_violations"]["future_neighbor"] or report["retrieval_violations"]["outcome_not_yet_known"]:
            findings.append("RAG_RETRIEVED_FUTURE")
            invalid = True
    for trade in trades:
        if abs(trade["identity_residual_price"]) > 1e-8:
            findings.append("COST_IDENTITY_BROKEN")
            invalid = True
            break
        if trade["year"] in holdout_years:
            findings.append("HOLDOUT_WAS_SCORED")
            invalid = True
            break
    ready = [index for index in decisions if row_ready(features, index)]
    sample = [index for index in ready if index < 2200][:3]
    causality = feature_causality_errors(
        table, features, sample, settings.vol_percentile_window, settings.slow_vol_bars
    )
    if causality:
        findings.extend(causality)
        invalid = True
    if ready:
        probe = ready[0]
        window = table.prefix(min(len(table), probe + 40))
        perturbed = future_perturbation_errors(
            window, features, probe, settings.vol_percentile_window, settings.slow_vol_bars
        )
        if perturbed:
            findings.extend(perturbed)
            invalid = True
    if meta.get("synthetic") and meta.get("event_join_allowed"):
        findings.append("SYNTHETIC_EVENT_JOIN")
        invalid = True
    findings.append("IN_SAMPLE_NOT_COMPUTED")
    findings.append("HOLDOUT_NOT_EXAMINED")
    return {
        "valid": not invalid,
        "invalid": invalid,
        "findings": findings,
        "causality_indices_checked": sample,
        "future_information_in_events": False,
    }


def _last_point(table, features, predictions, quality, meta, settings, audit_ok, cost) -> dict | None:
    expanding = [row for row in predictions if row["procedure"] == "expanding"]
    if not expanding:
        return None
    last = expanding[-1]
    index = last["bar_index"]
    regime = classify_regime(features, index)
    # Rebuild the retrieval distribution numbers already summarized on the row.
    # The full analogue list is not retained, so the package reports the verified counts
    # and the model estimates stored on the prediction row.
    retrieval = {
        "matches": last["neighbors"],
        "historical_cutoff_verified": last["cutoff_verified"],
        "distribution": {
            "median_forward_outcome": last.get("median_forward_outcome"),
            "positive_outcome_probability": last.get("positive_outcome_probability"),
            "negative_outcome_probability": last.get("negative_outcome_probability"),
            "labels": {
                "LONG_SUCCESS": last["knn_long"],
                "SHORT_SUCCESS": last["knn_short"],
                "NO_RESOLUTION": last["knn_none"],
                "AMBIGUOUS_PATH": None,
            },
        },
    }
    if meta.get("synthetic") or not meta.get("event_join_allowed", False):
        events = {
            "active_event": None,
            "historical_analogues": [],
            "future_information_detected": False,
            "unsafe_used": False,
            "note": "Event catalog was not joined. Synthetic clocks and unverified joins stay excluded.",
        }
    else:
        events = retrieve_events(last["decision_time"], catalog())
    logit = {
        "LONG_SUCCESS": last["logit_long"],
        "SHORT_SUCCESS": last["logit_short"],
        "NO_RESOLUTION": last["logit_none"],
        "AMBIGUOUS_PATH": max(0.0, 1.0 - last["logit_long"] - last["logit_short"] - last["logit_none"]),
    }
    agreed = last["agreed_side"] or None
    package = build_point_in_time(
        table.instrument,
        table.granularity,
        last["decision_time"],
        {
            "synthetic": meta.get("synthetic"),
            "financing_measured": settings.financing_is_measured,
        },
        regime,
        retrieval,
        events,
        logit,
        quality,
        {"slippage_measured": cost.slippage_measured},
        audit_ok,
        bool(agreed),
        agreed,
    )
    package["analogue_detail_retained"] = False
    package["analogue_detail_note"] = (
        "The package reports the outcome distribution of the retrieved neighbours. "
        "Individual analogue timestamps are not copied into this summary row."
    )
    return package


def _verdict(expanding, rolling, robustness, audit, settings, meta) -> dict:
    reasons = []
    if audit["invalid"]:
        lab = "INVALIDATED"
        reasons.append("AUDIT_FAILED")
    else:
        lab = _process_verdict(expanding, rolling, robustness, settings, reasons)
    if meta.get("synthetic") or meta.get("not_market_data"):
        market = "INSUFFICIENT_DATA"
        reasons.append("SYNTHETIC_OR_NON_MARKET_DATA")
    elif audit["invalid"] or not settings.financing_is_measured:
        market = "INSUFFICIENT_DATA"
        reasons.append("MARKET_CLAIM_BLOCKED")
    else:
        market = "NO_EDGE" if lab in {"NO_EDGE", "FRAGILE", "INSUFFICIENT_DATA"} else "INSUFFICIENT_DATA"
        reasons.append("MARKET_CLASSIFICATION_REQUIRES_A_POINT_IN_TIME_PACKAGE")
    reasons.append("HOLDOUT_NOT_EXAMINED")
    reasons.append("LIVE_EXECUTION_NOT_AUTHORIZED")
    return {
        "laboratory_process_verdict": lab,
        "market_classification": market,
        "reason_codes": reasons,
        "survives_as_live_authorization": False,
        "research_only": True,
    }


def _process_verdict(expanding, rolling, robustness, settings, reasons) -> str:
    trades = expanding.get("trades") or 0
    expectancy = expanding.get("expectancy_pips")
    rolling_expectancy = rolling.get("expectancy_pips")
    sharpe = expanding.get("sharpe_type")
    profit_factor = expanding.get("profit_factor")
    if trades == 0:
        reasons.append("NO_TRADES_UNDER_FROZEN_RULE")
        return "NO_EDGE"
    if trades < settings.min_trades_for_pattern_claim:
        reasons.append("TRADE_COUNT_BELOW_PRE_REGISTERED_MINIMUM")
        if expectancy is not None and expectancy <= 0:
            return "NO_EDGE"
        return "INSUFFICIENT_DATA"
    if expectancy is None or expectancy <= 0 or sharpe is None or sharpe <= 0 or profit_factor is None or profit_factor <= 1:
        reasons.append("OOS_RISK_ADJUSTED_RESULT_NOT_POSITIVE")
        return "NO_EDGE"
    if rolling_expectancy is None or rolling_expectancy < 0:
        reasons.append("ROLLING_WINDOW_DISAGREES")
        return "FRAGILE"
    if robustness.get("sign_flips_without_best_five"):
        reasons.append("CONCENTRATED_IN_BEST_FIVE")
        return "FRAGILE"
    share = robustness.get("best_year_share_of_net")
    if share is not None and share > 0.70:
        reasons.append("CONCENTRATED_IN_ONE_YEAR")
        return "FRAGILE"
    stressed = robustness.get("slippage_2x_expectancy_pips")
    if stressed is None or stressed < 0:
        reasons.append("FAILS_SLIPPAGE_STRESS")
        return "FRAGILE"
    years = expanding.get("by_year") or {}
    positive_years = [year for year, bucket in years.items() if (bucket.get("net_pips") or 0) > 0]
    if len(positive_years) < 2:
        reasons.append("NOT_POSITIVE_IN_TWO_YEARS")
        return "FRAGILE"
    reasons.append("HISTORICAL_PATTERN_UNDER_FROZEN_RULE_HOLDOUT_STILL_LOCKED")
    return "HISTORICAL_OOS_PATTERN_HOLDOUT_PENDING"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_equity(path: Path, trades: list[dict]) -> None:
    fields = ["procedure", "exit_time_index", "decision_time", "net_pips", "cumulative_net_pips"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for procedure in ("expanding", "rolling"):
            running = 0.0
            ordered = [trade for trade in trades if trade["procedure"] == procedure]
            ordered.sort(key=lambda trade: trade["decision_time"])
            for trade in ordered:
                running += trade["net_pips"]
                writer.writerow(
                    {
                        "procedure": procedure,
                        "exit_time_index": trade["exit_index"],
                        "decision_time": trade["decision_time"],
                        "net_pips": f"{trade['net_pips']:.6f}",
                        "cumulative_net_pips": f"{running:.6f}",
                    }
                )
