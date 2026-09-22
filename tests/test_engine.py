import random
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from aegis.align import asof_indices
from aegis.config import laboratory_settings
from aegis.costs import CostModel
from aegis.events import catalog, retrieve_events
from aegis.experiment import run_holdout_once, run_research
from aegis.features import compute_features
from aegis.memory import retrieve
from aegis.models import decide_side
from aegis.quality import assess_quality
from aegis.safety import DataIntegrityError, HoldoutLockError
from aegis.store import load_dataset, write_dataset
from aegis.table import CandleTable
from aegis.targets import path_label, simulate_side
from aegis.timeutil import bar_available_time, session_flags
from aegis.walkforward import build_folds, purge_training_indices


def _cost() -> CostModel:
    return CostModel(
        slippage_price_per_side=0.0,
        commission_price_round_turn=0.0,
        financing_price_per_hour=0.0,
        pip_size=0.0001,
        spread_measured_from_quotes=True,
        slippage_measured=False,
        financing_measured=False,
        commission_measured=False,
        financing_direction="adverse_only_assumption",
    )


def _flat_table(n=6) -> CandleTable:
    start = datetime(2020, 1, 6, tzinfo=timezone.utc)
    table = CandleTable(
        instrument="EUR_USD",
        granularity="H1",
        timestamp=[start + timedelta(hours=i) for i in range(n)],
        bid_o=[1.10] * n,
        bid_h=[1.1004] * n,
        bid_l=[1.0996] * n,
        bid_c=[1.10] * n,
        ask_o=[1.1002] * n,
        ask_h=[1.1006] * n,
        ask_l=[1.0998] * n,
        ask_c=[1.1002] * n,
        volume=[1.0] * n,
        complete=[True] * n,
    )
    return table


class TargetTests(unittest.TestCase):
    def test_long_target_before_stop(self):
        table = _flat_table()
        table.bid_h[2] = 1.1020
        table.ask_h[2] = 1.1022
        trade = simulate_side(table, 0, "LONG", 0.001, 1.0, 1.0, 3, _cost())
        self.assertEqual(trade["execution_label"], "LONG_SUCCESS")
        self.assertAlmostEqual(trade["identity_residual_price"], 0.0, places=12)
        self.assertGreater(trade["net_pips"], 0)

    def test_same_bar_conflict_is_not_a_success(self):
        table = _flat_table()
        table.bid_h[1] = 1.1030
        table.bid_l[1] = 1.0970
        trade = simulate_side(table, 0, "LONG", 0.001, 1.0, 1.0, 3, _cost())
        self.assertEqual(trade["execution_label"], "AMBIGUOUS_PATH")
        self.assertTrue(trade["ambiguous"])
        self.assertLess(trade["net_pips"], 0)

    def test_mid_label_does_not_use_a_later_bar_as_a_feature(self):
        table = _flat_table(8)
        label = path_label(table, 1, 0.001, 1.0, 1.0, 3)
        self.assertEqual(label["label"], "NO_RESOLUTION")
        self.assertEqual(label["label_end_index"], 4)


class WalkForwardTests(unittest.TestCase):
    def test_holdout_is_not_a_test_year(self):
        folds, holdout = build_folds([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018], 5, 1)
        self.assertEqual(holdout, [2018])
        self.assertTrue(all(fold.test_year != 2018 for fold in folds))
        self.assertTrue(all(2018 not in fold.train_years for fold in folds))
        expanding = [fold for fold in folds if fold.procedure == "expanding"]
        self.assertEqual(expanding[0].train_years, (2010, 2011, 2012, 2013, 2014))
        self.assertEqual(expanding[0].test_year, 2015)
        rolling = [fold for fold in folds if fold.procedure == "rolling" and fold.test_year == 2017]
        self.assertEqual(rolling[0].train_years, (2012, 2013, 2014, 2015, 2016))

    def test_purge_drops_labels_that_touch_the_test_window(self):
        kept, embargoed = purge_training_indices([1, 2, 3], {1: 4, 2: 10, 3: 11}, test_start_index=10)
        self.assertEqual(kept, [1])
        self.assertEqual(embargoed, [2, 3])


class MemoryAndEventTests(unittest.TestCase):
    def test_future_neighbor_is_not_returned(self):
        rows = [[0.0, 0.0], [1.0, 1.0]]
        meta = [
            {
                "decision_time": "2020-01-02T00:00:00Z",
                "label_available_time": "2020-01-03T00:00:00Z",
                "label": "LONG_SUCCESS",
                "forward_mid_return": 0.01,
            },
            {
                "decision_time": "2020-01-01T00:00:00Z",
                "label_available_time": "2020-01-01T12:00:00Z",
                "label": "SHORT_SUCCESS",
                "forward_mid_return": -0.01,
            },
        ]
        result = retrieve(rows, meta, [0.1, 0.1], "2020-01-01T18:00:00Z", k=5)
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["principal"][0]["decision_time"], "2020-01-01T00:00:00Z")
        self.assertGreater(result["violations"]["future_neighbor"], 0)
        self.assertFalse(result["historical_cutoff_verified"])

    def test_outcome_not_yet_known_is_excluded(self):
        rows = [[0.0]]
        meta = [
            {
                "decision_time": "2020-01-01T00:00:00Z",
                "label_available_time": "2020-01-05T00:00:00Z",
                "label": "LONG_SUCCESS",
                "forward_mid_return": 0.02,
            }
        ]
        result = retrieve(rows, meta, [0.0], "2020-01-02T00:00:00Z", k=3)
        self.assertEqual(result["matches"], 0)
        self.assertEqual(result["violations"]["outcome_not_yet_known"], 1)

    def test_unsafe_events_are_excluded(self):
        found = retrieve_events("2021-01-01T00:00:00Z", catalog())
        self.assertEqual(found["historical_analogues"], [])
        self.assertTrue(found["excluded"])
        self.assertFalse(found["unsafe_used"])

    def test_known_event_is_visible_only_after_public_time(self):
        record = {
            "event_id": "fixture",
            "category": "CENTRAL_BANK_SURPRISE",
            "title": "Fixture",
            "public_information_available_time": "2020-03-11T15:00:00Z",
            "unsafe_for_backtest": False,
            "knowable_summary": "A fixture with an explicit public time.",
            "time_precision": "utc_minute",
        }
        before = retrieve_events("2020-03-11T14:59:00Z", [record])
        after = retrieve_events("2020-03-11T15:00:00Z", [record])
        self.assertEqual(before["historical_analogues"], [])
        self.assertEqual(after["historical_analogues"][0]["event_id"], "fixture")
        self.assertTrue(before["future_information_detected"])


class AlignmentTests(unittest.TestCase):
    def test_asof_never_returns_a_future_print(self):
        decisions = [
            datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
            datetime(2020, 1, 2, 21, tzinfo=timezone.utc),
        ]
        available = [
            datetime(2020, 1, 1, 21, tzinfo=timezone.utc),
            datetime(2020, 1, 2, 21, tzinfo=timezone.utc),
        ]
        self.assertEqual(asof_indices(decisions, available), [None, 1])

    def test_daily_bar_is_not_known_at_its_open(self):
        opened = datetime(2020, 1, 2, 17, tzinfo=timezone.utc)
        known = bar_available_time(opened, "D")
        intraday = datetime(2020, 1, 3, 12, tzinfo=timezone.utc)
        self.assertGreater(known, intraday)
        self.assertIsNone(asof_indices([intraday], [known])[0])


class FeatureTests(unittest.TestCase):
    def test_future_does_not_change_the_past(self):
        table = _random_walk(400, "H1")
        features = compute_features(table, vol_window=40, slow_vol_bars=30)
        index = 200
        prefix = compute_features(table.prefix(index + 1), vol_window=40, slow_vol_bars=30)
        for name in ("ret_1_z", "vol_pct", "ma_dist_50_atr", "spread_pct"):
            if features[name][index] is None:
                continue
            self.assertAlmostEqual(features[name][index], prefix[name][index], places=9)
        mutated = table.copy()
        mutated.bid_c[-1] += 0.05
        mutated.ask_c[-1] += 0.05
        again = compute_features(mutated, vol_window=40, slow_vol_bars=30)
        self.assertAlmostEqual(features["ma_dist_20_atr"][index], again["ma_dist_20_atr"][index], places=9)

    def test_overlap_uses_local_clocks(self):
        winter = datetime(2020, 1, 15, 13, tzinfo=timezone.utc)
        flags = session_flags(winter, "H1")
        self.assertEqual(flags["overlap_lon_ny"], 1.0)


class StoreTests(unittest.TestCase):
    def test_dataset_is_immutable_and_hashed(self):
        table = _flat_table()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, "SAMPLE_V1", table, {"source": "SYNTHETIC", "synthetic": True, "not_market_data": True})
            with self.assertRaises(DataIntegrityError):
                write_dataset(root, "SAMPLE_V1", table, {"source": "SYNTHETIC"})
            loaded, manifest = load_dataset(root, "SAMPLE_V1")
            self.assertEqual(len(loaded), len(table))
            manifest_path = root / "SAMPLE_V1" / "manifest.json"
            text = manifest_path.read_text().replace(manifest["sha256"], "0" * 64)
            manifest_path.write_text(text)
            with self.assertRaises(DataIntegrityError):
                load_dataset(root, "SAMPLE_V1")


class DecisionRuleTests(unittest.TestCase):
    def test_disagreement_is_no_trade(self):
        logit = {"LONG_SUCCESS": 0.7, "SHORT_SUCCESS": 0.1, "NO_RESOLUTION": 0.1, "AMBIGUOUS_PATH": 0.1}
        knn = {"LONG_SUCCESS": 0.2, "SHORT_SUCCESS": 0.6, "NO_RESOLUTION": 0.1, "AMBIGUOUS_PATH": 0.1}
        base = {"LONG_SUCCESS": 0.3, "SHORT_SUCCESS": 0.3}
        self.assertIsNone(
            decide_side(logit, knn, base, 0.4, 0.05, 0.05, neighbor_count=25, min_neighbors=20)
        )

    def test_agreement_can_select_a_side(self):
        logit = {"LONG_SUCCESS": 0.62, "SHORT_SUCCESS": 0.1, "NO_RESOLUTION": 0.2, "AMBIGUOUS_PATH": 0.08}
        knn = {"LONG_SUCCESS": 0.58, "SHORT_SUCCESS": 0.12, "NO_RESOLUTION": 0.2, "AMBIGUOUS_PATH": 0.1}
        base = {"LONG_SUCCESS": 0.3, "SHORT_SUCCESS": 0.3}
        self.assertEqual(
            decide_side(logit, knn, base, 0.4, 0.05, 0.05, neighbor_count=25, min_neighbors=20),
            "LONG",
        )


def _random_walk(n: int, granularity: str, start: datetime | None = None, seed: int = 7) -> CandleTable:
    rng = random.Random(seed)
    start = start or datetime(2014, 1, 6, tzinfo=timezone.utc)
    step = timedelta(hours=1) if granularity == "H1" else timedelta(days=1)
    timestamps = []
    cursor = start
    while len(timestamps) < n:
        if granularity == "D" or cursor.weekday() < 5:
            timestamps.append(cursor)
        cursor += step
    mid = 1.2
    table = CandleTable(
        instrument="EUR_USD",
        granularity=granularity,
        timestamp=timestamps,
        bid_o=[],
        bid_h=[],
        bid_l=[],
        bid_c=[],
        ask_o=[],
        ask_h=[],
        ask_l=[],
        ask_c=[],
        volume=[1.0] * n,
        complete=[True] * n,
    )
    for _ in range(n):
        drift = rng.gauss(0, 0.001 if granularity == "D" else 0.0002)
        close = max(0.8, mid * (1 + drift))
        high = max(mid, close) * (1 + abs(rng.gauss(0, 0.0003)))
        low = min(mid, close) * (1 - abs(rng.gauss(0, 0.0003)))
        spread = 0.00012
        half = spread / 2
        table.bid_o.append(mid - half)
        table.bid_h.append(high - half)
        table.bid_l.append(low - half)
        table.bid_c.append(close - half)
        table.ask_o.append(mid + half)
        table.ask_h.append(high + half)
        table.ask_l.append(low + half)
        table.ask_c.append(close + half)
        mid = close
    return table


class EndToEndTests(unittest.TestCase):
    def test_small_research_run_does_not_score_holdout_or_authorize_trading(self):
        table = _random_walk(2400, "D", start=datetime(2010, 1, 4, tzinfo=timezone.utc), seed=11)
        settings = replace(
            laboratory_settings(),
            granularity="D",
            horizon_bars=5,
            stride_bars=5,
            min_train_years=2,
            holdout_years=1,
            vol_percentile_window=40,
            slow_vol_bars=20,
            knn_k=8,
            min_neighbors=5,
            logit_steps=40,
            logit_lr=0.2,
        )
        meta = {
            "dataset_id": "TEST_SYNTHETIC",
            "source": "SYNTHETIC",
            "synthetic": True,
            "not_market_data": True,
            "event_join_allowed": False,
            "sha256": "test",
            "notes": "unit test process",
        }
        result = run_research(table, meta, settings)
        self.assertTrue(result["audit"]["valid"], result["audit"]["findings"])
        self.assertFalse(result["holdout"]["examined"])
        self.assertFalse(result["in_sample_performance_computed"])
        self.assertEqual(result["verdict"]["market_classification"], "INSUFFICIENT_DATA")
        self.assertGreater(result["holdout"]["bars_excluded"], 100)
        self.assertNotIn(result["holdout"]["years"][0], [fold["test_year"] for fold in result["folds"]])
        self.assertFalse(result["verdict"]["survives_as_live_authorization"])
        self.assertTrue(result["research_only"])
        for trade in result["trades"]:
            self.assertNotIn(trade["year"], result["holdout"]["years"])
            self.assertAlmostEqual(trade["identity_residual_price"], 0.0, places=9)
        for fold in result["folds"]:
            self.assertNotIn(fold["test_year"], fold["train_years"])
            self.assertEqual(fold["retrieval_violations"]["future_neighbor"], 0)
            self.assertEqual(fold["retrieval_violations"]["outcome_not_yet_known"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp)
            once = run_holdout_once(table, meta, settings, lock_dir)
            self.assertTrue(once["examined_once"])
            self.assertFalse(once["live_execution_authorized"])
            with self.assertRaises(HoldoutLockError):
                run_holdout_once(table, meta, settings, lock_dir)


class QualityTests(unittest.TestCase):
    def test_crossed_market_is_unsafe(self):
        table = _flat_table()
        table.bid_c[2] = table.ask_c[2] + 0.01
        report = assess_quality(table)
        self.assertTrue(report["unsafe_for_research"])
        self.assertGreater(report["crossed_markets"], 0)


if __name__ == "__main__":
    unittest.main()
