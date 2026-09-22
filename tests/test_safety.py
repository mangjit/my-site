import unittest
from datetime import datetime, timezone

from aegis.oanda import OandaHistoricalClient, _parse_candle
from aegis.paper import append_observation
from aegis.safety import LIVE_HOST, PRACTICE_HOST, LiveExecutionForbidden, assert_read_only_request


class SafetyTests(unittest.TestCase):
    def test_live_host_is_refused(self):
        with self.assertRaises(LiveExecutionForbidden):
            OandaHistoricalClient(token="not-a-real-token", host=LIVE_HOST)

    def test_missing_token_is_refused(self):
        with self.assertRaises(LiveExecutionForbidden):
            OandaHistoricalClient(token="", host=PRACTICE_HOST)

    def test_repr_redacts_token(self):
        client = OandaHistoricalClient(token="super-secret-token", host=PRACTICE_HOST)
        self.assertNotIn("super-secret-token", repr(client))
        self.assertIn("REDACTED", repr(client))

    def test_order_paths_are_refused(self):
        for path in (
            "/v3/accounts/001-001-001-1/orders",
            "/v3/accounts/1/trades/2/close",
            "/v3/accounts/1/positions/EUR_USD/close",
        ):
            with self.assertRaises(LiveExecutionForbidden):
                assert_read_only_request(f"https://{PRACTICE_HOST}{path}")

    def test_candle_path_shape_is_allowed(self):
        assert_read_only_request(
            f"https://{PRACTICE_HOST}/v3/instruments/EUR_USD/candles?price=BA&granularity=H1"
        )

    def test_parser_drops_incomplete_and_missing_side(self):
        self.assertIsNone(_parse_candle({"complete": False, "time": "2020-01-01T00:00:00Z", "bid": {}, "ask": {}}))
        self.assertIsNone(
            _parse_candle(
                {
                    "complete": True,
                    "time": "2020-01-01T00:00:00.000000000Z",
                    "volume": 5,
                    "bid": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.1"},
                }
            )
        )
        parsed = _parse_candle(
            {
                "complete": True,
                "time": "2020-01-01T00:00:00.000000000Z",
                "volume": 5,
                "bid": {"o": "1.10000", "h": "1.20000", "l": "1.00000", "c": "1.10000"},
                "ask": {"o": "1.10010", "h": "1.20010", "l": "1.00010", "c": "1.10010"},
            }
        )
        self.assertIsNotNone(parsed)
        stamp, row = parsed
        self.assertEqual(stamp, datetime(2020, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(row["ask_c"], 1.10010)

    def test_paper_log_refuses_orders(self):
        with self.assertRaises(LiveExecutionForbidden):
            append_observation(
                self.id() and __import__("pathlib").Path("/tmp/should-not-matter.jsonl"),
                {"live_order": True},
            )


if __name__ == "__main__":
    unittest.main()
