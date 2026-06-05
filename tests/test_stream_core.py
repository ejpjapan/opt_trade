from __future__ import annotations

import math
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import pandas as pd
from zoneinfo import ZoneInfo

from option_stream import stream_core


class FakeTicker:
    def __init__(self, market_price=None, last=None, close=None):
        self._market_price = market_price
        self.last = last
        self.close = close

    def marketPrice(self):
        return self._market_price


class FakeIB:
    def __init__(self):
        self.calls = []

    def qualifyContracts(self, option):
        self.calls.append(option)
        return [SimpleNamespace(strike=option.strike)]


class FakeWrapper:
    def __init__(self):
        self.ib = FakeIB()


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 1, 1, 12, 0, tzinfo=tz)


class StreamCoreTests(unittest.TestCase):
    def test_best_price_prefers_market_then_fallbacks(self):
        self.assertEqual(stream_core._best_price(FakeTicker(market_price=101.5)), 101.5)
        self.assertEqual(stream_core._best_price(FakeTicker(market_price=0, last=99.25)), 99.25)
        self.assertEqual(stream_core._best_price(FakeTicker(market_price=float("nan"), close=98.75)), 98.75)
        self.assertIsNone(stream_core._best_price(None))

    def test_trend_color_and_tracker(self):
        self.assertEqual(stream_core._trend_color(None, 10), "black")
        self.assertEqual(stream_core._trend_color(10, 11), "green")
        self.assertEqual(stream_core._trend_color(11, 10), "red")
        self.assertEqual(stream_core._trend_color(10, 10, "blue"), "blue")

        tracker = stream_core.PriceTracker()
        tracker.update_price("SPX", 100)
        self.assertEqual(tracker.get_trend("SPX"), "black")
        tracker.update_price("SPX", 101)
        self.assertEqual(tracker.get_trend("SPX"), "green")
        tracker.update_price("SPX", 101)
        self.assertEqual(tracker.get_trend("SPX"), "green")
        tracker.update_price("SPX", 99)
        self.assertEqual(tracker.get_trend("SPX"), "red")

    def test_format_and_patch_helpers(self):
        self.assertEqual(
            stream_core._format_colored_value(12.345, "red", 1),
            "<span style='color:red'>12.3</span>",
        )
        self.assertEqual(
            stream_core._build_patch(pd.DataFrame({"A": [1, 2], "B": ["x", "y"]}), ["A", "B"]),
            {"A": [(0, 1), (1, 2)], "B": [(0, "x"), (1, "y")]},
        )

    def test_convert_and_select_expiries(self):
        ny = ZoneInfo("America/New_York")
        datestamps = stream_core.convert_to_datestamps([["20250101", "20250103"]])
        self.assertEqual(datestamps[0][0].strftime("%Y%m%d %H:%M"), "20250101 16:00")
        self.assertEqual(datestamps[0][0].tzinfo, ny)

        future_a = datetime(2030, 1, 3, 16, 0, tzinfo=ny)
        future_b = datetime(2029, 12, 31, 16, 0, tzinfo=ny)
        option_params_df = pd.DataFrame({"expirations_timestamps": [[future_a, future_b, future_a]]})
        selected = stream_core._select_option_expiries(option_params_df, 5)
        self.assertEqual(selected, [future_b, future_a])

    def test_get_theoretical_strike(self):
        ny = ZoneInfo("America/New_York")
        expiry = datetime(2026, 1, 31, 16, 0, tzinfo=ny)
        risk_free = pd.Series([5.0])

        with mock.patch.object(stream_core, "datetime", FixedDateTime):
            df = stream_core.get_theoretical_strike(
                [expiry],
                [4000.0],
                pd.Series([5.0], index=["20260131"]),
                [-1],
                0.013,
                [20.0],
            )

        self.assertEqual(df.index.tolist(), ["20260131"])
        self.assertEqual(df.at["20260131", "Days to Expiry"], 30)
        self.assertTrue(math.isfinite(float(df.at["20260131", "theoretical_strike"])))
        self.assertEqual(df.at["20260131", "spot_price"], 4000.0)

    def test_contract_cache_and_qualification(self):
        cache = stream_core.QualifiedContractsCache()
        wrapper = FakeWrapper()
        base_df = pd.DataFrame(
            {
                "expiry_date": ["20300117"],
                "theoretical_strike": [109.0],
            }
        )

        first = stream_core.qualify_all_contracts(wrapper, base_df.copy(), [100.0, 110.0, 120.0], cache)
        self.assertEqual(first.at[0, "closest_strike"], 110.0)
        self.assertEqual(first.at[0, "qualified_contracts"].strike, 110.0)
        self.assertEqual(len(wrapper.ib.calls), 1)

        second = stream_core.qualify_all_contracts(wrapper, base_df.copy(), [100.0, 110.0, 120.0], cache)
        self.assertEqual(second.at[0, "closest_strike"], 110.0)
        self.assertEqual(len(wrapper.ib.calls), 1)

    def test_recompute_and_build_display_df(self):
        raw = pd.DataFrame(
            {
                "expiry_date": ["20300102"],
                "option_life": [0.1],
                "trade_date": [datetime(2026, 1, 1, 12, 0, tzinfo=ZoneInfo("America/New_York"))],
                "sigma": [0.2],
                "dividend_yield": [0.013],
                "spot_price": [4000.0],
                "risk_free": [0.05],
                "time_discount": [0.1],
                "time_scale": [0.2],
                "strike_discount": [0.98],
                "theoretical_strike": [3900.0],
                "closest_strike": [3900.0],
                "bid": [12.34],
                "ask": [13.34],
                "implied_volatility": [0.2],
                "qualified_contracts": [None],
                "last_traded": [None],
                "volume": [None],
                "market": [None],
            }
        )

        raw["Bid"] = raw["bid"]
        raw["Ask"] = raw["ask"]
        stream_core._recompute_derived_fields(raw, base_capital=100000.0, leverage=1.0)
        self.assertEqual(raw.at[0, "Mid"], 12.84)
        self.assertTrue(raw.at[0, "Lots"] >= 0)
        self.assertAlmostEqual(raw.at[0, "Discount"], 3900.0 / 4000.0 - 1)

        display = stream_core._build_display_df(raw, leverage=1.0, base_capital=100000.0)
        self.assertEqual(display.at[0, "Expiry"], "02 Jan 2030")
        self.assertIn("BidDisplay", display.columns)
        self.assertTrue(display.at[0, "BidDisplay"].startswith("<span"))
        self.assertNotIn("qualified_contracts", display.columns)
        self.assertEqual(display.at[0, "Strike"], 3900.0)


if __name__ == "__main__":
    unittest.main()
