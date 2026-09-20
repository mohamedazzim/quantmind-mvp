"""Unit tests for Market Data Replay Feed (PRD v3.9)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from quantmind.data.splits import SplitZone
from quantmind.paper.feed import MarketFeedSecurityError, ReplayBar, ReplayFeed


def _make_dummy_market_data(num_bars: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01 09:15", periods=num_bars, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0 + np.arange(num_bars) * 0.1,
            "high": 100.5 + np.arange(num_bars) * 0.1,
            "low": 99.5 + np.arange(num_bars) * 0.1,
            "close": 100.2 + np.arange(num_bars) * 0.1,
            "volume": 1000.0,
            "open_interest": 500.0,
        }
    )


class TestReplayFeed:
    def test_chronological_streaming(self) -> None:
        df = _make_dummy_market_data(50)
        feed = ReplayFeed(df, symbol="NIFTY_FUT", lot_size=50, tick_size=0.05)

        bars = list(feed.stream_bars())
        assert len(bars) == 50
        for i in range(len(bars) - 1):
            assert bars[i].timestamp < bars[i + 1].timestamp
            assert bars[i].bar_index == i

    def test_session_boundaries_tracking(self) -> None:
        ts1 = pd.date_range("2023-01-01 09:15", periods=5, freq="1min")
        ts2 = pd.date_range("2023-01-02 09:15", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": list(ts1) + list(ts2),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
        )
        feed = ReplayFeed(df)
        bars = list(feed.stream_bars())
        assert len(bars) == 10
        assert bars[0].session_id == "2023-01-01"
        assert bars[5].session_id == "2023-01-02"

    def test_pause_and_resume_capability(self) -> None:
        df = _make_dummy_market_data(20)
        feed = ReplayFeed(df)

        it = feed.stream_bars()
        first_bar = next(it)
        assert first_bar.bar_index == 0

        feed.pause()
        assert feed.is_paused is True
        # Trying to stream while paused yields nothing
        assert list(feed.stream_bars()) == []

        feed.resume()
        assert feed.is_paused is False
        remaining_bars = list(feed.stream_bars())
        assert len(remaining_bars) == 19
        assert remaining_bars[0].bar_index == 1

    def test_reset_capability(self) -> None:
        df = _make_dummy_market_data(10)
        feed = ReplayFeed(df)

        bars1 = list(feed.stream_bars())
        assert len(bars1) == 10

        feed.reset()
        bars2 = list(feed.stream_bars())
        assert len(bars2) == 10
        assert bars1 == bars2

    def test_prohibit_final_holdout_access(self) -> None:
        df = _make_dummy_market_data(10)
        with pytest.raises(MarketFeedSecurityError, match="prohibited from accessing the sealed FINAL_HOLDOUT"):
            ReplayFeed(df, split_zone=SplitZone.FINAL_HOLDOUT)
