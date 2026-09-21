"""Unit tests for Paper Execution Model & Position Accounting (PRD v3.9)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from quantmind.backtest.engine import CostSchedule, CostSchedulePeriod
from quantmind.paper.engine import PaperReplayEngine
from quantmind.paper.feed import ReplayFeed
from quantmind.paper.models import PaperOrderStatus
from quantmind.research_integrity.qualification import (
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationStatus,
)
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.spec import StrategySpec


def _make_sample_record(spec: StrategySpec, dataset_version: str = "DS-TEST") -> StrategyQualificationRecord:
    norm_spec = normalize_strategy_spec(spec)
    strat_id = derive_strategy_id(norm_spec)
    import hashlib
    spec_hash = hashlib.sha256(norm_spec.canonical_json().encode()).hexdigest()

    return StrategyQualificationRecord.create(
        qualification_id="QUAL-EXEC-1",
        strategy_id=strat_id,
        strategy_spec_hash=spec_hash,
        dataset_version=dataset_version,
        dataset_sha256="dsha",
        split_manifest_version="m1",
        research_protocol_version="RP-2",
        population_hash="pophash",
        effective_trial_count=10.0,
        observed_sharpe=1.8,
        dsr=0.96,
        trade_count=120,
        holdout_state="PASSED",
        robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE,
    )


class TestPaperExecution:
    def test_next_bar_open_fill_timing(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        record = _make_sample_record(spec)

        timestamps = pd.date_range("2023-01-01 09:15", periods=10, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0],
                "high": [101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0, 119.0],
                "low": [99.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0],
                "close": [100.5, 102.5, 104.5, 106.5, 108.5, 110.5, 112.5, 114.5, 116.5, 118.5],
            }
        )
        feed = ReplayFeed(df, dataset_version="DS-TEST", lot_size=50, tick_size=0.05)
        engine = PaperReplayEngine(allow_fixture_feed=True)

        report = engine.run_replay(record, spec, feed)
        orders = engine.ledger.get_orders()
        fills = engine.ledger.get_fills()

        assert len(orders) > 0
        assert len(fills) > 0
        first_order = orders[0]
        first_fill = fills[0]
        assert first_order["status"] == PaperOrderStatus.FILLED.value
        # Fill timestamp must be later than signal timestamp
        assert pd.Timestamp(first_fill["fill_timestamp"]) > pd.Timestamp(first_order["signal_timestamp"])

    def test_multi_bar_holding(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 20, "session_window": [0.0, 1.0]})
        record = _make_sample_record(spec)

        timestamps = pd.date_range("2023-01-01 09:15", periods=20, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": 100.0 + np.arange(20),
                "high": 101.0 + np.arange(20),
                "low": 99.0 + np.arange(20),
                "close": 100.5 + np.arange(20),
            }
        )
        feed = ReplayFeed(df, dataset_version="DS-TEST")
        engine = PaperReplayEngine(allow_fixture_feed=True)

        # Hold for 3 bars
        report = engine.run_replay(record, spec, feed, hold_bars=3)
        assert report.trade_count >= 1

    def test_slippage_and_tick_rounding(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 5, "session_window": [0.0, 1.0]})
        record = _make_sample_record(spec)

        timestamps = pd.date_range("2023-01-01 09:15", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 101.0, 101.0, 101.0],
                "low": [99.0, 99.0, 99.0, 99.0, 99.0],
                "close": [100.5, 100.5, 100.5, 100.5, 100.5],
            }
        )
        feed = ReplayFeed(df, dataset_version="DS-TEST", tick_size=0.05, lot_size=50)
        # 10 bps slippage
        engine = PaperReplayEngine(slippage_bps_per_side=10.0, allow_fixture_feed=True)

        report = engine.run_replay(record, spec, feed)
        fills = engine.ledger.get_fills()
        assert len(fills) > 0
        for f in fills:
            price = f["fill_price"]
            # Price must align with 0.05 tick increment
            assert round(price % 0.05, 4) in (0.0, 0.05)

    def test_date_effective_cost_schedule(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 6, "session_window": [0.0, 1.0]})
        record = _make_sample_record(spec)

        timestamps = pd.date_range("2023-01-01 09:15", periods=6, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
                "low": [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
                "close": [100.5, 100.5, 100.5, 100.5, 100.5, 100.5],
            }
        )
        feed = ReplayFeed(df, dataset_version="DS-TEST")
        cost_sched = CostSchedule(
            schedule_id="SCHED-1",
            periods=(
                CostSchedulePeriod(effective_from=date(2023, 1, 1), effective_to=date(2023, 12, 31), round_trip_bps=2.0),
            ),
        )
        engine = PaperReplayEngine(cost_schedule=cost_sched, allow_fixture_feed=True)
        report = engine.run_replay(record, spec, feed)

        fills = engine.ledger.get_fills()
        assert len(fills) > 0
        for f in fills:
            assert f["cost"] > 0.0
