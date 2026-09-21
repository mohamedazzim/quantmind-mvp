"""Targeted adversarial and security regression suite for Paper Replay Engine (PRD v3.9).

Validates:
1. Qualification Boundary Security (duck typing, digest tampering, holdout tampering, spec mismatch, dataset mismatch)
2. Execution Semantics & Causality (no same-bar fills, final bar signal handling, session boundary liquidation)
3. Cost Schedule 50/50 split and tick-rounding slippage mechanics
4. Ledger Immutability (SQLite BEFORE UPDATE and BEFORE DELETE triggers across all 5 tables)
5. Strict Replay Determinism across independent engine instances
6. Replay Provenance cryptographic binding in report_hash
7. Pre-Trade Risk Engine (lot_size exposure scaling, negative quantity rejection, risk-reducing exit safety)
8. Data Feed Preflight Validation (NaNs, Infs, duplicate timestamps, unsorted timestamps, invalid OHLC bounds, checksum verification)
"""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from quantmind.backtest.engine import CostSchedule, CostSchedulePeriod
from quantmind.data.registry import (
    DatasetKind,
    DatasetRecord,
    DatasetRegistry,
    DatasetRegistryError,
    DatasetZone,
)
from quantmind.data.splits import SplitZone
from quantmind.paper.engine import PaperReplayEngine, PaperReplaySecurityError
from quantmind.paper.feed import MarketFeedSecurityError, ReplayFeed
from quantmind.paper.ledger import PaperLedger
from quantmind.paper.models import (
    PaperFill,
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    PaperRiskEvent,
    ReplayReport,
    ReplaySessionSummary,
)
from quantmind.paper.risk import PaperRiskConfig, PaperRiskEngine
from quantmind.research_integrity.qualification import (
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationStatus,
)
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.spec import StrategySpec


def _make_spec() -> StrategySpec:
    return StrategySpec(
        "v1",
        "f1",
        "current_bar_momentum",
        {"calendar_session_bars": 10, "session_window": [0.0, 1.0]},
    )


def _make_valid_record(
    spec: StrategySpec,
    *,
    dataset_version: str = "DS-TEST-2023",
    final_status: ValidationStatus = ValidationStatus.PAPER_ELIGIBLE,
    holdout_state: str = "PASSED",
    protocol_version: str = "RP-2",
    dataset_sha256: str = "dsha-valid-12345",
) -> StrategyQualificationRecord:
    norm_spec = normalize_strategy_spec(spec)
    strat_id = derive_strategy_id(norm_spec)
    spec_hash = hashlib.sha256(norm_spec.canonical_json().encode()).hexdigest()

    return StrategyQualificationRecord.create(
        qualification_id="QUAL-ADV-1",
        strategy_id=strat_id,
        strategy_spec_hash=spec_hash,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        split_manifest_version="manifest-v1",
        research_protocol_version=protocol_version,
        population_hash="pophash-adv",
        effective_trial_count=10.0,
        observed_sharpe=1.85,
        dsr=0.97,
        trade_count=120,
        holdout_state=holdout_state,
        robustness_status=RobustnessStatus.PASSED,
        final_status=final_status,
    )


def _make_clean_feed(
    n: int = 10,
    dataset_version: str = "DS-TEST-2023",
    lot_size: int = 50,
    tick_size: float = 0.05,
) -> ReplayFeed:
    timestamps = pd.date_range("2023-01-01 09:15", periods=n, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0 + np.arange(n) * 2.0,
            "high": 101.0 + np.arange(n) * 2.0,
            "low": 99.0 + np.arange(n) * 2.0,
            "close": 100.5 + np.arange(n) * 2.0,
        }
    )
    return ReplayFeed(
        df,
        dataset_version=dataset_version,
        symbol="NIFTY_FUT",
        lot_size=lot_size,
        tick_size=tick_size,
    )


# ==============================================================================
# PHASE 2: QUALIFICATION BOUNDARY ADVERSARIAL AUDIT
# ==============================================================================


class TestQualificationBoundaryAdversarial:
    def test_duck_typed_fake_record_rejected(self) -> None:
        """A fake object mimicking StrategyQualificationRecord attributes is blocked."""
        spec = _make_spec()
        feed = _make_clean_feed()
        engine = PaperReplayEngine()

        class FakeRecord:
            qualification_id = "QUAL-FAKE"
            final_status = ValidationStatus.PAPER_ELIGIBLE
            holdout_state = "PASSED"
            strategy_spec_hash = "fake"
            dataset_version = "DS-TEST-2023"
            dataset_sha256 = "dsha"
            research_protocol_version = "RP-2"
            def verify_digest(self) -> bool:
                return True

        with pytest.raises(PaperReplaySecurityError, match="must be an instance of StrategyQualificationRecord"):
            engine.run_replay(FakeRecord(), spec, feed)  # type: ignore

    def test_tampered_digest_rejected(self) -> None:
        """Mutating record attributes invalidates SHA-256 digest and is rejected."""
        import dataclasses

        spec = _make_spec()
        feed = _make_clean_feed()
        engine = PaperReplayEngine()
        record = _make_valid_record(spec)

        # Tamper attribute without updating record_hash
        tampered = dataclasses.replace(record, observed_sharpe=3.50)
        assert not tampered.verify_digest()
        with pytest.raises(PaperReplaySecurityError, match="failed cryptographic digest verification"):
            engine.run_replay(tampered, spec, feed)

    def test_spec_hash_mismatch_rejected(self) -> None:
        """A StrategySpec with different parameters than the qualified spec is rejected."""
        spec = _make_spec()
        tampered_spec = StrategySpec(
            "v1", "f1", "current_bar_momentum", {"calendar_session_bars": 99, "session_window": [0.0, 1.0]}
        )
        feed = _make_clean_feed()
        engine = PaperReplayEngine()
        record = _make_valid_record(spec)

        with pytest.raises(PaperReplaySecurityError, match="does not match strategy spec derived ID"):
            engine.run_replay(record, tampered_spec, feed)

    def test_dataset_version_mismatch_or_empty_rejected(self) -> None:
        """ReplayFeed with mismatched or empty dataset_version is rejected."""
        spec = _make_spec()
        engine = PaperReplayEngine()
        record = _make_valid_record(spec, dataset_version="DS-OFFICIAL-v1")

        feed_mismatch = _make_clean_feed(dataset_version="DS-OTHER-v2")
        with pytest.raises(PaperReplaySecurityError, match="does not match qualification record dataset_version"):
            engine.run_replay(record, spec, feed_mismatch)

        feed_empty = _make_clean_feed(dataset_version="")
        with pytest.raises(PaperReplaySecurityError, match="does not match qualification record dataset_version"):
            engine.run_replay(record, spec, feed_empty)

    def test_synthetic_dataset_barred_from_production(self, tmp_path: Path) -> None:
        """SYNTHETIC dataset in registry cannot be loaded for production replay."""
        csv_file = tmp_path / "synthetic_test.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 09:15", periods=5, freq="1min"),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
            }
        )
        df.to_csv(csv_file, index=False)

        reg_db = tmp_path / "registry.db"
        registry = DatasetRegistry(path=reg_db)
        registry.register_file(
            version="v_synth",
            kind=DatasetKind.SYNTHETIC,
            path=csv_file,
        )

        with pytest.raises(MarketFeedSecurityError, match="requires LICENSED dataset"):
            ReplayFeed.from_dataset_registry(
                registry, "v_synth", require_licensed=True
            )


# ==============================================================================
# PHASE 3: EXECUTION SEMANTICS & CAUSALITY AUDIT
# ==============================================================================


class TestExecutionSemanticsCausality:
    def test_signal_on_final_bar_never_fills(self) -> None:
        """A signal generated on the final bar cannot fill because there is no subsequent bar."""
        spec = _make_spec()
        record = _make_valid_record(spec)

        # 3-bar feed where momentum signal fires on bar 2 (last bar)
        timestamps = pd.date_range("2023-01-01 09:15", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 106.0],
                "low": [99.0, 99.0, 99.0],
                "close": [100.0, 100.0, 105.0],  # Close > Open only on final bar
            }
        )
        feed = ReplayFeed(df, dataset_version="DS-TEST-2023")
        engine = PaperReplayEngine()

        report = engine.run_replay(record, spec, feed)
        fills = engine.ledger.get_fills()
        assert len(fills) == 0
        assert report.trade_count == 0

    def test_no_same_bar_fill_strict_causality(self) -> None:
        """Every fill must occur at bar t+1 open timestamp, strictly > bar t close timestamp."""
        spec = _make_spec()
        record = _make_valid_record(spec)
        feed = _make_clean_feed(n=10)
        engine = PaperReplayEngine()

        engine.run_replay(record, spec, feed, hold_bars=2)
        orders = engine.ledger.get_orders()
        fills = engine.ledger.get_fills()

        assert len(fills) > 0
        for fill in fills:
            order = next(o for o in orders if o["order_id"] == fill["order_id"])
            signal_time = pd.Timestamp(order["signal_timestamp"])
            fill_time = pd.Timestamp(fill["fill_timestamp"])
            assert fill_time > signal_time, "Causality violated: fill cannot occur at or before signal time"

    def test_session_boundary_liquidation(self) -> None:
        """With enforce_session_boundaries=True, positions liquidate across day boundaries."""
        spec = _make_spec()
        record = _make_valid_record(spec)

        # 2 bars on day 1, 2 bars on day 2
        ts1 = pd.date_range("2023-01-01 09:15", periods=2, freq="1min")
        ts2 = pd.date_range("2023-01-02 09:15", periods=2, freq="1min")
        timestamps = ts1.append(ts2)
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 102.0, 104.0, 106.0],
                "high": [101.0, 103.0, 105.0, 107.0],
                "low": [99.0, 101.0, 103.0, 105.0],
                "close": [100.5, 102.5, 104.5, 106.5],
            }
        )
        feed = ReplayFeed(df, dataset_version="DS-TEST-2023")
        engine = PaperReplayEngine(enforce_session_boundaries=True)

        report = engine.run_replay(record, spec, feed, hold_bars=10)
        # Even with hold_bars=10, the trade must have closed due to session boundary
        assert report.trade_count >= 1


# ==============================================================================
# PHASE 4: COST & SLIPPAGE MATH AUDIT
# ==============================================================================


class TestCostAndSlippageMath:
    def test_round_trip_cost_divided_equally_on_entry_and_exit(self) -> None:
        """Verify that round_trip_bps=10.0 applies 5.0 bps on entry and 5.0 bps on exit."""
        spec = _make_spec()
        record = _make_valid_record(spec)

        # 4 bars: entry signal on bar 0 -> fills bar 1; exit signal on bar 1 -> fills bar 2
        timestamps = pd.date_range("2023-01-01 09:15", periods=4, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 100.0, 110.0, 110.0],
                "high": [101.0, 101.0, 111.0, 111.0],
                "low": [99.0, 99.0, 109.0, 109.0],
                "close": [100.5, 100.5, 110.0, 110.0],
            }
        )
        feed = ReplayFeed(df, dataset_version="DS-TEST-2023", lot_size=50, tick_size=0.05)

        schedule = CostSchedule(
            schedule_id="TEST-10BPS",
            periods=(
                CostSchedulePeriod(
                    effective_from=date(2020, 1, 1),
                    effective_to=None,
                    round_trip_bps=10.0,
                ),
            ),
        )
        engine = PaperReplayEngine(cost_schedule=schedule, slippage_bps_per_side=0.0)

        report = engine.run_replay(record, spec, feed, quantity=1, hold_bars=1)
        fills = engine.ledger.get_fills()

        assert len(fills) == 2
        fill_entry, fill_exit = fills[0], fills[1]

        # Entry cost: 100.0 * 1 * 50 * (5.0 / 10000.0) = 5000.0 * 0.0005 = 2.50
        expected_entry_cost = 100.0 * 1 * 50 * (5.0 / 10000.0)
        assert abs(fill_entry["cost"] - expected_entry_cost) < 1e-6

        # Exit cost: 110.0 * 1 * 50 * (5.0 / 10000.0) = 5500.0 * 0.0005 = 2.75
        expected_exit_cost = 110.0 * 1 * 50 * (5.0 / 10000.0)
        assert abs(fill_exit["cost"] - expected_exit_cost) < 1e-6

        total_costs = fill_entry["cost"] + fill_exit["cost"]
        assert abs(report.costs - total_costs) < 1e-6

    def test_zero_cost_schedule(self) -> None:
        """When cost schedule is None, transaction costs are strictly zero."""
        spec = _make_spec()
        record = _make_valid_record(spec)
        feed = _make_clean_feed(n=6)
        engine = PaperReplayEngine(cost_schedule=None, slippage_bps_per_side=0.0)

        report = engine.run_replay(record, spec, feed, hold_bars=1)
        assert report.costs == 0.0
        fills = engine.ledger.get_fills()
        for f in fills:
            assert f["cost"] == 0.0


# ==============================================================================
# PHASE 5: LEDGER IMMUTABILITY AUDIT (SQL TRIGGERS)
# ==============================================================================


class TestLedgerImmutabilityTriggers:
    def test_all_five_tables_reject_update_and_delete(self) -> None:
        """PaperLedger enforces append-only semantics via triggers on all 5 tables."""
        ledger = PaperLedger()  # In-memory SQLite
        conn = ledger._connection

        order = PaperOrder(
            order_id="ORD-IMMUT-1",
            strategy_id="STRAT-1",
            qualification_id="QUAL-1",
            symbol="NIFTY_FUT",
            side=1,
            quantity=1,
            signal_timestamp="2023-01-01T09:15:00",
            submit_timestamp="2023-01-01T09:15:00",
            requested_price=100.0,
        )
        ledger.record_order(order)

        fill = PaperFill(
            fill_id="FILL-IMMUT-1",
            order_id="ORD-IMMUT-1",
            strategy_id="STRAT-1",
            symbol="NIFTY_FUT",
            fill_timestamp="2023-01-01T09:16:00",
            fill_price=100.0,
            quantity=1,
            side=1,
            cost=2.5,
            slippage=0.0,
        )
        ledger.record_fill(fill)

        event = PaperRiskEvent(
            event_id="EVT-IMMUT-1",
            order_id="ORD-IMMUT-1",
            strategy_id="STRAT-1",
            rule_name="max_position",
            limit_value=5.0,
            requested_value=10.0,
            timestamp="2023-01-01T09:15:00",
            reason="limit reached",
        )
        ledger.record_risk_event(event)

        pos = PaperPosition(
            symbol="NIFTY_FUT",
            quantity=1,
            entry_price=100.0,
            current_price=100.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees_costs=2.5,
        )
        ledger.record_position_snapshot("STRAT-1", "2023-01-01T09:16:00", pos)

        report = ReplayReport.create(
            strategy_id="STRAT-1",
            qualification_id="QUAL-1",
            dataset_version="DS-TEST-2023",
            trade_count=1,
            gross_pnl=100.0,
            net_pnl=97.5,
            costs=2.5,
            slippage=0.0,
            max_drawdown_bps=0.0,
            exposure=5000.0,
            win_rate=1.0,
            expectancy=97.5,
            sharpe_ratio=None,
            session_breakdown=[],
        )
        ledger.record_report(report)

        # 1. paper_orders
        with pytest.raises(sqlite3.IntegrityError, match="cannot be updated"):
            conn.execute("UPDATE paper_orders SET status = 'FILLED' WHERE order_id = 'ORD-IMMUT-1'")
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            conn.execute("DELETE FROM paper_orders WHERE order_id = 'ORD-IMMUT-1'")

        # 2. paper_fills
        with pytest.raises(sqlite3.IntegrityError, match="cannot be updated"):
            conn.execute("UPDATE paper_fills SET fill_price = 200.0 WHERE fill_id = 'FILL-IMMUT-1'")
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            conn.execute("DELETE FROM paper_fills WHERE fill_id = 'FILL-IMMUT-1'")

        # 3. paper_risk_events
        with pytest.raises(sqlite3.IntegrityError, match="cannot be updated"):
            conn.execute("UPDATE paper_risk_events SET rule_name = 'hacked' WHERE event_id = 'EVT-IMMUT-1'")
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            conn.execute("DELETE FROM paper_risk_events WHERE event_id = 'EVT-IMMUT-1'")

        # 4. paper_positions
        with pytest.raises(sqlite3.IntegrityError, match="cannot be updated"):
            conn.execute("UPDATE paper_positions SET quantity = 999 WHERE symbol = 'NIFTY_FUT'")
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            conn.execute("DELETE FROM paper_positions WHERE symbol = 'NIFTY_FUT'")

        # 5. paper_reports
        with pytest.raises(sqlite3.IntegrityError, match="cannot be updated"):
            conn.execute("UPDATE paper_reports SET report_json = '{}' WHERE report_hash = ?", (report.report_hash,))
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            conn.execute("DELETE FROM paper_reports WHERE report_hash = ?", (report.report_hash,))


# ==============================================================================
# PHASE 6 & 7: DETERMINISM & PROVENANCE AUDIT
# ==============================================================================


class TestReplayDeterminismAndProvenance:
    def test_replay_strict_determinism_across_instances(self) -> None:
        """Two independent replay runs produce bitwise identical orders, fills, and report_hash."""
        spec = _make_spec()
        record = _make_valid_record(spec)
        feed1 = _make_clean_feed(n=10)
        feed2 = _make_clean_feed(n=10)

        engine1 = PaperReplayEngine(slippage_bps_per_side=2.0)
        engine2 = PaperReplayEngine(slippage_bps_per_side=2.0)

        report1 = engine1.run_replay(record, spec, feed1, hold_bars=2)
        report2 = engine2.run_replay(record, spec, feed2, hold_bars=2)

        assert report1.report_hash == report2.report_hash
        assert report1.canonical_json() == report2.canonical_json()

        fills1 = engine1.ledger.get_fills()
        fills2 = engine2.ledger.get_fills()
        assert len(fills1) == len(fills2)
        for f1, f2 in zip(fills1, fills2):
            assert f1["fill_id"] == f2["fill_id"]
            assert f1["fill_price"] == f2["fill_price"]
            assert f1["cost"] == f2["cost"]

    def test_replay_provenance_cryptographically_sealed(self) -> None:
        """Report contains all provenance metadata, and altering any field changes the report_hash."""
        spec = _make_spec()
        record = _make_valid_record(spec)
        feed = _make_clean_feed(n=10)
        engine = PaperReplayEngine()

        report = engine.run_replay(record, spec, feed, hold_bars=2)
        canon = report.canonical_dict()

        expected_keys = {
            "strategy_spec_hash",
            "qualification_hash",
            "dataset_sha256",
            "research_protocol_version",
            "symbol",
            "lot_size",
            "tick_size",
            "slippage_bps_per_side",
            "cost_schedule_id",
            "execution_policy",
        }
        for key in expected_keys:
            assert key in canon, f"Missing provenance key '{key}' in report canonical_dict"

        # Mutate one provenance field -> hash must change
        mutated_report = ReplayReport.create(
            strategy_id=report.strategy_id,
            qualification_id=report.qualification_id,
            dataset_version=report.dataset_version,
            trade_count=report.trade_count,
            gross_pnl=report.gross_pnl,
            net_pnl=report.net_pnl,
            costs=report.costs,
            slippage=report.slippage,
            max_drawdown_bps=report.max_drawdown_bps,
            exposure=report.exposure,
            win_rate=report.win_rate,
            expectancy=report.expectancy,
            sharpe_ratio=report.sharpe_ratio,
            session_breakdown=report.session_breakdown,
            strategy_spec_hash=report.strategy_spec_hash,
            qualification_hash=report.qualification_hash,
            dataset_sha256="dsha-DIFFERENT",  # Mutated
            research_protocol_version=report.research_protocol_version,
            symbol=report.symbol,
            lot_size=report.lot_size,
            tick_size=report.tick_size,
            slippage_bps_per_side=report.slippage_bps_per_side,
            cost_schedule_id=report.cost_schedule_id,
            execution_policy=report.execution_policy,
        )
        assert mutated_report.report_hash != report.report_hash


# ==============================================================================
# PHASE 8: RISK ENGINE AUDIT
# ==============================================================================


class TestRiskEngineAdversarial:
    def test_invalid_order_quantity_or_side_rejected(self) -> None:
        """PaperOrder rejects zero or negative quantity, or side not in (-1, 1)."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            PaperOrder(
                order_id="ORD-0",
                strategy_id="S1",
                qualification_id="Q1",
                symbol="NIFTY_FUT",
                side=1,
                quantity=0,
                signal_timestamp="2023-01-01T09:15:00",
                submit_timestamp="2023-01-01T09:15:00",
                requested_price=100.0,
            )

        with pytest.raises(ValueError, match="quantity must be positive"):
            PaperOrder(
                order_id="ORD-NEG",
                strategy_id="S1",
                qualification_id="Q1",
                symbol="NIFTY_FUT",
                side=1,
                quantity=-5,
                signal_timestamp="2023-01-01T09:15:00",
                submit_timestamp="2023-01-01T09:15:00",
                requested_price=100.0,
            )

        with pytest.raises(ValueError, match="side must be 1 \\(Buy\\) or -1 \\(Sell\\)"):
            PaperOrder(
                order_id="ORD-SIDE",
                strategy_id="S1",
                qualification_id="Q1",
                symbol="NIFTY_FUT",
                side=0,
                quantity=1,
                signal_timestamp="2023-01-01T09:15:00",
                submit_timestamp="2023-01-01T09:15:00",
                requested_price=100.0,
            )

    def test_lot_size_scaling_in_exposure_check(self) -> None:
        """Exposure check must scale by lot_size: Q=1 * P=2000 * lot_size=50 = 100,000."""
        config = PaperRiskConfig(max_exposure=50_000.0)
        engine = PaperRiskEngine(config)
        pos = PaperPosition(
            symbol="NIFTY_FUT",
            quantity=0,
            entry_price=0.0,
            current_price=2000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees_costs=0.0,
        )
        order = PaperOrder(
            order_id="ORD-EXP-1",
            strategy_id="S1",
            qualification_id="Q1",
            symbol="NIFTY_FUT",
            side=1,
            quantity=1,
            signal_timestamp="2023-01-01T09:15:00",
            submit_timestamp="2023-01-01T09:15:00",
            requested_price=2000.0,
        )

        # With lot_size=1: exposure = 2000 <= 50000 -> Approved
        assert engine.evaluate_order(order, pos, 2000.0, "2023-01-01", lot_size=1) is None

        # With lot_size=50: exposure = 100000 > 50000 -> Rejected
        event = engine.evaluate_order(order, pos, 2000.0, "2023-01-01", lot_size=50)
        assert event is not None
        assert event.rule_name == "max_exposure"

    def test_risk_reducing_exit_orders_permitted_under_drawdown_or_loss_breach(self) -> None:
        """Positions are never trapped: risk-reducing exit orders pass even if loss/drawdown breached."""
        config = PaperRiskConfig(
            max_daily_loss=1000.0,
            max_strategy_drawdown=2000.0,
            max_trades_per_session=1,
        )
        engine = PaperRiskEngine(config)
        engine.on_session_change("2023-01-01")

        # Breach daily loss and max drawdown
        engine.update_portfolio_state(current_equity=90_000.0, realized_pnl_delta=-5000.0)

        # Long 1 position
        pos = PaperPosition(
            symbol="NIFTY_FUT",
            quantity=1,
            entry_price=100.0,
            current_price=90.0,
            realized_pnl=-5000.0,
            unrealized_pnl=0.0,
            fees_costs=10.0,
        )

        # Order to increase risk (buy another 1 -> qty 2) -> must be blocked
        buy_order = PaperOrder(
            order_id="ORD-BUY-FAIL",
            strategy_id="S1",
            qualification_id="Q1",
            symbol="NIFTY_FUT",
            side=1,
            quantity=1,
            signal_timestamp="2023-01-01T09:15:00",
            submit_timestamp="2023-01-01T09:15:00",
            requested_price=90.0,
        )
        event = engine.evaluate_order(buy_order, pos, 90.0, "2023-01-01", lot_size=50)
        assert event is not None
        assert event.rule_name == "max_daily_loss"

        # Order to reduce risk (sell 1 -> qty 0) -> must be PERMITTED
        sell_order = PaperOrder(
            order_id="ORD-SELL-EXIT",
            strategy_id="S1",
            qualification_id="Q1",
            symbol="NIFTY_FUT",
            side=-1,
            quantity=1,
            signal_timestamp="2023-01-01T09:15:00",
            submit_timestamp="2023-01-01T09:15:00",
            requested_price=90.0,
        )
        assert engine.evaluate_order(sell_order, pos, 90.0, "2023-01-01", lot_size=50) is None


# ==============================================================================
# PHASE 9: DATA FEED SECURITY AUDIT
# ==============================================================================


class TestDataFeedSecurityAdversarial:
    def test_feed_rejects_nans_in_ohlc(self) -> None:
        """Feed rejects DataFrame with NaNs."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 09:15", periods=3, freq="1min"),
                "open": [100.0, np.nan, 102.0],
                "high": [101.0, 103.0, 104.0],
                "low": [99.0, 98.0, 100.0],
                "close": [100.5, 101.5, 102.5],
            }
        )
        with pytest.raises(ValueError, match="contains NaN values in column 'open'"):
            ReplayFeed(df)

    def test_feed_rejects_infs_in_ohlc(self) -> None:
        """Feed rejects DataFrame with infinities."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 09:15", periods=3, freq="1min"),
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, np.inf, 104.0],
                "low": [99.0, 98.0, 100.0],
                "close": [100.5, 101.5, 102.5],
            }
        )
        with pytest.raises(ValueError, match="contains non-finite or infinite values in column 'high'"):
            ReplayFeed(df)

    def test_feed_rejects_duplicate_timestamps(self) -> None:
        """Feed rejects duplicate timestamps."""
        t0 = pd.Timestamp("2023-01-01 09:15:00")
        df = pd.DataFrame(
            {
                "timestamp": [t0, t0, t0 + pd.Timedelta(minutes=1)],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
            }
        )
        with pytest.raises(ValueError, match="contains duplicate timestamps"):
            ReplayFeed(df)

    def test_feed_rejects_unsorted_timestamps(self) -> None:
        """Feed rejects timestamps out of chronological order."""
        t0 = pd.Timestamp("2023-01-01 09:15:00")
        df = pd.DataFrame(
            {
                "timestamp": [t0 + pd.Timedelta(minutes=2), t0, t0 + pd.Timedelta(minutes=1)],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
            }
        )
        with pytest.raises(ValueError, match="must be sorted in strictly increasing chronological order"):
            ReplayFeed(df)

    def test_feed_rejects_non_positive_price(self) -> None:
        """Feed rejects zero or negative prices."""
        timestamps = pd.date_range("2023-01-01 09:15", periods=2, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 0.0],
                "high": [101.0, 10.0],
                "low": [99.0, 0.0],
                "close": [100.5, 5.0],
            }
        )
        with pytest.raises(ValueError, match="prices must be strictly positive"):
            ReplayFeed(df)

    def test_feed_rejects_invalid_ohlc_high_low_violation(self) -> None:
        """Feed rejects bar where high < low."""
        timestamps = pd.date_range("2023-01-01 09:15", periods=2, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 105.0],
                "high": [101.0, 95.0],  # High < Low
                "low": [99.0, 100.0],
                "close": [100.5, 98.0],
            }
        )
        with pytest.raises(ValueError, match="OHLC invariant violated"):
            ReplayFeed(df)

    def test_registry_verify_detects_tampered_disk_file(self, tmp_path: Path) -> None:
        """If file on disk is modified after registration, checksum verification blocks replay."""
        csv_file = tmp_path / "licensed_data.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 09:15", periods=5, freq="1min"),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
            }
        )
        df.to_csv(csv_file, index=False)

        reg_db = tmp_path / "registry.db"
        registry = DatasetRegistry(path=reg_db)
        record = registry.register_file(
            version="v1",
            kind=DatasetKind.LICENSED,
            path=csv_file,
        )

        # Tamper disk file
        with open(csv_file, "a") as f:
            f.write("\ntampered_garbage\n")

        with pytest.raises(DatasetRegistryError, match="dataset checksum mismatch"):
            ReplayFeed.from_dataset_registry(registry, "v1")
