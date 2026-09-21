"""Tests for PaperEvaluationService (PRD v4.0 Milestone 5).

Covers:
- Basic: valid evaluation, snapshot persistence, idempotency
- Boundary: T_start / T_cutoff inclusion / exclusion
- Future leakage: adversarial T1/T2 invariant (mandatory gate)
- Sessions: complete vs incomplete
- Baseline: authoritative loading, rejection of unregistered
- Observed report: separate from baseline, future-coverage guard
- Provenance: tampered inputs rejected
- Integrity: invalid fill prices, duplicate IDs
- Detector: no event / one event / multiple / deterministic order / idempotent
- Read-only: PaperLedger unmodified after evaluation
- Metrics pipeline: trade count, net P&L, risk events, slippage, determinism
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from quantmind.paper.evaluation.ledger import EvaluationLedger
from quantmind.paper.evaluation.models import (
    MonitoringConfig,
    PaperEvaluationBaseline,
)
from quantmind.paper.evaluation.service import (
    EvaluationResult,
    PaperEvaluationService,
    PaperEvaluationServiceError,
)
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
from quantmind.research_integrity.qualification import (
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_ID = "STRAT_M5_TEST"
DATASET_VERSION = "nifty_2020_v1"
DATASET_SHA256 = "ab" * 32  # 64 hex chars
QUALIFICATION_ID = "QUAL_M5_001"

T_START = "2024-01-10T09:15:00"
T1 = "2024-01-10T09:15:01"       # inside window
T_MID = "2024-01-10T12:00:00"   # inside window
T_CUTOFF = "2024-01-10T15:30:00"
T_AFTER = "2024-01-10T15:30:01"  # just after cutoff
T_BEFORE = "2024-01-09T15:30:00" # before window


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_config(
    max_drawdown_expansion_limit: float = 2.0,
    min_rolling_sharpe_30d: float = -0.5,
    max_slippage_drift_ratio: float = 3.0,
    max_risk_rejection_rate: float = 0.5,
) -> MonitoringConfig:
    return MonitoringConfig(
        protocol_version="MP-1.0",
        max_drawdown_expansion_limit=max_drawdown_expansion_limit,
        min_rolling_sharpe_30d=min_rolling_sharpe_30d,
        max_slippage_drift_ratio=max_slippage_drift_ratio,
        max_risk_rejection_rate=max_risk_rejection_rate,
        window_size_sessions=30,
        min_evaluation_trades=5,
    )


def _make_qual_record(
    strategy_id: str = STRATEGY_ID,
    dataset_sha256: str = DATASET_SHA256,
    final_status: ValidationStatus = ValidationStatus.PAPER_ELIGIBLE,
    robustness_status: RobustnessStatus = RobustnessStatus.PASSED,
    holdout_state: str = "PASSED",
) -> StrategyQualificationRecord:
    return StrategyQualificationRecord.create(
        qualification_id=QUALIFICATION_ID,
        strategy_id=strategy_id,
        strategy_spec_hash="spec" + "a" * 60,
        dataset_version=DATASET_VERSION,
        dataset_sha256=dataset_sha256,
        split_manifest_version="v1",
        research_protocol_version="rp-1.0",
        population_hash="pop" + "a" * 61,
        effective_trial_count=100.0,
        observed_sharpe=1.5,
        dsr=0.95,
        trade_count=200,
        holdout_state=holdout_state,
        robustness_status=robustness_status,
        final_status=final_status,
        created_at="2024-01-01T00:00:00",
    )


def _make_baseline_report(
    qual_record: StrategyQualificationRecord | None = None,
    strategy_id: str = STRATEGY_ID,
    session_start: str = "2024-01-01T09:15:00",
    session_end: str = "2024-01-01T15:30:00",
    max_drawdown_bps: float = 50.0,
    slippage_bps_per_side: float = 2.0,
) -> ReplayReport:
    if qual_record is None:
        qual_record = _make_qual_record(strategy_id=strategy_id)
    session = ReplaySessionSummary(
        session_id="BASE_S001",
        start_ts=session_start,
        end_ts=session_end,
        trades=10,
        gross_pnl=1000.0,
        net_pnl=900.0,
    )
    return ReplayReport.create(
        strategy_id=strategy_id,
        qualification_id=qual_record.qualification_id,
        dataset_version=qual_record.dataset_version,
        trade_count=10,
        gross_pnl=1000.0,
        net_pnl=900.0,
        costs=100.0,
        slippage=10.0,
        max_drawdown_bps=max_drawdown_bps,
        exposure=0.6,
        win_rate=0.6,
        expectancy=90.0,
        sharpe_ratio=1.5,
        session_breakdown=(session,),
        created_at="2024-01-01T16:00:00",
        qualification_hash=qual_record.record_hash,
        dataset_sha256=qual_record.dataset_sha256,
        split_zone="FORWARD_PAPER",
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="csh" + "a" * 61,
        risk_config_hash="rch" + "a" * 61,
        lot_size=50,
        initial_capital=100_000.0,
        slippage_bps_per_side=slippage_bps_per_side,
    )


def _make_observed_report(
    qual_record: StrategyQualificationRecord | None = None,
    strategy_id: str = STRATEGY_ID,
    session_start: str = T_START,
    session_end: str = T_CUTOFF,
    trades: int = 5,
    net_pnl: float = 450.0,
) -> ReplayReport:
    if qual_record is None:
        qual_record = _make_qual_record(strategy_id=strategy_id)
    session = ReplaySessionSummary(
        session_id="OBS_S001",
        start_ts=session_start,
        end_ts=session_end,
        trades=trades,
        gross_pnl=net_pnl + 50.0,
        net_pnl=net_pnl,
    )
    return ReplayReport.create(
        strategy_id=strategy_id,
        qualification_id=qual_record.qualification_id,
        dataset_version=qual_record.dataset_version,
        trade_count=trades,
        gross_pnl=net_pnl + 50.0,
        net_pnl=net_pnl,
        costs=50.0,
        slippage=5.0,
        max_drawdown_bps=30.0,
        exposure=0.5,
        win_rate=0.6,
        expectancy=90.0,
        sharpe_ratio=1.2,
        session_breakdown=(session,),
        created_at="2024-01-10T16:00:00",
        qualification_hash=qual_record.record_hash,
        dataset_sha256=qual_record.dataset_sha256,
        split_zone="FORWARD_PAPER",
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="csh" + "a" * 61,
        risk_config_hash="rch" + "a" * 61,
        lot_size=50,
        initial_capital=100_000.0,
        slippage_bps_per_side=2.0,
    )


def _make_baseline(
    qual_record: StrategyQualificationRecord,
    baseline_report: ReplayReport,
) -> PaperEvaluationBaseline:
    return PaperEvaluationBaseline.create(
        strategy_id=qual_record.strategy_id,
        qualification_hash=qual_record.record_hash,
        baseline_replay_report_hash=baseline_report.report_hash,
        baseline_dataset_version=baseline_report.dataset_version,
        baseline_dataset_sha256=baseline_report.dataset_sha256,
        baseline_split_zone=baseline_report.split_zone,
        baseline_execution_policy=baseline_report.execution_policy,
        baseline_cost_schedule_hash=baseline_report.cost_schedule_hash,
        baseline_risk_config_hash=baseline_report.risk_config_hash,
        created_at="2024-01-01T16:00:00",
    )


def _setup_ledgers(
    qual_record: StrategyQualificationRecord | None = None,
    baseline_report: ReplayReport | None = None,
    register_baseline: bool = True,
) -> tuple[PaperLedger, EvaluationLedger, StrategyQualificationRecord, ReplayReport]:
    """Create in-memory PaperLedger + EvaluationLedger with optional registered baseline."""
    if qual_record is None:
        qual_record = _make_qual_record()
    if baseline_report is None:
        baseline_report = _make_baseline_report(qual_record=qual_record)

    paper_ledger = PaperLedger()
    paper_ledger.record_report(baseline_report)

    eval_ledger = EvaluationLedger()

    if register_baseline:
        baseline = _make_baseline(qual_record, baseline_report)
        eval_ledger.register_baseline(baseline, replay_report=baseline_report)

    return paper_ledger, eval_ledger, qual_record, baseline_report


def _add_fill(
    paper_ledger: PaperLedger,
    *,
    fill_id: str,
    order_id: str,
    strategy_id: str = STRATEGY_ID,
    fill_timestamp: str,
    fill_price: float = 21000.0,
    quantity: int = 1,
    side: int = 1,
    cost: float = 20.0,
    slippage: float = 5.0,
) -> None:
    """Record a submitted order + fill pair in PaperLedger."""
    order = PaperOrder(
        order_id=order_id,
        strategy_id=strategy_id,
        qualification_id=QUALIFICATION_ID,
        symbol="NIFTY_FUT",
        side=side,
        quantity=quantity,
        order_type="NEXT_BAR_OPEN",
        signal_timestamp=fill_timestamp,
        submit_timestamp=fill_timestamp,
        requested_price=fill_price,
        status=PaperOrderStatus.FILLED,
    )
    paper_ledger.record_order(order)
    fill = PaperFill(
        fill_id=fill_id,
        order_id=order_id,
        strategy_id=strategy_id,
        symbol="NIFTY_FUT",
        fill_timestamp=fill_timestamp,
        fill_price=fill_price,
        quantity=quantity,
        side=side,
        cost=cost,
        slippage=slippage,
    )
    paper_ledger.record_fill(fill)


def _add_position(
    paper_ledger: PaperLedger,
    *,
    timestamp: str,
    strategy_id: str = STRATEGY_ID,
    realized_pnl: float = 0.0,
    symbol: str = "NIFTY_FUT",
) -> None:
    pos = PaperPosition(
        symbol=symbol,
        quantity=1,
        entry_price=21000.0,
        current_price=21100.0,
        realized_pnl=realized_pnl,
        unrealized_pnl=100.0,
        fees_costs=20.0,
    )
    paper_ledger.record_position_snapshot(timestamp, strategy_id, pos)


def _add_risk_event(
    paper_ledger: PaperLedger,
    *,
    event_id: str,
    order_id: str,
    timestamp: str,
    strategy_id: str = STRATEGY_ID,
) -> None:
    ev = PaperRiskEvent(
        event_id=event_id,
        order_id=order_id,
        strategy_id=strategy_id,
        rule_name="MAX_POSITION_SIZE",
        limit_value=10.0,
        requested_value=15.0,
        timestamp=timestamp,
        reason="Exceeds max position size",
    )
    paper_ledger.record_risk_event(ev)


def _svc() -> PaperEvaluationService:
    return PaperEvaluationService()


# ===========================================================================
# BASIC TESTS
# ===========================================================================


class TestBasicEvaluation:
    """Basic valid evaluation scenarios."""

    def test_valid_window_evaluates(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F001", order_id="O001", fill_timestamp=T1)
        _add_position(paper_ledger, timestamp=T1, realized_pnl=200.0)
        obs = _make_observed_report(qual_record=qual_record)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        assert isinstance(result, EvaluationResult)
        assert result.snapshot.strategy_id == STRATEGY_ID
        assert result.snapshot.verify_digest()

    def test_snapshot_persisted_in_ledger(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        retrieved = eval_ledger.get_snapshot(result.snapshot.snapshot_hash)
        assert retrieved is not None
        assert retrieved.snapshot_hash == result.snapshot.snapshot_hash

    def test_repeat_evaluation_is_idempotent(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        obs = _make_observed_report(qual_record=qual_record)
        kwargs = dict(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        r1 = _svc().evaluate_window(**kwargs)
        r2 = _svc().evaluate_window(**kwargs)
        assert r1.snapshot.snapshot_hash == r2.snapshot.snapshot_hash
        assert r1.snapshot.canonical_dict() == r2.snapshot.canonical_dict()

    def test_empty_window_produces_zero_snapshot(self):
        """Empty window (no fills, positions, orders, risk events) is allowed."""
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.total_trades == 0
        assert result.snapshot.net_pnl == 0.0
        assert result.snapshot.risk_event_count == 0

    def test_evaluation_result_has_degradation_property(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert isinstance(result.has_degradation, bool)


# ===========================================================================
# BOUNDARY TESTS
# ===========================================================================


class TestBoundaryInclusion:
    """Temporal boundary: T_start and T_cutoff are inclusive."""

    def test_fill_at_T_start_included(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_START", order_id="O_START", fill_timestamp=T_START)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.total_trades == 1

    def test_fill_at_T_cutoff_included(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_END", order_id="O_END", fill_timestamp=T_CUTOFF)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.total_trades == 1

    def test_fill_just_after_cutoff_excluded(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_AFTER", order_id="O_AFTER", fill_timestamp=T_AFTER)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.total_trades == 0

    def test_fill_just_before_start_excluded(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_BEFORE", order_id="O_BEFORE", fill_timestamp=T_BEFORE)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.total_trades == 0

    def test_risk_event_at_T_cutoff_included(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_risk_event(paper_ledger, event_id="RE_END", order_id="O_RE", timestamp=T_CUTOFF)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.risk_event_count == 1

    def test_risk_event_after_cutoff_excluded(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_risk_event(paper_ledger, event_id="RE_AFTER", order_id="O_RA", timestamp=T_AFTER)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.risk_event_count == 0

    def test_position_at_T_cutoff_included(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_position(paper_ledger, timestamp=T_CUTOFF, realized_pnl=500.0)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.net_pnl == 500.0

    def test_position_after_cutoff_excluded(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_position(paper_ledger, timestamp=T_AFTER, realized_pnl=99999.0)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.net_pnl == 0.0


# ===========================================================================
# FUTURE LEAKAGE INVARIANT TESTS (M5 MANDATORY GATE)
# ===========================================================================


class TestFutureLeakage:
    """Future evidence must have zero effect on Snapshot[T_START, T_CUTOFF]."""

    def _evaluate(self, paper_ledger, eval_ledger, qual_record) -> EvaluationResult:
        obs = _make_observed_report(qual_record=qual_record)
        return _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )

    def test_future_fill_does_not_change_snapshot(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_IN", order_id="O_IN", fill_timestamp=T1)
        r1 = self._evaluate(paper_ledger, eval_ledger, qual_record)
        # Add future fill
        _add_fill(paper_ledger, fill_id="F_FUTURE", order_id="O_FUTURE", fill_timestamp=T_AFTER)
        r2 = self._evaluate(paper_ledger, eval_ledger, qual_record)
        assert r1.snapshot.snapshot_hash == r2.snapshot.snapshot_hash
        assert r1.snapshot.total_trades == r2.snapshot.total_trades

    def test_future_order_does_not_change_snapshot(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        r1 = self._evaluate(paper_ledger, eval_ledger, qual_record)
        # Add rejected order after cutoff
        order = PaperOrder(
            order_id="O_FUTURE_REJ",
            strategy_id=STRATEGY_ID,
            qualification_id=QUALIFICATION_ID,
            symbol="NIFTY_FUT",
            side=1,
            quantity=1,
            submit_timestamp=T_AFTER,
            signal_timestamp=T_AFTER,
            requested_price=21000.0,
            status=PaperOrderStatus.REJECTED,
            rejection_reason="limit",
        )
        paper_ledger.record_order(order)
        r2 = self._evaluate(paper_ledger, eval_ledger, qual_record)
        assert r1.snapshot.snapshot_hash == r2.snapshot.snapshot_hash

    def test_future_position_does_not_change_snapshot(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_position(paper_ledger, timestamp=T1, realized_pnl=100.0)
        r1 = self._evaluate(paper_ledger, eval_ledger, qual_record)
        # Add position snapshot after cutoff
        _add_position(paper_ledger, timestamp=T_AFTER, realized_pnl=99999.0)
        r2 = self._evaluate(paper_ledger, eval_ledger, qual_record)
        assert r1.snapshot.snapshot_hash == r2.snapshot.snapshot_hash
        assert r1.snapshot.net_pnl == r2.snapshot.net_pnl

    def test_future_risk_event_does_not_change_snapshot(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        r1 = self._evaluate(paper_ledger, eval_ledger, qual_record)
        _add_risk_event(paper_ledger, event_id="RE_FUTURE", order_id="O_RF", timestamp=T_AFTER)
        r2 = self._evaluate(paper_ledger, eval_ledger, qual_record)
        assert r1.snapshot.snapshot_hash == r2.snapshot.snapshot_hash
        assert r1.snapshot.risk_event_count == r2.snapshot.risk_event_count

    def test_adversarial_t1_t2_invariant(self):
        """M5 mandatory gate: Evaluate [T_START, T_CUTOFF], add T2 evidence, re-evaluate — identical."""
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_W", order_id="O_W", fill_timestamp=T1)
        _add_position(paper_ledger, timestamp=T1, realized_pnl=250.0)
        obs = _make_observed_report(qual_record=qual_record)
        kwargs = dict(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        # Step 1: Evaluate [T_START, T_CUTOFF]
        r1 = _svc().evaluate_window(**kwargs)
        hash1 = r1.snapshot.snapshot_hash
        canon1 = r1.snapshot.canonical_dict()
        event_hashes1 = sorted(ev.event_hash for ev in r1.degradation_events)

        # Step 2: Add T2 evidence (after T_CUTOFF)
        _add_fill(paper_ledger, fill_id="F_T2", order_id="O_T2", fill_timestamp=T_AFTER)
        _add_position(paper_ledger, timestamp=T_AFTER, realized_pnl=99999.0)
        _add_risk_event(paper_ledger, event_id="RE_T2", order_id="O_T2RE", timestamp=T_AFTER)

        # Step 3: Re-evaluate same window
        r2 = _svc().evaluate_window(**kwargs)
        hash2 = r2.snapshot.snapshot_hash
        canon2 = r2.snapshot.canonical_dict()
        event_hashes2 = sorted(ev.event_hash for ev in r2.degradation_events)

        # All outputs must be identical
        assert hash1 == hash2, "snapshot_hash changed after adding future evidence"
        assert canon1 == canon2, "canonical snapshot JSON changed after adding future evidence"
        assert event_hashes1 == event_hashes2, "event hashes changed after adding future evidence"


# ===========================================================================
# SESSION TESTS
# ===========================================================================


class TestSessionBoundary:
    """Completed vs incomplete session semantics."""

    def test_complete_session_included(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        obs = _make_observed_report(
            qual_record=qual_record,
            session_start=T_START,
            session_end=T_CUTOFF,
            trades=5,
        )
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        metrics = json.loads(result.snapshot.metrics_json)
        assert metrics["completed_sessions"] == 1

    def test_incomplete_session_rejected(self):
        """Observed report with session ending after T_cutoff is rejected."""
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        obs = _make_observed_report(
            qual_record=qual_record,
            session_start=T_START,
            session_end=T_AFTER,  # extends past T_CUTOFF
        )
        with pytest.raises(PaperEvaluationServiceError, match="after T_cutoff"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_session_starting_before_window_excluded_from_completed(self):
        """Session starting before T_start is excluded from completed session count."""
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        obs = _make_observed_report(
            qual_record=qual_record,
            session_start=T_BEFORE,  # starts before window
            session_end=T_CUTOFF,
        )
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        metrics = json.loads(result.snapshot.metrics_json)
        assert metrics["completed_sessions"] == 0


# ===========================================================================
# BASELINE TESTS
# ===========================================================================


class TestBaselineProvenance:
    """Authoritative baseline loading and rejection of unregistered baselines."""

    def test_authoritative_baseline_loaded_from_ledger(self):
        paper_ledger, eval_ledger, qual_record, baseline_report = _setup_ledgers()
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.qualification_hash == qual_record.record_hash

    def test_no_registered_baseline_raises(self):
        qual_record = _make_qual_record()
        baseline_report = _make_baseline_report(qual_record=qual_record)
        paper_ledger = PaperLedger()
        paper_ledger.record_report(baseline_report)
        eval_ledger = EvaluationLedger()  # no baseline registered
        with pytest.raises(PaperEvaluationServiceError, match="No authoritative baseline"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )

    def test_missing_baseline_report_in_paper_ledger_raises(self):
        qual_record = _make_qual_record()
        baseline_report = _make_baseline_report(qual_record=qual_record)
        paper_ledger = PaperLedger()  # baseline report NOT recorded here
        eval_ledger = EvaluationLedger()
        baseline = _make_baseline(qual_record, baseline_report)
        eval_ledger.register_baseline(baseline, replay_report=baseline_report)
        with pytest.raises(PaperEvaluationServiceError, match="not found in PaperLedger"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )


# ===========================================================================
# OBSERVED REPORT TESTS
# ===========================================================================


class TestObservedReport:
    """Observed report validation and baseline/observed separation."""

    def test_baseline_and_observed_reports_may_differ(self):
        paper_ledger, eval_ledger, qual_record, baseline_report = _setup_ledgers()
        obs = _make_observed_report(qual_record=qual_record)
        assert obs.report_hash != baseline_report.report_hash
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        assert result.snapshot.replay_report_hash == obs.report_hash

    def test_observed_report_wrong_strategy_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        other_qual = _make_qual_record(strategy_id="OTHER_STRAT")
        obs = _make_observed_report(strategy_id="OTHER_STRAT", qual_record=other_qual)
        with pytest.raises(PaperEvaluationServiceError):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_future_coverage_report_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        obs = _make_observed_report(
            qual_record=qual_record,
            session_start=T_START,
            session_end=T_AFTER,
        )
        with pytest.raises(PaperEvaluationServiceError, match="after T_cutoff"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_observed_report_wrong_split_zone_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        obs = _make_observed_report(qual_record=qual_record)
        # Build an observed report with wrong split_zone
        bad_obs = dataclasses.replace(obs, split_zone="IN_SAMPLE", report_hash="")
        bad_obs = dataclasses.replace(bad_obs, report_hash=bad_obs.compute_report_hash())
        with pytest.raises(PaperEvaluationServiceError, match="split_zone"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
                observed_replay_report=bad_obs,
            )

    def test_no_observed_report_binds_to_baseline_hash(self):
        """When no observed report supplied, snapshot.replay_report_hash == baseline hash."""
        paper_ledger, eval_ledger, qual_record, baseline_report = _setup_ledgers()
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.replay_report_hash == baseline_report.report_hash


# ===========================================================================
# PROVENANCE TESTS
# ===========================================================================


class TestProvenance:
    """Provenance validation: tampered or mismatched inputs are rejected."""

    def test_tampered_qualification_record_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        bad_qual = dataclasses.replace(qual_record, observed_sharpe=99.0)
        with pytest.raises(PaperEvaluationServiceError, match="digest verification failed"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=bad_qual,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )

    def test_qualification_hash_mismatch_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        with pytest.raises(PaperEvaluationServiceError, match="mismatch"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash="wrong" + "a" * 59,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )

    def test_strategy_id_mismatch_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        with pytest.raises(PaperEvaluationServiceError):
            _svc().evaluate_window(
                strategy_id="WRONG_STRAT",
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )

    def test_non_paper_eligible_status_rejected(self):
        paper_ledger = PaperLedger()
        eval_ledger = EvaluationLedger()
        bad_qual = _make_qual_record(
            final_status=ValidationStatus.REJECTED,
            robustness_status=RobustnessStatus.FAILED,
            holdout_state="FAILED",
        )
        with pytest.raises(PaperEvaluationServiceError, match="PAPER_ELIGIBLE"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=bad_qual.record_hash,
                qualification_record=bad_qual,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )

    def test_window_start_after_end_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        with pytest.raises(PaperEvaluationServiceError, match="window_start_ts"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_CUTOFF,
                window_end_ts=T_START,  # reversed
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )

    def test_empty_strategy_id_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        with pytest.raises(PaperEvaluationServiceError, match="strategy_id cannot be empty"):
            _svc().evaluate_window(
                strategy_id="",
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )


# ===========================================================================
# DATA INTEGRITY TESTS
# ===========================================================================


class TestDataIntegrity:
    """Slice integrity validation."""

    def test_fill_with_zero_price_rejected(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        # Bypass model validation to insert invalid fill directly via connection
        paper_ledger._connection.execute(
            """
            INSERT INTO paper_orders (order_id, strategy_id, qualification_id, symbol, side,
                quantity, order_type, signal_timestamp, submit_timestamp, requested_price,
                status, rejection_reason)
            VALUES ('O_BAD', ?, ?, 'NIFTY_FUT', 1, 1, 'NEXT_BAR_OPEN', ?, ?, 0.0, 'FILLED', NULL)
            """,
            (STRATEGY_ID, QUALIFICATION_ID, T1, T1),
        )
        paper_ledger._connection.execute(
            """
            INSERT INTO paper_fills (fill_id, order_id, strategy_id, symbol, fill_timestamp,
                fill_price, quantity, side, cost, slippage)
            VALUES ('F_BAD', 'O_BAD', ?, 'NIFTY_FUT', ?, 0.0, 1, 1, 5.0, 1.0)
            """,
            (STRATEGY_ID, T1),
        )
        with pytest.raises(PaperEvaluationServiceError, match="non-positive fill_price"):
            _svc().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual_record.record_hash,
                qualification_record=qual_record,
                window_start_ts=T_START,
                window_end_ts=T_CUTOFF,
                monitoring_config=_make_config(),
                paper_ledger=paper_ledger,
                evaluation_ledger=eval_ledger,
            )

    def test_evidence_outside_window_silently_excluded(self):
        """Old evidence is excluded, not an error — zero trades in window."""
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_OLD", order_id="O_OLD", fill_timestamp=T_BEFORE)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.total_trades == 0


# ===========================================================================
# DETECTOR PIPELINE TESTS
# ===========================================================================


class TestDetectorPipeline:
    """DegradationDetector integration through the service."""

    def test_no_degradation_event_for_healthy_window(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.has_degradation is False
        assert len(result.degradation_events) == 0

    def test_drawdown_expansion_event_detected_and_persisted(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        # Config with near-zero expansion limit forces trigger on any drawdown
        tight_config = _make_config(max_drawdown_expansion_limit=0.001)
        _add_position(paper_ledger, timestamp=T1, realized_pnl=0.0)
        _add_position(paper_ledger, timestamp=T_MID, realized_pnl=-5000.0)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=tight_config,
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.has_degradation
        events = eval_ledger.list_degradation_events(STRATEGY_ID)
        assert any(ev.rule_name == "RULE_DD_EXPANSION_CRITICAL" for ev in events)

    def test_degradation_events_idempotent_on_repeat(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        tight_config = _make_config(max_drawdown_expansion_limit=0.001)
        _add_position(paper_ledger, timestamp=T1, realized_pnl=0.0)
        _add_position(paper_ledger, timestamp=T_MID, realized_pnl=-5000.0)
        kwargs = dict(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=tight_config,
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        r1 = _svc().evaluate_window(**kwargs)
        r2 = _svc().evaluate_window(**kwargs)
        hashes1 = sorted(ev.event_hash for ev in r1.degradation_events)
        hashes2 = sorted(ev.event_hash for ev in r2.degradation_events)
        assert hashes1 == hashes2

    def test_degradation_events_are_alphabetically_ordered(self):
        """Events must be sorted by rule_name (deterministic order from detector)."""
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        # Trigger drawdown expansion + slippage anomaly simultaneously
        tight_config = _make_config(
            max_drawdown_expansion_limit=0.001,
            max_slippage_drift_ratio=0.001,
        )
        _add_fill(
            paper_ledger, fill_id="F_SLP", order_id="O_SLP",
            fill_timestamp=T1, slippage=500.0,
        )
        _add_position(paper_ledger, timestamp=T1, realized_pnl=0.0)
        _add_position(paper_ledger, timestamp=T_MID, realized_pnl=-5000.0)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=tight_config,
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        rule_names = [ev.rule_name for ev in result.degradation_events]
        assert rule_names == sorted(rule_names), (
            f"Events not alphabetically ordered: {rule_names}"
        )

    def test_multiple_events_persisted_independently(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        tight_config = _make_config(
            max_drawdown_expansion_limit=0.001,
            max_slippage_drift_ratio=0.001,
        )
        _add_fill(
            paper_ledger, fill_id="F_SLP2", order_id="O_SLP2",
            fill_timestamp=T1, slippage=500.0,
        )
        _add_position(paper_ledger, timestamp=T1, realized_pnl=0.0)
        _add_position(paper_ledger, timestamp=T_MID, realized_pnl=-5000.0)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=tight_config,
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        # Both DD expansion and slippage rules should fire
        assert len(result.degradation_events) >= 2
        # Each has a distinct event_hash
        hashes = [ev.event_hash for ev in result.degradation_events]
        assert len(hashes) == len(set(hashes))


# ===========================================================================
# READ-ONLY TESTS
# ===========================================================================


class TestReadOnly:
    """Verify evaluation never mutates PaperLedger."""

    def test_paper_ledger_orders_fills_risk_unchanged_after_evaluation(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_RO", order_id="O_RO", fill_timestamp=T1)
        _add_risk_event(paper_ledger, event_id="RE_RO", order_id="O_RO_RE", timestamp=T1)
        orders_before = len(paper_ledger.get_orders(STRATEGY_ID))
        fills_before = len(paper_ledger.get_fills(STRATEGY_ID))
        risk_before = len(paper_ledger.get_risk_events(STRATEGY_ID))

        _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )

        assert len(paper_ledger.get_orders(STRATEGY_ID)) == orders_before
        assert len(paper_ledger.get_fills(STRATEGY_ID)) == fills_before
        assert len(paper_ledger.get_risk_events(STRATEGY_ID)) == risk_before

    def test_evaluation_does_not_add_positions(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_position(paper_ledger, timestamp=T1, realized_pnl=100.0)
        pos_before = len(paper_ledger.get_positions(STRATEGY_ID))

        _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )

        assert len(paper_ledger.get_positions(STRATEGY_ID)) == pos_before

    def test_evaluation_does_not_modify_reports(self):
        paper_ledger, eval_ledger, qual_record, baseline_report = _setup_ledgers()
        reports_before = len(paper_ledger.list_reports(STRATEGY_ID))

        _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )

        assert len(paper_ledger.list_reports(STRATEGY_ID)) == reports_before


# ===========================================================================
# METRICS PIPELINE TESTS
# ===========================================================================


class TestMetricsPipeline:
    """Verify metric values are correctly computed and populated in snapshot."""

    def test_total_trades_from_fills(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F1", order_id="O1", fill_timestamp=T1)
        _add_fill(paper_ledger, fill_id="F2", order_id="O2", fill_timestamp=T_MID)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.total_trades == 2

    def test_net_pnl_from_latest_position_snapshot(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        # Two snapshots for same symbol — latest wins
        _add_position(paper_ledger, timestamp=T1, realized_pnl=100.0)
        _add_position(paper_ledger, timestamp=T_MID, realized_pnl=750.0)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        # Latest realized_pnl for NIFTY_FUT is 750.0
        assert result.snapshot.net_pnl == 750.0

    def test_risk_event_count_in_snapshot(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_risk_event(paper_ledger, event_id="RE1", order_id="O1", timestamp=T1)
        _add_risk_event(paper_ledger, event_id="RE2", order_id="O2", timestamp=T_MID)
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.risk_event_count == 2

    def test_realized_slippage_bps_positive_when_fills_have_slippage(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(
            paper_ledger, fill_id="F_SLP", order_id="O_SLP", fill_timestamp=T1,
            fill_price=21000.0, quantity=1, slippage=10.5, cost=20.0,
        )
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        assert result.snapshot.realized_slippage_bps > 0.0

    def test_snapshot_hash_is_deterministic(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        _add_fill(paper_ledger, fill_id="F_DET", order_id="O_DET", fill_timestamp=T1)
        kwargs = dict(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        r1 = _svc().evaluate_window(**kwargs)
        r2 = _svc().evaluate_window(**kwargs)
        assert r1.snapshot.snapshot_hash == r2.snapshot.snapshot_hash

    def test_metrics_json_contains_auxiliary_fields(self):
        paper_ledger, eval_ledger, qual_record, _ = _setup_ledgers()
        result = _svc().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual_record.record_hash,
            qualification_record=qual_record,
            window_start_ts=T_START,
            window_end_ts=T_CUTOFF,
            monitoring_config=_make_config(),
            paper_ledger=paper_ledger,
            evaluation_ledger=eval_ledger,
        )
        aux = json.loads(result.snapshot.metrics_json)
        assert "submitted_orders" in aux
        assert "rejected_orders" in aux
        assert "completed_sessions" in aux
        assert "consecutive_inactive_sessions" in aux
        assert "peak_gross_exposure" in aux
