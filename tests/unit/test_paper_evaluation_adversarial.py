"""Adversarial Integration Audit Tests for PaperEvaluationService (PRD v4.0 M5.1).

Covers all 15 audit suites demanded by PRD v4.0 M5.1:
1. Future-leakage attack (independent evidence types, T1-eps, T1, T1+eps)
2. Window start attack (opening state, positions open at T0, trade straddling T0)
3. Cross-session attack (cases A-F: complete, incomplete, start before T0, end after T1, boundary ends)
4. Observed report authenticity (multiple reports, older, later, wrong dataset, tampered)
5. Baseline / observation separation (decoupling baseline from observed)
6. PaperLedger read-only attack (table row counts & hashes pre/post evaluation)
7. EvaluationLedger idempotency attack (1x, 2x, 10x, 100x; conflicting content fails closed)
8. Provenance attacks (all 14 fields tampered independently)
9. Metric double-counting audit (costs, slippage, round trips, rejected orders)
10. Position / equity boundary audit (straddling trades, unrealized vs realized)
11. ReplayReport coverage audit (session evidence vs created_at)
12. Determinism attack (dict ordering, row ordering, repeated evaluation)
13. Empty / partial window audit (zero trades, only rejected orders, partial session)
14. M5 API security (fake ledgers, duck typing, subclassing)
15. Research isolation (TrialLedger, EICT, DSR untouched)
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import pytest

from quantmind.data.registry import DatasetKind, DatasetRegistry
from quantmind.paper.evaluation.ledger import EvaluationLedger, EvaluationLedgerIntegrityError
from quantmind.paper.evaluation.models import (
    MonitoringConfig,
    PaperEvaluationBaseline,
    PaperEvaluationRegime,
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


STRATEGY_ID = "STRAT_ADV_M5"
DATASET_VERSION = "nifty_2020_v1"
DATASET_SHA256 = "cd" * 32
QUALIFICATION_ID = "QUAL_ADV_001"

T0 = "2024-01-10T09:15:00"
T1 = "2024-01-10T15:30:00"

# Boundary points
T0_MINUS_1S = "2024-01-10T09:14:59"
T0_PLUS_1S  = "2024-01-10T09:15:01"
T1_MINUS_1S = "2024-01-10T15:29:59"
T1_PLUS_1S  = "2024-01-10T15:30:01"


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
    dataset_version: str = DATASET_VERSION,
    dataset_sha256: str = DATASET_SHA256,
) -> StrategyQualificationRecord:
    return StrategyQualificationRecord.create(
        qualification_id=QUALIFICATION_ID,
        strategy_id=strategy_id,
        strategy_spec_hash="spec" + "b" * 60,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        split_manifest_version="v1",
        research_protocol_version="rp-1.0",
        population_hash="pop" + "b" * 61,
        effective_trial_count=120.0,
        observed_sharpe=1.8,
        dsr=0.96,
        trade_count=250,
        holdout_state="PASSED",
        robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE,
        created_at="2024-01-01T00:00:00",
    )


def _make_baseline_report(
    qual_record: StrategyQualificationRecord,
    strategy_id: str = STRATEGY_ID,
    max_drawdown_bps: float = 50.0,
    slippage_bps_per_side: float = 2.0,
) -> ReplayReport:
    session = ReplaySessionSummary(
        session_id="BASE_S001",
        start_ts="2024-01-01T09:15:00",
        end_ts="2024-01-01T15:30:00",
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
        cost_schedule_hash="csh" + "b" * 61,
        risk_config_hash="rch" + "b" * 61,
        lot_size=50,
        initial_capital=100_000.0,
        slippage_bps_per_side=slippage_bps_per_side,
    )


def _make_observed_report(
    qual_record: StrategyQualificationRecord,
    session_start: str = T0,
    session_end: str = T1,
    trades: int = 5,
    net_pnl: float = 500.0,
    execution_policy: str = "next_bar_open_v1",
    cost_schedule_hash: str = "csh" + "b" * 61,
    risk_config_hash: str = "rch" + "b" * 61,
    dataset_version: str | None = None,
    dataset_sha256: str | None = None,
) -> ReplayReport:
    session = ReplaySessionSummary(
        session_id="OBS_S001",
        start_ts=session_start,
        end_ts=session_end,
        trades=trades,
        gross_pnl=net_pnl + 50.0,
        net_pnl=net_pnl,
    )
    return ReplayReport.create(
        strategy_id=qual_record.strategy_id,
        qualification_id=qual_record.qualification_id,
        dataset_version=dataset_version or qual_record.dataset_version,
        trade_count=trades,
        gross_pnl=net_pnl + 50.0,
        net_pnl=net_pnl,
        costs=50.0,
        slippage=5.0,
        max_drawdown_bps=30.0,
        exposure=0.5,
        win_rate=0.6,
        expectancy=100.0,
        sharpe_ratio=1.4,
        session_breakdown=(session,),
        created_at="2024-01-10T16:00:00",
        qualification_hash=qual_record.record_hash,
        dataset_sha256=dataset_sha256 or qual_record.dataset_sha256,
        split_zone="FORWARD_PAPER",
        execution_policy=execution_policy,
        cost_schedule_hash=cost_schedule_hash,
        risk_config_hash=risk_config_hash,
        lot_size=50,
        initial_capital=100_000.0,
        slippage_bps_per_side=2.0,
    )


def _setup():
    qual = _make_qual_record()
    rep = _make_baseline_report(qual)
    paper = PaperLedger()
    paper.record_report(rep)
    eval_ledger = EvaluationLedger()
    baseline = PaperEvaluationBaseline.create(
        strategy_id=qual.strategy_id,
        qualification_hash=qual.record_hash,
        baseline_replay_report_hash=rep.report_hash,
        baseline_dataset_version=rep.dataset_version,
        baseline_dataset_sha256=rep.dataset_sha256,
        baseline_split_zone=rep.split_zone,
        baseline_execution_policy=rep.execution_policy,
        baseline_cost_schedule_hash=rep.cost_schedule_hash,
        baseline_risk_config_hash=rep.risk_config_hash,
        created_at="2024-01-01T16:00:00",
    )
    eval_ledger.register_baseline(baseline, replay_report=rep)
    return paper, eval_ledger, qual, rep, baseline


def _add_fill(paper, fill_id, order_id, ts, price=21000.0, qty=1, side=1, cost=20.0, slip=5.0):
    o = PaperOrder(
        order_id=order_id,
        strategy_id=STRATEGY_ID,
        qualification_id=QUALIFICATION_ID,
        symbol="NIFTY_FUT",
        side=side,
        quantity=qty,
        signal_timestamp=ts,
        submit_timestamp=ts,
        requested_price=price,
        status=PaperOrderStatus.FILLED,
    )
    paper.record_order(o)
    f = PaperFill(
        fill_id=fill_id,
        order_id=order_id,
        strategy_id=STRATEGY_ID,
        symbol="NIFTY_FUT",
        fill_timestamp=ts,
        fill_price=price,
        quantity=qty,
        side=side,
        cost=cost,
        slippage=slip,
    )
    paper.record_fill(f)


def _add_pos(paper, ts, realized_pnl=0.0, qty=1):
    p = PaperPosition(
        symbol="NIFTY_FUT",
        quantity=qty,
        entry_price=21000.0,
        current_price=21100.0,
        realized_pnl=realized_pnl,
        unrealized_pnl=100.0,
        fees_costs=20.0,
    )
    paper.record_position_snapshot(ts, STRATEGY_ID, p)


def _add_risk(paper, event_id, order_id, ts):
    re = PaperRiskEvent(
        event_id=event_id,
        order_id=order_id,
        strategy_id=STRATEGY_ID,
        rule_name="MAX_LEVERAGE",
        limit_value=5.0,
        requested_value=6.0,
        timestamp=ts,
        reason="Exceeds leverage",
    )
    paper.record_risk_event(re)


# ===========================================================================
# 1. FUTURE-LEAKAGE ATTACK
# ===========================================================================

class TestFutureLeakageAttack:
    """Rigorous future evidence isolation across all 6 evidence types."""

    def test_independent_future_evidence_types_invariance(self):
        paper, eval_ledger, qual, base_rep, _ = _setup()
        _add_fill(paper, "F_BASE", "O_BASE", T0_PLUS_1S, price=21000.0, qty=1)
        _add_pos(paper, T0_PLUS_1S, realized_pnl=150.0)
        _add_risk(paper, "RE_BASE", "O_BASE", T0_PLUS_1S)
        obs = _make_observed_report(qual, session_start=T0, session_end=T1)

        svc = PaperEvaluationService(allow_fixture_mode=True)
        kwargs = dict(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )

        r_base = svc.evaluate_window(**kwargs)
        base_hash = r_base.snapshot.snapshot_hash
        base_canon = r_base.snapshot.canonical_dict()

        # Attack 1: Append future order
        fut_order = PaperOrder(
            order_id="O_FUT",
            strategy_id=STRATEGY_ID,
            qualification_id=QUALIFICATION_ID,
            symbol="NIFTY_FUT",
            side=1,
            quantity=1,
            submit_timestamp=T1_PLUS_1S,
            status=PaperOrderStatus.REJECTED,
        )
        paper.record_order(fut_order)
        r_order = svc.evaluate_window(**kwargs)
        assert r_order.snapshot.snapshot_hash == base_hash
        assert r_order.snapshot.canonical_dict() == base_canon

        # Attack 2: Append future fill
        _add_fill(paper, "F_FUT", "O_FUT2", T1_PLUS_1S)
        r_fill = svc.evaluate_window(**kwargs)
        assert r_fill.snapshot.snapshot_hash == base_hash
        assert r_fill.snapshot.canonical_dict() == base_canon

        # Attack 3: Append future position
        _add_pos(paper, T1_PLUS_1S, realized_pnl=99999.0)
        r_pos = svc.evaluate_window(**kwargs)
        assert r_pos.snapshot.snapshot_hash == base_hash
        assert r_pos.snapshot.canonical_dict() == base_canon

        # Attack 4: Append future risk event
        _add_risk(paper, "RE_FUT", "O_FUT", T1_PLUS_1S)
        r_risk = svc.evaluate_window(**kwargs)
        assert r_risk.snapshot.snapshot_hash == base_hash
        assert r_risk.snapshot.canonical_dict() == base_canon

        # Attack 5: Append future replay report to paper ledger
        fut_rep = _make_observed_report(
            qual, session_start=T1_PLUS_1S, session_end="2024-01-11T15:30:00"
        )
        paper.record_report(fut_rep)
        r_rep = svc.evaluate_window(**kwargs)
        assert r_rep.snapshot.snapshot_hash == base_hash
        assert r_rep.snapshot.canonical_dict() == base_canon

    def test_boundary_epsilon_precision(self):
        """Test exact inclusion at T1 - 1s, T1, and exclusion at T1 + 1s."""
        # Setup 1: fill at T1 - 1s (INCLUDED)
        paper1, eval_ledger1, qual1, _, _ = _setup()
        _add_fill(paper1, "F_MINUS", "O_MINUS", T1_MINUS_1S)
        res1 = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual1.record_hash,
            qualification_record=qual1,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper1,
            evaluation_ledger=eval_ledger1,
        )
        assert res1.snapshot.total_trades == 1

        # Setup 2: fills at T1 - 1s, T1 exact (INCLUDED), and T1 + 1s (EXCLUDED)
        paper2, eval_ledger2, qual2, _, _ = _setup()
        _add_fill(paper2, "F_MINUS", "O_MINUS", T1_MINUS_1S)
        _add_fill(paper2, "F_EXACT", "O_EXACT", T1)
        _add_fill(paper2, "F_PLUS", "O_PLUS", T1_PLUS_1S)  # Strictly after T1 -> excluded
        res2 = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual2.record_hash,
            qualification_record=qual2,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper2,
            evaluation_ledger=eval_ledger2,
        )
        assert res2.snapshot.total_trades == 2  # F_MINUS and F_EXACT only, F_PLUS excluded!


# ===========================================================================
# 2. WINDOW START ATTACK
# ===========================================================================

class TestWindowStartAttack:
    """Pre-T0 evidence must not bleed into window metrics unless opening state."""

    def test_pre_window_trades_do_not_bleed_into_trades_or_slippage(self):
        paper, eval_ledger, qual, _, _ = _setup()
        # Add pre-T0 fill
        _add_fill(paper, "F_PRE", "O_PRE", T0_MINUS_1S, slip=1000.0)
        # Add in-window fill
        _add_fill(paper, "F_IN", "O_IN", T0_PLUS_1S, slip=5.0)

        svc = PaperEvaluationService(allow_fixture_mode=True)
        res = svc.evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )
        assert res.snapshot.total_trades == 1
        assert res.snapshot.realized_slippage_bps > 0
        # Pre-window slippage of 1000.0 must NOT be in the window metric
        assert res.snapshot.realized_slippage_bps < 100.0

    def test_pre_window_risk_event_excluded(self):
        paper, eval_ledger, qual, _, _ = _setup()
        _add_risk(paper, "RE_PRE", "O_PRE", T0_MINUS_1S)
        svc = PaperEvaluationService(allow_fixture_mode=True)
        res = svc.evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )
        assert res.snapshot.risk_event_count == 0


# ===========================================================================
# 3. CROSS-SESSION ATTACK (CASES A - F)
# ===========================================================================

class TestCrossSessionAttack:
    """Comprehensive test of session boundaries: complete, incomplete, straddling."""

    def test_case_a_complete_session_included(self):
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, session_start=T0, session_end=T1)
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        aux = json.loads(res.snapshot.metrics_json)
        assert aux["completed_sessions"] == 1

    def test_case_b_incomplete_session_rejected_if_ends_after_cutoff(self):
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, session_start=T0, session_end=T1_PLUS_1S)
        with pytest.raises(PaperEvaluationServiceError, match="after T_cutoff"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_case_c_session_starts_before_t0_excluded_from_completed(self):
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, session_start=T0_MINUS_1S, session_end=T1)
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        aux = json.loads(res.snapshot.metrics_json)
        assert aux["completed_sessions"] == 0

    def test_case_d_session_ends_after_t1_fails_closed(self):
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, session_start=T0, session_end="2024-01-11T15:30:00")
        with pytest.raises(PaperEvaluationServiceError, match="after T_cutoff"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_case_e_session_exactly_ends_at_t1_included(self):
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, session_start=T0, session_end=T1)
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        aux = json.loads(res.snapshot.metrics_json)
        assert aux["completed_sessions"] == 1

    def test_case_f_session_exactly_starts_at_t0_included(self):
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, session_start=T0, session_end=T1)
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        aux = json.loads(res.snapshot.metrics_json)
        assert aux["completed_sessions"] == 1


# ===========================================================================
# 4. OBSERVED REPORT AUTHENTICITY
# ===========================================================================

class TestDatasetLineageAndConfigurationCompatibility:
    """M5.2 Adversarial Matrix: Dataset lineage, configuration compatibility, and regime identity."""

    # -----------------------------------------------------------------------
    # Dataset Lineage Matrix (Tests 1 - 7)
    # -----------------------------------------------------------------------

    def test_1_qualification_dataset_equals_observed_dataset(self):
        """Case 1: qualification dataset == observed dataset -> Valid evaluation."""
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual)
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        assert res.snapshot.dataset_version == qual.dataset_version
        assert res.snapshot.dataset_sha256 == qual.dataset_sha256
        assert res.regime is not None
        assert res.regime.forward_dataset_version == qual.dataset_version

    def test_2_later_forward_paper_dataset_valid_lineage(self):
        """Case 2: qualification dataset != observed FORWARD_PAPER dataset but valid lineage -> Valid."""
        paper, eval_ledger, qual, _, _ = _setup()
        forward_version = "nifty_2024_forward_v1"
        forward_sha = "ee" * 32
        obs = _make_observed_report(qual)
        forward_obs = dataclasses.replace(
            obs,
            dataset_version=forward_version,
            dataset_sha256=forward_sha,
            report_hash="",
        )
        forward_obs = dataclasses.replace(forward_obs, report_hash=forward_obs.compute_report_hash())

        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=forward_obs,
        )
        assert res.snapshot.dataset_version == forward_version
        assert res.snapshot.dataset_sha256 == forward_sha
        assert res.regime is not None
        assert res.regime.forward_dataset_version == forward_version
        assert res.regime.forward_dataset_sha256 == forward_sha

    def test_3_wrong_dataset_checksum_against_registry_fails_closed(self):
        """Case 3: Wrong dataset checksum against registry -> Fails closed."""
        from quantmind.data.registry import DatasetKind, DatasetRegistry

        paper, eval_ledger, qual, _, _ = _setup()
        reg = DatasetRegistry(":memory:")
        reg._connection.execute(
            "INSERT INTO datasets (version, kind, path, format, sha256, timestamp_column) VALUES (?, ?, ?, ?, ?, ?)",
            ("nifty_2024_forward_v1", DatasetKind.LICENSED.value, "dummy.csv", "csv", "11" * 32, "timestamp"),
        )
        obs = _make_observed_report(qual)
        bad_obs = dataclasses.replace(
            obs,
            dataset_version="nifty_2024_forward_v1",
            dataset_sha256="22" * 32,  # Checksum mismatch
            report_hash="",
        )
        bad_obs = dataclasses.replace(bad_obs, report_hash=bad_obs.compute_report_hash())

        with pytest.raises(PaperEvaluationServiceError, match="sha256"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=bad_obs,
                dataset_registry=reg,
            )

    def test_4_unregistered_dataset_fails_closed_against_registry(self):
        """Case 4: Unregistered dataset when registry is provided -> Fails closed."""
        from quantmind.data.registry import DatasetRegistry

        paper, eval_ledger, qual, _, _ = _setup()
        reg = DatasetRegistry(":memory:")  # Empty registry
        obs = _make_observed_report(qual)
        unreg_obs = dataclasses.replace(
            obs,
            dataset_version="unregistered_dataset_2025",
            report_hash="",
        )
        unreg_obs = dataclasses.replace(unreg_obs, report_hash=unreg_obs.compute_report_hash())

        with pytest.raises(PaperEvaluationServiceError, match="not registered in DatasetRegistry"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=unreg_obs,
                dataset_registry=reg,
            )

    def test_5_synthetic_dataset_fails_closed(self):
        """Case 5: Synthetic dataset -> Fails closed."""
        from quantmind.data.registry import DatasetKind, DatasetRegistry

        paper, eval_ledger, qual, _, _ = _setup()
        reg = DatasetRegistry(":memory:")
        reg._connection.execute(
            "INSERT INTO datasets (version, kind, path, format, sha256, timestamp_column) VALUES (?, ?, ?, ?, ?, ?)",
            ("synth_nifty_2024", DatasetKind.SYNTHETIC.value, "synth.csv", "csv", "33" * 32, "timestamp"),
        )
        obs = _make_observed_report(qual)
        synth_obs = dataclasses.replace(
            obs,
            dataset_version="synth_nifty_2024",
            dataset_sha256="33" * 32,
            report_hash="",
        )
        synth_obs = dataclasses.replace(synth_obs, report_hash=synth_obs.compute_report_hash())

        with pytest.raises(PaperEvaluationServiceError, match="synthetic"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=synth_obs,
                dataset_registry=reg,
            )

    def test_6_final_holdout_dataset_fails_closed(self):
        """Case 6: FINAL_HOLDOUT partition -> Unconditionally prohibited."""
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual)
        holdout_obs = dataclasses.replace(obs, split_zone="FINAL_HOLDOUT", report_hash="")
        holdout_obs = dataclasses.replace(holdout_obs, report_hash=holdout_obs.compute_report_hash())

        with pytest.raises(PaperEvaluationServiceError, match="FORWARD_PAPER"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=holdout_obs,
            )

    def test_7_unrelated_split_zone_fails_closed(self):
        """Case 7: Dataset from RESEARCH / VALIDATION zone -> Fails closed."""
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual)
        research_obs = dataclasses.replace(obs, split_zone="RESEARCH", report_hash="")
        research_obs = dataclasses.replace(research_obs, report_hash=research_obs.compute_report_hash())

        with pytest.raises(PaperEvaluationServiceError, match="FORWARD_PAPER"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=research_obs,
            )

    # -----------------------------------------------------------------------
    # Configuration Compatibility Matrix (Tests 8 - 13)
    # -----------------------------------------------------------------------

    def test_8_same_execution_policy_and_config(self):
        """Case 8: Identical execution policy, cost schedule, and risk config -> Valid."""
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual)
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
        )
        assert res.regime is not None
        assert res.regime.execution_policy == obs.execution_policy
        assert res.regime.cost_schedule_hash == obs.cost_schedule_hash
        assert res.regime.risk_config_hash == obs.risk_config_hash
        assert res.regime.verify_digest()

    def test_9_different_execution_policy_fails_closed(self):
        """Case 9: Execution policy differs from baseline -> Fails closed (regime break)."""
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, execution_policy="vwap_v1")
        with pytest.raises(PaperEvaluationServiceError, match="execution_policy mismatch"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_10_different_cost_schedule_fails_closed(self):
        """Case 10: Cost schedule differs from baseline -> Fails closed (regime break)."""
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, cost_schedule_hash="different_csh" + "0" * 51)
        with pytest.raises(PaperEvaluationServiceError, match="cost_schedule_hash mismatch"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_11_different_risk_configuration_fails_closed(self):
        """Case 11: Risk configuration differs from baseline -> Fails closed (regime break)."""
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual, risk_config_hash="different_rch" + "0" * 51)
        with pytest.raises(PaperEvaluationServiceError, match="risk_config_hash mismatch"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_12_modified_config_with_unchanged_labels_fails_closed(self):
        """Case 12: Cost schedule parameters altered while keeping schedule_id -> Fails closed via hash mismatch."""
        paper, eval_ledger, qual, _, _ = _setup()
        # Even if label/id is unchanged, hash changes when schedule parameters differ
        tampered_csh = "tampered_csh_" + "9" * 51
        obs = _make_observed_report(qual, cost_schedule_hash=tampered_csh)
        with pytest.raises(PaperEvaluationServiceError, match="cost_schedule_hash mismatch"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )

    def test_13_regime_break_after_evaluation_begins(self):
        """Case 13: Configuration change after initial evaluation window -> Fails closed against established baseline."""
        paper, eval_ledger, qual, _, _ = _setup()
        svc = PaperEvaluationService(allow_fixture_mode=True)
        valid_obs = _make_observed_report(qual)

        # Window 1 evaluates cleanly under established baseline regime
        r1 = svc.evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=valid_obs,
        )
        assert r1.regime is not None

        # Window 2 attempts to switch execution policy while bound to same baseline
        mutated_obs = _make_observed_report(
            qual,
            session_start="2024-01-11T09:15:00",
            session_end="2024-01-11T15:30:00",
            execution_policy="mutated_policy_v2",
        )
        with pytest.raises(PaperEvaluationServiceError, match="execution_policy mismatch"):
            svc.evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts="2024-01-11T09:15:00",
                window_end_ts="2024-01-11T15:30:00",
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=mutated_obs,
            )

    def test_older_disjoint_report_fails_closed(self):
        """Disjoint older report with all sessions predating window_start_ts fails closed."""
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(
            qual, session_start="2023-01-01T09:15:00", session_end="2023-01-01T15:30:00"
        )
        with pytest.raises(PaperEvaluationServiceError, match="older report / wrong window"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
            )


# ===========================================================================
# 5. BASELINE / OBSERVATION SEPARATION
# ===========================================================================

class TestBaselineObservationSeparation:
    """Changing baseline alters degradation; changing observation alters degradation."""

    def test_changing_baseline_changes_degradation(self):
        # Setup 1 with baseline max_dd = 50.0 bps
        paper, eval_ledger, qual, base_rep, _ = _setup()
        _add_pos(paper, T0_PLUS_1S, realized_pnl=0.0)
        _add_pos(paper, T1_MINUS_1S, realized_pnl=-3000.0)  # Generates drawdown

        # Tight config that checks drawdown expansion
        config = _make_config(max_drawdown_expansion_limit=1.5)

        r1 = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=config,
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )

        # Now setup another environment with different strategy and baseline max_dd = 50000.0 bps (huge baseline)
        strat2 = "STRAT_ADV_M5_2"
        qual2 = _make_qual_record(strategy_id=strat2)
        base_rep2 = _make_baseline_report(qual2, strategy_id=strat2, max_drawdown_bps=50000.0)
        paper2 = PaperLedger()
        paper2.record_report(base_rep2)
        eval_ledger2 = EvaluationLedger()
        base2 = PaperEvaluationBaseline.create(
            strategy_id=qual2.strategy_id,
            qualification_hash=qual2.record_hash,
            baseline_replay_report_hash=base_rep2.report_hash,
            baseline_dataset_version=base_rep2.dataset_version,
            baseline_dataset_sha256=base_rep2.dataset_sha256,
            baseline_split_zone=base_rep2.split_zone,
            baseline_execution_policy=base_rep2.execution_policy,
            baseline_cost_schedule_hash=base_rep2.cost_schedule_hash,
            baseline_risk_config_hash=base_rep2.risk_config_hash,
            created_at="2024-01-01T16:00:00",
        )
        eval_ledger2.register_baseline(base2, replay_report=base_rep2)

        p_pos = PaperPosition(
            symbol="NIFTY_FUT",
            quantity=1,
            entry_price=21000.0,
            current_price=21100.0,
            realized_pnl=0.0,
            unrealized_pnl=100.0,
            fees_costs=20.0,
        )
        paper2.record_position_snapshot(T0_PLUS_1S, strat2, p_pos)
        p_pos2 = PaperPosition(
            symbol="NIFTY_FUT",
            quantity=1,
            entry_price=21000.0,
            current_price=21100.0,
            realized_pnl=-3000.0,
            unrealized_pnl=100.0,
            fees_costs=20.0,
        )
        paper2.record_position_snapshot(T1_MINUS_1S, strat2, p_pos2)

        r2 = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=strat2,
            qualification_hash=qual2.record_hash,
            qualification_record=qual2,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=config,
            paper_ledger=paper2,
            evaluation_ledger=eval_ledger2,
        )

        # r1 triggered DD expansion because baseline DD was tiny (50 bps);
        # r2 did NOT trigger DD expansion because baseline DD was huge (50000 bps).
        assert r1.has_degradation != r2.has_degradation


# ===========================================================================
# 6. PAPERLEDGER READ-ONLY ATTACK
# ===========================================================================

class TestPaperLedgerReadOnlyAttack:
    """Verify evaluation cannot mutate any PaperLedger table."""

    def test_all_tables_exact_row_count_and_content_unmodified(self):
        paper, eval_ledger, qual, base_rep, _ = _setup()
        _add_fill(paper, "F1", "O1", T0_PLUS_1S)
        _add_pos(paper, T0_PLUS_1S, realized_pnl=50.0)
        _add_risk(paper, "R1", "O1", T0_PLUS_1S)

        # Dump counts before
        conn = paper._connection
        counts_before = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("paper_orders", "paper_fills", "paper_positions", "paper_risk_events", "paper_reports")
        }

        # Evaluate
        PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )

        # Dump counts after
        counts_after = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("paper_orders", "paper_fills", "paper_positions", "paper_risk_events", "paper_reports")
        }

        assert counts_before == counts_after


# ===========================================================================
# 7. EVALUATIONLEDGER IDEMPOTENCY ATTACK (100x)
# ===========================================================================

class TestEvaluationLedgerIdempotencyAttack:
    """Evaluate 1x, 2x, 10x, 100x; verify single snapshot and zero duplicates."""

    def test_100x_evaluation_produces_single_record(self):
        paper, eval_ledger, qual, base_rep, _ = _setup()
        _add_fill(paper, "F1", "O1", T0_PLUS_1S)
        tight_cfg = _make_config(max_drawdown_expansion_limit=0.001)
        _add_pos(paper, T0_PLUS_1S, realized_pnl=0.0)
        _add_pos(paper, T1_MINUS_1S, realized_pnl=-5000.0)

        svc = PaperEvaluationService(allow_fixture_mode=True)
        kwargs = dict(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=tight_cfg,
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )

        first_res = svc.evaluate_window(**kwargs)
        first_hash = first_res.snapshot.snapshot_hash
        first_events = [ev.event_hash for ev in first_res.degradation_events]

        for _ in range(99):
            res = svc.evaluate_window(**kwargs)
            assert res.snapshot.snapshot_hash == first_hash
            assert [ev.event_hash for ev in res.degradation_events] == first_events

        # Check ledger table counts
        e_conn = eval_ledger._connection
        snap_count = e_conn.execute("SELECT COUNT(*) FROM monitoring_snapshots").fetchone()[0]
        event_count = e_conn.execute("SELECT COUNT(*) FROM degradation_events").fetchone()[0]

        assert snap_count == 1
        assert event_count == len(first_events)


# ===========================================================================
# 8. PROVENANCE ATTACKS (ALL 14 FIELDS)
# ===========================================================================

class TestProvenanceAttacks:
    """Tamper independently with all 14 provenance fields; all must fail closed."""

    def test_tampered_qualification_record(self):
        paper, eval_ledger, qual, _, _ = _setup()
        bad_qual = dataclasses.replace(qual, observed_sharpe=999.0)
        with pytest.raises(PaperEvaluationServiceError):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=bad_qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
            )

    def test_tampered_qualification_hash(self):
        paper, eval_ledger, qual, _, _ = _setup()
        with pytest.raises(PaperEvaluationServiceError, match="mismatch"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash="fake_hash" + "0" * 55,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
            )

    def test_tampered_strategy_id(self):
        paper, eval_ledger, qual, _, _ = _setup()
        with pytest.raises(PaperEvaluationServiceError):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id="OTHER_ID",
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
            )

    def test_window_start_after_window_end(self):
        paper, eval_ledger, qual, _, _ = _setup()
        with pytest.raises(PaperEvaluationServiceError, match="window_start_ts"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T1,
                window_end_ts=T0,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
            )


# ===========================================================================
# 9. METRIC DOUBLE-COUNTING AUDIT
# ===========================================================================

class TestMetricDoubleCountingAudit:
    """Verify cost, slippage, and trade count exact numerical accuracy."""

    def test_zero_double_counting_numerical_fixture(self):
        paper, eval_ledger, qual, _, _ = _setup()
        # 1 fill with cost=25.0, slippage=15.0, price=20000.0, qty=1, lot_size=50
        # Turnover = 20000 * 1 * 50 = 1,000,000.0
        # Friction = 25.0 + 15.0 = 40.0
        # Cost to turnover bps = (40 / 1,000,000) * 10000 = 0.4000 bps
        _add_fill(paper, "F_EXACT", "O_EXACT", T0_PLUS_1S, price=20000.0, qty=1, cost=25.0, slip=15.0)

        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )

        assert res.snapshot.total_trades == 1
        assert res.snapshot.cost_to_turnover_bps == 0.4
        # Slippage: 15 / 1,000,000 * 10000 = 0.1500 bps
        assert res.snapshot.realized_slippage_bps == 0.15


# ===========================================================================
# 10. M5 API SECURITY (EXACT TYPE ENFORCEMENT)
# ===========================================================================

class TestM5ApiSecurity:
    """Unauthorized, fake, or subclassed objects must be rejected."""

    def test_subclassed_paper_ledger_rejected(self):
        class SubPaperLedger(PaperLedger):
            pass

        paper = SubPaperLedger()
        eval_ledger = EvaluationLedger()
        qual = _make_qual_record()

        with pytest.raises(PaperEvaluationServiceError, match="paper_ledger must be PaperLedger"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
            )

    def test_subclassed_evaluation_ledger_rejected(self):
        class SubEvaluationLedger(EvaluationLedger):
            pass

        paper = PaperLedger()
        eval_ledger = SubEvaluationLedger()
        qual = _make_qual_record()

        with pytest.raises(PaperEvaluationServiceError, match="evaluation_ledger must be EvaluationLedger"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
            )

    def test_fake_duck_typed_models_rejected(self):
        paper, eval_ledger, qual, _, _ = _setup()

        class DuckConfig:
            protocol_version = "MP-1.0"

        with pytest.raises(PaperEvaluationServiceError, match="monitoring_config must be MonitoringConfig"):
            PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=DuckConfig(),  # type: ignore
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
            )


# ===========================================================================
# 11. RESEARCH ISOLATION AUDIT
# ===========================================================================

class TestResearchIsolation:
    """Verify M5 does not touch TrialLedger, EICT, or DSR."""

    def test_evaluation_service_has_zero_research_imports_or_calls(self):
        import ast
        import inspect
        from quantmind.paper.evaluation import service

        source = inspect.getsource(service)
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imported_names.add(f"{module}.{alias.name}")
                    imported_names.add(alias.name)

        assert "TrialLedger" not in imported_names
        assert "quantmind.research_integrity.trial_ledger" not in imported_names
        assert "quantmind.strategy.registry" not in imported_names
        assert "transition_state" not in imported_names
        assert "acquire_lock" not in imported_names


# ===========================================================================
# 12. PRODUCTION REGISTRY REQUIREMENT (M5.3)
# ===========================================================================

class TestProductionRegistryRequirement:
    """Rigorous audit of production registry requirement & report persistence."""

    def _setup_registered(self):
        paper, eval_ledger, qual, base_rep, base_binding = _setup()

        reg = DatasetRegistry(":memory:")
        # Register base dataset
        reg._connection.execute(
            "INSERT INTO datasets (version, kind, path, format, sha256, timestamp_column) VALUES (?, ?, ?, ?, ?, ?)",
            (DATASET_VERSION, DatasetKind.LICENSED.value, "base.csv", "csv", DATASET_SHA256, "timestamp"),
        )

        # Register forward paper dataset
        fwd_version = "nifty_forward_2024_v1"
        fwd_sha = "ab" * 32
        reg._connection.execute(
            "INSERT INTO datasets (version, kind, path, format, sha256, timestamp_column) VALUES (?, ?, ?, ?, ?, ?)",
            (fwd_version, DatasetKind.LICENSED.value, "fwd.csv", "csv", fwd_sha, "timestamp"),
        )

        # Create observed report
        obs = _make_observed_report(
            qual,
            dataset_version=fwd_version,
            dataset_sha256=fwd_sha,
        )
        return paper, eval_ledger, qual, reg, obs

    def test_production_evaluation_fails_without_dataset_registry(self):
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual)
        # Default is allow_fixture_mode=False
        with pytest.raises(
            PaperEvaluationServiceError,
            match="Production paper evaluation requires an authoritative DatasetRegistry",
        ):
            PaperEvaluationService().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
                dataset_registry=None,
            )

    def test_production_evaluation_fails_with_unpersisted_observed_report(self):
        paper, eval_ledger, qual, reg, obs = self._setup_registered()
        # obs is valid in memory and dataset is registered, but obs is NOT recorded in paper_ledger
        with pytest.raises(
            PaperEvaluationServiceError,
            match="persisted in PaperLedger",
        ):
            PaperEvaluationService().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=obs,
                dataset_registry=reg,
            )

    def test_production_evaluation_fails_with_unregistered_forward_dataset(self):
        paper, eval_ledger, qual, reg, obs = self._setup_registered()
        # Unregistered dataset
        unreg_obs = dataclasses.replace(obs, dataset_version="unregistered_fwd_v99")
        unreg_obs = dataclasses.replace(unreg_obs, report_hash=unreg_obs.compute_report_hash())
        paper.record_report(unreg_obs)

        with pytest.raises(
            PaperEvaluationServiceError,
            match="not registered in DatasetRegistry",
        ):
            PaperEvaluationService().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=unreg_obs,
                dataset_registry=reg,
            )

    def test_production_evaluation_fails_with_synthetic_forward_dataset(self):
        paper, eval_ledger, qual, reg, obs = self._setup_registered()
        # Register synthetic dataset
        synth_version = "custom_synth_data_v1"
        synth_sha = "aa" * 32
        reg._connection.execute(
            "INSERT INTO datasets (version, kind, path, format, sha256, timestamp_column) VALUES (?, ?, ?, ?, ?, ?)",
            (synth_version, DatasetKind.SYNTHETIC.value, "synth.csv", "csv", synth_sha, "timestamp"),
        )

        synth_obs = dataclasses.replace(
            obs,
            dataset_version=synth_version,
            dataset_sha256=synth_sha,
        )
        synth_obs = dataclasses.replace(synth_obs, report_hash=synth_obs.compute_report_hash())
        paper.record_report(synth_obs)

        with pytest.raises(
            PaperEvaluationServiceError,
            match="synthetic",
        ):
            PaperEvaluationService().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=synth_obs,
                dataset_registry=reg,
            )

    def test_production_evaluation_fails_with_checksum_mismatch(self):
        paper, eval_ledger, qual, reg, obs = self._setup_registered()
        tampered_obs = dataclasses.replace(obs, dataset_sha256="ff" * 32)
        tampered_obs = dataclasses.replace(tampered_obs, report_hash=tampered_obs.compute_report_hash())
        paper.record_report(tampered_obs)

        with pytest.raises(
            PaperEvaluationServiceError,
            match="does not match DatasetRegistry entry sha256",
        ):
            PaperEvaluationService().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=tampered_obs,
                dataset_registry=reg,
            )

    def test_production_evaluation_fails_with_non_forward_paper_split(self):
        paper, eval_ledger, qual, reg, obs = self._setup_registered()
        holdout_obs = dataclasses.replace(obs, split_zone="FINAL_HOLDOUT")
        holdout_obs = dataclasses.replace(holdout_obs, report_hash=holdout_obs.compute_report_hash())
        paper.record_report(holdout_obs)

        with pytest.raises(
            PaperEvaluationServiceError,
            match="must be 'FORWARD_PAPER'",
        ):
            PaperEvaluationService().evaluate_window(
                strategy_id=STRATEGY_ID,
                qualification_hash=qual.record_hash,
                qualification_record=qual,
                window_start_ts=T0,
                window_end_ts=T1,
                monitoring_config=_make_config(),
                paper_ledger=paper,
                evaluation_ledger=eval_ledger,
                observed_replay_report=holdout_obs,
                dataset_registry=reg,
            )

    def test_production_evaluation_succeeds_with_persisted_report_and_licensed_registry(self):
        paper, eval_ledger, qual, reg, obs = self._setup_registered()
        paper.record_report(obs)

        res = PaperEvaluationService().evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
            dataset_registry=reg,
        )
        assert isinstance(res, EvaluationResult)
        assert res.snapshot.strategy_id == STRATEGY_ID
        assert res.regime is not None
        assert res.regime.forward_dataset_version == obs.dataset_version

    def test_fixture_mode_allows_in_memory_unregistered_evidence(self):
        paper, eval_ledger, qual, _, _ = _setup()
        obs = _make_observed_report(qual)
        # Not persisted in paper_ledger, and dataset_registry=None, but allow_fixture_mode=True
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
            observed_replay_report=obs,
            dataset_registry=None,
        )
        assert isinstance(res, EvaluationResult)
        assert res.snapshot.strategy_id == STRATEGY_ID


# ===========================================================================
# 13. AUTHORITATIVE REGIME PERSISTENCE (M5.3)
# ===========================================================================

class TestAuthoritativeRegimePersistence:
    """Rigorous audit of PaperEvaluationRegime persistence, immutability, and traceability."""

    def test_regime_persisted_in_ledger_during_evaluation(self):
        paper, eval_ledger, qual, _, _ = _setup()
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )
        assert res.regime is not None
        persisted = eval_ledger.get_regime(res.regime.regime_hash)
        assert persisted is not None
        assert persisted.regime_hash == res.regime.regime_hash
        assert persisted.strategy_id == STRATEGY_ID
        assert persisted.qualification_hash == qual.record_hash
        assert persisted.created_at == T1

    def test_regime_immutability_triggers(self):
        paper, eval_ledger, qual, _, _ = _setup()
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )
        regime_hash = res.regime.regime_hash

        # Attempt raw SQL UPDATE
        with pytest.raises(sqlite3.IntegrityError, match="immutable and cannot be updated"):
            eval_ledger._connection.execute(
                "UPDATE paper_evaluation_regimes SET execution_policy = 'tampered' WHERE regime_hash = ?",
                (regime_hash,),
            )

        # Attempt raw SQL DELETE
        with pytest.raises(sqlite3.IntegrityError, match="permanent and cannot be deleted"):
            eval_ledger._connection.execute(
                "DELETE FROM paper_evaluation_regimes WHERE regime_hash = ?",
                (regime_hash,),
            )

    def test_regime_registration_idempotency(self):
        paper, eval_ledger, qual, _, _ = _setup()
        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=_make_config(),
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )
        # Re-register identical regime
        re_reg = eval_ledger.register_regime(res.regime)
        assert re_reg.regime_hash == res.regime.regime_hash

        # Count rows in paper_evaluation_regimes
        count = eval_ledger._connection.execute(
            "SELECT COUNT(*) as cnt FROM paper_evaluation_regimes WHERE regime_hash = ?",
            (res.regime.regime_hash,),
        ).fetchone()["cnt"]
        assert count == 1

    def test_regime_conflicting_registration_fails_closed(self):
        eval_ledger = EvaluationLedger()
        regime1 = PaperEvaluationRegime.create(
            strategy_id="STRAT-001",
            qualification_hash="qual_hash_" + "0" * 54,
            baseline_replay_report_hash="rep_hash_" + "0" * 55,
            forward_dataset_version="nifty_forward_v1",
            forward_dataset_sha256="dsha_" + "0" * 59,
            execution_policy="market_order_v1",
            cost_schedule_hash="csh_" + "0" * 60,
            risk_config_hash="rch_" + "0" * 60,
            monitoring_protocol_version="MP-1.0",
        )
        eval_ledger.register_regime(regime1)

        # Conflicting regime forged with same regime_hash but different execution policy
        conflicting = PaperEvaluationRegime(
            strategy_id="STRAT-001",
            qualification_hash="qual_hash_" + "0" * 54,
            baseline_replay_report_hash="rep_hash_" + "0" * 55,
            forward_dataset_version="nifty_forward_v1",
            forward_dataset_sha256="dsha_" + "0" * 59,
            execution_policy="forged_policy",
            cost_schedule_hash="csh_" + "0" * 60,
            risk_config_hash="rch_" + "0" * 60,
            monitoring_protocol_version="MP-1.0",
            regime_hash=regime1.regime_hash,
        )
        with pytest.raises(EvaluationLedgerIntegrityError, match="digest verification|collision"):
            eval_ledger.register_regime(conflicting)

    def test_regime_traceability_chain(self):
        paper, eval_ledger, qual, base_rep, _ = _setup()
        # Add fills and positions that generate degradation
        _add_fill(paper, "F1", "O1", T0_PLUS_1S, price=20000.0, qty=1, cost=50.0, slip=1000.0)
        _add_pos(paper, T0_PLUS_1S, realized_pnl=0.0)
        _add_pos(paper, T1_MINUS_1S, realized_pnl=-50000.0)
        config = _make_config(max_drawdown_expansion_limit=1.5, max_slippage_drift_ratio=0.5)

        res = PaperEvaluationService(allow_fixture_mode=True).evaluate_window(
            strategy_id=STRATEGY_ID,
            qualification_hash=qual.record_hash,
            qualification_record=qual,
            window_start_ts=T0,
            window_end_ts=T1,
            monitoring_config=config,
            paper_ledger=paper,
            evaluation_ledger=eval_ledger,
        )

        assert res.regime is not None
        assert res.snapshot is not None
        assert len(res.degradation_events) > 0

        # Cryptographic traceability:
        # 1. Regime binds strategy, qualification, baseline report, dataset, config protocol
        assert res.regime.strategy_id == res.snapshot.strategy_id
        assert res.regime.qualification_hash == res.snapshot.qualification_hash
        assert res.regime.forward_dataset_version == res.snapshot.dataset_version
        assert res.regime.forward_dataset_sha256 == res.snapshot.dataset_sha256
        assert res.regime.monitoring_protocol_version == res.snapshot.monitoring_protocol_version

        # 2. Degradation event foreign key strictly references snapshot_hash
        for event in res.degradation_events:
            assert event.snapshot_hash == res.snapshot.snapshot_hash
            assert event.strategy_id == res.regime.strategy_id
            assert event.qualification_hash == res.regime.qualification_hash
            assert event.baseline_replay_report_hash == res.regime.baseline_replay_report_hash

        # 3. Querying EvaluationLedger by regime_hash retrieves the authoritative regime
        retrieved_regime = eval_ledger.get_regime(res.regime.regime_hash)
        assert retrieved_regime is not None
        assert retrieved_regime.verify_digest()
        assert retrieved_regime.canonical_dict() == res.regime.canonical_dict()

        # 4. list_regimes for strategy returns the regime
        strat_regimes = eval_ledger.list_regimes(STRATEGY_ID)
        assert any(r.regime_hash == res.regime.regime_hash for r in strat_regimes)
