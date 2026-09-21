"""Unit tests for deterministic DegradationDetector (PRD v4.0 Milestone 4).

Tests every degradation rule, boundary conditions, edge cases, fail-closed
provenance verification, and determinism / purity invariants.
"""

from __future__ import annotations

import json
from typing import Any
import pytest

from quantmind.paper.evaluation.detector import (
    DegradationDetector,
    MP1_DROPOUT_WINDOW_SESSIONS,
    RULE_DD_EXPANSION_CRITICAL,
    RULE_RISK_REJECTION_SPIKE,
    RULE_SHARPE_COLLAPSE,
    RULE_SLIPPAGE_ANOMALY,
    RULE_TRADE_DROPOUT,
    detect_degradations,
)
from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringConfig,
    MonitoringProvenanceError,
    MonitoringSnapshot,
)
from quantmind.paper.models import ReplayReport, ReplaySessionSummary
from quantmind.research_integrity.qualification import (
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Factories
# ---------------------------------------------------------------------------


def _make_valid_config(
    protocol_version: str = "MP-1.0",
    max_drawdown_expansion_limit: float = 1.5,
    min_rolling_sharpe_30d: float = 1.0,
    max_slippage_drift_ratio: float = 1.25,
    max_risk_rejection_rate: float = 0.05,
    window_size_sessions: int = 30,
    min_evaluation_trades: int = 10,
) -> MonitoringConfig:
    return MonitoringConfig(
        protocol_version=protocol_version,
        max_drawdown_expansion_limit=max_drawdown_expansion_limit,
        min_rolling_sharpe_30d=min_rolling_sharpe_30d,
        max_slippage_drift_ratio=max_slippage_drift_ratio,
        max_risk_rejection_rate=max_risk_rejection_rate,
        window_size_sessions=window_size_sessions,
        min_evaluation_trades=min_evaluation_trades,
    )


def _make_valid_qualification_record(
    strategy_id: str = "STRAT-M4-TEST",
    final_status: ValidationStatus = ValidationStatus.PAPER_ELIGIBLE,
) -> StrategyQualificationRecord:
    return StrategyQualificationRecord.create(
        qualification_id="QUAL-M4-001",
        strategy_id=strategy_id,
        strategy_spec_hash="spec_hash_1234567890abcdef",
        dataset_version="DS-NIFTY-2026",
        dataset_sha256="dataset_sha_1234567890abcdef",
        split_manifest_version="SPLIT-2026-V1",
        research_protocol_version="RP-2.0",
        population_hash="pop_hash_1234567890abcdef",
        effective_trial_count=12.5,
        observed_sharpe=2.15,
        dsr=0.985,
        trade_count=150,
        holdout_state="UNLOCKED_TEST",
        robustness_status=RobustnessStatus.PASSED,
        final_status=final_status,
        created_at="2026-03-01T00:00:00Z",
    )


def _make_valid_replay_report(
    strategy_id: str = "STRAT-M4-TEST",
    qualification_hash: str = "",
    max_drawdown_bps: float = 500.0,
    slippage_bps_per_side: float = 4.0,
    report_hash: str = "rep_hash_9999999999abcdef",
) -> ReplayReport:
    sessions = (
        ReplaySessionSummary(
            session_id="SESS-001",
            start_ts="2026-03-01T09:15:00Z",
            end_ts="2026-03-01T15:30:00Z",
            trades=5,
            gross_pnl=2500.0,
            net_pnl=2200.0,
        ),
    )
    return ReplayReport.create(
        strategy_id=strategy_id,
        qualification_id="QUAL-M4-001",
        dataset_version="DS-NIFTY-2026",
        trade_count=50,
        gross_pnl=15000.0,
        net_pnl=12000.0,
        costs=1800.0,
        slippage=1200.0,
        max_drawdown_bps=max_drawdown_bps,
        exposure=50000.0,
        win_rate=0.55,
        expectancy=240.0,
        sharpe_ratio=1.85,
        session_breakdown=sessions,
        created_at="2026-03-01T15:30:00Z",
        qualification_hash=qualification_hash,
        slippage_bps_per_side=slippage_bps_per_side,
        split_zone="FORWARD_PAPER",
    )


def _make_valid_snapshot(
    config: MonitoringConfig,
    qual_rec: StrategyQualificationRecord,
    rep_report: ReplayReport | None = None,
    max_drawdown_bps: float = 400.0,
    realized_sharpe: float | None = 1.5,
    realized_slippage_bps: float = 4.5,
    total_trades: int = 25,
    net_pnl: float = 5000.0,
    metrics_json: str = "{}",
    window_end_ts: str = "2026-03-31T15:30:00Z",
) -> MonitoringSnapshot:
    rep_hash = rep_report.report_hash if rep_report else "dummy_rep_hash_12345678"
    return MonitoringSnapshot.create(
        strategy_id=qual_rec.strategy_id,
        qualification_hash=qual_rec.record_hash,
        replay_report_hash=rep_hash,
        dataset_version=qual_rec.dataset_version,
        dataset_sha256=qual_rec.dataset_sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version=config.protocol_version,
        monitoring_config_hash=config.compute_config_hash(),
        window_start_ts="2026-03-01T09:15:00Z",
        window_end_ts=window_end_ts,
        total_trades=total_trades,
        net_pnl=net_pnl,
        max_drawdown_bps=max_drawdown_bps,
        realized_sharpe=realized_sharpe,
        realized_slippage_bps=realized_slippage_bps,
        cost_to_turnover_bps=8.5,
        risk_event_count=0,
        metrics_json=metrics_json,
    )


# ---------------------------------------------------------------------------
# 1. Rule Tests: RULE_DD_EXPANSION_CRITICAL
# ---------------------------------------------------------------------------


class TestDrawdownExpansionRule:
    def test_drawdown_below_threshold_no_event(self) -> None:
        config = _make_valid_config(max_drawdown_expansion_limit=1.5)
        qual = _make_valid_qualification_record()
        # Baseline DD = 500.0 bps, Limit = 1.5 -> Threshold = 750.0 bps
        # Observed DD = 749.9 bps (below)
        snap = _make_valid_snapshot(config, qual, max_drawdown_bps=749.9)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 0

    def test_drawdown_exactly_at_threshold_no_event(self) -> None:
        config = _make_valid_config(max_drawdown_expansion_limit=1.5)
        qual = _make_valid_qualification_record()
        # Baseline DD = 500.0 bps, Limit = 1.5 -> Threshold = 750.0 bps
        # Observed DD = 750.0 bps (strict > required, boundary equality does not trigger)
        snap = _make_valid_snapshot(config, qual, max_drawdown_bps=750.0)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 0

    def test_drawdown_above_threshold_triggers_event(self) -> None:
        config = _make_valid_config(max_drawdown_expansion_limit=1.5)
        qual = _make_valid_qualification_record()
        # Observed DD = 750.01 bps (above 750.0)
        snap = _make_valid_snapshot(config, qual, max_drawdown_bps=750.01)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 1
        ev = events[0]
        assert ev.rule_name == RULE_DD_EXPANSION_CRITICAL
        assert ev.threshold_value == 750.0
        assert ev.observed_value == 750.01
        assert ev.strategy_id == qual.strategy_id
        assert ev.qualification_hash == qual.record_hash
        assert ev.snapshot_hash == snap.snapshot_hash
        assert ev.monitoring_protocol_version == config.protocol_version
        assert ev.monitoring_config_hash == config.compute_config_hash()
        assert ev.timestamp == snap.window_end_ts
        assert ev.verify_digest() is True

    def test_tampered_snapshot_rejected(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual, max_drawdown_bps=800.0)
        # Tamper snapshot net_pnl
        tampered = MonitoringSnapshot(
            strategy_id=snap.strategy_id,
            qualification_hash=snap.qualification_hash,
            replay_report_hash=snap.replay_report_hash,
            dataset_version=snap.dataset_version,
            dataset_sha256=snap.dataset_sha256,
            split_zone=snap.split_zone,
            monitoring_protocol_version=snap.monitoring_protocol_version,
            monitoring_config_hash=snap.monitoring_config_hash,
            window_start_ts=snap.window_start_ts,
            window_end_ts=snap.window_end_ts,
            total_trades=snap.total_trades,
            net_pnl=999999.0,  # tampered
            max_drawdown_bps=snap.max_drawdown_bps,
            realized_sharpe=snap.realized_sharpe,
            realized_slippage_bps=snap.realized_slippage_bps,
            cost_to_turnover_bps=snap.cost_to_turnover_bps,
            risk_event_count=snap.risk_event_count,
            metrics_json=snap.metrics_json,
            created_at=snap.created_at,
            snapshot_hash=snap.snapshot_hash,
        )
        with pytest.raises(MonitoringProvenanceError, match="Snapshot digest verification failed"):
            detect_degradations(tampered, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)

    def test_tampered_config_rejected(self) -> None:
        config = _make_valid_config(max_drawdown_expansion_limit=1.5)
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual, max_drawdown_bps=800.0)
        # Caller supplies a different config with different hash
        other_config = _make_valid_config(max_drawdown_expansion_limit=2.0)
        with pytest.raises(MonitoringProvenanceError, match="Snapshot monitoring_config_hash mismatch"):
            detect_degradations(snap, other_config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)


# ---------------------------------------------------------------------------
# 2. Rule Tests: RULE_SHARPE_COLLAPSE
# ---------------------------------------------------------------------------


class TestSharpeCollapseRule:
    def test_sharpe_below_threshold_triggers_event(self) -> None:
        config = _make_valid_config(min_rolling_sharpe_30d=1.0)
        qual = _make_valid_qualification_record()
        # Observed Sharpe = 0.95 (< 1.0)
        snap = _make_valid_snapshot(config, qual, realized_sharpe=0.95)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 1
        ev = events[0]
        assert ev.rule_name == RULE_SHARPE_COLLAPSE
        assert ev.threshold_value == 1.0
        assert ev.observed_value == 0.95
        assert ev.verify_digest() is True

    def test_sharpe_exactly_at_threshold_no_event(self) -> None:
        config = _make_valid_config(min_rolling_sharpe_30d=1.0)
        qual = _make_valid_qualification_record()
        # Observed Sharpe = 1.0 (strict < required)
        snap = _make_valid_snapshot(config, qual, realized_sharpe=1.0)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 0

    def test_sharpe_above_threshold_no_event(self) -> None:
        config = _make_valid_config(min_rolling_sharpe_30d=1.0)
        qual = _make_valid_qualification_record()
        # Observed Sharpe = 1.5 (> 1.0)
        snap = _make_valid_snapshot(config, qual, realized_sharpe=1.5)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 0

    def test_sharpe_none_no_event(self) -> None:
        config = _make_valid_config(min_rolling_sharpe_30d=1.0)
        qual = _make_valid_qualification_record()
        # Sharpe is None (insufficient sessions / zero variance in M3)
        snap = _make_valid_snapshot(config, qual, realized_sharpe=None)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 0

    def test_negative_sharpe_triggers_event(self) -> None:
        config = _make_valid_config(min_rolling_sharpe_30d=0.0)
        qual = _make_valid_qualification_record()
        # Observed Sharpe = -0.5 (< 0.0)
        snap = _make_valid_snapshot(config, qual, realized_sharpe=-0.5)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 1
        assert events[0].rule_name == RULE_SHARPE_COLLAPSE
        assert events[0].observed_value == -0.5


# ---------------------------------------------------------------------------
# 3. Rule Tests: RULE_SLIPPAGE_ANOMALY
# ---------------------------------------------------------------------------


class TestSlippageAnomalyRule:
    def test_slippage_below_threshold_no_event(self) -> None:
        config = _make_valid_config(max_slippage_drift_ratio=1.25)
        qual = _make_valid_qualification_record()
        # Configured = 4.0 bps, ratio = 1.25 -> Threshold = 5.0 bps
        # Observed = 4.9 bps (below)
        snap = _make_valid_snapshot(config, qual, realized_slippage_bps=4.9)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 0

    def test_slippage_equal_threshold_no_event(self) -> None:
        config = _make_valid_config(max_slippage_drift_ratio=1.25)
        qual = _make_valid_qualification_record()
        # Observed = 5.0 bps (equal)
        snap = _make_valid_snapshot(config, qual, realized_slippage_bps=5.0)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 0

    def test_slippage_above_threshold_triggers_event(self) -> None:
        config = _make_valid_config(max_slippage_drift_ratio=1.25)
        qual = _make_valid_qualification_record()
        # Observed = 5.01 bps (above 5.0)
        snap = _make_valid_snapshot(config, qual, realized_slippage_bps=5.01)
        events = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events) == 1
        ev = events[0]
        assert ev.rule_name == RULE_SLIPPAGE_ANOMALY
        assert ev.threshold_value == 5.0
        assert ev.observed_value == 5.01
        assert ev.verify_digest() is True

    def test_configured_slippage_zero_policy(self) -> None:
        config = _make_valid_config(max_slippage_drift_ratio=1.25)
        qual = _make_valid_qualification_record()
        # Modeled slippage = 0.0 bps. Threshold is 0.0 bps.
        # If realized is 0.0 -> equality, no event
        snap_zero = _make_valid_snapshot(config, qual, realized_slippage_bps=0.0)
        events_zero = detect_degradations(snap_zero, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=0.0)
        assert len(events_zero) == 0

        # If realized is > 0.0 (e.g. 0.1 bps) -> any unexpected slippage triggers!
        snap_positive = _make_valid_snapshot(config, qual, realized_slippage_bps=0.1)
        events_pos = detect_degradations(snap_positive, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=0.0)
        assert len(events_pos) == 1
        assert events_pos[0].rule_name == RULE_SLIPPAGE_ANOMALY
        assert events_pos[0].threshold_value == 0.0
        assert events_pos[0].observed_value == 0.1

    def test_authoritative_replay_report_extracts_slippage(self) -> None:
        config = _make_valid_config(max_slippage_drift_ratio=1.25)
        qual = _make_valid_qualification_record()
        rep = _make_valid_replay_report(
            strategy_id=qual.strategy_id,
            qualification_hash=qual.record_hash,
            max_drawdown_bps=500.0,
            slippage_bps_per_side=4.0,
        )
        snap = _make_valid_snapshot(config, qual, rep_report=rep, realized_slippage_bps=5.5)
        events = detect_degradations(snap, config, qual, baseline_replay_report=rep)
        assert len(events) == 1
        assert events[0].rule_name == RULE_SLIPPAGE_ANOMALY
        assert events[0].threshold_value == 5.0


# ---------------------------------------------------------------------------
# 4. Rule Tests: RULE_RISK_REJECTION_SPIKE
# ---------------------------------------------------------------------------


class TestRiskRejectionSpikeRule:
    def test_rejection_rate_below_threshold_no_event(self) -> None:
        config = _make_valid_config(max_risk_rejection_rate=0.05)
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            submitted_orders=20, rejected_orders=0,
        )
        assert len(events) == 0

    def test_rejection_rate_equal_threshold_no_event(self) -> None:
        config = _make_valid_config(max_risk_rejection_rate=0.05)
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        # 1 rejected out of 20 = 0.05 (exactly equal)
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            submitted_orders=20, rejected_orders=1,
        )
        assert len(events) == 0

    def test_rejection_rate_above_threshold_triggers_event(self) -> None:
        config = _make_valid_config(max_risk_rejection_rate=0.05)
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        # 2 rejected out of 20 = 0.10 (> 0.05)
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            submitted_orders=20, rejected_orders=2,
        )
        assert len(events) == 1
        ev = events[0]
        assert ev.rule_name == RULE_RISK_REJECTION_SPIKE
        assert ev.threshold_value == 0.05
        assert ev.observed_value == 0.10
        assert ev.verify_digest() is True

    def test_fewer_than_10_submitted_orders_never_triggers(self) -> None:
        config = _make_valid_config(max_risk_rejection_rate=0.05)
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        # 9 submitted orders, 5 rejected (55% rejection rate!), but sample < 10
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            submitted_orders=9, rejected_orders=5,
        )
        assert len(events) == 0

    def test_exactly_10_submitted_orders_evaluated(self) -> None:
        config = _make_valid_config(max_risk_rejection_rate=0.05)
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        # 10 submitted orders, 1 rejected (10% > 5%) -> triggers!
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            submitted_orders=10, rejected_orders=1,
        )
        assert len(events) == 1
        assert events[0].rule_name == RULE_RISK_REJECTION_SPIKE
        assert events[0].observed_value == 0.10

    def test_metrics_json_embedded_rejection_context(self) -> None:
        config = _make_valid_config(max_risk_rejection_rate=0.05)
        qual = _make_valid_qualification_record()
        # Context embedded directly in snapshot.metrics_json
        m_json = json.dumps({"submitted_orders": 20, "rejection_rate": 0.15})
        snap = _make_valid_snapshot(config, qual, metrics_json=m_json)
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
        )
        assert len(events) == 1
        assert events[0].rule_name == RULE_RISK_REJECTION_SPIKE
        assert events[0].observed_value == 0.15


# ---------------------------------------------------------------------------
# 5. Rule Tests: RULE_TRADE_DROPOUT
# ---------------------------------------------------------------------------


class TestTradeDropoutRule:
    def test_dropout_29_sessions_no_event(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            consecutive_inactive_sessions=29,
        )
        assert len(events) == 0

    def test_dropout_30_sessions_triggers_event(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            consecutive_inactive_sessions=30,
        )
        assert len(events) == 1
        ev = events[0]
        assert ev.rule_name == RULE_TRADE_DROPOUT
        assert ev.threshold_value == 30.0
        assert ev.observed_value == 30.0
        assert ev.verify_digest() is True

    def test_dropout_31_sessions_triggers_event(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            consecutive_inactive_sessions=31,
        )
        assert len(events) == 1
        assert events[0].rule_name == RULE_TRADE_DROPOUT
        assert events[0].observed_value == 31.0

    def test_trade_resets_dropout_condition(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        # Even after historical inactivity, an active trade resets consecutive inactive count to 0
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
            consecutive_inactive_sessions=0,
        )
        assert len(events) == 0


# ---------------------------------------------------------------------------
# 6. Multi-Rule Snapshot & Deterministic Ordering
# ---------------------------------------------------------------------------


class TestMultiRuleEvaluation:
    def test_two_breaches_generate_two_events_sorted_lexically(self) -> None:
        config = _make_valid_config(
            max_drawdown_expansion_limit=1.5,
            min_rolling_sharpe_30d=1.0,
        )
        qual = _make_valid_qualification_record()
        # Both Drawdown (800 > 750) and Sharpe (0.8 < 1.0) breach
        snap = _make_valid_snapshot(
            config, qual,
            max_drawdown_bps=800.0,
            realized_sharpe=0.8,
        )
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0, configured_slippage_bps=4.0,
        )
        assert len(events) == 2
        rule_names = [ev.rule_name for ev in events]
        assert rule_names == [RULE_DD_EXPANSION_CRITICAL, RULE_SHARPE_COLLAPSE]
        # Verify both have distinct valid hashes
        assert events[0].event_hash != events[1].event_hash
        assert events[0].verify_digest() is True
        assert events[1].verify_digest() is True

    def test_all_five_breaches_generate_five_events_sorted(self) -> None:
        config = _make_valid_config(
            max_drawdown_expansion_limit=1.5,
            min_rolling_sharpe_30d=1.0,
            max_slippage_drift_ratio=1.25,
            max_risk_rejection_rate=0.05,
        )
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(
            config, qual,
            max_drawdown_bps=800.0,     # DD breach
            realized_sharpe=0.5,        # Sharpe breach
            realized_slippage_bps=6.0,  # Slippage breach (6.0 > 5.0)
        )
        events = detect_degradations(
            snap, config, qual,
            baseline_max_dd_bps=500.0,
            configured_slippage_bps=4.0,
            submitted_orders=20,
            rejected_orders=3,           # Rejection breach (0.15 > 0.05)
            consecutive_inactive_sessions=35,  # Dropout breach (35 >= 30)
        )
        assert len(events) == 5
        expected_order = sorted([
            RULE_DD_EXPANSION_CRITICAL,
            RULE_SHARPE_COLLAPSE,
            RULE_SLIPPAGE_ANOMALY,
            RULE_RISK_REJECTION_SPIKE,
            RULE_TRADE_DROPOUT,
        ])
        assert [ev.rule_name for ev in events] == expected_order


# ---------------------------------------------------------------------------
# 7. Fail-Closed Security & Provenance Verification
# ---------------------------------------------------------------------------


class TestSecurityAndProvenanceChecks:
    def test_fake_snapshot_rejected(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()

        class FakeSnapshot:
            pass

        with pytest.raises(MonitoringProvenanceError, match="snapshot must be an instance of MonitoringSnapshot"):
            detect_degradations(FakeSnapshot(), config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)  # type: ignore

    def test_fake_config_rejected(self) -> None:
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(_make_valid_config(), qual)

        class FakeConfig:
            pass

        with pytest.raises(MonitoringProvenanceError, match="config must be an instance of MonitoringConfig"):
            detect_degradations(snap, FakeConfig(), qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)  # type: ignore

    def test_wrong_split_zone_rejected(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = MonitoringSnapshot.create(
            strategy_id=qual.strategy_id,
            qualification_hash=qual.record_hash,
            replay_report_hash="rep_123",
            dataset_version=qual.dataset_version,
            dataset_sha256=qual.dataset_sha256,
            split_zone="TRAIN",  # ILLEGAL split zone for forward paper evaluation
            monitoring_protocol_version=config.protocol_version,
            monitoring_config_hash=config.compute_config_hash(),
            window_start_ts="2026-03-01T00:00:00Z",
            window_end_ts="2026-03-31T00:00:00Z",
            total_trades=20,
            net_pnl=1000.0,
            max_drawdown_bps=200.0,
            realized_sharpe=1.5,
            realized_slippage_bps=4.0,
            cost_to_turnover_bps=5.0,
            risk_event_count=0,
            metrics_json="{}",
        )
        with pytest.raises(MonitoringProvenanceError, match="Invalid split_zone"):
            detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)

    def test_tampered_qualification_record_rejected(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual)
        # Tamper qualification record observed_sharpe
        tampered_qual = StrategyQualificationRecord(
            qualification_id=qual.qualification_id,
            strategy_id=qual.strategy_id,
            strategy_spec_hash=qual.strategy_spec_hash,
            dataset_version=qual.dataset_version,
            dataset_sha256=qual.dataset_sha256,
            split_manifest_version=qual.split_manifest_version,
            research_protocol_version=qual.research_protocol_version,
            population_hash=qual.population_hash,
            effective_trial_count=qual.effective_trial_count,
            observed_sharpe=9.99,  # tampered
            dsr=qual.dsr,
            trade_count=qual.trade_count,
            holdout_state=qual.holdout_state,
            robustness_status=qual.robustness_status,
            final_status=qual.final_status,
            created_at=qual.created_at,
            reasons=qual.reasons,
            record_hash=qual.record_hash,
        )
        with pytest.raises(MonitoringProvenanceError, match="Qualification record digest verification failed"):
            detect_degradations(snap, config, tampered_qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)

    def test_mismatched_qualification_strategy_id_rejected(self) -> None:
        config = _make_valid_config()
        qual_other = _make_valid_qualification_record(strategy_id="STRAT-OTHER")
        snap = _make_valid_snapshot(config, _make_valid_qualification_record(strategy_id="STRAT-M4-TEST"))
        with pytest.raises(MonitoringProvenanceError, match="Qualification strategy_id mismatch"):
            detect_degradations(snap, config, qual_other, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)

    def test_unqualified_strategy_final_status_rejected(self) -> None:
        config = _make_valid_config()
        qual_rejected = _make_valid_qualification_record(final_status=ValidationStatus.REJECTED)
        snap = _make_valid_snapshot(config, qual_rejected)
        with pytest.raises(MonitoringProvenanceError, match="Qualification record final_status must be PAPER_ELIGIBLE"):
            detect_degradations(snap, config, qual_rejected, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)

    def test_conflicting_explicit_and_report_baseline_rejected(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        rep = _make_valid_replay_report(
            strategy_id=qual.strategy_id,
            qualification_hash=qual.record_hash,
            max_drawdown_bps=500.0,
            slippage_bps_per_side=4.0,
        )
        snap = _make_valid_snapshot(config, qual, rep_report=rep)
        with pytest.raises(MonitoringProvenanceError, match="Conflicting baseline max drawdown supplied"):
            detect_degradations(
                snap, config, qual,
                baseline_replay_report=rep,
                baseline_max_dd_bps=999.0,  # contradicts 500.0 in authoritative report
            )


# ---------------------------------------------------------------------------
# 8. Property & Determinism Invariants
# ---------------------------------------------------------------------------


class TestPropertyAndDeterminismInvariants:
    def test_same_authoritative_inputs_produce_identical_events_and_hashes(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual, max_drawdown_bps=800.0, realized_sharpe=0.5)

        events1 = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        events2 = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.canonical_dict() == e2.canonical_dict()
            assert e1.event_hash == e2.event_hash
            assert e1.derived_event_id == e2.derived_event_id

    def test_changing_only_observed_metric_changes_only_relevant_event(self) -> None:
        config = _make_valid_config(min_rolling_sharpe_30d=1.0)
        qual = _make_valid_qualification_record()

        snap_dd_breach = _make_valid_snapshot(config, qual, max_drawdown_bps=800.0, realized_sharpe=1.5)
        events_dd = detect_degradations(snap_dd_breach, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events_dd) == 1
        assert events_dd[0].rule_name == RULE_DD_EXPANSION_CRITICAL

        snap_sharpe_breach = _make_valid_snapshot(config, qual, max_drawdown_bps=400.0, realized_sharpe=0.5)
        events_sh = detect_degradations(snap_sharpe_breach, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        assert len(events_sh) == 1
        assert events_sh[0].rule_name == RULE_SHARPE_COLLAPSE

    def test_changing_unrelated_threshold_changes_event_hash_via_config_hash(self) -> None:
        qual = _make_valid_qualification_record()
        config1 = _make_valid_config(max_drawdown_expansion_limit=1.5)
        config2 = _make_valid_config(max_drawdown_expansion_limit=1.6)

        snap1 = _make_valid_snapshot(config1, qual, max_drawdown_bps=900.0)
        snap2 = _make_valid_snapshot(config2, qual, max_drawdown_bps=900.0)

        ev1 = detect_degradations(snap1, config1, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)[0]
        ev2 = detect_degradations(snap2, config2, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)[0]

        assert ev1.threshold_value != ev2.threshold_value
        assert ev1.monitoring_config_hash != ev2.monitoring_config_hash
        assert ev1.event_hash != ev2.event_hash

    def test_degradation_detector_wrapper_matches_direct_function(self) -> None:
        config = _make_valid_config()
        qual = _make_valid_qualification_record()
        snap = _make_valid_snapshot(config, qual, max_drawdown_bps=800.0)

        detector = DegradationDetector(config)
        events_class = detector.evaluate(snap, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)
        events_func = detect_degradations(snap, config, qual, baseline_max_dd_bps=500.0, configured_slippage_bps=4.0)

        assert len(events_class) == len(events_func)
        assert events_class[0].event_hash == events_func[0].event_hash
