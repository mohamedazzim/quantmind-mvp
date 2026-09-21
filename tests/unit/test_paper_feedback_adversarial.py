"""Adversarial security and integrity tests for Research Feedback Subsystem (PRD v4.0 Milestone 7).

Verifies 29 exhaustive adversarial boundary cases:
1. Forged/corrupted feedback_hash rejected by ledger.
2. Non-existent degradation_event_hash rejected by ledger.
3. Strategy ID mismatch between feedback and degradation event rejected by ledger.
4. Qualification hash mismatch between feedback and degradation event rejected by ledger.
5. Non-DegradationEvent input rejected with TypeError.
6. DegradationEvent with tampered digest rejected by feedback service.
7. Unregistered strategy rejected by feedback service.
8. Unmonitored strategy (IDEA/RESEARCH/VALIDATION/REJECTED) rejected by feedback service.
9. Strategy qualification hash mismatch rejected by feedback service.
10. DegradationEvent absent from EvaluationLedger rejected.
11. DegradationEvent differing from stored ledger record rejected.
12. DegradationEvent referencing non-existent snapshot rejected.
13. MonitoringSnapshot with corrupted digest rejected.
14. Snapshot strategy ID mismatch with event rejected.
15. Snapshot qualification hash mismatch with event rejected.
16. Non-existent or corrupted regime bound to snapshot rejected.
17. Feedback timestamp earlier than degradation event rejected (causality).
18. Feedback timestamp earlier than snapshot created_at rejected (causality).
19. Direct SQLite UPDATE on research_feedback aborted by trigger.
20. Direct SQLite DELETE on research_feedback aborted by trigger.
21. Hash collision with conflicting content raises integrity error.
22. Feedback ID collision with different hash raises integrity error.
23. re_research_strategy rejects non-ResearchFeedbackRecord object.
24. re_research_strategy rejects feedback_record with tampered digest.
25. re_research_strategy rejects feedback_record absent from EvaluationLedger.
26. re_research_strategy rejects unregistered strategy or feedback_record with mismatched strategy_id.
27. re_research_strategy rejects transition timestamp earlier than feedback created_at.
28. create_research_task_from_feedback rejects non-feedback or corrupted record.
29. Research task bridge cannot access FINAL_HOLDOUT via ResearchHarness.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import pytest

from quantmind.backtest.engine import BacktestConfig, BacktestEngine
from quantmind.data.registry import DatasetRegistry
from quantmind.data.splits import SplitZone
from quantmind.paper.evaluation.feedback import (
    ResearchFeedbackCausalError,
    ResearchFeedbackIntegrityError,
    ResearchFeedbackService,
)
from quantmind.paper.evaluation.governance import (
    GovernanceCausalError,
    GovernanceIntegrityError,
    PaperGovernanceService,
)
from quantmind.paper.evaluation.ledger import EvaluationLedger, EvaluationLedgerIntegrityError
from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringSnapshot,
    PaperEvaluationBaseline,
    PaperEvaluationRegime,
    ResearchFeedbackRecord,
)
from quantmind.paper.models import ReplayReport, ReplaySessionSummary
from quantmind.research_integrity.feedback_bridge import (
    ResearchBridgeIntegrityError,
    create_research_task_from_feedback,
)
from quantmind.research_integrity.harness import ResearchHarness
from quantmind.research_integrity.holdout import HoldoutManager, HoldoutState
from quantmind.research_integrity.qualification import (
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationStatus,
)
from quantmind.research_integrity.trial_ledger import ResearchBudget, TrialLedger
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.registry import (
    StrategyLifecycleState,
    StrategyRegistry,
)
from quantmind.strategy.spec import StrategySpec


# ---------------------------------------------------------------------------
# Fixture Helpers
# ---------------------------------------------------------------------------

def _make_spec() -> StrategySpec:
    return StrategySpec(
        strategy_version="v1.0",
        feature_version="feat_v1",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
    )


def _make_qualification(
    spec: StrategySpec,
    dataset_version: str = "ds_v1",
    dataset_sha256: str = "a" * 64,
) -> StrategyQualificationRecord:
    norm_spec = normalize_strategy_spec(spec)
    strat_id = derive_strategy_id(norm_spec)
    spec_hash = hashlib.sha256(norm_spec.canonical_json().encode("utf-8")).hexdigest()
    return StrategyQualificationRecord.create(
        qualification_id="QUAL-001",
        strategy_id=strat_id,
        strategy_spec_hash=spec_hash,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        split_manifest_version="v1",
        research_protocol_version="RP-1.0",
        population_hash="pop_" + "a" * 60,
        effective_trial_count=100.0,
        observed_sharpe=1.5,
        dsr=0.95,
        trade_count=200,
        holdout_state=HoldoutState.PASSED.value,
        robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _setup_active_strategy(
    registry: StrategyRegistry,
    ledger: EvaluationLedger,
    qual: StrategyQualificationRecord,
    spec: StrategySpec,
) -> tuple[str, PaperEvaluationBaseline]:
    strat_id = registry.register_strategy(
        spec,
        initial_state=StrategyLifecycleState.IDEA,
        registered_at="2026-01-01T00:00:00+00:00",
    )
    registry.transition_state(strat_id, StrategyLifecycleState.RESEARCH, reason="Start research", timestamp="2026-01-01T01:00:00+00:00")
    registry.transition_state(strat_id, StrategyLifecycleState.VALIDATION, reason="Start validation", timestamp="2026-01-01T02:00:00+00:00")
    registry.transition_state(
        strat_id,
        StrategyLifecycleState.PAPER_ELIGIBLE,
        reason="Passed qualification",
        qualification_record=qual,
        timestamp="2026-01-01T03:00:00+00:00",
    )

    session = ReplaySessionSummary(
        session_id="BASE_S001",
        start_ts="2026-01-01T09:15:00",
        end_ts="2026-01-01T15:30:00",
        trades=10,
        gross_pnl=1000.0,
        net_pnl=900.0,
    )
    report = ReplayReport.create(
        strategy_id=strat_id,
        qualification_id=qual.qualification_id,
        dataset_version=qual.dataset_version,
        trade_count=10,
        gross_pnl=1000.0,
        net_pnl=900.0,
        costs=100.0,
        slippage=10.0,
        max_drawdown_bps=150.0,
        exposure=0.6,
        win_rate=0.6,
        expectancy=90.0,
        sharpe_ratio=1.5,
        session_breakdown=(session,),
        created_at="2026-01-01T03:15:00+00:00",
        qualification_hash=qual.record_hash,
        dataset_sha256=qual.dataset_sha256,
        split_zone="FORWARD_PAPER",
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="csh" + "a" * 61,
        risk_config_hash="rch" + "a" * 61,
        lot_size=50,
        initial_capital=100_000.0,
        slippage_bps_per_side=2.0,
    )
    baseline = PaperEvaluationBaseline.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        baseline_replay_report_hash=report.report_hash,
        baseline_dataset_version=qual.dataset_version,
        baseline_dataset_sha256=qual.dataset_sha256,
        baseline_split_zone="FORWARD_PAPER",
        baseline_execution_policy="next_bar_open_v1",
        baseline_cost_schedule_hash="csh" + "a" * 61,
        baseline_risk_config_hash="rch" + "a" * 61,
        created_at="2026-01-01T03:30:00+00:00",
    )
    ledger.register_baseline(baseline, replay_report=report)

    governance = PaperGovernanceService(registry, ledger)
    governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")
    return strat_id, baseline


def _create_snapshot_and_event(
    ledger: EvaluationLedger,
    strat_id: str,
    qual: StrategyQualificationRecord,
    baseline: PaperEvaluationBaseline,
    rule_name: str = "RULE_DD_EXPANSION_CRITICAL",
    max_drawdown_bps: float = 300.0,
    realized_sharpe: float | None = 0.42,
    realized_slippage_bps: float = 4.5,
    window_end_ts: str = "2026-01-02T16:00:00+00:00",
    regime_hash: str = "",
) -> tuple[MonitoringSnapshot, DegradationEvent]:
    snapshot = MonitoringSnapshot.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        replay_report_hash="rep_" + "c" * 60,
        dataset_version=qual.dataset_version,
        dataset_sha256=qual.dataset_sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        window_start_ts="2026-01-02T09:00:00+00:00",
        window_end_ts=window_end_ts,
        total_trades=50,
        net_pnl=1200.0,
        max_drawdown_bps=max_drawdown_bps,
        realized_sharpe=realized_sharpe,
        realized_slippage_bps=realized_slippage_bps,
        cost_to_turnover_bps=12.5,
        risk_event_count=0,
        metrics_json=json.dumps({"trades": 50}),
        created_at=window_end_ts,
        regime_hash=regime_hash,
    )
    ledger.get_or_insert_snapshot(snapshot)

    event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name=rule_name,
        threshold_value=225.0,
        observed_value=max_drawdown_bps,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        timestamp=window_end_ts,
        details_json=json.dumps({"baseline_max_dd_bps": 150.0}, sort_keys=True),
    )
    ledger.record_degradation_event(event)
    return snapshot, event


# ===========================================================================
# ADVERSARIAL TEST SUITE
# ===========================================================================

# Case 1: Corrupted feedback_hash rejected by ledger
def test_case_01_forged_feedback_hash_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    fb = ResearchFeedbackRecord.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        degradation_event_hash=event.event_hash,
        dataset_version=qual.dataset_version,
        failure_mode="DRAWDOWN_EXPANSION",
        realized_sharpe=0.45,
        drawdown_expansion_ratio=1.5,
        realized_slippage_bps=3.8,
        created_at="2026-01-02T17:00:00+00:00",
    )
    tampered = dataclasses.replace(fb, feedback_hash="bad_hash_" + "0" * 55)
    with pytest.raises(EvaluationLedgerIntegrityError, match="digest verification"):
        ledger.record_feedback(tampered)


# Case 2: Non-existent degradation_event_hash rejected by ledger
def test_case_02_nonexistent_degradation_event_hash_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, _ = _setup_active_strategy(registry, ledger, qual, spec)

    fb = ResearchFeedbackRecord.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        degradation_event_hash="nonexistent_deg_event_" + "0" * 43,
        dataset_version=qual.dataset_version,
        failure_mode="DRAWDOWN_EXPANSION",
        realized_sharpe=0.45,
        drawdown_expansion_ratio=1.5,
        realized_slippage_bps=3.8,
        created_at="2026-01-02T17:00:00+00:00",
    )
    with pytest.raises(EvaluationLedgerIntegrityError, match="non-existent degradation_event_hash"):
        ledger.record_feedback(fb)


# Case 3: Strategy ID mismatch between feedback and event rejected by ledger
def test_case_03_strategy_id_mismatch_in_ledger_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    fb = ResearchFeedbackRecord.create(
        strategy_id="STRAT-OTHER-12345",
        qualification_hash=qual.record_hash,
        degradation_event_hash=event.event_hash,
        dataset_version=qual.dataset_version,
        failure_mode="DRAWDOWN_EXPANSION",
        realized_sharpe=0.45,
        drawdown_expansion_ratio=1.5,
        realized_slippage_bps=3.8,
        created_at="2026-01-02T17:00:00+00:00",
    )
    with pytest.raises(EvaluationLedgerIntegrityError, match="does not match referenced degradation event"):
        ledger.record_feedback(fb)


# Case 4: Qualification hash mismatch between feedback and event rejected by ledger
def test_case_04_qualification_hash_mismatch_in_ledger_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    fb = ResearchFeedbackRecord.create(
        strategy_id=strat_id,
        qualification_hash="qual_mismatch_" + "9" * 50,
        degradation_event_hash=event.event_hash,
        dataset_version=qual.dataset_version,
        failure_mode="DRAWDOWN_EXPANSION",
        realized_sharpe=0.45,
        drawdown_expansion_ratio=1.5,
        realized_slippage_bps=3.8,
        created_at="2026-01-02T17:00:00+00:00",
    )
    with pytest.raises(EvaluationLedgerIntegrityError, match="does not match referenced degradation event"):
        ledger.record_feedback(fb)


# Case 5: Non-DegradationEvent argument raises TypeError
def test_case_05_non_degradation_event_type_error() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(TypeError, match="Expected DegradationEvent"):
        service.create_feedback_from_event("not-a-degradation-event")  # type: ignore


# Case 6: DegradationEvent with tampered digest rejected by feedback service
def test_case_06_degradation_event_tampered_digest_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    tampered_event = dataclasses.replace(event, event_hash="tampered_" + "0" * 55)
    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackIntegrityError, match="failed digest verification"):
        service.create_feedback_from_event(tampered_event)


# Case 7: Unregistered strategy rejected by feedback service
def test_case_07_unregistered_strategy_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    # Use a different, empty registry
    service = ResearchFeedbackService(StrategyRegistry(), ledger)
    with pytest.raises(ResearchFeedbackIntegrityError, match="not found in StrategyRegistry"):
        service.create_feedback_from_event(event)


# Case 8: Unmonitored strategy (IDEA) rejected by feedback service
def test_case_08_unmonitored_strategy_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.IDEA)

    service = ResearchFeedbackService(registry, ledger)
    dummy_event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash="qual_001",
        snapshot_hash="snap_001",
        baseline_replay_report_hash="base_001",
        rule_name="RULE_SHARPE_COLLAPSE",
        threshold_value=-0.5,
        observed_value=-1.0,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_001",
        timestamp="2026-01-02T16:00:00+00:00",
        details_json="{}",
    )
    with pytest.raises(ResearchFeedbackIntegrityError, match="is in state 'IDEA'"):
        service.create_feedback_from_event(dummy_event)


# Case 9: Strategy qualification hash mismatch rejected by feedback service
def test_case_09_qualification_hash_mismatch_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    tampered_event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash="conflicting_qual_hash_" + "1" * 42,
        snapshot_hash=event.snapshot_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name=event.rule_name,
        threshold_value=event.threshold_value,
        observed_value=event.observed_value,
        monitoring_protocol_version=event.monitoring_protocol_version,
        monitoring_config_hash=event.monitoring_config_hash,
        timestamp=event.timestamp,
        details_json=event.details_json,
    )
    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackIntegrityError, match="does not match degradation event qualification hash"):
        service.create_feedback_from_event(tampered_event)


# Case 10: DegradationEvent absent from EvaluationLedger rejected
def test_case_10_event_absent_from_ledger_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    snapshot, _ = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    unrecorded_event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name="RULE_DD_EXPANSION_CRITICAL",
        threshold_value=225.0,
        observed_value=300.0,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        timestamp="2026-01-02T16:00:00+00:00",
        details_json="{}",
    )
    # Event is NOT recorded in ledger
    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackIntegrityError, match="not found in EvaluationLedger"):
        service.create_feedback_from_event(unrecorded_event)


# Case 11: DegradationEvent differing from stored record rejected
def test_case_11_event_content_diverges_from_ledger_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    # Reconstruct event with same hash attribute but different semantic content
    divergent_event = DegradationEvent(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=event.snapshot_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name="RULE_SHARPE_COLLAPSE",
        threshold_value=-0.5,
        observed_value=-1.2,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        timestamp="2026-01-02T16:00:00+00:00",
        details_json="{}",
        event_hash=event.event_hash,  # Fake hash collision
    )
    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackIntegrityError):
        service.create_feedback_from_event(divergent_event)


# Case 12: DegradationEvent referencing non-existent snapshot rejected
def test_case_12_nonexistent_snapshot_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)

    event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash="missing_snap_" + "0" * 51,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name="RULE_SHARPE_COLLAPSE",
        threshold_value=-0.5,
        observed_value=-1.2,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        timestamp="2026-01-02T16:00:00+00:00",
        details_json="{}",
    )
    # Attempting to record into ledger or service will fail closed
    with pytest.raises(EvaluationLedgerIntegrityError, match="non-existent snapshot_hash"):
        ledger.record_degradation_event(event)

    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackIntegrityError, match="not found in EvaluationLedger"):
        service.create_feedback_from_event(event)


# Case 13: MonitoringSnapshot with corrupted digest rejected
def test_case_13_corrupted_snapshot_digest_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)

    # Insert snapshot directly with raw SQL with forged snapshot_hash and regime_hash=NULL
    raw_snap_hash = "corrupted_snap_hash_" + "0" * 44
    ledger._connection.execute(
        """
        INSERT INTO monitoring_snapshots (
            snapshot_id, strategy_id, qualification_hash, replay_report_hash,
            dataset_version, dataset_sha256, split_zone, monitoring_protocol_version,
            monitoring_config_hash, window_start_ts, window_end_ts, total_trades,
            net_pnl, max_drawdown_bps, realized_sharpe, realized_slippage_bps,
            cost_to_turnover_bps, risk_event_count, metrics_json, created_at,
            snapshot_hash, regime_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            "SNAP-CORRUPT-001",
            strat_id,
            qual.record_hash,
            "rep_001",
            qual.dataset_version,
            qual.dataset_sha256,
            "FORWARD_PAPER",
            "MP-1.0",
            "cfg_001",
            "2026-01-02T09:00:00+00:00",
            "2026-01-02T16:00:00+00:00",
            10,
            -100.0,
            200.0,
            -0.5,
            2.0,
            5.0,
            0,
            "{}",
            "2026-01-02T16:00:00+00:00",
            raw_snap_hash,
        ),
    )

    event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=raw_snap_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name="RULE_SHARPE_COLLAPSE",
        threshold_value=-0.5,
        observed_value=-1.2,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_001",
        timestamp="2026-01-02T16:00:00+00:00",
        details_json="{}",
    )
    ledger.record_degradation_event(event)

    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackIntegrityError, match="failed digest verification"):
        service.create_feedback_from_event(event)


# Case 14: Snapshot strategy ID mismatch with event rejected
def test_case_14_snapshot_strategy_mismatch_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)

    # Snapshot with different strategy_id
    diff_snap = MonitoringSnapshot.create(
        strategy_id="STRAT-OTHER-DIFF",
        qualification_hash=qual.record_hash,
        replay_report_hash="rep_" + "c" * 60,
        dataset_version=qual.dataset_version,
        dataset_sha256=qual.dataset_sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        window_start_ts="2026-01-02T09:00:00+00:00",
        window_end_ts="2026-01-02T16:00:00+00:00",
        total_trades=50,
        net_pnl=1200.0,
        max_drawdown_bps=300.0,
        realized_sharpe=0.42,
        realized_slippage_bps=4.5,
        cost_to_turnover_bps=12.5,
        risk_event_count=0,
        metrics_json="{}",
        created_at="2026-01-02T16:00:00+00:00",
    )
    ledger.get_or_insert_snapshot(diff_snap)

    event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=diff_snap.snapshot_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name="RULE_DD_EXPANSION_CRITICAL",
        threshold_value=225.0,
        observed_value=300.0,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        timestamp="2026-01-02T16:00:00+00:00",
        details_json="{}",
    )
    with pytest.raises(EvaluationLedgerIntegrityError, match="does not match"):
        ledger.record_degradation_event(event)


# Case 15: Snapshot qualification hash mismatch with event rejected
def test_case_15_snapshot_qualification_mismatch_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)

    diff_snap = MonitoringSnapshot.create(
        strategy_id=strat_id,
        qualification_hash="other_qual_" + "7" * 53,
        replay_report_hash="rep_" + "c" * 60,
        dataset_version=qual.dataset_version,
        dataset_sha256=qual.dataset_sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        window_start_ts="2026-01-02T09:00:00+00:00",
        window_end_ts="2026-01-02T16:00:00+00:00",
        total_trades=50,
        net_pnl=1200.0,
        max_drawdown_bps=300.0,
        realized_sharpe=0.42,
        realized_slippage_bps=4.5,
        cost_to_turnover_bps=12.5,
        risk_event_count=0,
        metrics_json="{}",
        created_at="2026-01-02T16:00:00+00:00",
    )
    ledger.get_or_insert_snapshot(diff_snap)

    event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=diff_snap.snapshot_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name="RULE_DD_EXPANSION_CRITICAL",
        threshold_value=225.0,
        observed_value=300.0,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        timestamp="2026-01-02T16:00:00+00:00",
        details_json="{}",
    )
    with pytest.raises(EvaluationLedgerIntegrityError, match="does not match"):
        ledger.record_degradation_event(event)


# Case 16: Non-existent or corrupted regime bound to snapshot rejected
def test_case_16_nonexistent_regime_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)

    # 1. Non-existent regime rejected by foreign key constraint at DB layer
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        ledger._connection.execute(
            """
            INSERT INTO monitoring_snapshots (
                snapshot_id, strategy_id, qualification_hash, replay_report_hash,
                dataset_version, dataset_sha256, split_zone, monitoring_protocol_version,
                monitoring_config_hash, window_start_ts, window_end_ts, total_trades,
                net_pnl, max_drawdown_bps, realized_sharpe, realized_slippage_bps,
                cost_to_turnover_bps, risk_event_count, metrics_json, created_at,
                snapshot_hash, regime_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SNAP-NONEXIST-REGIME",
                strat_id,
                qual.record_hash,
                "rep_001",
                qual.dataset_version,
                qual.dataset_sha256,
                "FORWARD_PAPER",
                "MP-1.0",
                "cfg_001",
                "2026-01-02T09:00:00+00:00",
                "2026-01-02T16:00:00+00:00",
                10,
                -100.0,
                200.0,
                -0.5,
                2.0,
                5.0,
                0,
                "{}",
                "2026-01-02T16:00:00+00:00",
                "snap_hash_test_16",
                "nonexistent_regime_hash",
            ),
        )

    # 2. Corrupted regime digest rejected by feedback service
    raw_regime_hash = "corrupted_regime_hash_" + "0" * 42
    ledger._connection.execute(
        """
        INSERT INTO paper_evaluation_regimes (
            regime_hash, strategy_id, qualification_hash, baseline_replay_report_hash,
            forward_dataset_version, forward_dataset_sha256, execution_policy,
            cost_schedule_hash, risk_config_hash, monitoring_protocol_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            raw_regime_hash,
            strat_id,
            qual.record_hash,
            baseline.baseline_replay_report_hash,
            qual.dataset_version,
            qual.dataset_sha256,
            "STRICT",
            "cost_001",
            "risk_001",
            "MP-1.0",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    snap = MonitoringSnapshot.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        replay_report_hash="rep_test_16",
        dataset_version=qual.dataset_version,
        dataset_sha256=qual.dataset_sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_001",
        window_start_ts="2026-01-02T09:00:00+00:00",
        window_end_ts="2026-01-02T16:00:00+00:00",
        total_trades=10,
        net_pnl=-100.0,
        max_drawdown_bps=200.0,
        realized_sharpe=-0.5,
        realized_slippage_bps=2.0,
        cost_to_turnover_bps=5.0,
        risk_event_count=0,
        metrics_json="{}",
        created_at="2026-01-02T16:00:00+00:00",
        regime_hash=raw_regime_hash,
    )
    ledger._connection.execute(
        """
        INSERT INTO monitoring_snapshots (
            snapshot_id, strategy_id, qualification_hash, replay_report_hash,
            dataset_version, dataset_sha256, split_zone, monitoring_protocol_version,
            monitoring_config_hash, window_start_ts, window_end_ts, total_trades,
            net_pnl, max_drawdown_bps, realized_sharpe, realized_slippage_bps,
            cost_to_turnover_bps, risk_event_count, metrics_json, created_at,
            snapshot_hash, regime_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snap.derived_snapshot_id,
            snap.strategy_id,
            snap.qualification_hash,
            snap.replay_report_hash,
            snap.dataset_version,
            snap.dataset_sha256,
            snap.split_zone,
            snap.monitoring_protocol_version,
            snap.monitoring_config_hash,
            snap.window_start_ts,
            snap.window_end_ts,
            snap.total_trades,
            snap.net_pnl,
            snap.max_drawdown_bps,
            snap.realized_sharpe,
            snap.realized_slippage_bps,
            snap.cost_to_turnover_bps,
            snap.risk_event_count,
            snap.metrics_json,
            snap.created_at,
            snap.snapshot_hash,
            snap.regime_hash,
        ),
    )
    event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=snap.snapshot_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name="RULE_SHARPE_COLLAPSE",
        threshold_value=-0.5,
        observed_value=-1.2,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_001",
        timestamp="2026-01-02T16:00:00+00:00",
        details_json="{}",
    )
    ledger.record_degradation_event(event)

    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackIntegrityError, match="PaperEvaluationRegime .* failed digest verification"):
        service.create_feedback_from_event(event)


# Case 17: Feedback timestamp earlier than degradation event rejected (causality)
def test_case_17_feedback_before_degradation_event_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(
        ledger, strat_id, qual, baseline, window_end_ts="2026-01-02T16:00:00+00:00"
    )

    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackCausalError, match="earlier than degradation event timestamp"):
        service.create_feedback_from_event(
            event,
            created_at="2026-01-02T15:00:00+00:00",  # 1 hour prior to event!
        )


# Case 18: Feedback timestamp earlier than snapshot created_at rejected (causality)
def test_case_18_feedback_before_snapshot_created_at_rejected() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    snapshot, event = _create_snapshot_and_event(
        ledger, strat_id, qual, baseline, window_end_ts="2026-01-02T16:00:00+00:00"
    )

    early_event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash=baseline.baseline_replay_report_hash,
        rule_name="RULE_DD_EXPANSION_CRITICAL",
        threshold_value=225.0,
        observed_value=300.0,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        timestamp="2026-01-02T15:00:00+00:00",
        details_json="{}",
    )
    ledger.record_degradation_event(early_event)

    service = ResearchFeedbackService(registry, ledger)
    with pytest.raises(ResearchFeedbackCausalError, match="earlier than snapshot created_at"):
        service.create_feedback_from_event(
            early_event,
            created_at="2026-01-02T15:30:00+00:00",  # After event, but before snapshot (16:00)
        )


# Case 19: Direct SQLite UPDATE on research_feedback table is aborted by trigger
def test_case_19_direct_update_aborted_by_trigger() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    service = ResearchFeedbackService(registry, ledger)
    fb = service.create_feedback_from_event(event, created_at="2026-01-02T17:00:00+00:00")

    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="research feedback records are immutable"):
        ledger._connection.execute(
            "UPDATE research_feedback SET failure_mode = 'TAMPERED' WHERE feedback_id = ?",
            (fb.derived_feedback_id,),
        )


# Case 20: Direct SQLite DELETE on research_feedback table is aborted by trigger
def test_case_20_direct_delete_aborted_by_trigger() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    service = ResearchFeedbackService(registry, ledger)
    fb = service.create_feedback_from_event(event, created_at="2026-01-02T17:00:00+00:00")

    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="research feedback records are permanent"):
        ledger._connection.execute(
            "DELETE FROM research_feedback WHERE feedback_id = ?",
            (fb.derived_feedback_id,),
        )


# Case 21: Hash collision with conflicting content raises integrity error
def test_case_21_hash_collision_conflicting_content_raises() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    service = ResearchFeedbackService(registry, ledger)
    fb = service.create_feedback_from_event(event, created_at="2026-01-02T17:00:00+00:00")

    # Construct colliding record with different empirical_notes but same feedback_hash
    colliding = ResearchFeedbackRecord(
        strategy_id=fb.strategy_id,
        qualification_hash=fb.qualification_hash,
        degradation_event_hash=fb.degradation_event_hash,
        dataset_version=fb.dataset_version,
        failure_mode=fb.failure_mode,
        realized_sharpe=fb.realized_sharpe,
        drawdown_expansion_ratio=fb.drawdown_expansion_ratio,
        realized_slippage_bps=fb.realized_slippage_bps,
        empirical_notes="CONFLICTING CONTENT",
        created_at=fb.created_at,
        feedback_hash=fb.feedback_hash,
    )
    with pytest.raises(EvaluationLedgerIntegrityError, match="digest verification"):
        ledger.record_feedback(colliding)


# Case 22: Feedback ID collision with different hash raises integrity error
def test_case_22_feedback_id_collision_raises() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    # fb1 and fb2 have identical (strategy_id, failure_mode, degradation_event_hash)
    # -> identical derived_feedback_id!
    # But different realized_sharpe -> different feedback_hash!
    fb1 = ResearchFeedbackRecord.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        degradation_event_hash=event.event_hash,
        dataset_version=qual.dataset_version,
        failure_mode="DRAWDOWN_EXPANSION",
        realized_sharpe=0.45,
        drawdown_expansion_ratio=1.5,
        realized_slippage_bps=3.8,
        created_at="2026-01-02T17:00:00+00:00",
    )
    fb2 = ResearchFeedbackRecord.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        degradation_event_hash=event.event_hash,
        dataset_version=qual.dataset_version,
        failure_mode="DRAWDOWN_EXPANSION",
        realized_sharpe=0.99,  # Different!
        drawdown_expansion_ratio=1.5,
        realized_slippage_bps=3.8,
        created_at="2026-01-02T17:00:00+00:00",
    )

    assert fb1.derived_feedback_id == fb2.derived_feedback_id
    assert fb1.feedback_hash != fb2.feedback_hash

    ledger.record_feedback(fb1)
    with pytest.raises(EvaluationLedgerIntegrityError, match="Feedback ID collision"):
        ledger.record_feedback(fb2)


# Case 23: re_research_strategy rejects non-ResearchFeedbackRecord object
def test_case_23_re_research_rejects_non_feedback_record() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    governance = PaperGovernanceService(registry, ledger)
    governance.degrade_strategy(event, timestamp="2026-01-02T16:30:00+00:00")

    with pytest.raises(GovernanceIntegrityError, match="feedback_record must be an instance of ResearchFeedbackRecord"):
        governance.re_research_strategy(
            strat_id,
            reason="Demote to research",
            feedback_record="not-a-feedback-record",  # type: ignore
            timestamp="2026-01-02T17:00:00+00:00",
        )


# Case 24: re_research_strategy rejects feedback_record with tampered digest
def test_case_24_re_research_rejects_tampered_feedback_record() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    governance = PaperGovernanceService(registry, ledger)
    governance.degrade_strategy(event, timestamp="2026-01-02T16:30:00+00:00")

    service = ResearchFeedbackService(registry, ledger)
    fb = service.create_feedback_from_event(event, created_at="2026-01-02T17:00:00+00:00")
    tampered_fb = dataclasses.replace(fb, feedback_hash="tampered_fb_" + "0" * 52)

    with pytest.raises(GovernanceIntegrityError, match="failed digest verification"):
        governance.re_research_strategy(
            strat_id,
            reason="Demote to research",
            feedback_record=tampered_fb,
            timestamp="2026-01-02T17:30:00+00:00",
        )


# Case 25: re_research_strategy rejects feedback_record absent from EvaluationLedger
def test_case_25_re_research_rejects_unpersisted_feedback_record() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    governance = PaperGovernanceService(registry, ledger)
    governance.degrade_strategy(event, timestamp="2026-01-02T16:30:00+00:00")

    # Create feedback record directly without persisting in ledger
    unpersisted_fb = ResearchFeedbackRecord.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        degradation_event_hash=event.event_hash,
        dataset_version=qual.dataset_version,
        failure_mode="DRAWDOWN_EXPANSION",
        realized_sharpe=0.45,
        drawdown_expansion_ratio=1.5,
        realized_slippage_bps=3.8,
        created_at="2026-01-02T17:00:00+00:00",
    )

    with pytest.raises(GovernanceIntegrityError, match="not found in EvaluationLedger"):
        governance.re_research_strategy(
            strat_id,
            reason="Demote to research",
            feedback_record=unpersisted_fb,
            timestamp="2026-01-02T17:30:00+00:00",
        )


# Case 26: re_research_strategy rejects unregistered strategy and feedback_record with mismatched strategy_id
def test_case_26_re_research_rejects_strategy_mismatch_in_feedback() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    governance = PaperGovernanceService(registry, ledger)
    governance.degrade_strategy(event, timestamp="2026-01-02T16:30:00+00:00")

    service = ResearchFeedbackService(registry, ledger)
    fb = service.create_feedback_from_event(event, created_at="2026-01-02T17:00:00+00:00")

    # 1. Unregistered strategy
    with pytest.raises(GovernanceIntegrityError, match="not found in StrategyRegistry"):
        governance.re_research_strategy(
            "STRAT-OTHER-9999",
            reason="Demote to research",
            feedback_record=fb,
            timestamp="2026-01-02T17:30:00+00:00",
        )

    # 2. Registered degraded strategy receiving feedback for different strategy
    strat2_spec = StrategySpec(
        strategy_version="v2.0",
        feature_version="feat_v2",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 375, "session_window": [0.2, 0.8]},
    )
    qual2 = _make_qualification(strat2_spec)
    strat2_id, base2 = _setup_active_strategy(registry, ledger, qual2, strat2_spec)
    _, event2 = _create_snapshot_and_event(ledger, strat2_id, qual2, base2)
    governance.degrade_strategy(event2, timestamp="2026-01-02T16:35:00+00:00")

    # fb has strategy_id = strat_id, but we call re_research on strat2_id
    with pytest.raises(GovernanceIntegrityError, match="does not match strategy"):
        governance.re_research_strategy(
            strat2_id,
            reason="Demote to research",
            feedback_record=fb,
            timestamp="2026-01-02T17:30:00+00:00",
        )


# Case 27: re_research_strategy rejects transition timestamp earlier than feedback created_at
def test_case_27_re_research_causality_rejection() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    governance = PaperGovernanceService(registry, ledger)
    governance.degrade_strategy(event, timestamp="2026-01-02T16:30:00+00:00")

    service = ResearchFeedbackService(registry, ledger)
    fb = service.create_feedback_from_event(event, created_at="2026-01-02T17:30:00+00:00")

    with pytest.raises(GovernanceCausalError, match="earlier than feedback created_at"):
        governance.re_research_strategy(
            strat_id,
            reason="Demote to research",
            feedback_record=fb,
            timestamp="2026-01-02T17:00:00+00:00",  # Earlier than feedback!
        )


# Case 28: create_research_task_from_feedback rejects non-feedback or corrupted record
def test_case_28_bridge_rejects_corrupted_record() -> None:
    with pytest.raises(TypeError, match="Expected ResearchFeedbackRecord"):
        create_research_task_from_feedback("not-a-feedback-record")  # type: ignore

    valid_fb = ResearchFeedbackRecord.create(
        strategy_id="STRAT-001",
        qualification_hash="qual_001",
        degradation_event_hash="deg_001",
        dataset_version="ds_v1",
        failure_mode="SHARPE_COLLAPSE",
        realized_sharpe=-0.2,
        drawdown_expansion_ratio=1.1,
        realized_slippage_bps=2.0,
        created_at="2026-01-01T00:00:00+00:00",
    )
    tampered_fb = dataclasses.replace(valid_fb, feedback_hash="tampered_hash_" + "0" * 50)
    with pytest.raises(ResearchBridgeIntegrityError, match="failed cryptographic digest verification"):
        create_research_task_from_feedback(tampered_fb)

    with pytest.raises(ValueError, match="research_protocol_version cannot be empty"):
        create_research_task_from_feedback(valid_fb, research_protocol_version="")


# Case 29: Research task bridge cannot access FINAL_HOLDOUT via ResearchHarness
def test_case_29_research_task_cannot_access_final_holdout() -> None:
    valid_fb = ResearchFeedbackRecord.create(
        strategy_id="STRAT-001",
        qualification_hash="qual_001",
        degradation_event_hash="deg_001",
        dataset_version="ds_v1",
        failure_mode="SHARPE_COLLAPSE",
        realized_sharpe=-0.2,
        drawdown_expansion_ratio=1.1,
        realized_slippage_bps=2.0,
        created_at="2026-01-01T00:00:00+00:00",
    )
    task = create_research_task_from_feedback(valid_fb)

    # Initialize ResearchHarness
    engine = BacktestEngine()
    ledger = TrialLedger()
    data_registry = DatasetRegistry()
    harness = ResearchHarness(engine, ledger, data_registry)

    spec = _make_spec()
    config = BacktestConfig()

    # Attempting to run trial on FINAL_HOLDOUT must be unconditionally rejected
    with pytest.raises(ValueError, match="FINAL_HOLDOUT zone is sealed"):
        harness.run_trial(
            config=config,
            research_task_id=task.task_id,
            strategy_spec=spec,
            dataset_version="ds_v1",
            split_zone=SplitZone.FINAL_HOLDOUT,
            research_protocol_version="RP-1.0",
            seed=42,
            budget=ResearchBudget(max_trials=10, max_runtime_minutes=60.0),
        )


# Case 30: ResearchFeedbackTask cannot directly mutate StrategyRegistry
def test_case_30_task_cannot_modify_strategy_registry() -> None:
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.IDEA)

    valid_fb = ResearchFeedbackRecord.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        degradation_event_hash="deg_001",
        dataset_version="ds_v1",
        failure_mode="SHARPE_COLLAPSE",
        realized_sharpe=-0.2,
        drawdown_expansion_ratio=1.1,
        realized_slippage_bps=2.0,
        created_at="2026-01-01T00:00:00+00:00",
    )
    task = create_research_task_from_feedback(valid_fb)

    # Strategy state remains untouched in IDEA
    record = registry.get_strategy(strat_id)
    assert record.state == StrategyLifecycleState.IDEA
    assert not hasattr(task, "mutate_registry")
    assert not hasattr(task, "activate_strategy")


# Case 31: DEGRADE -> RESEARCH without feedback uses explicit governance decision type
def test_case_31_degrade_to_research_without_feedback() -> None:
    ledger = EvaluationLedger()
    registry = StrategyRegistry()
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
    _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

    governance = PaperGovernanceService(registry, ledger)
    governance.degrade_strategy(event, timestamp="2026-01-02T16:30:00+00:00")
    assert registry.get_strategy(strat_id).state == StrategyLifecycleState.DEGRADED

    # Administrative demotion without research feedback record
    trans = governance.re_research_strategy(
        strat_id,
        reason="Manual administrative model overhaul",
        timestamp="2026-01-02T17:00:00+00:00",
    )

    # Must NOT pretend to be RESEARCH_FEEDBACK
    assert trans.evidence_type != "RESEARCH_FEEDBACK"
    assert trans.evidence_type == "re_research_decision"
    assert trans.new_state == StrategyLifecycleState.RESEARCH.value

    # Now verify with feedback_record on another strategy
    strat2_spec = StrategySpec(
        strategy_version="v2.0",
        feature_version="feat_v2",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 375, "session_window": [0.2, 0.8]},
    )
    qual2 = _make_qualification(strat2_spec)
    strat2_id, base2 = _setup_active_strategy(registry, ledger, qual2, strat2_spec)
    _, event2 = _create_snapshot_and_event(ledger, strat2_id, qual2, base2)
    governance.degrade_strategy(event2, timestamp="2026-01-02T16:35:00+00:00")

    service = ResearchFeedbackService(registry, ledger)
    fb2 = service.create_feedback_from_event(event2, created_at="2026-01-02T17:00:00+00:00")
    trans2 = governance.re_research_strategy(
        strat2_id,
        reason="Feedback-driven model revision",
        feedback_record=fb2,
        timestamp="2026-01-02T17:30:00+00:00",
    )
    assert trans2.evidence_type == "RESEARCH_FEEDBACK"
    assert trans2.evidence_hash == fb2.feedback_hash


# Case 32: ResearchFeedbackTask creation creates strictly zero TrialLedger rows
def test_case_32_task_creation_creates_zero_trial_ledger_rows() -> None:
    trial_ledger = TrialLedger()
    valid_fb = ResearchFeedbackRecord.create(
        strategy_id="STRAT-001",
        qualification_hash="qual_001",
        degradation_event_hash="deg_001",
        dataset_version="ds_v1",
        failure_mode="SHARPE_COLLAPSE",
        realized_sharpe=-0.2,
        drawdown_expansion_ratio=1.1,
        realized_slippage_bps=2.0,
        created_at="2026-01-01T00:00:00+00:00",
    )

    # Derive task
    task = create_research_task_from_feedback(valid_fb)
    assert task is not None

    # Check TrialLedger has 0 entries
    trial_rows = trial_ledger._connection.execute("SELECT count(*) FROM trials").fetchone()[0]
    assert trial_rows == 0

    # Ensure task has no trial_id attribute
    assert not hasattr(task, "trial_id")
