"""Unit tests for Paper Governance Subsystem (PRD v4.0 M6).

Covers:
- Strategy activation: PAPER_ELIGIBLE -> PAPER_ACTIVE with baseline verification.
- Strategy degradation: PAPER_ACTIVE -> DEGRADED with DegradationEvent verification.
- Strategy retirement: flat position requirement, permanent state.
- Strategy re-research: DEGRADED -> RESEARCH, qualification invalidation.
- Ledger audit logging: paper_evaluation_transitions immutability and provenance.
- Causal ordering and timestamp enforcement.
- Replay engine integration with PaperExecutionLifecycleContext (entry gating).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from quantmind.paper.engine import PaperReplayEngine, PaperReplaySecurityError
from quantmind.paper.evaluation.governance import (
    GovernanceCausalError,
    GovernanceIntegrityError,
    PaperGovernanceService,
)
from quantmind.paper.evaluation.ledger import EvaluationLedger
from quantmind.paper.evaluation.models import (
    DegradationEvent,
    PaperEvaluationBaseline,
    PaperEvaluationTransition,
    PaperExecutionLifecycleContext,
)
from quantmind.paper.feed import ReplayFeed
from quantmind.paper.ledger import PaperLedger
from quantmind.research_integrity.holdout import HoldoutState
from quantmind.research_integrity.qualification import (
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationStatus,
)
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.registry import (
    StrategyLifecycleState,
    StrategyRegistry,
    StrategyRegistryError,
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


def _setup_eligible_strategy(
    registry: StrategyRegistry,
    spec: StrategySpec,
    qual: StrategyQualificationRecord,
) -> str:
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
    return strat_id


def _make_replay_report(strat_id: str, qual: StrategyQualificationRecord) -> "ReplayReport":
    from quantmind.paper.models import ReplayReport, ReplaySessionSummary
    session = ReplaySessionSummary(
        session_id="BASE_S001",
        start_ts="2026-01-01T09:15:00",
        end_ts="2026-01-01T15:30:00",
        trades=10,
        gross_pnl=1000.0,
        net_pnl=900.0,
    )
    return ReplayReport.create(
        strategy_id=strat_id,
        qualification_id=qual.qualification_id,
        dataset_version=qual.dataset_version,
        trade_count=10,
        gross_pnl=1000.0,
        net_pnl=900.0,
        costs=100.0,
        slippage=10.0,
        max_drawdown_bps=50.0,
        exposure=0.6,
        win_rate=0.6,
        expectancy=90.0,
        sharpe_ratio=1.5,
        session_breakdown=(session,),
        created_at="2026-01-01T16:00:00+00:00",
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


def _make_baseline(strat_id: str, qual: StrategyQualificationRecord) -> tuple["PaperEvaluationBaseline", "ReplayReport"]:  # type: ignore[type-arg]
    from quantmind.paper.models import ReplayReport
    report = _make_replay_report(strat_id, qual)
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
        created_at="2026-01-01T16:30:00+00:00",
    )
    return baseline, report


def _make_snapshot(
    strat_id: str,
    qual: "StrategyQualificationRecord",
    ledger: "EvaluationLedger",
    window_start: str = "2026-01-02T00:00:00+00:00",
    window_end: str = "2026-01-02T10:00:00+00:00",
) -> "MonitoringSnapshot":
    from quantmind.paper.evaluation.models import MonitoringSnapshot
    snapshot = MonitoringSnapshot.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        replay_report_hash="report_" + "b" * 57,
        dataset_version=qual.dataset_version,
        dataset_sha256=qual.dataset_sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        window_start_ts=window_start,
        window_end_ts=window_end,
        total_trades=5,
        net_pnl=-300.0,
        max_drawdown_bps=200.0,
        realized_sharpe=-1.2,
        realized_slippage_bps=3.0,
        cost_to_turnover_bps=5.0,
        risk_event_count=0,
        metrics_json="{}",
        created_at="2026-01-02T10:00:01+00:00",
    )
    ledger.get_or_insert_snapshot(snapshot)
    return snapshot


def _make_degradation_event(
    strat_id: str,
    qual: "StrategyQualificationRecord",
    ledger: "EvaluationLedger",
    rule_name: str = "RULE_SHARPE_COLLAPSE",
    timestamp: str = "2026-01-02T10:00:00+00:00",
) -> "DegradationEvent":
    snapshot = _make_snapshot(strat_id, qual, ledger)
    return DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual.record_hash,
        snapshot_hash=snapshot.snapshot_hash,
        rule_name=rule_name,
        threshold_value=-0.5,
        observed_value=-1.2,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        baseline_replay_report_hash="report_" + "b" * 57,
        timestamp=timestamp,
        details_json=json.dumps({"reason": "Sharpe collapse breach"}),
    )


# ---------------------------------------------------------------------------
# Tests: Strategy Activation
# ---------------------------------------------------------------------------

def test_activate_strategy_success() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)

    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)

    trans = governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    assert trans.old_state == "PAPER_ELIGIBLE"
    assert trans.new_state == "PAPER_ACTIVE"
    assert trans.strategy_id == strat_id
    assert trans.qualification_hash == qual.record_hash
    assert trans.evidence_hash == baseline.binding_hash

    record = registry.get_strategy(strat_id)
    assert record.state == StrategyLifecycleState.PAPER_ACTIVE
    assert record.latest_transition_hash == trans.transition_hash

    # Verify audit in ledger
    latest = ledger.get_latest_transition(strat_id)
    assert latest is not None
    assert latest.transition_hash == trans.transition_hash


def test_activate_strategy_fails_without_authoritative_baseline() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)

    with pytest.raises(GovernanceIntegrityError, match="no authoritative baseline binding found"):
        governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    # State must remain PAPER_ELIGIBLE
    assert registry.get_strategy(strat_id).state == StrategyLifecycleState.PAPER_ELIGIBLE


def test_activate_strategy_fails_if_not_paper_eligible() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.IDEA)

    with pytest.raises(GovernanceIntegrityError, match="current state is 'IDEA', requires 'PAPER_ELIGIBLE'"):
        governance.activate_strategy(strat_id)


# ---------------------------------------------------------------------------
# Tests: Strategy Degradation
# ---------------------------------------------------------------------------

def test_degrade_strategy_success() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)

    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)
    governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    deg_event = _make_degradation_event(strat_id, qual, ledger, timestamp="2026-01-02T10:00:00+00:00")

    trans = governance.degrade_strategy(deg_event, timestamp="2026-01-02T10:05:00+00:00")

    assert trans.old_state == "PAPER_ACTIVE"
    assert trans.new_state == "DEGRADED"
    assert trans.evidence_hash == deg_event.event_hash
    assert trans.snapshot_hash == deg_event.snapshot_hash

    record = registry.get_strategy(strat_id)
    assert record.state == StrategyLifecycleState.DEGRADED
    assert record.latest_transition_hash == trans.transition_hash


def test_degrade_strategy_rejects_tampered_event() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)
    governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    deg_event = _make_degradation_event(strat_id, qual, ledger)
    # Tamper event
    tampered = DegradationEvent(
        strategy_id=deg_event.strategy_id,
        qualification_hash=deg_event.qualification_hash,
        snapshot_hash=deg_event.snapshot_hash,
        rule_name=deg_event.rule_name,
        threshold_value=deg_event.threshold_value,
        observed_value=-999.0,  # Tampered!
        monitoring_protocol_version=deg_event.monitoring_protocol_version,
        monitoring_config_hash=deg_event.monitoring_config_hash,
        timestamp=deg_event.timestamp,
        details_json=deg_event.details_json,
        event_hash=deg_event.event_hash,
        baseline_replay_report_hash=deg_event.baseline_replay_report_hash,
    )

    with pytest.raises(GovernanceIntegrityError, match="failed digest verification"):
        governance.degrade_strategy(tampered)


def test_degrade_strategy_rejects_future_event_timestamp() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)
    governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    # Event is from 2026-01-05, but governance tries to evaluate at 2026-01-02
    deg_event = _make_degradation_event(strat_id, qual, ledger, timestamp="2026-01-05T00:00:00+00:00")

    with pytest.raises(GovernanceCausalError, match="is in the future relative to"):
        governance.degrade_strategy(deg_event, timestamp="2026-01-02T00:00:00+00:00")


# ---------------------------------------------------------------------------
# Tests: Strategy Retirement & Flatness
# ---------------------------------------------------------------------------

def test_retire_strategy_flat_position_success() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)
    governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    trans = governance.retire_strategy(
        strat_id,
        reason="Model decommissioned",
        position_quantity=0.0,
        timestamp="2026-01-02T12:00:00+00:00",
    )

    assert trans.old_state == "PAPER_ACTIVE"
    assert trans.new_state == "RETIRED"
    assert registry.get_strategy(strat_id).state == StrategyLifecycleState.RETIRED


def test_retire_strategy_non_flat_position_fails() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)
    governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    with pytest.raises(GovernanceIntegrityError, match="Strategy must be flat"):
        governance.retire_strategy(
            strat_id,
            reason="Decommissioning",
            position_quantity=5.0,  # Non-zero position!
        )

    # State must not have transitioned to RETIRED
    assert registry.get_strategy(strat_id).state == StrategyLifecycleState.PAPER_ACTIVE


# ---------------------------------------------------------------------------
# Tests: Demotion to Research (Re-Research)
# ---------------------------------------------------------------------------

def test_re_research_strategy_success_and_invalidates_qualification() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)
    governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    deg_event = _make_degradation_event(strat_id, qual, ledger, timestamp="2026-01-02T10:00:00+00:00")
    governance.degrade_strategy(deg_event, timestamp="2026-01-02T10:05:00+00:00")

    # Demote from DEGRADED to RESEARCH
    trans = governance.re_research_strategy(
        strat_id,
        reason="Refactoring alpha parameters",
        position_quantity=0.0,
        timestamp="2026-01-03T10:00:00+00:00",
    )

    assert trans.old_state == "DEGRADED"
    assert trans.new_state == "RESEARCH"

    record = registry.get_strategy(strat_id)
    assert record.state == StrategyLifecycleState.RESEARCH
    # Qualification must be invalidated!
    assert record.qualification_id is None
    assert record.qualification_hash is None


def test_re_research_strategy_fails_if_not_degraded() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)

    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)

    with pytest.raises(GovernanceIntegrityError, match="requires 'DEGRADED'"):
        governance.re_research_strategy(strat_id, reason="Demote", position_quantity=0.0)


# ---------------------------------------------------------------------------
# Tests: Replay Engine Lifecycle Context & Entry Gating
# ---------------------------------------------------------------------------

def _make_test_feed() -> ReplayFeed:
    import pandas as pd
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 09:15:00",
            "2026-01-01 09:16:00",
            "2026-01-01 09:17:00",
        ]),
        "open":   [100.0, 104.0, 105.0],
        "high":   [105.0, 106.0, 107.0],
        "low":    [99.0,  103.0, 104.0],
        "close":  [104.0, 105.0, 106.0],
        "volume": [1000.0, 1200.0, 1100.0],
    })
    return ReplayFeed(
        data,
        symbol="TEST_SYM",
        dataset_version="ds_v1",
        split_zone="FORWARD_PAPER",
    )


def test_replay_engine_allows_entries_when_active() -> None:
    engine = PaperReplayEngine()
    spec = _make_spec()
    qual = _make_qualification(spec)
    feed = _make_test_feed()

    context = PaperExecutionLifecycleContext(
        strategy_id=qual.strategy_id,
        authorized_state="PAPER_ACTIVE",
        authorization_timestamp="2026-01-01T04:00:00+00:00",
    )

    report = engine.run_fixture_replay(
        qualification_record=qual,
        strategy_spec=spec,
        feed=feed,
        lifecycle_context=context,
    )
    assert report.strategy_id == qual.strategy_id


def test_replay_engine_halts_entries_when_degraded() -> None:
    engine = PaperReplayEngine()
    spec = _make_spec()
    qual = _make_qualification(spec)
    feed = _make_test_feed()

    context = PaperExecutionLifecycleContext(
        strategy_id=qual.strategy_id,
        authorized_state="DEGRADED",
        authorization_timestamp="2026-01-02T10:00:00+00:00",
    )

    report = engine.run_fixture_replay(
        qualification_record=qual,
        strategy_spec=spec,
        feed=feed,
        lifecycle_context=context,
    )
    # When degraded, new entries must be suppressed -> trade_count should be 0
    assert report.trade_count == 0


def test_replay_engine_prohibits_execution_when_retired() -> None:
    engine = PaperReplayEngine()
    spec = _make_spec()
    qual = _make_qualification(spec)
    feed = _make_test_feed()

    context = PaperExecutionLifecycleContext(
        strategy_id=qual.strategy_id,
        authorized_state="RETIRED",
        authorization_timestamp="2026-01-02T12:00:00+00:00",
    )

    with pytest.raises(PaperReplaySecurityError, match="Paper replay prohibited for strategy.*RETIRED"):
        engine.run_fixture_replay(
            qualification_record=qual,
            strategy_spec=spec,
            feed=feed,
            lifecycle_context=context,
        )
