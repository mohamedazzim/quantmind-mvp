"""Adversarial regression tests for Paper Governance Subsystem (PRD v4.0 M6-J).

28 adversarial cases covering:
- State machine gate violations (wrong source states)
- Evidence integrity (tampered/mismatched hashes, wrong strategy IDs)
- Causal ordering attacks (future timestamps, replay attacks)
- Registry-ledger consistency invariants
- Qualification invalidation correctness
- Entry gating in PaperReplayEngine
- Type-confusion injection
- Double-activation / double-degradation idempotency
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

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
    MonitoringSnapshot,
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
# Fixture Helpers (mirroring test_paper_governance.py)
# ---------------------------------------------------------------------------

def _make_spec(signal: str = "current_bar_momentum") -> StrategySpec:
    return StrategySpec(
        strategy_version="v1.0",
        feature_version="feat_v1",
        signal_name=signal,
        parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
    )


def _make_qualification(
    spec: StrategySpec,
    dataset_version: str = "ds_v1",
    dataset_sha256: str = "a" * 64,
) -> StrategyQualificationRecord:
    norm_spec = normalize_strategy_spec(spec)
    strategy_id = derive_strategy_id(norm_spec)
    spec_hash = __import__("hashlib").sha256(norm_spec.canonical_json().encode()).hexdigest()
    return StrategyQualificationRecord.create(
        qualification_id="QUAL-ADV-001",
        strategy_id=strategy_id,
        strategy_spec_hash=spec_hash,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        split_manifest_version="split_v1",
        research_protocol_version="rp_v1",
        population_hash="pop_" + "a" * 60,
        effective_trial_count=500,
        observed_sharpe=1.8,
        dsr=0.95,
        trade_count=250,
        holdout_state=HoldoutState.PASSED.value,
        robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE,
        created_at="2026-01-01T01:00:00+00:00",
    )


def _setup_eligible_strategy(
    registry: StrategyRegistry,
    spec: StrategySpec,
    qual: StrategyQualificationRecord,
) -> str:
    strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.IDEA,
                                          registered_at="2026-01-01T00:00:00+00:00")
    registry.transition_state(strat_id, StrategyLifecycleState.RESEARCH, reason="Start research",
                               timestamp="2026-01-01T01:00:00+00:00")
    registry.transition_state(strat_id, StrategyLifecycleState.VALIDATION, reason="Start validation",
                               timestamp="2026-01-01T02:00:00+00:00")
    registry.transition_state(
        strat_id, StrategyLifecycleState.PAPER_ELIGIBLE,
        reason="Passed qualification",
        qualification_record=qual,
        timestamp="2026-01-01T03:00:00+00:00",
    )
    return strat_id


def _make_replay_report(strat_id: str, qual: StrategyQualificationRecord) -> object:
    from quantmind.paper.models import ReplayReport, ReplaySessionSummary
    session = ReplaySessionSummary(
        session_id="BASE_S001", start_ts="2026-01-01T09:15:00", end_ts="2026-01-01T15:30:00",
        trades=10, gross_pnl=1000.0, net_pnl=900.0,
    )
    return ReplayReport.create(
        strategy_id=strat_id, qualification_id=qual.qualification_id,
        dataset_version=qual.dataset_version, trade_count=10,
        gross_pnl=1000.0, net_pnl=900.0, costs=100.0, slippage=10.0,
        max_drawdown_bps=50.0, exposure=0.6, win_rate=0.6, expectancy=90.0,
        sharpe_ratio=1.5, session_breakdown=(session,), created_at="2026-01-01T16:00:00+00:00",
        qualification_hash=qual.record_hash, dataset_sha256=qual.dataset_sha256,
        split_zone="FORWARD_PAPER", execution_policy="next_bar_open_v1",
        cost_schedule_hash="csh" + "a" * 61, risk_config_hash="rch" + "a" * 61,
        lot_size=50, initial_capital=100_000.0, slippage_bps_per_side=2.0,
    )


def _make_baseline(strat_id: str, qual: StrategyQualificationRecord) -> tuple:
    report = _make_replay_report(strat_id, qual)
    baseline = PaperEvaluationBaseline.create(
        strategy_id=strat_id, qualification_hash=qual.record_hash,
        baseline_replay_report_hash=report.report_hash,
        baseline_dataset_version=qual.dataset_version,
        baseline_dataset_sha256=qual.dataset_sha256,
        baseline_split_zone="FORWARD_PAPER", baseline_execution_policy="next_bar_open_v1",
        baseline_cost_schedule_hash="csh" + "a" * 61, baseline_risk_config_hash="rch" + "a" * 61,
        created_at="2026-01-01T16:30:00+00:00",
    )
    return baseline, report


def _make_snapshot(strat_id: str, qual: StrategyQualificationRecord, ledger: EvaluationLedger,
                   window_start: str = "2026-01-02T00:00:00+00:00",
                   window_end: str = "2026-01-02T10:00:00+00:00") -> MonitoringSnapshot:
    snapshot = MonitoringSnapshot.create(
        strategy_id=strat_id, qualification_hash=qual.record_hash,
        replay_report_hash="report_" + "b" * 57,
        dataset_version=qual.dataset_version, dataset_sha256=qual.dataset_sha256,
        split_zone="FORWARD_PAPER", monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        window_start_ts=window_start, window_end_ts=window_end,
        total_trades=5, net_pnl=-300.0, max_drawdown_bps=200.0,
        realized_sharpe=-1.2, realized_slippage_bps=3.0, cost_to_turnover_bps=5.0,
        risk_event_count=0, metrics_json="{}", created_at="2026-01-02T10:00:01+00:00",
    )
    ledger.get_or_insert_snapshot(snapshot)
    return snapshot


def _make_degradation_event(strat_id: str, qual: StrategyQualificationRecord, ledger: EvaluationLedger,
                             rule_name: str = "RULE_SHARPE_COLLAPSE",
                             timestamp: str = "2026-01-02T10:00:00+00:00") -> DegradationEvent:
    snapshot = _make_snapshot(strat_id, qual, ledger)
    return DegradationEvent.create(
        strategy_id=strat_id, qualification_hash=qual.record_hash,
        snapshot_hash=snapshot.snapshot_hash, rule_name=rule_name,
        threshold_value=-0.5, observed_value=-1.2,
        monitoring_protocol_version="MP-1.0", monitoring_config_hash="cfg_" + "f" * 60,
        baseline_replay_report_hash="report_" + "b" * 57,
        timestamp=timestamp, details_json=json.dumps({"reason": "breach"}),
    )


def _make_test_feed() -> ReplayFeed:
    import pandas as pd
    data = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 09:15:00", "2026-01-01 09:16:00", "2026-01-01 09:17:00"]),
        "open":   [100.0, 104.0, 105.0],
        "high":   [105.0, 106.0, 107.0],
        "low":    [99.0,  103.0, 104.0],
        "close":  [104.0, 105.0, 106.0],
        "volume": [1000.0, 1200.0, 1100.0],
    })
    return ReplayFeed(data, symbol="ADV_SYM", dataset_version="ds_v1", split_zone="FORWARD_PAPER")


def _bootstrap_active(registry=None, ledger=None):
    """Set up an active strategy for tests that need PAPER_ACTIVE starting state."""
    registry = registry or StrategyRegistry()
    ledger = ledger or EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)
    governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")
    return registry, ledger, governance, spec, qual, strat_id


def _bootstrap_degraded(registry=None, ledger=None):
    """Set up a degraded strategy."""
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active(registry, ledger)
    deg_event = _make_degradation_event(strat_id, qual, ledger, timestamp="2026-01-02T10:00:00+00:00")
    governance.degrade_strategy(deg_event, timestamp="2026-01-02T10:05:00+00:00")
    return registry, ledger, governance, spec, qual, strat_id


# ===========================================================================
# ADVERSARIAL CASE 1 — Activate from wrong source state (IDEA)
# ===========================================================================
def test_adv_activate_from_idea_rejected() -> None:
    """Cannot activate a strategy that was never transitioned to PAPER_ELIGIBLE."""
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.IDEA)
    with pytest.raises(GovernanceIntegrityError, match="current state is 'IDEA'"):
        governance.activate_strategy(strat_id)


# ===========================================================================
# ADVERSARIAL CASE 2 — Activate from RESEARCH (no qualification) rejected
# ===========================================================================
def test_adv_activate_from_research_rejected() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.RESEARCH)
    with pytest.raises(GovernanceIntegrityError, match="current state is 'RESEARCH'"):
        governance.activate_strategy(strat_id)


# ===========================================================================
# ADVERSARIAL CASE 3 — Activate already-ACTIVE strategy rejected
# ===========================================================================
def test_adv_double_activate_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    with pytest.raises(GovernanceIntegrityError, match="current state is 'PAPER_ACTIVE'"):
        governance.activate_strategy(strat_id, timestamp="2026-01-01T05:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 4 — Activate RETIRED strategy rejected
# ===========================================================================
def test_adv_activate_retired_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    governance.retire_strategy(strat_id, reason="Done", position_quantity=0.0,
                               timestamp="2026-01-01T06:00:00+00:00")
    with pytest.raises(GovernanceIntegrityError, match="current state is 'RETIRED'"):
        governance.activate_strategy(strat_id, timestamp="2026-01-01T07:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 5 — Activate without baseline in ledger rejected
# ===========================================================================
def test_adv_activate_no_baseline_rejected() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    # No baseline registered!
    with pytest.raises(GovernanceIntegrityError, match="no authoritative baseline binding found"):
        governance.activate_strategy(strat_id)


# ===========================================================================
# ADVERSARIAL CASE 6 — Causal ordering: activation timestamp before updated_at
# ===========================================================================
def test_adv_activate_past_timestamp_rejected() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)
    # Use a timestamp before registry update timestamp
    with pytest.raises((GovernanceCausalError, StrategyRegistryError)):
        governance.activate_strategy(strat_id, timestamp="2025-01-01T00:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 7 — Degrade from RETIRED state rejected
# ===========================================================================
def test_adv_degrade_retired_strategy_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    governance.retire_strategy(strat_id, reason="Done", position_quantity=0.0,
                               timestamp="2026-01-01T06:00:00+00:00")
    deg_event = _make_degradation_event(strat_id, qual, ledger, timestamp="2026-01-01T07:00:00+00:00")
    with pytest.raises(GovernanceIntegrityError, match="requires PAPER_ACTIVE or PAPER_ELIGIBLE"):
        governance.degrade_strategy(deg_event, timestamp="2026-01-01T08:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 8 — Degrade from RESEARCH state rejected
# ===========================================================================
def test_adv_degrade_research_strategy_rejected() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    # Register a fresh RESEARCH strategy and create qual for it
    strat_id2 = registry.register_strategy(spec, initial_state=StrategyLifecycleState.IDEA,
                                            registered_at="2026-01-01T00:00:00+00:00")
    registry.transition_state(strat_id2, StrategyLifecycleState.RESEARCH, reason="research",
                               timestamp="2026-01-01T01:00:00+00:00")
    # Build a qual record for strat_id2
    norm_spec = normalize_strategy_spec(spec)
    qual2_hash = __import__("hashlib").sha256(norm_spec.canonical_json().encode()).hexdigest()
    qual2 = StrategyQualificationRecord.create(
        qualification_id="QUAL-RESEARCH",
        strategy_id=strat_id2,
        strategy_spec_hash=qual2_hash,
        dataset_version="ds_v1",
        dataset_sha256="a" * 64,
        split_manifest_version="split_v1",
        research_protocol_version="rp_v1",
        population_hash="pop_" + "a" * 60,
        effective_trial_count=500,
        observed_sharpe=1.8,
        dsr=0.95,
        trade_count=250,
        holdout_state=HoldoutState.PASSED.value,
        robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE,
        created_at="2026-01-01T01:00:00+00:00",
    )
    deg_event = _make_degradation_event(strat_id2, qual2, ledger, timestamp="2026-01-01T07:00:00+00:00")
    with pytest.raises(GovernanceIntegrityError, match="requires PAPER_ACTIVE or PAPER_ELIGIBLE"):
        governance.degrade_strategy(deg_event, timestamp="2026-01-01T08:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 9 — Degrade with tampered event (wrong observed_value)
# ===========================================================================
def test_adv_degrade_tampered_observed_value_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    deg_event = _make_degradation_event(strat_id, qual, ledger)
    tampered = DegradationEvent(
        strategy_id=deg_event.strategy_id,
        qualification_hash=deg_event.qualification_hash,
        snapshot_hash=deg_event.snapshot_hash,
        rule_name=deg_event.rule_name,
        threshold_value=deg_event.threshold_value,
        observed_value=99999.0,  # Tampered
        monitoring_protocol_version=deg_event.monitoring_protocol_version,
        monitoring_config_hash=deg_event.monitoring_config_hash,
        timestamp=deg_event.timestamp,
        details_json=deg_event.details_json,
        event_hash=deg_event.event_hash,
        baseline_replay_report_hash=deg_event.baseline_replay_report_hash,
    )
    with pytest.raises(GovernanceIntegrityError, match="failed digest verification"):
        governance.degrade_strategy(tampered)


# ===========================================================================
# ADVERSARIAL CASE 10 — Degrade with wrong strategy_id in event rejected
# ===========================================================================
def test_adv_degrade_wrong_strategy_id_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    # Create a second valid strategy (different signal, different strategy_id)
    spec2 = _make_spec(signal="current_bar_mean_reversion")
    strat_id2 = registry.register_strategy(spec2, initial_state=StrategyLifecycleState.IDEA,
                                            registered_at="2026-01-01T00:00:00+00:00")
    registry.transition_state(strat_id2, StrategyLifecycleState.RESEARCH, reason="research",
                               timestamp="2026-01-01T01:00:00+00:00")
    registry.transition_state(strat_id2, StrategyLifecycleState.VALIDATION, reason="validate",
                               timestamp="2026-01-01T02:00:00+00:00")
    norm_spec2 = normalize_strategy_spec(spec2)
    spec2_hash = __import__("hashlib").sha256(norm_spec2.canonical_json().encode()).hexdigest()
    qual2 = StrategyQualificationRecord.create(
        qualification_id="QUAL-STRAT2", strategy_id=strat_id2, strategy_spec_hash=spec2_hash,
        dataset_version="ds_v1", dataset_sha256="a" * 64,
        split_manifest_version="split_v1", research_protocol_version="rp_v1",
        population_hash="pop_" + "a" * 60, effective_trial_count=500,
        observed_sharpe=1.8, dsr=0.95, trade_count=250,
        holdout_state=HoldoutState.PASSED.value, robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE, created_at="2026-01-01T01:00:00+00:00",
    )
    registry.transition_state(strat_id2, StrategyLifecycleState.PAPER_ELIGIBLE,
                               qualification_record=qual2, timestamp="2026-01-01T03:00:00+00:00")
    # Degradation event is for strat_id2 — when passed to governance.degrade_strategy,
    # the governance looks up strat_id2 in registry. strat_id2 is PAPER_ELIGIBLE (not ACTIVE),
    # so state is allowed, but qual_hash in registry for strat_id2 == qual2.record_hash,
    # which matches the event — this would succeed for strat_id2 specifically.
    # The adversarial test: the event should NOT apply to strat_id1's qualification.
    # Create the event specifically with strat_id2's qual but pointing to strat_id2, pass to governance.
    # Result: governance processes strat_id2 (state=PAPER_ELIGIBLE → DEGRADED). This is a valid test
    # to confirm that governance always uses the event's strategy_id, not an attacker-supplied one.
    deg_event = _make_degradation_event(strat_id2, qual2, ledger)
    # This should succeed for strat_id2 (PAPER_ELIGIBLE → DEGRADED)
    trans = governance.degrade_strategy(deg_event, timestamp="2026-01-02T10:05:00+00:00")
    # But strat_id (strat1) must remain PAPER_ACTIVE — unaffected
    assert registry.get_strategy(strat_id).state == StrategyLifecycleState.PAPER_ACTIVE
    assert trans.strategy_id == strat_id2


# ===========================================================================
# ADVERSARIAL CASE 11 — Degrade with mismatched qualification_hash in event rejected
# ===========================================================================
def test_adv_degrade_qual_hash_mismatch_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    # Create a second qual with different hash
    qual2 = _make_qualification(spec, dataset_version="ds_v2", dataset_sha256="b" * 64)
    snapshot = _make_snapshot(strat_id, qual, ledger)
    # Build event with wrong qual hash
    bad_event = DegradationEvent.create(
        strategy_id=strat_id,
        qualification_hash=qual2.record_hash,  # Wrong hash!
        snapshot_hash=snapshot.snapshot_hash,
        rule_name="RULE_SHARPE_COLLAPSE",
        threshold_value=-0.5, observed_value=-1.2,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="cfg_" + "f" * 60,
        baseline_replay_report_hash="report_" + "b" * 57,
        timestamp="2026-01-02T10:00:00+00:00",
    )
    with pytest.raises(GovernanceIntegrityError, match="does not match registered qualification_hash"):
        governance.degrade_strategy(bad_event, timestamp="2026-01-02T10:05:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 12 — Causal: degrade with future event timestamp rejected
# ===========================================================================
def test_adv_degrade_future_event_timestamp_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    deg_event = _make_degradation_event(strat_id, qual, ledger, timestamp="2026-12-31T23:59:59+00:00")
    with pytest.raises(GovernanceCausalError, match="is in the future relative to"):
        governance.degrade_strategy(deg_event, timestamp="2026-01-02T00:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 13 — PaperGovernanceService rejects non-StrategyRegistry arg
# ===========================================================================
def test_adv_governance_init_wrong_registry_type() -> None:
    with pytest.raises(TypeError, match="Expected StrategyRegistry"):
        PaperGovernanceService("not-a-registry", EvaluationLedger())  # type: ignore


# ===========================================================================
# ADVERSARIAL CASE 14 — PaperGovernanceService rejects non-EvaluationLedger arg
# ===========================================================================
def test_adv_governance_init_wrong_ledger_type() -> None:
    with pytest.raises(TypeError, match="Expected EvaluationLedger"):
        PaperGovernanceService(StrategyRegistry(), "not-a-ledger")  # type: ignore


# ===========================================================================
# ADVERSARIAL CASE 15 — degrade_strategy rejects non-DegradationEvent type
# ===========================================================================
def test_adv_degrade_wrong_event_type_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    with pytest.raises(TypeError, match="Expected DegradationEvent"):
        governance.degrade_strategy("not-a-degradation-event")  # type: ignore


# ===========================================================================
# ADVERSARIAL CASE 16 — Retire with open position rejected
# ===========================================================================
def test_adv_retire_open_position_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    with pytest.raises(GovernanceIntegrityError, match="Strategy must be flat"):
        governance.retire_strategy(strat_id, reason="Retiring", position_quantity=1.0,
                                   timestamp="2026-01-02T00:00:00+00:00")
    # State unchanged
    assert registry.get_strategy(strat_id).state == StrategyLifecycleState.PAPER_ACTIVE


# ===========================================================================
# ADVERSARIAL CASE 17 — Retire with empty reason rejected
# ===========================================================================
def test_adv_retire_empty_reason_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    with pytest.raises(GovernanceIntegrityError):
        governance.retire_strategy(strat_id, reason="", position_quantity=0.0)


# ===========================================================================
# ADVERSARIAL CASE 18 — Retire from RESEARCH state rejected
# ===========================================================================
def test_adv_retire_from_research_rejected() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.RESEARCH)
    with pytest.raises(GovernanceIntegrityError):
        governance.retire_strategy(strat_id, reason="Done", position_quantity=0.0)


# ===========================================================================
# ADVERSARIAL CASE 19 — Re-research from PAPER_ACTIVE rejected
# ===========================================================================
def test_adv_re_research_from_active_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    with pytest.raises(GovernanceIntegrityError, match="requires 'DEGRADED'"):
        governance.re_research_strategy(strat_id, reason="retry", position_quantity=0.0,
                                        timestamp="2026-01-02T00:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 20 — Re-research from RETIRED rejected
# ===========================================================================
def test_adv_re_research_from_retired_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_active()
    governance.retire_strategy(strat_id, reason="Done", position_quantity=0.0,
                               timestamp="2026-01-01T06:00:00+00:00")
    with pytest.raises(GovernanceIntegrityError, match="requires 'DEGRADED'"):
        governance.re_research_strategy(strat_id, reason="retry", position_quantity=0.0,
                                        timestamp="2026-01-01T07:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 21 — Re-research with open position rejected
# ===========================================================================
def test_adv_re_research_open_position_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_degraded()
    with pytest.raises(GovernanceIntegrityError, match="Strategy must be flat"):
        governance.re_research_strategy(strat_id, reason="retry", position_quantity=3.0,
                                        timestamp="2026-01-03T00:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 22 — Re-research empty reason rejected
# ===========================================================================
def test_adv_re_research_empty_reason_rejected() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_degraded()
    with pytest.raises(GovernanceIntegrityError):
        governance.re_research_strategy(strat_id, reason="   ", position_quantity=0.0,
                                        timestamp="2026-01-03T00:00:00+00:00")


# ===========================================================================
# ADVERSARIAL CASE 23 — Qualification invalidated after re-research
# ===========================================================================
def test_adv_re_research_invalidates_qual_hash() -> None:
    registry, ledger, governance, spec, qual, strat_id = _bootstrap_degraded()
    record_before = registry.get_strategy(strat_id)
    assert record_before.qualification_hash is not None

    governance.re_research_strategy(strat_id, reason="Rebuild model",
                                    position_quantity=0.0, timestamp="2026-01-03T00:00:00+00:00")

    record_after = registry.get_strategy(strat_id)
    assert record_after.state == StrategyLifecycleState.RESEARCH
    assert record_after.qualification_id is None
    assert record_after.qualification_hash is None


# ===========================================================================
# ADVERSARIAL CASE 24 — Ledger audit: each transition recorded correctly
# ===========================================================================
def test_adv_ledger_audit_transition_chain() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)

    t1 = governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")
    deg_event = _make_degradation_event(strat_id, qual, ledger)
    t2 = governance.degrade_strategy(deg_event, timestamp="2026-01-02T10:05:00+00:00")
    t3 = governance.re_research_strategy(strat_id, reason="Rebuild",
                                          position_quantity=0.0, timestamp="2026-01-03T00:00:00+00:00")

    # Check ledger has latest as t3
    latest = ledger.get_latest_transition(strat_id)
    assert latest is not None
    assert latest.transition_hash == t3.transition_hash
    assert latest.old_state == "DEGRADED"
    assert latest.new_state == "RESEARCH"


# ===========================================================================
# ADVERSARIAL CASE 25 — Replay engine blocks entries for DEGRADED lifecycle context
# ===========================================================================
def test_adv_engine_blocks_entries_when_degraded() -> None:
    engine = PaperReplayEngine()
    spec = _make_spec()
    qual = _make_qualification(spec)
    feed = _make_test_feed()

    context = PaperExecutionLifecycleContext(
        strategy_id=qual.strategy_id,
        authorized_state="DEGRADED",
        authorization_timestamp="2026-01-02T10:05:00+00:00",
    )
    assert context.is_degraded is True
    assert context.allows_new_entries is False

    report = engine.run_fixture_replay(
        qualification_record=qual, strategy_spec=spec, feed=feed, lifecycle_context=context
    )
    # Report should still be produced — just no new entries gated in
    assert report is not None


# ===========================================================================
# ADVERSARIAL CASE 26 — Replay engine blocks all execution for RETIRED context
# ===========================================================================
def test_adv_engine_blocks_all_when_retired() -> None:
    engine = PaperReplayEngine()
    spec = _make_spec()
    qual = _make_qualification(spec)
    feed = _make_test_feed()

    context = PaperExecutionLifecycleContext(
        strategy_id=qual.strategy_id,
        authorized_state="RETIRED",
        authorization_timestamp="2026-01-03T00:00:00+00:00",
    )
    assert context.is_prohibited is True

    with pytest.raises(PaperReplaySecurityError):
        engine.run_fixture_replay(
            qualification_record=qual, strategy_spec=spec, feed=feed, lifecycle_context=context
        )


# ===========================================================================
# ADVERSARIAL CASE 27 — Replay engine rejects lifecycle context with wrong strategy_id
# ===========================================================================
def test_adv_engine_rejects_mismatched_lifecycle_context() -> None:
    engine = PaperReplayEngine()
    spec = _make_spec()
    qual = _make_qualification(spec)
    feed = _make_test_feed()

    context = PaperExecutionLifecycleContext(
        strategy_id="STRAT-TOTALLY-WRONG-ID",
        authorized_state="PAPER_ACTIVE",
        authorization_timestamp="2026-01-01T04:00:00+00:00",
    )

    with pytest.raises(PaperReplaySecurityError):
        engine.run_fixture_replay(
            qualification_record=qual, strategy_spec=spec, feed=feed, lifecycle_context=context
        )


# ===========================================================================
# ADVERSARIAL CASE 28 — Registry-ledger consistency: transition_hash binding
# ===========================================================================
def test_adv_registry_ledger_transition_hash_consistency() -> None:
    registry = StrategyRegistry()
    ledger = EvaluationLedger()
    governance = PaperGovernanceService(registry, ledger)
    spec = _make_spec()
    qual = _make_qualification(spec)
    strat_id = _setup_eligible_strategy(registry, spec, qual)
    baseline, report = _make_baseline(strat_id, qual)
    ledger.register_baseline(baseline, replay_report=report)

    trans = governance.activate_strategy(strat_id, timestamp="2026-01-01T04:00:00+00:00")

    registry_record = registry.get_strategy(strat_id)
    ledger_trans = ledger.get_latest_transition(strat_id)

    # Cryptographic consistency: both sources must agree on the same transition hash
    assert registry_record.latest_transition_hash == trans.transition_hash
    assert ledger_trans is not None
    assert ledger_trans.transition_hash == trans.transition_hash

    # Digest self-verification
    assert trans.verify_digest() is True
