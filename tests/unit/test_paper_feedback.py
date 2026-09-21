"""Unit tests for Research Feedback Subsystem (PRD v4.0 Milestone 7).

Covers:
- EvaluationLedger research_feedback table persistence, querying, and idempotency.
- ResearchFeedbackService: creation of ResearchFeedbackRecord from DegradationEvent.
- Rule mapping to canonical failure modes.
- Extraction and population of realized performance/execution metrics.
- Governance integration: re_research_strategy binding to ResearchFeedbackRecord.
- Research task bridge: create_research_task_from_feedback.
"""

from __future__ import annotations

import hashlib
import json
import pytest

from quantmind.paper.evaluation.feedback import (
    RULE_FAILURE_MODE_MAP,
    ResearchFeedbackCausalError,
    ResearchFeedbackIntegrityError,
    ResearchFeedbackService,
)
from quantmind.paper.evaluation.governance import PaperGovernanceService
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
    ResearchFeedbackTask,
    create_research_task_from_feedback,
    derive_feedback_task_id,
)
from quantmind.research_integrity.harness import ResearchTask
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
    details_dict: dict | None = None,
    window_end_ts: str = "2026-01-02T16:00:00+00:00",
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
    )
    ledger.get_or_insert_snapshot(snapshot)

    details = details_dict or {
        "baseline_max_dd_bps": 150.0,
        "max_drawdown_expansion_limit": 1.5,
        "threshold_dd_bps": 225.0,
    }

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
        details_json=json.dumps(details, sort_keys=True),
    )
    ledger.record_degradation_event(event)
    return snapshot, event


# ---------------------------------------------------------------------------
# 1. EvaluationLedger Persistence Tests
# ---------------------------------------------------------------------------

class TestEvaluationLedgerResearchFeedback:
    def test_record_and_retrieve_feedback(self) -> None:
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
            drawdown_expansion_ratio=1.6667,
            realized_slippage_bps=3.8,
            empirical_notes="Drawdown expansion observed",
            created_at="2026-01-02T17:00:00+00:00",
        )

        inserted = ledger.record_feedback(fb)
        assert inserted.feedback_hash == fb.feedback_hash
        assert inserted.derived_feedback_id == fb.derived_feedback_id

        # Query by hash
        retrieved = ledger.get_feedback(fb.feedback_hash)
        assert retrieved is not None
        assert retrieved.canonical_dict() == fb.canonical_dict()

        # Query by ID
        by_id = ledger.get_feedback_by_id(fb.derived_feedback_id)
        assert by_id is not None
        assert by_id.feedback_hash == fb.feedback_hash

        # List
        fb_list = ledger.list_feedback(strat_id)
        assert len(fb_list) == 1
        assert fb_list[0].feedback_hash == fb.feedback_hash

    def test_idempotent_insert_duplicate(self) -> None:
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
            drawdown_expansion_ratio=1.6667,
            realized_slippage_bps=3.8,
            empirical_notes="Notes",
            created_at="2026-01-02T17:00:00+00:00",
        )
        rec1 = ledger.record_feedback(fb)
        rec2 = ledger.record_feedback(fb)
        assert rec1.feedback_hash == rec2.feedback_hash
        assert len(ledger.list_feedback(strat_id)) == 1


# ---------------------------------------------------------------------------
# 2. ResearchFeedbackService Tests
# ---------------------------------------------------------------------------

class TestResearchFeedbackService:
    @pytest.mark.parametrize(
        ("rule_name", "expected_failure_mode"),
        [
            ("RULE_DD_EXPANSION_CRITICAL", "DRAWDOWN_EXPANSION"),
            ("RULE_SHARPE_COLLAPSE", "SHARPE_COLLAPSE"),
            ("RULE_SLIPPAGE_ANOMALY", "SLIPPAGE_ANOMALY"),
            ("RULE_RISK_REJECTION_SPIKE", "RISK_REJECTION_SPIKE"),
            ("RULE_TRADE_DROPOUT", "TRADE_DROPOUT"),
            ("CUSTOM_RULE_UNKNOWN", "CUSTOM_RULE_UNKNOWN"),
        ],
    )
    def test_rule_failure_mode_mapping(self, rule_name: str, expected_failure_mode: str) -> None:
        ledger = EvaluationLedger()
        registry = StrategyRegistry()
        spec = _make_spec()
        qual = _make_qualification(spec)
        strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
        _, event = _create_snapshot_and_event(
            ledger, strat_id, qual, baseline, rule_name=rule_name
        )

        service = ResearchFeedbackService(registry, ledger)
        fb = service.create_feedback_from_event(
            event,
            empirical_notes="Automated test",
            created_at="2026-01-02T18:00:00+00:00",
        )
        assert fb.failure_mode == expected_failure_mode
        assert fb.strategy_id == strat_id
        assert fb.qualification_hash == qual.record_hash
        assert fb.degradation_event_hash == event.event_hash
        assert fb.verify_digest() is True

    def test_metrics_populated_from_authoritative_snapshot(self) -> None:
        ledger = EvaluationLedger()
        registry = StrategyRegistry()
        spec = _make_spec()
        qual = _make_qualification(spec)
        strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
        _, event = _create_snapshot_and_event(
            ledger,
            strat_id,
            qual,
            baseline,
            rule_name="RULE_DD_EXPANSION_CRITICAL",
            max_drawdown_bps=300.0,
            realized_sharpe=0.42,
            realized_slippage_bps=4.5,
            details_dict={"baseline_max_dd_bps": 150.0},
        )

        service = ResearchFeedbackService(registry, ledger)
        fb = service.create_feedback_from_event(
            event,
            empirical_notes="Detailed analysis",
            created_at="2026-01-02T18:00:00+00:00",
        )
        assert fb.realized_sharpe == 0.42
        assert fb.realized_slippage_bps == 4.5
        assert fb.drawdown_expansion_ratio == 2.0  # 300 / 150


# ---------------------------------------------------------------------------
# 3. Governance Integration: re_research_strategy with Feedback
# ---------------------------------------------------------------------------

class TestGovernanceReResearchWithFeedback:
    def test_re_research_binds_feedback_evidence(self) -> None:
        ledger = EvaluationLedger()
        registry = StrategyRegistry()
        spec = _make_spec()
        qual = _make_qualification(spec)
        strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
        _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

        governance = PaperGovernanceService(registry, ledger)
        service = ResearchFeedbackService(registry, ledger)

        # Degrade first
        governance.degrade_strategy(
            event,
            timestamp="2026-01-02T16:30:00+00:00",
        )
        assert registry.get_strategy(strat_id).state == StrategyLifecycleState.DEGRADED

        # Create feedback record
        fb = service.create_feedback_from_event(
            event,
            empirical_notes="Investigate momentum regime change",
            created_at="2026-01-02T17:00:00+00:00",
        )

        # Re-research demotion with feedback_record
        trans = governance.re_research_strategy(
            strat_id,
            reason="Demoted to research for parameter revision",
            feedback_record=fb,
            timestamp="2026-01-02T17:30:00+00:00",
        )

        assert trans.evidence_type == "RESEARCH_FEEDBACK"
        assert trans.evidence_hash == fb.feedback_hash
        assert trans.new_state == StrategyLifecycleState.RESEARCH.value
        assert registry.get_strategy(strat_id).state == StrategyLifecycleState.RESEARCH
        # Qualification hash in registry is cleared upon return to research
        assert registry.get_strategy(strat_id).qualification_hash is None

    def test_re_research_without_feedback_maintains_backward_compat(self) -> None:
        ledger = EvaluationLedger()
        registry = StrategyRegistry()
        spec = _make_spec()
        qual = _make_qualification(spec)
        strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
        _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

        governance = PaperGovernanceService(registry, ledger)
        governance.degrade_strategy(event, timestamp="2026-01-02T16:30:00+00:00")

        trans = governance.re_research_strategy(
            strat_id,
            reason="Demoted without explicit feedback record",
            timestamp="2026-01-02T17:30:00+00:00",
        )
        assert trans.evidence_type == "re_research_decision"
        assert trans.new_state == StrategyLifecycleState.RESEARCH.value


# ---------------------------------------------------------------------------
# 4. Research Task Bridge Tests
# ---------------------------------------------------------------------------

class TestResearchFeedbackBridge:
    def test_create_research_task_from_feedback(self) -> None:
        ledger = EvaluationLedger()
        registry = StrategyRegistry()
        spec = _make_spec()
        qual = _make_qualification(spec)
        strat_id, baseline = _setup_active_strategy(registry, ledger, qual, spec)
        _, event = _create_snapshot_and_event(ledger, strat_id, qual, baseline)

        service = ResearchFeedbackService(registry, ledger)
        fb = service.create_feedback_from_event(
            event,
            empirical_notes="Empirical failure notes",
            created_at="2026-01-02T17:00:00+00:00",
        )

        task = create_research_task_from_feedback(fb, research_protocol_version="RP-1.0")

        # Verify task is a ResearchTask subclass
        assert isinstance(task, ResearchTask)
        assert isinstance(task, ResearchFeedbackTask)
        assert task.task_id.startswith("TASK-RFB-")
        assert task.strategy_id == strat_id
        assert task.qualification_hash == qual.record_hash
        assert task.feedback_hash == fb.feedback_hash
        assert task.failure_mode == fb.failure_mode
        assert task.drawdown_expansion_ratio == fb.drawdown_expansion_ratio
        assert task.realized_sharpe == fb.realized_sharpe

        # Format context contains structured summary
        ctx = task.format_research_context()
        assert "RESEARCH HYPOTHESIS FORMULATION CONTEXT" in ctx
        assert fb.strategy_id in ctx
        assert fb.failure_mode in ctx

        # To dict
        d = task.to_dict()
        assert d["task_id"] == task.task_id
        assert d["failure_mode"] == fb.failure_mode
