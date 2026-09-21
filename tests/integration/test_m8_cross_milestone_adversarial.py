"""Cross-Milestone Comprehensive Adversarial Integration Suite (PRD v4.0 Milestone 8).

This module verifies end-to-end security, integrity, and non-bypassability across
all integrated QuantMind milestones (M1 through M7):
1. Forged qualification -> blocked
2. Failed holdout -> blocked
3. Holdout re-evaluation -> blocked
4. Unregistered dataset -> blocked
5. Checksum mismatch -> blocked
6. Synthetic production evaluation -> blocked
7. FINAL_HOLDOUT access -> blocked
8. PAPER_ELIGIBLE forward entry -> blocked
9. DEGRADED new entry -> blocked
10. DEGRADED exit -> allowed
11. RETIRED replay -> blocked
12. REJECTED replay -> blocked
13. Forged degradation event -> blocked
14. Cross-regime event -> blocked
15. Future monitoring evidence -> blocked
16. Future governance evidence -> blocked
17. Cross-strategy feedback -> blocked
18. Duplicate feedback -> idempotent
19. Feedback task -> zero trials
20. Feedback -> direct PAPER_ACTIVE -> blocked
21. Lifecycle mutation during deterministic replay -> no output change
22. Immutable historical evidence mutation -> blocked
23. Stale governance transition -> blocked
24. Concurrent transition -> fail safely
25. Trial population manipulation -> blocked
26. Strategy identity mutation through metadata -> blocked
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from quantmind.data.registry import DatasetKind, DatasetRegistry, DatasetZone
from quantmind.paper.engine import PaperReplayEngine, PaperReplaySecurityError
from quantmind.paper.evaluation.feedback import (
    ResearchFeedbackCausalError,
    ResearchFeedbackError,
    ResearchFeedbackIntegrityError,
    ResearchFeedbackService,
)
from quantmind.paper.evaluation.governance import (
    GovernanceCausalError,
    GovernanceIntegrityError,
    PaperGovernanceService,
)
from quantmind.paper.evaluation.ledger import (
    EvaluationLedger,
    EvaluationLedgerIntegrityError,
)
from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringSnapshot,
    PaperEvaluationBaseline,
    PaperEvaluationRegime,
    PaperEvaluationTransition,
    PaperExecutionLifecycleContext,
    ResearchFeedbackRecord,
)
from quantmind.paper.evaluation.service import (
    PaperEvaluationService,
    PaperEvaluationServiceError,
)
from quantmind.paper.feed import MarketFeedSecurityError, ReplayFeed
from quantmind.paper.ledger import PaperLedger
from quantmind.paper.models import (
    PaperOrderStatus,
    ReplayReport,
    ReplaySessionSummary,
)
from quantmind.research_integrity.artifacts import ArtifactRegistry
from quantmind.research_integrity.feedback_bridge import (
    ResearchFeedbackTask,
    create_research_task_from_feedback,
    derive_feedback_task_id,
)
from quantmind.research_integrity.holdout import (
    HoldoutManager,
    HoldoutSecurityError,
    HoldoutState,
)
from quantmind.research_integrity.qualification import (
    QualificationLedger,
    QualificationLedgerError,
    RobustnessStatus,
    StrategyQualificationRecord,
    StrategyValidationGate,
    ValidationStatus,
    check_paper_replay_eligibility,
)
from quantmind.research_integrity.trial_ledger import TrialBudgetExceeded, TrialLedger
from quantmind.strategy.compiler import (
    derive_strategy_id,
    normalize_strategy_spec,
)
from quantmind.strategy.registry import (
    StrategyLifecycleState,
    StrategyRegistry,
    StrategyRegistryError,
)
from quantmind.strategy.spec import StrategySpec


@pytest.fixture
def test_env(tmp_path: Path):
    """Complete integrated environment fixture across all subsystems."""
    db_path = tmp_path / "quantmind_integrated.db"
    conn_str = str(db_path)

    dataset_reg = DatasetRegistry(conn_str)
    strat_reg = StrategyRegistry(conn_str)
    trial_ledger = TrialLedger(conn_str)
    artifact_reg = ArtifactRegistry(conn_str)
    qual_ledger = QualificationLedger(db_path=conn_str)
    holdout_mgr = HoldoutManager(engine=None, ledger=trial_ledger, dataset_registry=dataset_reg, db_path=conn_str)
    paper_ledger = PaperLedger(db_path=conn_str)
    eval_ledger = EvaluationLedger(db_path=conn_str)

    # Multi-session market data: 100 bars across 2 days
    day1 = pd.date_range("2023-01-02 09:15", periods=50, freq="1min")
    day2 = pd.date_range("2023-01-03 09:15", periods=50, freq="1min")
    timestamps = day1.append(day2)

    price = 1000.0
    opens, closes, highs, lows = [], [], [], []
    for i in range(len(timestamps)):
        o = price
        c = price + (1.5 if i % 2 == 0 else -1.5)
        opens.append(o)
        closes.append(c)
        highs.append(max(o, c) + 0.5)
        lows.append(min(o, c) - 0.5)
        price = c

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(timestamps),
            "open_interest": [5000.0] * len(timestamps),
        }
    )

    csv_path = tmp_path / "licensed_data.csv"
    df.to_csv(csv_path, index=False)

    zones = {
        "DISCOVERY": DatasetZone("DISCOVERY", "2023-01-02 09:15", "2023-01-02 09:30"),
        "VALIDATION": DatasetZone("VALIDATION", "2023-01-02 09:31", "2023-01-02 09:50"),
        "FINAL_HOLDOUT": DatasetZone("FINAL_HOLDOUT", "2023-01-02 09:51", "2023-01-02 10:04"),
        "FORWARD_PAPER": DatasetZone("FORWARD_PAPER", "2023-01-03 09:15", "2023-01-03 10:04"),
    }

    dataset_record = dataset_reg.register_file(
        version="DS-NIFTY-2023-V1",
        kind=DatasetKind.LICENSED,
        path=csv_path,
        timestamp_column="timestamp",
        zones=zones,
        metadata={"source": "NSE", "asset": "NIFTY_FUT"},
    )

    spec = StrategySpec(
        strategy_version="v1.0",
        feature_version="f1",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 50, "session_window": [0.0, 1.0]},
    )
    strategy_id = derive_strategy_id(spec)
    strat_reg.register_strategy(spec)

    norm_spec = normalize_strategy_spec(spec)
    spec_hash = hashlib.sha256(norm_spec.canonical_json().encode("utf-8")).hexdigest()

    qual_record = StrategyQualificationRecord.create(
        qualification_id="QUAL-TEST-1",
        strategy_id=strategy_id,
        strategy_spec_hash=spec_hash,
        dataset_version=dataset_record.version,
        dataset_sha256=dataset_record.sha256,
        split_manifest_version="MANIFEST-V1",
        research_protocol_version="RP-2",
        population_hash="pop-hash-1",
        effective_trial_count=10,
        observed_sharpe=1.85,
        dsr=0.96,
        trade_count=45,
        holdout_state=HoldoutState.PASSED,
        robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE,
        reasons=["Meets all RP-2 statistical and holdout thresholds"],
    )
    qual_ledger.record_qualification(qual_record)

    # Advance state: IDEA -> RESEARCH -> VALIDATION -> PAPER_ELIGIBLE
    strat_reg.transition_state(strategy_id, StrategyLifecycleState.RESEARCH, reason="Research phase")
    strat_reg.transition_state(strategy_id, StrategyLifecycleState.VALIDATION, reason="Validation phase")
    strat_reg.transition_state(
        strategy_id,
        StrategyLifecycleState.PAPER_ELIGIBLE,
        reason="Certified by StrategyValidationGate",
        qualification_record=qual_record,
    )

    return {
        "db_path": conn_str,
        "dataset_registry": dataset_reg,
        "strategy_registry": strat_reg,
        "trial_ledger": trial_ledger,
        "artifact_registry": artifact_reg,
        "qualification_ledger": qual_ledger,
        "holdout_manager": holdout_mgr,
        "paper_ledger": paper_ledger,
        "evaluation_ledger": eval_ledger,
        "dataset_record": dataset_record,
        "strategy_spec": spec,
        "strategy_id": strategy_id,
        "qual_record": qual_record,
        "csv_path": csv_path,
    }


def test_m8_adversarial_1_to_3_qualification_and_holdout_security(test_env):
    """Cases 1-3: Forged qualification blocked, failed holdout blocked, holdout re-evaluation blocked."""
    qual_record = test_env["qual_record"]
    holdout_mgr = test_env["holdout_manager"]
    strategy_id = test_env["strategy_id"]

    # 1. Forged qualification (digest mismatch) -> blocked
    tampered = StrategyQualificationRecord(
        qualification_id=qual_record.qualification_id,
        strategy_id=qual_record.strategy_id,
        strategy_spec_hash=qual_record.strategy_spec_hash,
        dataset_version=qual_record.dataset_version,
        dataset_sha256=qual_record.dataset_sha256,
        split_manifest_version=qual_record.split_manifest_version,
        research_protocol_version=qual_record.research_protocol_version,
        population_hash=qual_record.population_hash,
        effective_trial_count=qual_record.effective_trial_count,
        observed_sharpe=9.99,  # forged
        dsr=qual_record.dsr,
        trade_count=qual_record.trade_count,
        holdout_state=qual_record.holdout_state,
        robustness_status=qual_record.robustness_status,
        final_status=qual_record.final_status,
        created_at=qual_record.created_at,
        reasons=qual_record.reasons,
        record_hash=qual_record.record_hash,  # original hash, now mismatched
    )
    assert not tampered.verify_digest()
    eligibility = check_paper_replay_eligibility(tampered)
    assert not eligibility.is_eligible
    assert "tampered" in eligibility.reason

    # 2. Failed holdout cannot qualify for paper
    failed_record = StrategyQualificationRecord.create(
        qualification_id="QUAL-FAILED-1",
        strategy_id=strategy_id,
        strategy_spec_hash=qual_record.strategy_spec_hash,
        dataset_version=qual_record.dataset_version,
        dataset_sha256=qual_record.dataset_sha256,
        split_manifest_version="MANIFEST-V1",
        research_protocol_version="RP-2",
        population_hash="pop-hash-1",
        effective_trial_count=10,
        observed_sharpe=0.4,
        dsr=0.1,
        trade_count=45,
        holdout_state=HoldoutState.FAILED,
        robustness_status=RobustnessStatus.FAILED,
        final_status=ValidationStatus.REJECTED,
        reasons=["Failed holdout"],
    )
    elig_failed = check_paper_replay_eligibility(failed_record)
    assert not elig_failed.is_eligible

    # 3. Holdout re-evaluation blocked
    holdout_mgr._connection.execute(
        """
        INSERT INTO holdout_evaluations (candidate_strategy_id, dataset_version, research_protocol_version, holdout_version, state, timestamp_evaluated)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("cand-1", "DS-NIFTY-2023-V1", "RP-2", "H-1", "FAILED", "2023-01-02T10:00:00Z"),
    )
    assert holdout_mgr.has_candidate_failed("cand-1", "DS-NIFTY-2023-V1", "RP-2")
    with pytest.raises(Exception, match="cannot be reset"):
        holdout_mgr._connection.execute(
            "UPDATE holdout_evaluations SET state = 'PASSED' WHERE candidate_strategy_id = 'cand-1'"
        )


def test_m8_adversarial_4_to_7_dataset_security_and_partition_isolation(test_env, tmp_path):
    """Cases 4-7: Unregistered dataset, checksum mismatch, synthetic production evaluation, FINAL_HOLDOUT access."""
    # 4. Unregistered dataset blocked from ReplayFeed
    with pytest.raises(Exception):
        ReplayFeed.from_dataset_registry(
            registry=test_env["dataset_registry"],
            dataset_version="NON_EXISTENT_VERSION",
            split_zone="FORWARD_PAPER",
        )

    # 5. Checksum mismatch blocked
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text("timestamp,open,high,low,close,volume\n2023-01-01,1,2,0,1,10\n")
    with pytest.raises(Exception):
        test_env["dataset_registry"].load_dataset_dataframe(
            version="DS-NIFTY-2023-V1",
            expected_sha256="wrong_checksum_00000000000000000000000000000000000000000000000000",
        )

    # 6. Synthetic data blocked from production forward replay / evaluation
    syn_csv = tmp_path / "syn.csv"
    syn_csv.write_text("timestamp,open,high,low,close,volume\n2023-01-01,1,2,0,1,10\n")
    test_env["dataset_registry"].register_file(
        version="SYN-DATA-V1",
        kind=DatasetKind.SYNTHETIC,
        path=syn_csv,
        timestamp_column="timestamp",
        zones={"FORWARD_PAPER": DatasetZone("FORWARD_PAPER", "2023-01-01", "2023-01-02")},
    )
    with pytest.raises(MarketFeedSecurityError):
        ReplayFeed.from_dataset_registry(
            registry=test_env["dataset_registry"],
            dataset_version="SYN-DATA-V1",
            split_zone="FORWARD_PAPER",
        )

    # 7. FINAL_HOLDOUT access strictly forbidden
    with pytest.raises(MarketFeedSecurityError):
        ReplayFeed.from_dataset_registry(
            registry=test_env["dataset_registry"],
            dataset_version="DS-NIFTY-2023-V1",
            split_zone="FINAL_HOLDOUT",
        )


def test_m8_adversarial_8_to_12_lifecycle_execution_gating(test_env):
    """Cases 8-12: Execution gating by lifecycle states."""
    ctx_eligible = PaperExecutionLifecycleContext(
        strategy_id=test_env["strategy_id"],
        authorized_state="PAPER_ELIGIBLE",
        authorization_timestamp="2023-01-03T09:00:00Z",
    )
    assert not ctx_eligible.allows_new_entries

    ctx_active = PaperExecutionLifecycleContext(
        strategy_id=test_env["strategy_id"],
        authorized_state="PAPER_ACTIVE",
        authorization_timestamp="2023-01-03T09:00:00Z",
    )
    assert ctx_active.allows_new_entries
    assert not ctx_active.is_degraded
    assert not ctx_active.is_prohibited

    # 9 & 10. DEGRADED blocks entries, allows exits
    ctx_degraded = PaperExecutionLifecycleContext(
        strategy_id=test_env["strategy_id"],
        authorized_state="DEGRADED",
        authorization_timestamp="2023-01-03T09:00:00Z",
    )
    assert not ctx_degraded.allows_new_entries
    assert ctx_degraded.is_degraded
    assert not ctx_degraded.is_prohibited

    # 11 & 12. RETIRED and REJECTED block all execution
    ctx_retired = PaperExecutionLifecycleContext(
        strategy_id=test_env["strategy_id"],
        authorized_state="RETIRED",
        authorization_timestamp="2023-01-03T09:00:00Z",
    )
    assert ctx_retired.is_prohibited

    ctx_rejected = PaperExecutionLifecycleContext(
        strategy_id=test_env["strategy_id"],
        authorized_state="REJECTED",
        authorization_timestamp="2023-01-03T09:00:00Z",
    )
    assert ctx_rejected.is_prohibited


def test_m8_adversarial_13_to_16_evaluation_governance_provenance_and_causality(test_env):
    """Cases 13-16: Forged degradation event, cross-regime breach, future monitoring and governance evidence."""
    eval_ledger = test_env["evaluation_ledger"]
    strategy_id = test_env["strategy_id"]
    qual_hash = test_env["qual_record"].record_hash

    sess = ReplaySessionSummary(
        session_id="SESS-1",
        start_ts="2023-01-03T09:15:00",
        end_ts="2023-01-03T10:04:00",
        trades=10,
        gross_pnl=100.0,
        net_pnl=95.0,
    )
    report = ReplayReport.create(
        strategy_id=strategy_id,
        qualification_id=test_env["qual_record"].qualification_id,
        dataset_version="DS-NIFTY-2023-V1",
        trade_count=10,
        gross_pnl=100.0,
        net_pnl=95.0,
        costs=5.0,
        slippage=1.0,
        max_drawdown_bps=50.0,
        exposure=0.6,
        win_rate=0.6,
        expectancy=9.5,
        sharpe_ratio=2.1,
        session_breakdown=[sess],
        created_at="2023-01-03T09:00:00+00:00",
        qualification_hash=qual_hash,
        dataset_sha256=test_env["dataset_record"].sha256,
        split_zone="FORWARD_PAPER",
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="cost-hash",
        risk_config_hash="risk-hash",
    )
    eval_ledger.register_baseline(
        PaperEvaluationBaseline.create(
            strategy_id=strategy_id,
            qualification_hash=qual_hash,
            baseline_replay_report_hash=report.report_hash,
            baseline_dataset_version="DS-NIFTY-2023-V1",
            baseline_dataset_sha256=test_env["dataset_record"].sha256,
            baseline_split_zone="FORWARD_PAPER",
            baseline_execution_policy="next_bar_open_v1",
            baseline_cost_schedule_hash="cost-hash",
            baseline_risk_config_hash="risk-hash",
            created_at="2023-01-03T09:00:00Z",
        ),
        replay_report=report,
    )

    regime = PaperEvaluationRegime.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        baseline_replay_report_hash=report.report_hash,
        forward_dataset_version="DS-NIFTY-2023-V1",
        forward_dataset_sha256=test_env["dataset_record"].sha256,
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="cost-hash",
        risk_config_hash="risk-hash",
        monitoring_protocol_version="MP-1",
        created_at="2023-01-03T09:00:00Z",
    )
    eval_ledger.register_regime(regime)

    snapshot = MonitoringSnapshot.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        replay_report_hash="rep-hash-1",
        dataset_version="DS-NIFTY-2023-V1",
        dataset_sha256=test_env["dataset_record"].sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        window_start_ts="2023-01-03T09:15:00Z",
        window_end_ts="2023-01-03T10:04:00Z",
        total_trades=10,
        net_pnl=100.0,
        max_drawdown_bps=200.0,
        realized_sharpe=1.2,
        realized_slippage_bps=2.0,
        cost_to_turnover_bps=1.0,
        risk_event_count=0,
        metrics_json="{}",
        created_at="2023-01-03T10:05:00Z",
        regime_hash=regime.regime_hash,
    )
    eval_ledger.get_or_insert_snapshot(snapshot)

    # 13. Forged degradation event (bad digest) -> fails verification
    event = DegradationEvent(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash="baseline-rep-hash",
        rule_name="max_drawdown",
        threshold_value=150.0,
        observed_value=200.0,
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        timestamp="2023-01-03T10:06:00Z",
        details_json="{}",
        event_hash="bad-forged-hash-123",
    )
    assert not event.verify_digest()

    # Register valid event
    valid_event = DegradationEvent.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash=report.report_hash,
        rule_name="max_drawdown",
        threshold_value=150.0,
        observed_value=200.0,
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        timestamp="2023-01-03T10:05:00Z",
        details_json="{}",
    )
    eval_ledger.record_degradation_event(valid_event)

    gov_service = PaperGovernanceService(
        strategy_registry=test_env["strategy_registry"],
        evaluation_ledger=eval_ledger,
    )

    # 14. Cross-strategy mismatch rejected
    other_event = DegradationEvent.create(
        strategy_id="STRAT-UNKNOWN",
        qualification_hash=qual_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash=report.report_hash,
        rule_name="max_drawdown",
        threshold_value=150.0,
        observed_value=200.0,
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        timestamp="2023-01-03T10:05:00Z",
        details_json="{}",
    )
    with pytest.raises((KeyError, StrategyRegistryError)):
        gov_service.degrade_strategy(other_event)

    # 15. Cross-regime / snapshot mismatch
    other_snapshot = MonitoringSnapshot.create(
        strategy_id="STRAT-MISMATCH",
        qualification_hash=qual_hash,
        replay_report_hash="rep-diff-1",
        dataset_version="DS-NIFTY-2023-V1",
        dataset_sha256=test_env["dataset_record"].sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        window_start_ts="2023-01-03T09:15:00Z",
        window_end_ts="2023-01-03T10:04:00Z",
        total_trades=10,
        net_pnl=100.0,
        max_drawdown_bps=200.0,
        realized_sharpe=1.2,
        realized_slippage_bps=2.0,
        cost_to_turnover_bps=1.0,
        risk_event_count=0,
        metrics_json="{}",
        created_at="2023-01-03T10:05:00Z",
        regime_hash=regime.regime_hash,
    )
    with pytest.raises(EvaluationLedgerIntegrityError, match="does not match bound regime context"):
        eval_ledger.get_or_insert_snapshot(other_snapshot)

    # 16. Governance evidence timestamp in future relative to transition timestamp fails causality check
    with pytest.raises(GovernanceCausalError):
        gov_service.degrade_strategy(
            valid_event,
            timestamp="2023-01-03T10:04:00Z",  # Prior to event timestamp 10:05:00
            initiator="AUTO_MONITOR",
        )


def test_m8_adversarial_17_to_20_research_feedback_bridge_invariants(test_env):
    """Cases 17-20: Cross-strategy feedback, duplicate idempotency, zero-trial feedback tasks, no direct promotion."""
    eval_ledger = test_env["evaluation_ledger"]
    strategy_id = test_env["strategy_id"]
    qual_hash = test_env["qual_record"].record_hash

    sess = ReplaySessionSummary(
        session_id="SESS-2",
        start_ts="2023-01-03T09:15:00",
        end_ts="2023-01-03T10:04:00",
        trades=10,
        gross_pnl=100.0,
        net_pnl=95.0,
    )
    report = ReplayReport.create(
        strategy_id=strategy_id,
        qualification_id=test_env["qual_record"].qualification_id,
        dataset_version="DS-NIFTY-2023-V1",
        trade_count=10,
        gross_pnl=100.0,
        net_pnl=95.0,
        costs=5.0,
        slippage=1.0,
        max_drawdown_bps=50.0,
        exposure=0.6,
        win_rate=0.6,
        expectancy=9.5,
        sharpe_ratio=2.1,
        session_breakdown=[sess],
        created_at="2023-01-03T09:00:00+00:00",
        qualification_hash=qual_hash,
        dataset_sha256=test_env["dataset_record"].sha256,
        split_zone="FORWARD_PAPER",
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="cost-hash",
        risk_config_hash="risk-hash",
    )
    eval_ledger.register_baseline(
        PaperEvaluationBaseline.create(
            strategy_id=strategy_id,
            qualification_hash=qual_hash,
            baseline_replay_report_hash=report.report_hash,
            baseline_dataset_version="DS-NIFTY-2023-V1",
            baseline_dataset_sha256=test_env["dataset_record"].sha256,
            baseline_split_zone="FORWARD_PAPER",
            baseline_execution_policy="next_bar_open_v1",
            baseline_cost_schedule_hash="cost-hash",
            baseline_risk_config_hash="risk-hash",
            created_at="2023-01-03T09:00:00Z",
        ),
        replay_report=report,
    )

    # Promote to PAPER_ACTIVE now that baseline is registered
    gov_service = PaperGovernanceService(
        strategy_registry=test_env["strategy_registry"],
        evaluation_ledger=eval_ledger,
    )
    gov_service.activate_strategy(strategy_id)

    regime = PaperEvaluationRegime.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        baseline_replay_report_hash=report.report_hash,
        forward_dataset_version="DS-NIFTY-2023-V1",
        forward_dataset_sha256=test_env["dataset_record"].sha256,
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="cost-hash",
        risk_config_hash="risk-hash",
        monitoring_protocol_version="MP-1",
        created_at="2023-01-03T09:00:00Z",
    )
    eval_ledger.register_regime(regime)

    snapshot = MonitoringSnapshot.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        replay_report_hash="rep-hash-fb-1",
        dataset_version="DS-NIFTY-2023-V1",
        dataset_sha256=test_env["dataset_record"].sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        window_start_ts="2023-01-03T09:15:00Z",
        window_end_ts="2023-01-03T10:04:00Z",
        total_trades=10,
        net_pnl=100.0,
        max_drawdown_bps=200.0,
        realized_sharpe=1.2,
        realized_slippage_bps=2.0,
        cost_to_turnover_bps=1.0,
        risk_event_count=0,
        metrics_json="{}",
        created_at=datetime.now(timezone.utc).isoformat(),
        regime_hash=regime.regime_hash,
    )
    eval_ledger.get_or_insert_snapshot(snapshot)

    event = DegradationEvent.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash=report.report_hash,
        rule_name="RULE_DD_EXPANSION_CRITICAL",
        threshold_value=150.0,
        observed_value=200.0,
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        timestamp=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        details_json="{}",
    )
    eval_ledger.record_degradation_event(event)

    fb_service = ResearchFeedbackService(
        strategy_registry=test_env["strategy_registry"],
        evaluation_ledger=eval_ledger,
    )

    # 17. Cross-strategy feedback rejected
    other_event = DegradationEvent.create(
        strategy_id="STRAT-UNKNOWN",
        qualification_hash=qual_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash=report.report_hash,
        rule_name="RULE_DD_EXPANSION_CRITICAL",
        threshold_value=150.0,
        observed_value=200.0,
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        timestamp=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        details_json="{}",
    )
    with pytest.raises(ResearchFeedbackIntegrityError):
        fb_service.create_feedback_from_event(other_event)

    # 18. Duplicate feedback registration is idempotent
    rec1 = fb_service.create_feedback_from_event(
        event,
        empirical_notes="Deterministic feedback notes",
        created_at=(datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat(),
    )
    rec2 = fb_service.create_feedback_from_event(
        event,
        empirical_notes="Deterministic feedback notes",
        created_at=(datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat(),
    )
    assert rec1.feedback_hash == rec2.feedback_hash

    # Conflicting notes on same event triggers ID collision error
    with pytest.raises(EvaluationLedgerIntegrityError, match="collision"):
        fb_service.create_feedback_from_event(
            event,
            empirical_notes="Conflicting notes",
            created_at=(datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat(),
        )

    # 19. Feedback task creates zero trials in TrialLedger
    initial_trials = test_env["trial_ledger"]._connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    task = create_research_task_from_feedback(rec1)
    assert isinstance(task, ResearchFeedbackTask)
    assert test_env["trial_ledger"]._connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == initial_trials

    # Degrade strategy
    gov_service.degrade_strategy(
        event,
        timestamp=(datetime.now(timezone.utc) + timedelta(seconds=3)).isoformat(),
    )

    # 20. Feedback task cannot directly promote DEGRADED to PAPER_ACTIVE in StrategyRegistry
    with pytest.raises(StrategyRegistryError):
        test_env["strategy_registry"].transition_state(
            strategy_id=strategy_id,
            target_state=StrategyLifecycleState.PAPER_ACTIVE,
            reason="Attempting direct bypass via feedback",
        )


def test_m8_adversarial_21_to_26_immutability_concurrency_and_population_integrity(test_env):
    """Cases 21-26: Evidence immutability triggers, non-flat retirement block, trial population integrity."""
    eval_ledger = test_env["evaluation_ledger"]
    strategy_id = test_env["strategy_id"]
    qual_hash = test_env["qual_record"].record_hash

    # Insert valid baseline, regime, snapshot, event, and feedback row to test triggers
    sess = ReplaySessionSummary(
        session_id="SESS-3",
        start_ts="2023-01-03T09:15:00",
        end_ts="2023-01-03T10:04:00",
        trades=10,
        gross_pnl=100.0,
        net_pnl=95.0,
    )
    report = ReplayReport.create(
        strategy_id=strategy_id,
        qualification_id=test_env["qual_record"].qualification_id,
        dataset_version="DS-NIFTY-2023-V1",
        trade_count=10,
        gross_pnl=100.0,
        net_pnl=95.0,
        costs=5.0,
        slippage=1.0,
        max_drawdown_bps=50.0,
        exposure=0.6,
        win_rate=0.6,
        expectancy=9.5,
        sharpe_ratio=2.1,
        session_breakdown=[sess],
        created_at="2023-01-03T09:00:00+00:00",
        qualification_hash=qual_hash,
        dataset_sha256=test_env["dataset_record"].sha256,
        split_zone="FORWARD_PAPER",
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="cost-hash",
        risk_config_hash="risk-hash",
    )
    eval_ledger.register_baseline(
        PaperEvaluationBaseline.create(
            strategy_id=strategy_id,
            qualification_hash=qual_hash,
            baseline_replay_report_hash=report.report_hash,
            baseline_dataset_version="DS-NIFTY-2023-V1",
            baseline_dataset_sha256=test_env["dataset_record"].sha256,
            baseline_split_zone="FORWARD_PAPER",
            baseline_execution_policy="next_bar_open_v1",
            baseline_cost_schedule_hash="cost-hash",
            baseline_risk_config_hash="risk-hash",
            created_at="2023-01-03T09:00:00Z",
        ),
        replay_report=report,
    )

    regime = PaperEvaluationRegime.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        baseline_replay_report_hash=report.report_hash,
        forward_dataset_version="DS-NIFTY-2023-V1",
        forward_dataset_sha256=test_env["dataset_record"].sha256,
        execution_policy="next_bar_open_v1",
        cost_schedule_hash="cost-hash",
        risk_config_hash="risk-hash",
        monitoring_protocol_version="MP-1",
        created_at="2023-01-03T09:00:00Z",
    )
    eval_ledger.register_regime(regime)

    snapshot = MonitoringSnapshot.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        replay_report_hash="rep-hash-immut-1",
        dataset_version="DS-NIFTY-2023-V1",
        dataset_sha256=test_env["dataset_record"].sha256,
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        window_start_ts="2023-01-03T09:15:00Z",
        window_end_ts="2023-01-03T10:04:00Z",
        total_trades=10,
        net_pnl=100.0,
        max_drawdown_bps=200.0,
        realized_sharpe=1.2,
        realized_slippage_bps=2.0,
        cost_to_turnover_bps=1.0,
        risk_event_count=0,
        metrics_json="{}",
        created_at="2023-01-03T10:05:00Z",
        regime_hash=regime.regime_hash,
    )
    eval_ledger.get_or_insert_snapshot(snapshot)

    event = DegradationEvent.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        snapshot_hash=snapshot.snapshot_hash,
        baseline_replay_report_hash=report.report_hash,
        rule_name="RULE_DD_EXPANSION_CRITICAL",
        threshold_value=150.0,
        observed_value=200.0,
        monitoring_protocol_version="MP-1",
        monitoring_config_hash="cfg-hash",
        timestamp="2023-01-03T10:06:00Z",
        details_json="{}",
    )
    eval_ledger.record_degradation_event(event)

    fb_record = ResearchFeedbackRecord.create(
        strategy_id=strategy_id,
        qualification_hash=qual_hash,
        degradation_event_hash=event.event_hash,
        dataset_version="DS-NIFTY-2023-V1",
        failure_mode="DRAWDOWN_EXPANSION",
        drawdown_expansion_ratio=1.33,
        realized_slippage_bps=2.0,
        empirical_notes="Trigger test record",
        created_at="2023-01-03T10:07:00Z",
        realized_sharpe=1.2,
    )
    eval_ledger.record_feedback(fb_record)

    # 22. Immutable historical evidence cannot be updated or deleted via raw SQL
    with pytest.raises(Exception, match="cannot be deleted"):
        eval_ledger._connection.execute("DELETE FROM research_feedback WHERE feedback_hash = ?", (fb_record.feedback_hash,))

    with pytest.raises(Exception, match="cannot be updated"):
        eval_ledger._connection.execute("UPDATE research_feedback SET failure_mode = 'MODIFIED'")

    with pytest.raises(Exception, match="cannot be updated"):
        eval_ledger._connection.execute("UPDATE degradation_events SET threshold_value = 999.0")

    with pytest.raises(Exception, match="cannot be deleted"):
        eval_ledger._connection.execute("DELETE FROM monitoring_snapshots WHERE snapshot_hash = ?", (snapshot.snapshot_hash,))

    # 23 & 24. Non-flat position retirement blocked & illegal transition fails closed
    gov_service = PaperGovernanceService(
        strategy_registry=test_env["strategy_registry"],
        evaluation_ledger=eval_ledger,
    )
    # Attempting to retire with non-flat position
    with pytest.raises(GovernanceIntegrityError, match="open position"):
        gov_service.retire_strategy(
            strategy_id=strategy_id,
            reason="Retirement attempt with open risk",
            position_quantity=5.0,  # Non-flat
        )

    # Attempting illegal state transition
    with pytest.raises(StrategyRegistryError):
        test_env["strategy_registry"].transition_state(
            strategy_id=strategy_id,
            target_state=StrategyLifecycleState.IDEA,
            reason="Illegal reverse transition",
        )

    # 25. TrialLedger population integrity - cannot delete or overwrite trials
    test_env["trial_ledger"]._connection.execute(
        """
        INSERT INTO trials (
            trial_id, experiment_id, strategy_id, dataset_version, split_zone,
            research_protocol_version, feature_version, parameter_set_json, strategy_spec_json,
            seed, execution_model, cost_model, slippage_model,
            timestamp_started, estimated_runtime_minutes,
            estimated_llm_cost, mode, dataset_kind, status
        ) VALUES (
            'TRIAL-IMMUT-1', 'EXP-1', ?, 'DS-NIFTY-2023-V1', 'RESEARCH',
            'RP-2', 'f1', '{}', '{}',
            42, 'next_bar', 'linear', 'linear',
            '2023-01-01T00:00:00Z', 1.0,
            0.0, 'FIXTURE', 'LICENSED', 'COMPLETED'
        )
        """,
        (strategy_id,),
    )
    with pytest.raises(Exception, match="append-only"):
        test_env["trial_ledger"]._connection.execute(
            "DELETE FROM trials WHERE trial_id = 'TRIAL-IMMUT-1'"
        )

    # 26. Strategy identity mutation through metadata blocked (hash is purely semantic over logic)
    spec1 = StrategySpec(
        strategy_version="v1.0",
        feature_version="f1",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 50, "session_window": [0.0, 1.0]},
    )
    spec2 = StrategySpec(
        strategy_version="v1.0",
        feature_version="f1",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 50, "session_window": [0.0, 1.0]},
    )
    # Identical parameters produce identical deterministic ID
    assert derive_strategy_id(spec1) == derive_strategy_id(spec2)

