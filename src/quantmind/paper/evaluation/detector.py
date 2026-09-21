"""Deterministic Degradation Detector (PRD v4.0 Milestone 4).

This module implements the pure, deterministic, observational degradation detector
for forward paper trading evaluation under Monitoring Protocol MP-1.0.

ARCHITECTURAL PRINCIPLES:
1. Pure Observation: The detector inspects MonitoringSnapshot against MonitoringConfig
   and authoritative qualification evidence. It has zero execution authority, makes
   zero database writes, performs zero lifecycle mutations, and invokes zero clocks.
2. Fail-Closed Provenance: All inputs are strictly type-checked and digest-verified.
   Untrusted, tampered, or mismatched inputs raise MonitoringProvenanceError immediately.
3. Cryptographic Determinism: Identical inputs produce identical DegradationEvents
   with identical event hashes. Events are deterministically sorted by rule_name.
4. Boundary Strictness: Strict inequalities are enforced (> or <); boundary equality
   never triggers a degradation event.
5. Zero Future Leakage / Wall-Clock Dependence: Event timestamps are derived strictly
   from the snapshot observation timestamp (snapshot.window_end_ts).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringConfig,
    MonitoringProvenanceError,
    MonitoringSnapshot,
    PaperEvaluationBaseline,
)
from quantmind.paper.models import ReplayReport
from quantmind.research_integrity.qualification import (
    StrategyQualificationRecord,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Rule Name Constants & Protocol Parameters
# ---------------------------------------------------------------------------

RULE_DD_EXPANSION_CRITICAL: str = "RULE_DD_EXPANSION_CRITICAL"
RULE_SHARPE_COLLAPSE: str = "RULE_SHARPE_COLLAPSE"
RULE_SLIPPAGE_ANOMALY: str = "RULE_SLIPPAGE_ANOMALY"
RULE_RISK_REJECTION_SPIKE: str = "RULE_RISK_REJECTION_SPIKE"
RULE_TRADE_DROPOUT: str = "RULE_TRADE_DROPOUT"

# MP-1 Protocol Specification Constants
MP1_DROPOUT_WINDOW_SESSIONS: int = 30
MIN_RISK_SUBMITTED_ORDERS: int = 10


# ---------------------------------------------------------------------------
# Detector Implementation
# ---------------------------------------------------------------------------


class DegradationDetector:
    """Deterministic, observational strategy degradation detector (MP-1).

    Evaluates a forward paper trading snapshot against qualification evidence
    and protocol configuration thresholds to detect empirical degradation.
    """

    def __init__(
        self,
        config: MonitoringConfig,
        baseline: PaperEvaluationBaseline | None = None,
    ) -> None:
        if type(config) is not MonitoringConfig:
            raise MonitoringProvenanceError(
                f"config must be an instance of MonitoringConfig (exact type required), got {type(config)}"
            )
        if baseline is not None and type(baseline) is not PaperEvaluationBaseline:
            raise MonitoringProvenanceError(
                f"baseline must be an instance of PaperEvaluationBaseline (exact type required), got {type(baseline)}"
            )
        self._config = config
        self._baseline = baseline

    @property
    def config(self) -> MonitoringConfig:
        return self._config

    @property
    def baseline(self) -> PaperEvaluationBaseline | None:
        return self._baseline

    def evaluate(
        self,
        snapshot: MonitoringSnapshot,
        qualification_record: StrategyQualificationRecord,
        baseline_replay_report: ReplayReport | None = None,
        *,
        baseline: PaperEvaluationBaseline | None = None,
        baseline_max_dd_bps: float | None = None,
        configured_slippage_bps: float | None = None,
        consecutive_inactive_sessions: int | None = None,
        submitted_orders: int | None = None,
        rejected_orders: int | None = None,
        rejection_rate: float | None = None,
    ) -> list[DegradationEvent]:
        """Evaluate a snapshot and return zero or more deterministic DegradationEvents."""
        effective_baseline = baseline if baseline is not None else self._baseline
        return detect_degradations(
            snapshot=snapshot,
            config=self._config,
            qualification_record=qualification_record,
            baseline_replay_report=baseline_replay_report,
            baseline=effective_baseline,
            baseline_max_dd_bps=baseline_max_dd_bps,
            configured_slippage_bps=configured_slippage_bps,
            consecutive_inactive_sessions=consecutive_inactive_sessions,
            submitted_orders=submitted_orders,
            rejected_orders=rejected_orders,
            rejection_rate=rejection_rate,
        )


def detect_degradations(
    snapshot: MonitoringSnapshot,
    config: MonitoringConfig,
    qualification_record: StrategyQualificationRecord,
    baseline_replay_report: ReplayReport | None = None,
    *,
    baseline: PaperEvaluationBaseline | None = None,
    baseline_max_dd_bps: float | None = None,
    configured_slippage_bps: float | None = None,
    consecutive_inactive_sessions: int | None = None,
    submitted_orders: int | None = None,
    rejected_orders: int | None = None,
    rejection_rate: float | None = None,
) -> list[DegradationEvent]:
    """Pure functional evaluation of paper trading performance against MP-1 degradation rules.

    Args:
        snapshot: Authoritative MonitoringSnapshot for the evaluation window.
        config: Authoritative MonitoringConfig.
        qualification_record: Authoritative StrategyQualificationRecord.
        baseline_replay_report: Optional authoritative ReplayReport for baseline replay evidence.
        baseline_max_dd_bps: Explicit baseline max drawdown in bps (must match replay report if both supplied).
        configured_slippage_bps: Explicit modeled slippage in bps (must match replay report if both supplied).
        consecutive_inactive_sessions: Explicit count of consecutive sessions with 0 trades.
        submitted_orders: Explicit total submitted order count in evaluation window.
        rejected_orders: Explicit risk-rejected order count in evaluation window.
        rejection_rate: Explicit risk rejection rate in evaluation window.

    Returns:
        List of DegradationEvent objects, sorted deterministically by rule_name.

    Raises:
        MonitoringProvenanceError: If any input fails provenance, digest, or type verification.
    """
    # -----------------------------------------------------------------------
    # 1. Fail-Closed Provenance Verification
    # -----------------------------------------------------------------------
    _verify_input_provenance(
        snapshot=snapshot,
        config=config,
        qualification_record=qualification_record,
        baseline_replay_report=baseline_replay_report,
        baseline=baseline,
    )

    # -----------------------------------------------------------------------
    # 2. Extract and Validate Authoritative Baseline Inputs
    # -----------------------------------------------------------------------
    effective_base_dd = _resolve_baseline_max_dd(
        baseline_replay_report=baseline_replay_report,
        explicit_base_dd=baseline_max_dd_bps,
        qualification_record=qualification_record,
    )

    effective_slip_bps = _resolve_configured_slippage(
        baseline_replay_report=baseline_replay_report,
        explicit_slip=configured_slippage_bps,
    )

    # Parse auxiliary context from snapshot.metrics_json if present
    aux_metrics = _parse_metrics_json(snapshot.metrics_json)

    # Resolve execution & rejection metrics
    effective_submitted, effective_rejection_rate = _resolve_risk_metrics(
        submitted_orders=submitted_orders,
        rejected_orders=rejected_orders,
        rejection_rate=rejection_rate,
        aux_metrics=aux_metrics,
    )

    # Resolve consecutive inactive sessions
    effective_inactive_sessions = _resolve_inactive_sessions(
        consecutive_inactive_sessions=consecutive_inactive_sessions,
        aux_metrics=aux_metrics,
    )

    # -----------------------------------------------------------------------
    # 3. Rule Evaluations
    # -----------------------------------------------------------------------
    events: list[DegradationEvent] = []
    timestamp = snapshot.window_end_ts

    # RULE 1: Drawdown Expansion Critical
    # Trigger: observed_dd_bps > baseline_max_dd_bps * max_drawdown_expansion_limit
    threshold_dd = effective_base_dd * config.max_drawdown_expansion_limit
    observed_dd = float(snapshot.max_drawdown_bps)
    if observed_dd > threshold_dd:
        details = json.dumps(
            {
                "baseline_max_dd_bps": round(float(effective_base_dd), 4),
                "max_drawdown_expansion_limit": round(float(config.max_drawdown_expansion_limit), 6),
                "threshold_dd_bps": round(float(threshold_dd), 4),
            },
            sort_keys=True,
        )
        events.append(
            DegradationEvent.create(
                strategy_id=snapshot.strategy_id,
                qualification_hash=snapshot.qualification_hash,
                snapshot_hash=snapshot.snapshot_hash,
                rule_name=RULE_DD_EXPANSION_CRITICAL,
                threshold_value=threshold_dd,
                observed_value=observed_dd,
                monitoring_protocol_version=snapshot.monitoring_protocol_version,
                monitoring_config_hash=snapshot.monitoring_config_hash,
                timestamp=timestamp,
                details_json=details,
            )
        )

    # RULE 2: Sharpe Collapse
    # Trigger: rolling_sharpe_30d < min_rolling_sharpe_30d (strict <)
    # If realized_sharpe is None, rule cannot trigger.
    if snapshot.realized_sharpe is not None:
        threshold_sharpe = float(config.min_rolling_sharpe_30d)
        observed_sharpe = float(snapshot.realized_sharpe)
        if observed_sharpe < threshold_sharpe:
            details = json.dumps(
                {
                    "min_rolling_sharpe_30d": round(float(threshold_sharpe), 6),
                    "realized_sharpe": round(float(observed_sharpe), 4),
                },
                sort_keys=True,
            )
            events.append(
                DegradationEvent.create(
                    strategy_id=snapshot.strategy_id,
                    qualification_hash=snapshot.qualification_hash,
                    snapshot_hash=snapshot.snapshot_hash,
                    rule_name=RULE_SHARPE_COLLAPSE,
                    threshold_value=threshold_sharpe,
                    observed_value=observed_sharpe,
                    monitoring_protocol_version=snapshot.monitoring_protocol_version,
                    monitoring_config_hash=snapshot.monitoring_config_hash,
                    timestamp=timestamp,
                    details_json=details,
                )
            )

    # RULE 3: Slippage Anomaly
    # Trigger: realized_slippage_bps > configured_slippage_bps * max_slippage_drift_ratio (strict >)
    # If configured_slippage_bps == 0.0, threshold is 0.0 bps; any realized_slippage_bps > 0.0 triggers.
    threshold_slip = effective_slip_bps * config.max_slippage_drift_ratio
    observed_slip = float(snapshot.realized_slippage_bps)
    if observed_slip > threshold_slip:
        details = json.dumps(
            {
                "configured_slippage_bps": round(float(effective_slip_bps), 4),
                "max_slippage_drift_ratio": round(float(config.max_slippage_drift_ratio), 6),
                "threshold_slippage_bps": round(float(threshold_slip), 4),
            },
            sort_keys=True,
        )
        events.append(
            DegradationEvent.create(
                strategy_id=snapshot.strategy_id,
                qualification_hash=snapshot.qualification_hash,
                snapshot_hash=snapshot.snapshot_hash,
                rule_name=RULE_SLIPPAGE_ANOMALY,
                threshold_value=threshold_slip,
                observed_value=observed_slip,
                monitoring_protocol_version=snapshot.monitoring_protocol_version,
                monitoring_config_hash=snapshot.monitoring_config_hash,
                timestamp=timestamp,
                details_json=details,
            )
        )

    # RULE 4: Risk Rejection Spike
    # Trigger: rejection_rate > max_risk_rejection_rate (strict >)
    # Gated by minimum sample: minimum submitted orders = 10.
    if effective_submitted >= MIN_RISK_SUBMITTED_ORDERS:
        threshold_rate = float(config.max_risk_rejection_rate)
        observed_rate = float(effective_rejection_rate)
        if observed_rate > threshold_rate:
            details = json.dumps(
                {
                    "max_risk_rejection_rate": round(float(threshold_rate), 6),
                    "rejection_rate": round(float(observed_rate), 6),
                    "submitted_orders": int(effective_submitted),
                },
                sort_keys=True,
            )
            events.append(
                DegradationEvent.create(
                    strategy_id=snapshot.strategy_id,
                    qualification_hash=snapshot.qualification_hash,
                    snapshot_hash=snapshot.snapshot_hash,
                    rule_name=RULE_RISK_REJECTION_SPIKE,
                    threshold_value=threshold_rate,
                    observed_value=observed_rate,
                    monitoring_protocol_version=snapshot.monitoring_protocol_version,
                    monitoring_config_hash=snapshot.monitoring_config_hash,
                    timestamp=timestamp,
                    details_json=details,
                )
            )

    # RULE 5: Trade Dropout
    # Trigger: actual_trades == 0 for >= 30 consecutive completed sessions
    if effective_inactive_sessions >= MP1_DROPOUT_WINDOW_SESSIONS:
        threshold_dropout = float(MP1_DROPOUT_WINDOW_SESSIONS)
        observed_dropout = float(effective_inactive_sessions)
        details = json.dumps(
            {
                "consecutive_inactive_sessions": int(effective_inactive_sessions),
                "dropout_window_sessions": int(MP1_DROPOUT_WINDOW_SESSIONS),
            },
            sort_keys=True,
        )
        events.append(
            DegradationEvent.create(
                strategy_id=snapshot.strategy_id,
                qualification_hash=snapshot.qualification_hash,
                snapshot_hash=snapshot.snapshot_hash,
                rule_name=RULE_TRADE_DROPOUT,
                threshold_value=threshold_dropout,
                observed_value=observed_dropout,
                monitoring_protocol_version=snapshot.monitoring_protocol_version,
                monitoring_config_hash=snapshot.monitoring_config_hash,
                timestamp=timestamp,
                details_json=details,
            )
        )

    # Deterministic sorting by rule_name
    events.sort(key=lambda ev: ev.rule_name)
    return events


# ---------------------------------------------------------------------------
# Internal Provenance & Resolution Helpers
# ---------------------------------------------------------------------------


def _verify_input_provenance(
    snapshot: MonitoringSnapshot,
    config: MonitoringConfig,
    qualification_record: StrategyQualificationRecord,
    baseline_replay_report: ReplayReport | None,
    baseline: PaperEvaluationBaseline | None = None,
) -> None:
    """Enforce fail-closed provenance checks across all supplied inputs."""
    # 1. Exact Type Checks
    if type(snapshot) is not MonitoringSnapshot:
        raise MonitoringProvenanceError(
            f"snapshot must be an instance of MonitoringSnapshot (exact type required), got {type(snapshot)}"
        )
    if type(config) is not MonitoringConfig:
        raise MonitoringProvenanceError(
            f"config must be an instance of MonitoringConfig (exact type required), got {type(config)}"
        )
    if type(qualification_record) is not StrategyQualificationRecord:
        raise MonitoringProvenanceError(
            f"qualification_record must be an instance of StrategyQualificationRecord (exact type required), got {type(qualification_record)}"
        )
    if baseline_replay_report is not None and type(baseline_replay_report) is not ReplayReport:
        raise MonitoringProvenanceError(
            f"baseline_replay_report must be an instance of ReplayReport (exact type required), got {type(baseline_replay_report)}"
        )
    if baseline is not None and type(baseline) is not PaperEvaluationBaseline:
        raise MonitoringProvenanceError(
            f"baseline must be an instance of PaperEvaluationBaseline (exact type required), got {type(baseline)}"
        )

    # 2. Snapshot Digest & Integrity
    if not snapshot.verify_digest():
        raise MonitoringProvenanceError(
            "Snapshot digest verification failed: snapshot content does not match snapshot_hash (tampered snapshot)"
        )

    # 3. Config Hash Binding
    expected_config_hash = config.compute_config_hash()
    if snapshot.monitoring_config_hash != expected_config_hash:
        raise MonitoringProvenanceError(
            f"Snapshot monitoring_config_hash mismatch: expected {expected_config_hash}, got {snapshot.monitoring_config_hash}"
        )

    # 4. Protocol Version Compatibility
    if snapshot.monitoring_protocol_version != config.protocol_version:
        raise MonitoringProvenanceError(
            f"Protocol version mismatch: snapshot {snapshot.monitoring_protocol_version} != config {config.protocol_version}"
        )

    # 5. Split Zone Isolation
    if snapshot.split_zone != "FORWARD_PAPER":
        raise MonitoringProvenanceError(
            f"Invalid split_zone: expected FORWARD_PAPER, got '{snapshot.split_zone}'"
        )

    # 6. Qualification Provenance Binding
    if not qualification_record.verify_digest():
        raise MonitoringProvenanceError(
            "Qualification record digest verification failed (tampered qualification record)"
        )
    if qualification_record.strategy_id != snapshot.strategy_id:
        raise MonitoringProvenanceError(
            f"Qualification strategy_id mismatch: {qualification_record.strategy_id} != {snapshot.strategy_id}"
        )
    if qualification_record.record_hash != snapshot.qualification_hash:
        raise MonitoringProvenanceError(
            f"Qualification record_hash mismatch: expected {snapshot.qualification_hash}, got {qualification_record.record_hash}"
        )
    if qualification_record.final_status != ValidationStatus.PAPER_ELIGIBLE:
        raise MonitoringProvenanceError(
            f"Qualification record final_status must be PAPER_ELIGIBLE, got {qualification_record.final_status.value}"
        )

    # 7. Baseline Replay Report Binding (if supplied)
    if baseline_replay_report is not None:
        computed_report_hash = baseline_replay_report.compute_report_hash()
        if baseline_replay_report.report_hash != computed_report_hash:
            raise MonitoringProvenanceError(
                "Baseline ReplayReport digest verification failed (tampered replay report)"
            )
        if baseline_replay_report.strategy_id != snapshot.strategy_id:
            raise MonitoringProvenanceError(
                f"Baseline ReplayReport strategy_id mismatch: {baseline_replay_report.strategy_id} != {snapshot.strategy_id}"
            )
        if baseline is not None and baseline_replay_report.report_hash != baseline.baseline_replay_report_hash:
            raise MonitoringProvenanceError(
                f"Supplied ReplayReport {baseline_replay_report.report_hash} is not the explicitly bound baseline {baseline.baseline_replay_report_hash}"
            )
        if baseline_replay_report.report_hash != snapshot.replay_report_hash:
            raise MonitoringProvenanceError(
                f"Baseline ReplayReport hash mismatch: expected {snapshot.replay_report_hash}, got {baseline_replay_report.report_hash}"
            )
        if baseline_replay_report.qualification_hash != snapshot.qualification_hash:
            raise MonitoringProvenanceError(
                f"Baseline ReplayReport qualification_hash mismatch: expected {snapshot.qualification_hash}, got {baseline_replay_report.qualification_hash}"
            )

    # 8. Authoritative Baseline Binding Verification (if supplied)
    if baseline is not None:
        if not baseline.verify_digest():
            raise MonitoringProvenanceError(
                "Baseline binding digest verification failed (tampered baseline binding)"
            )
        if baseline.strategy_id != snapshot.strategy_id:
            raise MonitoringProvenanceError(
                f"Baseline binding strategy_id mismatch: {baseline.strategy_id} != {snapshot.strategy_id}"
            )
        if baseline.qualification_hash != snapshot.qualification_hash:
            raise MonitoringProvenanceError(
                f"Baseline binding qualification_hash mismatch: {baseline.qualification_hash} != {snapshot.qualification_hash}"
            )
        if baseline.baseline_dataset_version != snapshot.dataset_version:
            raise MonitoringProvenanceError(
                f"Baseline binding dataset_version mismatch: {baseline.baseline_dataset_version} != {snapshot.dataset_version}"
            )
        if baseline.baseline_dataset_sha256 != snapshot.dataset_sha256:
            raise MonitoringProvenanceError(
                f"Baseline binding dataset_sha256 mismatch: {baseline.baseline_dataset_sha256} != {snapshot.dataset_sha256}"
            )
        if baseline.baseline_split_zone != snapshot.split_zone:
            raise MonitoringProvenanceError(
                f"Baseline binding split_zone mismatch: {baseline.baseline_split_zone} != {snapshot.split_zone}"
            )
        if baseline.baseline_replay_report_hash != snapshot.replay_report_hash:
            raise MonitoringProvenanceError(
                f"Snapshot replay_report_hash does not match explicitly bound baseline: {snapshot.replay_report_hash} != {baseline.baseline_replay_report_hash}"
            )
        if baseline_replay_report is not None:
            if baseline_replay_report.report_hash != baseline.baseline_replay_report_hash:
                raise MonitoringProvenanceError(
                    f"Supplied ReplayReport {baseline_replay_report.report_hash} is not the explicitly bound baseline {baseline.baseline_replay_report_hash}"
                )
            if baseline_replay_report.dataset_version != baseline.baseline_dataset_version:
                raise MonitoringProvenanceError(
                    "Baseline ReplayReport dataset_version does not match bound baseline"
                )
            if baseline_replay_report.dataset_sha256 != baseline.baseline_dataset_sha256:
                raise MonitoringProvenanceError(
                    "Baseline ReplayReport dataset_sha256 does not match bound baseline"
                )
            if baseline_replay_report.execution_policy != baseline.baseline_execution_policy:
                raise MonitoringProvenanceError(
                    "Baseline ReplayReport execution_policy does not match bound baseline"
                )
            if baseline_replay_report.cost_schedule_hash != baseline.baseline_cost_schedule_hash:
                raise MonitoringProvenanceError(
                    "Baseline ReplayReport cost_schedule_hash does not match bound baseline"
                )
            if baseline_replay_report.risk_config_hash != baseline.baseline_risk_config_hash:
                raise MonitoringProvenanceError(
                    "Baseline ReplayReport risk_config_hash does not match bound baseline"
                )


def _resolve_baseline_max_dd(
    baseline_replay_report: ReplayReport | None,
    explicit_base_dd: float | None,
    qualification_record: StrategyQualificationRecord,
) -> float:
    """Resolve and validate baseline max drawdown strictly from authoritative evidence."""
    if baseline_replay_report is not None:
        rep_dd = float(baseline_replay_report.max_drawdown_bps)
        if explicit_base_dd is not None and round(float(explicit_base_dd), 4) != round(rep_dd, 4):
            raise MonitoringProvenanceError(
                f"Conflicting baseline max drawdown supplied: explicit {explicit_base_dd} != authoritative report {rep_dd}"
            )
        base_dd = rep_dd
    elif hasattr(qualification_record, "baseline_max_drawdown_bps"):
        base_dd = float(getattr(qualification_record, "baseline_max_drawdown_bps"))
    elif explicit_base_dd is not None:
        base_dd = float(explicit_base_dd)
    else:
        raise MonitoringProvenanceError(
            "Baseline max drawdown is required for RULE_DD_EXPANSION_CRITICAL but absent from authoritative evidence"
        )

    if base_dd <= 0.0:
        raise MonitoringProvenanceError(
            f"Baseline max drawdown must be strictly positive (> 0.0 bps), got {base_dd}"
        )
    return base_dd


def _resolve_configured_slippage(
    baseline_replay_report: ReplayReport | None,
    explicit_slip: float | None,
) -> float:
    """Resolve and validate modeled slippage strictly from authoritative evidence."""
    if baseline_replay_report is not None:
        rep_slip = float(baseline_replay_report.slippage_bps_per_side)
        if explicit_slip is not None and round(float(explicit_slip), 4) != round(rep_slip, 4):
            raise MonitoringProvenanceError(
                f"Conflicting configured slippage supplied: explicit {explicit_slip} != authoritative report {rep_slip}"
            )
        slip = rep_slip
    elif explicit_slip is not None:
        slip = float(explicit_slip)
    else:
        raise MonitoringProvenanceError(
            "Configured slippage is required for RULE_SLIPPAGE_ANOMALY but absent from authoritative evidence"
        )

    if slip < 0.0:
        raise MonitoringProvenanceError(
            f"Configured slippage cannot be negative, got {slip}"
        )
    return slip


def _parse_metrics_json(metrics_json: str) -> dict[str, Any]:
    """Safely parse snapshot metrics JSON into a dictionary."""
    if not metrics_json or not metrics_json.strip():
        return {}
    try:
        data = json.loads(metrics_json)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _resolve_risk_metrics(
    submitted_orders: int | None,
    rejected_orders: int | None,
    rejection_rate: float | None,
    aux_metrics: Mapping[str, Any],
) -> tuple[int, float]:
    """Resolve submitted orders and rejection rate from arguments or auxiliary snapshot metrics."""
    # Submitted orders
    if submitted_orders is not None:
        submitted = max(0, int(submitted_orders))
    else:
        submitted = max(0, int(aux_metrics.get("submitted_orders", 0)))

    # Rejection rate
    if rejection_rate is not None:
        rate = float(rejection_rate)
    elif rejected_orders is not None:
        rej = max(0, int(rejected_orders))
        rate = (rej / submitted) if submitted > 0 else 0.0
    else:
        rate = float(aux_metrics.get("rejection_rate", 0.0))

    return submitted, max(0.0, rate)


def _resolve_inactive_sessions(
    consecutive_inactive_sessions: int | None,
    aux_metrics: Mapping[str, Any],
) -> int:
    """Resolve consecutive inactive completed sessions from arguments or auxiliary snapshot metrics."""
    if consecutive_inactive_sessions is not None:
        return max(0, int(consecutive_inactive_sessions))
    return max(0, int(aux_metrics.get("consecutive_inactive_sessions", 0)))
