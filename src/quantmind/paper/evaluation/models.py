"""Paper Evaluation & Monitoring Evidence Models (PRD v4.0).

This module defines the immutable data models for paper evaluation, monitoring,
and empirical degradation evidence:
- MonitoringConfig: Versioned monitoring protocol configuration.
- MonitoringSnapshot: Time-windowed performance and execution snapshot.
- DegradationEvent: Deterministic evidence of threshold breach.
- PaperEvaluationTransition: Auditable record of lifecycle transition request.
- ResearchFeedbackRecord: Empirical post-mortem evidence for future research.

ARCHITECTURAL PRINCIPLES:
1. Evidence is Immutable: All models are frozen dataclasses. Once created and
   hashed, records cannot be mutated.
2. Hashes are Strictly Deterministic: Every semantic hash (compute_*_hash) is
   computed via SHA-256 over a canonical JSON dictionary with sorted keys and
   stable numeric precision.
3. IDs are Administrative & Deterministic: Primary key identifiers are derived
   deterministically from semantic content via SHA-256 prefixes. Random UUIDs
   are strictly prohibited.
4. Semantic Hashes Exclude Non-Semantic Metadata: Administrative IDs and wall-clock
   timestamps (e.g. created_at) are excluded from canonical dictionaries to preserve
   temporal determinism.
5. Observational Evidence Only: These models represent measured historical facts.
   They possess zero direct execution authority and cannot modify orders, fills,
   positions, or account state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from quantmind.paper.models import ReplayReport


class MonitoringConfigError(ValueError):
    """Raised when an invalid MonitoringConfig is specified."""


class MonitoringProvenanceError(ValueError):
    """Raised when evidence verification or digest validation fails."""


# ---------------------------------------------------------------------------
# 1. MonitoringConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringConfig:
    """Versioned monitoring protocol configuration.

    Defines the quantitative thresholds and observation window settings for
    evaluating forward paper trading degradation.
    """

    protocol_version: str
    max_drawdown_expansion_limit: float
    min_rolling_sharpe_30d: float
    max_slippage_drift_ratio: float
    max_risk_rejection_rate: float
    window_size_sessions: int
    min_evaluation_trades: int

    def __post_init__(self) -> None:
        if not self.protocol_version or not self.protocol_version.strip():
            raise MonitoringConfigError("protocol_version cannot be empty")
        if self.max_drawdown_expansion_limit <= 0.0:
            raise MonitoringConfigError(
                f"max_drawdown_expansion_limit must be strictly positive, got {self.max_drawdown_expansion_limit}"
            )
        if self.max_slippage_drift_ratio <= 0.0:
            raise MonitoringConfigError(
                f"max_slippage_drift_ratio must be strictly positive, got {self.max_slippage_drift_ratio}"
            )
        if not (0.0 <= self.max_risk_rejection_rate <= 1.0):
            raise MonitoringConfigError(
                f"max_risk_rejection_rate must be in [0.0, 1.0], got {self.max_risk_rejection_rate}"
            )
        if self.window_size_sessions <= 0:
            raise MonitoringConfigError(
                f"window_size_sessions must be strictly positive, got {self.window_size_sessions}"
            )
        if self.min_evaluation_trades <= 0:
            raise MonitoringConfigError(
                f"min_evaluation_trades must be strictly positive, got {self.min_evaluation_trades}"
            )

    def canonical_dict(self) -> dict[str, Any]:
        """Produce deterministic sorted dictionary of configuration parameters."""
        return {
            "max_drawdown_expansion_limit": round(float(self.max_drawdown_expansion_limit), 6),
            "max_risk_rejection_rate": round(float(self.max_risk_rejection_rate), 6),
            "max_slippage_drift_ratio": round(float(self.max_slippage_drift_ratio), 6),
            "min_evaluation_trades": int(self.min_evaluation_trades),
            "min_rolling_sharpe_30d": round(float(self.min_rolling_sharpe_30d), 6),
            "protocol_version": str(self.protocol_version),
            "window_size_sessions": int(self.window_size_sessions),
        }

    def canonical_json(self) -> str:
        """Produce deterministic canonical JSON string."""
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_config_hash(self) -> str:
        """Compute cryptographic SHA-256 digest over canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 2. MonitoringSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Immutable time-windowed evaluation snapshot of forward paper trading performance.

    Binds execution provenance (qualification, replay report, dataset) with measured
    rolling performance, execution quality, and risk metrics.
    """

    strategy_id: str
    qualification_hash: str
    replay_report_hash: str
    dataset_version: str
    dataset_sha256: str
    split_zone: str
    monitoring_protocol_version: str
    monitoring_config_hash: str
    window_start_ts: str
    window_end_ts: str
    total_trades: int
    net_pnl: float
    max_drawdown_bps: float
    realized_sharpe: float | None
    realized_slippage_bps: float
    cost_to_turnover_bps: float
    risk_event_count: int
    metrics_json: str
    created_at: str
    snapshot_hash: str
    regime_hash: str = ""

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.qualification_hash:
            raise ValueError("qualification_hash cannot be empty")
        if not self.replay_report_hash:
            raise ValueError("replay_report_hash cannot be empty")
        if not self.dataset_version:
            raise ValueError("dataset_version cannot be empty")
        if not self.dataset_sha256:
            raise ValueError("dataset_sha256 cannot be empty")
        if not self.monitoring_protocol_version:
            raise ValueError("monitoring_protocol_version cannot be empty")
        if not self.monitoring_config_hash:
            raise ValueError("monitoring_config_hash cannot be empty")
        if not self.window_start_ts:
            raise ValueError("window_start_ts cannot be empty")
        if not self.window_end_ts:
            raise ValueError("window_end_ts cannot be empty")
        if self.total_trades < 0:
            raise ValueError("total_trades cannot be negative")
        if self.risk_event_count < 0:
            raise ValueError("risk_event_count cannot be negative")

    @property
    def derived_snapshot_id(self) -> str:
        """Derive deterministic administrative identifier via SHA-256."""
        return self.derive_id(
            self.strategy_id,
            self.window_start_ts,
            self.window_end_ts,
            self.monitoring_config_hash,
        )

    @staticmethod
    def derive_id(
        strategy_id: str,
        window_start_ts: str,
        window_end_ts: str,
        monitoring_config_hash: str,
    ) -> str:
        """Static helper to compute deterministic snapshot ID."""
        seed = f"{strategy_id}:{window_start_ts}:{window_end_ts}:{monitoring_config_hash}".encode("utf-8")
        digest = hashlib.sha256(seed).hexdigest()[:16]
        strat_prefix = strategy_id[:12]
        return f"SNAP-{strat_prefix}-{digest}"

    def canonical_dict(self) -> dict[str, Any]:
        """Produce deterministic sorted dictionary of semantic measurement fields.

        Explicitly EXCLUDES administrative IDs (snapshot_id) and wall-clock creation
        timestamps (created_at) to guarantee temporal determinism across re-evaluations.
        """
        return {
            "cost_to_turnover_bps": round(float(self.cost_to_turnover_bps), 4),
            "dataset_sha256": str(self.dataset_sha256),
            "dataset_version": str(self.dataset_version),
            "max_drawdown_bps": round(float(self.max_drawdown_bps), 4),
            "metrics_json": str(self.metrics_json),
            "monitoring_config_hash": str(self.monitoring_config_hash),
            "monitoring_protocol_version": str(self.monitoring_protocol_version),
            "net_pnl": round(float(self.net_pnl), 4),
            "qualification_hash": str(self.qualification_hash),
            "realized_sharpe": (
                round(float(self.realized_sharpe), 4)
                if self.realized_sharpe is not None
                else None
            ),
            "realized_slippage_bps": round(float(self.realized_slippage_bps), 4),
            "replay_report_hash": str(self.replay_report_hash),
            "risk_event_count": int(self.risk_event_count),
            "split_zone": str(self.split_zone),
            "strategy_id": str(self.strategy_id),
            "total_trades": int(self.total_trades),
            "window_end_ts": str(self.window_end_ts),
            "window_start_ts": str(self.window_start_ts),
        }
        if self.regime_hash:
            d["regime_hash"] = str(self.regime_hash)
        return d

    def canonical_json(self) -> str:
        """Produce deterministic canonical JSON string."""
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 digest over semantic canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def verify_digest(self) -> bool:
        """Verify that snapshot_hash matches the computed semantic digest."""
        return self.snapshot_hash == self.compute_hash()

    @classmethod
    def create(
        cls,
        *,
        strategy_id: str,
        qualification_hash: str,
        replay_report_hash: str,
        dataset_version: str,
        dataset_sha256: str,
        split_zone: str,
        monitoring_protocol_version: str,
        monitoring_config_hash: str,
        window_start_ts: str,
        window_end_ts: str,
        total_trades: int,
        net_pnl: float,
        max_drawdown_bps: float,
        realized_sharpe: float | None,
        realized_slippage_bps: float,
        cost_to_turnover_bps: float,
        risk_event_count: int,
        metrics_json: str,
        created_at: str | None = None,
        regime_hash: str = "",
    ) -> MonitoringSnapshot:
        """Factory method to instantiate an immutable MonitoringSnapshot with computed hash."""
        now_ts = created_at or datetime.now(timezone.utc).isoformat()
        temp = cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            replay_report_hash=replay_report_hash,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            split_zone=split_zone,
            monitoring_protocol_version=monitoring_protocol_version,
            monitoring_config_hash=monitoring_config_hash,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            total_trades=total_trades,
            net_pnl=net_pnl,
            max_drawdown_bps=max_drawdown_bps,
            realized_sharpe=realized_sharpe,
            realized_slippage_bps=realized_slippage_bps,
            cost_to_turnover_bps=cost_to_turnover_bps,
            risk_event_count=risk_event_count,
            metrics_json=metrics_json,
            created_at=now_ts,
            snapshot_hash="",
            regime_hash=regime_hash,
        )
        digest = temp.compute_hash()
        return cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            replay_report_hash=replay_report_hash,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            split_zone=split_zone,
            monitoring_protocol_version=monitoring_protocol_version,
            monitoring_config_hash=monitoring_config_hash,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            total_trades=total_trades,
            net_pnl=net_pnl,
            max_drawdown_bps=max_drawdown_bps,
            realized_sharpe=realized_sharpe,
            realized_slippage_bps=realized_slippage_bps,
            cost_to_turnover_bps=cost_to_turnover_bps,
            risk_event_count=risk_event_count,
            metrics_json=metrics_json,
            created_at=now_ts,
            snapshot_hash=digest,
            regime_hash=regime_hash,
        )


# ---------------------------------------------------------------------------
# 3. DegradationEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DegradationEvent:
    """Immutable evidence record created when a monitoring rule threshold is breached.

    Binds the underlying MonitoringSnapshot hash, authoritative baseline replay report hash,
    configured rule threshold, observed metric measurement, and protocol configuration.
    """

    strategy_id: str
    qualification_hash: str
    snapshot_hash: str
    rule_name: str
    threshold_value: float
    observed_value: float
    monitoring_protocol_version: str
    monitoring_config_hash: str
    timestamp: str
    details_json: str
    event_hash: str
    baseline_replay_report_hash: str

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.qualification_hash:
            raise ValueError("qualification_hash cannot be empty")
        if not self.snapshot_hash:
            raise ValueError("snapshot_hash cannot be empty")
        if not self.baseline_replay_report_hash:
            raise ValueError("baseline_replay_report_hash cannot be empty")
        if not self.rule_name:
            raise ValueError("rule_name cannot be empty")
        if not self.monitoring_protocol_version:
            raise ValueError("monitoring_protocol_version cannot be empty")
        if not self.monitoring_config_hash:
            raise ValueError("monitoring_config_hash cannot be empty")
        if not self.timestamp:
            raise ValueError("timestamp cannot be empty")

    @property
    def derived_event_id(self) -> str:
        """Derive deterministic administrative identifier via SHA-256."""
        return self.derive_id(
            self.strategy_id,
            self.rule_name,
            self.timestamp,
            self.snapshot_hash,
            self.baseline_replay_report_hash,
        )

    @staticmethod
    def derive_id(
        strategy_id: str,
        rule_name: str,
        timestamp: str,
        snapshot_hash: str,
        baseline_replay_report_hash: str = "",
    ) -> str:
        """Static helper to compute deterministic degradation event ID."""
        seed = f"{strategy_id}:{rule_name}:{timestamp}:{snapshot_hash}:{baseline_replay_report_hash}".encode("utf-8")
        digest = hashlib.sha256(seed).hexdigest()[:16]
        strat_prefix = strategy_id[:12]
        return f"DEG-{strat_prefix}-{digest}"

    def canonical_dict(self) -> dict[str, Any]:
        """Produce deterministic sorted dictionary of degradation event fields.

        Excludes administrative event_id and event_hash.
        """
        return {
            "baseline_replay_report_hash": str(self.baseline_replay_report_hash),
            "details_json": str(self.details_json),
            "monitoring_config_hash": str(self.monitoring_config_hash),
            "monitoring_protocol_version": str(self.monitoring_protocol_version),
            "observed_value": round(float(self.observed_value), 6),
            "qualification_hash": str(self.qualification_hash),
            "rule_name": str(self.rule_name),
            "snapshot_hash": str(self.snapshot_hash),
            "strategy_id": str(self.strategy_id),
            "threshold_value": round(float(self.threshold_value), 6),
            "timestamp": str(self.timestamp),
        }

    def canonical_json(self) -> str:
        """Produce deterministic canonical JSON string."""
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 digest over semantic canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def verify_digest(self) -> bool:
        """Verify that event_hash matches the computed semantic digest."""
        return self.event_hash == self.compute_hash()

    @classmethod
    def create(
        cls,
        *,
        strategy_id: str,
        qualification_hash: str,
        snapshot_hash: str,
        rule_name: str,
        threshold_value: float,
        observed_value: float,
        monitoring_protocol_version: str,
        monitoring_config_hash: str,
        timestamp: str,
        baseline_replay_report_hash: str,
        details_json: str = "{}",
    ) -> DegradationEvent:
        """Factory method to instantiate an immutable DegradationEvent with computed hash."""
        temp = cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            snapshot_hash=snapshot_hash,
            rule_name=rule_name,
            threshold_value=threshold_value,
            observed_value=observed_value,
            monitoring_protocol_version=monitoring_protocol_version,
            monitoring_config_hash=monitoring_config_hash,
            timestamp=timestamp,
            details_json=details_json,
            event_hash="",
            baseline_replay_report_hash=baseline_replay_report_hash,
        )
        digest = temp.compute_hash()
        return cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            snapshot_hash=snapshot_hash,
            rule_name=rule_name,
            threshold_value=threshold_value,
            observed_value=observed_value,
            monitoring_protocol_version=monitoring_protocol_version,
            monitoring_config_hash=monitoring_config_hash,
            timestamp=timestamp,
            details_json=details_json,
            event_hash=digest,
            baseline_replay_report_hash=baseline_replay_report_hash,
        )


# ---------------------------------------------------------------------------
# 4. PaperEvaluationTransition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperEvaluationTransition:
    """Immutable audit record representing a lifecycle transition request or execution.

    Cryptographically links the target lifecycle state change to its underlying
    evidence digest (e.g. DegradationEvent hash or StrategyQualificationRecord hash).
    """

    strategy_id: str
    old_state: str
    new_state: str
    initiator: str
    evidence_type: str
    evidence_hash: str
    reason: str
    timestamp: str
    transition_hash: str
    qualification_hash: str = ""
    snapshot_hash: str = ""
    regime_hash: str = ""

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.old_state:
            raise ValueError("old_state cannot be empty")
        if not self.new_state:
            raise ValueError("new_state cannot be empty")
        if not self.initiator:
            raise ValueError("initiator cannot be empty")
        if not self.evidence_type:
            raise ValueError("evidence_type cannot be empty")
        if not self.evidence_hash:
            raise ValueError("evidence_hash cannot be empty")
        if not self.timestamp:
            raise ValueError("timestamp cannot be empty")

    @property
    def derived_transition_id(self) -> str:
        """Derive deterministic administrative identifier via SHA-256."""
        return self.derive_id(
            self.strategy_id,
            self.old_state,
            self.new_state,
            self.timestamp,
            self.evidence_hash,
        )

    @staticmethod
    def derive_id(
        strategy_id: str,
        old_state: str,
        new_state: str,
        timestamp: str,
        evidence_hash: str,
    ) -> str:
        """Static helper to compute deterministic transition ID."""
        seed = f"{strategy_id}:{old_state}:{new_state}:{timestamp}:{evidence_hash}".encode("utf-8")
        digest = hashlib.sha256(seed).hexdigest()[:16]
        strat_prefix = strategy_id[:12]
        return f"TRANS-{strat_prefix}-{digest}"

    def canonical_dict(self) -> dict[str, Any]:
        """Produce deterministic sorted dictionary of transition fields.

        Excludes administrative transition_id and transition_hash.
        """
        d = {
            "evidence_hash": str(self.evidence_hash),
            "evidence_type": str(self.evidence_type),
            "initiator": str(self.initiator),
            "new_state": str(self.new_state),
            "old_state": str(self.old_state),
            "reason": str(self.reason),
            "strategy_id": str(self.strategy_id),
            "timestamp": str(self.timestamp),
        }
        if self.qualification_hash:
            d["qualification_hash"] = str(self.qualification_hash)
        if self.snapshot_hash:
            d["snapshot_hash"] = str(self.snapshot_hash)
        if self.regime_hash:
            d["regime_hash"] = str(self.regime_hash)
        return d

    def canonical_json(self) -> str:
        """Produce deterministic canonical JSON string."""
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 digest over semantic canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def verify_digest(self) -> bool:
        """Verify that transition_hash matches the computed semantic digest."""
        return self.transition_hash == self.compute_hash()

    @classmethod
    def create(
        cls,
        *,
        strategy_id: str,
        old_state: str,
        new_state: str,
        initiator: str,
        evidence_type: str,
        evidence_hash: str,
        reason: str,
        timestamp: str,
        qualification_hash: str = "",
        snapshot_hash: str = "",
        regime_hash: str = "",
    ) -> PaperEvaluationTransition:
        """Factory method to instantiate an immutable PaperEvaluationTransition with computed hash."""
        temp = cls(
            strategy_id=strategy_id,
            old_state=old_state,
            new_state=new_state,
            initiator=initiator,
            evidence_type=evidence_type,
            evidence_hash=evidence_hash,
            reason=reason,
            timestamp=timestamp,
            transition_hash="",
            qualification_hash=qualification_hash,
            snapshot_hash=snapshot_hash,
            regime_hash=regime_hash,
        )
        digest = temp.compute_hash()
        return cls(
            strategy_id=strategy_id,
            old_state=old_state,
            new_state=new_state,
            initiator=initiator,
            evidence_type=evidence_type,
            evidence_hash=evidence_hash,
            reason=reason,
            timestamp=timestamp,
            transition_hash=digest,
            qualification_hash=qualification_hash,
            snapshot_hash=snapshot_hash,
            regime_hash=regime_hash,
        )


@dataclass(frozen=True)
class PaperExecutionLifecycleContext:
    """Explicit immutable replay lifecycle authorization context (PRD v4.0 M6).

    Guarantees that historical paper replay relies on a deterministic, captured
    lifecycle authorization rather than querying mutable external registries on every bar.
    """

    strategy_id: str
    authorized_state: str  # "PAPER_ACTIVE", "PAPER_ELIGIBLE", "DEGRADED", "RETIRED", "REJECTED"
    authorization_timestamp: str
    governance_transition_hash: str = ""
    regime_hash: str = ""

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.authorized_state:
            raise ValueError("authorized_state cannot be empty")
        if not self.authorization_timestamp:
            raise ValueError("authorization_timestamp cannot be empty")

    @property
    def is_degraded(self) -> bool:
        """When True, new entry orders are suppressed (HALT_NEW_ENTRIES), while existing positions exit normally."""
        return self.authorized_state == "DEGRADED"

    @property
    def allows_new_entries(self) -> bool:
        """Only PAPER_ACTIVE strategies can take new entries."""
        return self.authorized_state == "PAPER_ACTIVE"

    @property
    def is_prohibited(self) -> bool:
        """RETIRED and REJECTED states prohibit replay execution entirely."""
        return self.authorized_state in ("REJECTED", "RETIRED")


# ---------------------------------------------------------------------------
# 5. ResearchFeedbackRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchFeedbackRecord:
    """Immutable post-mortem observational feedback package.

    Captures empirical degradation evidence (failure mode, realized Sharpe,
    drawdown expansion, slippage divergence) to inform future research hypotheses.

    CRITICAL INVARIANT:
    This record is strictly observational research feedback. It is NOT a
    TrialLedger trial, contains no trial ID, and cannot enter EICT or DSR.
    """

    strategy_id: str
    qualification_hash: str
    degradation_event_hash: str
    dataset_version: str
    failure_mode: str
    realized_sharpe: float | None
    drawdown_expansion_ratio: float
    realized_slippage_bps: float
    empirical_notes: str
    created_at: str
    feedback_hash: str

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.qualification_hash:
            raise ValueError("qualification_hash cannot be empty")
        if not self.degradation_event_hash:
            raise ValueError("degradation_event_hash cannot be empty")
        if not self.dataset_version:
            raise ValueError("dataset_version cannot be empty")
        if not self.failure_mode:
            raise ValueError("failure_mode cannot be empty")

    @property
    def derived_feedback_id(self) -> str:
        """Derive deterministic administrative identifier via SHA-256."""
        seed = f"{self.strategy_id}:{self.failure_mode}:{self.degradation_event_hash}".encode("utf-8")
        digest = hashlib.sha256(seed).hexdigest()[:16]
        strat_prefix = self.strategy_id[:12]
        return f"RFB-{strat_prefix}-{digest}"

    def canonical_dict(self) -> dict[str, Any]:
        """Produce deterministic sorted dictionary of feedback fields.

        Excludes administrative feedback_id, created_at, and feedback_hash.
        """
        return {
            "dataset_version": str(self.dataset_version),
            "degradation_event_hash": str(self.degradation_event_hash),
            "drawdown_expansion_ratio": round(float(self.drawdown_expansion_ratio), 4),
            "empirical_notes": str(self.empirical_notes),
            "failure_mode": str(self.failure_mode),
            "qualification_hash": str(self.qualification_hash),
            "realized_sharpe": (
                round(float(self.realized_sharpe), 4)
                if self.realized_sharpe is not None
                else None
            ),
            "realized_slippage_bps": round(float(self.realized_slippage_bps), 4),
            "strategy_id": str(self.strategy_id),
        }

    def canonical_json(self) -> str:
        """Produce deterministic canonical JSON string."""
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 digest over semantic canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def verify_digest(self) -> bool:
        """Verify that feedback_hash matches the computed semantic digest."""
        return self.feedback_hash == self.compute_hash()

    @classmethod
    def create(
        cls,
        *,
        strategy_id: str,
        qualification_hash: str,
        degradation_event_hash: str,
        dataset_version: str,
        failure_mode: str,
        realized_sharpe: float | None,
        drawdown_expansion_ratio: float,
        realized_slippage_bps: float,
        empirical_notes: str = "",
        created_at: str | None = None,
    ) -> ResearchFeedbackRecord:
        """Factory method to instantiate an immutable ResearchFeedbackRecord with computed hash."""
        now_ts = created_at or datetime.now(timezone.utc).isoformat()
        temp = cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            degradation_event_hash=degradation_event_hash,
            dataset_version=dataset_version,
            failure_mode=failure_mode,
            realized_sharpe=realized_sharpe,
            drawdown_expansion_ratio=drawdown_expansion_ratio,
            realized_slippage_bps=realized_slippage_bps,
            empirical_notes=empirical_notes,
            created_at=now_ts,
            feedback_hash="",
        )
        digest = temp.compute_hash()
        return cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            degradation_event_hash=degradation_event_hash,
            dataset_version=dataset_version,
            failure_mode=failure_mode,
            realized_sharpe=realized_sharpe,
            drawdown_expansion_ratio=drawdown_expansion_ratio,
            realized_slippage_bps=realized_slippage_bps,
            empirical_notes=empirical_notes,
            created_at=now_ts,
            feedback_hash=digest,
        )


# ---------------------------------------------------------------------------
# 6. PaperEvaluationBaseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperEvaluationBaseline:
    """Immutable binding between a qualified strategy and its authoritative baseline ReplayReport.

    Enforces the essential invariant:
    ONE qualified strategy + ONE explicitly bound baseline ReplayReport = authoritative monitoring baseline.
    """

    strategy_id: str
    qualification_hash: str
    baseline_replay_report_hash: str
    baseline_dataset_version: str
    baseline_dataset_sha256: str
    baseline_split_zone: str
    baseline_execution_policy: str
    baseline_cost_schedule_hash: str
    baseline_risk_config_hash: str
    created_at: str
    binding_hash: str

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.qualification_hash:
            raise ValueError("qualification_hash cannot be empty")
        if not self.baseline_replay_report_hash:
            raise ValueError("baseline_replay_report_hash cannot be empty")
        if not self.baseline_dataset_version:
            raise ValueError("baseline_dataset_version cannot be empty")
        if not self.baseline_dataset_sha256:
            raise ValueError("baseline_dataset_sha256 cannot be empty")
        if not self.baseline_split_zone:
            raise ValueError("baseline_split_zone cannot be empty")
        if not self.baseline_execution_policy:
            raise ValueError("baseline_execution_policy cannot be empty")

    def canonical_dict(self) -> dict[str, Any]:
        """Produce deterministic sorted dictionary of baseline binding parameters."""
        return {
            "baseline_cost_schedule_hash": str(self.baseline_cost_schedule_hash),
            "baseline_dataset_sha256": str(self.baseline_dataset_sha256),
            "baseline_dataset_version": str(self.baseline_dataset_version),
            "baseline_execution_policy": str(self.baseline_execution_policy),
            "baseline_replay_report_hash": str(self.baseline_replay_report_hash),
            "baseline_risk_config_hash": str(self.baseline_risk_config_hash),
            "baseline_split_zone": str(self.baseline_split_zone),
            "qualification_hash": str(self.qualification_hash),
            "strategy_id": str(self.strategy_id),
        }

    def canonical_json(self) -> str:
        """Produce deterministic canonical JSON string."""
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 digest over semantic canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def verify_digest(self) -> bool:
        """Verify that binding_hash matches the computed digest."""
        return self.binding_hash == self.compute_hash()

    @classmethod
    def create(
        cls,
        *,
        strategy_id: str,
        qualification_hash: str,
        baseline_replay_report_hash: str,
        baseline_dataset_version: str,
        baseline_dataset_sha256: str,
        baseline_split_zone: str = "FORWARD_PAPER",
        baseline_execution_policy: str = "next_bar_open_v1",
        baseline_cost_schedule_hash: str = "",
        baseline_risk_config_hash: str = "",
        created_at: str | None = None,
    ) -> PaperEvaluationBaseline:
        """Factory method to instantiate an immutable PaperEvaluationBaseline with computed hash."""
        now_ts = created_at or datetime.now(timezone.utc).isoformat()
        temp = cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            baseline_replay_report_hash=baseline_replay_report_hash,
            baseline_dataset_version=baseline_dataset_version,
            baseline_dataset_sha256=baseline_dataset_sha256,
            baseline_split_zone=baseline_split_zone,
            baseline_execution_policy=baseline_execution_policy,
            baseline_cost_schedule_hash=baseline_cost_schedule_hash,
            baseline_risk_config_hash=baseline_risk_config_hash,
            created_at=now_ts,
            binding_hash="",
        )
        digest = temp.compute_hash()
        return cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            baseline_replay_report_hash=baseline_replay_report_hash,
            baseline_dataset_version=baseline_dataset_version,
            baseline_dataset_sha256=baseline_dataset_sha256,
            baseline_split_zone=baseline_split_zone,
            baseline_execution_policy=baseline_execution_policy,
            baseline_cost_schedule_hash=baseline_cost_schedule_hash,
            baseline_risk_config_hash=baseline_risk_config_hash,
            created_at=now_ts,
            binding_hash=digest,
        )

    @classmethod
    def from_replay_report(
        cls,
        report: ReplayReport,
        created_at: str | None = None,
    ) -> PaperEvaluationBaseline:
        """Construct an authoritative baseline binding directly from a validated ReplayReport."""
        return cls.create(
            strategy_id=report.strategy_id,
            qualification_hash=report.qualification_hash,
            baseline_replay_report_hash=report.report_hash,
            baseline_dataset_version=report.dataset_version,
            baseline_dataset_sha256=report.dataset_sha256,
            baseline_split_zone=report.split_zone,
            baseline_execution_policy=report.execution_policy,
            baseline_cost_schedule_hash=report.cost_schedule_hash,
            baseline_risk_config_hash=report.risk_config_hash,
            created_at=created_at or report.created_at,
        )


# ---------------------------------------------------------------------------
# 6. PaperEvaluationRegime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperEvaluationRegime:
    """Immutable identity defining the execution, risk, and dataset regime for evaluation.

    Binds the strategy and qualification to the active forward dataset and execution
    parameters. A change in execution policy, cost schedule, or risk configuration
    represents a regime break requiring a distinct regime identity.
    """

    strategy_id: str
    qualification_hash: str
    baseline_replay_report_hash: str
    forward_dataset_version: str
    forward_dataset_sha256: str
    execution_policy: str
    cost_schedule_hash: str
    risk_config_hash: str
    monitoring_protocol_version: str
    regime_hash: str
    created_at: str = ""

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "baseline_replay_report_hash": str(self.baseline_replay_report_hash),
            "cost_schedule_hash": str(self.cost_schedule_hash),
            "execution_policy": str(self.execution_policy),
            "forward_dataset_sha256": str(self.forward_dataset_sha256),
            "forward_dataset_version": str(self.forward_dataset_version),
            "monitoring_protocol_version": str(self.monitoring_protocol_version),
            "qualification_hash": str(self.qualification_hash),
            "risk_config_hash": str(self.risk_config_hash),
            "strategy_id": str(self.strategy_id),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_regime_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def compute_hash(self) -> str:
        """Alias for compute_regime_hash for interface parity."""
        return self.compute_regime_hash()

    def verify_digest(self) -> bool:
        return self.regime_hash == self.compute_regime_hash()

    @classmethod
    def create(
        cls,
        *,
        strategy_id: str,
        qualification_hash: str,
        baseline_replay_report_hash: str,
        forward_dataset_version: str,
        forward_dataset_sha256: str,
        execution_policy: str,
        cost_schedule_hash: str,
        risk_config_hash: str,
        monitoring_protocol_version: str,
        created_at: str = "",
    ) -> PaperEvaluationRegime:
        canonical_bytes = json.dumps(
            {
                "baseline_replay_report_hash": str(baseline_replay_report_hash),
                "cost_schedule_hash": str(cost_schedule_hash),
                "execution_policy": str(execution_policy),
                "forward_dataset_sha256": str(forward_dataset_sha256),
                "forward_dataset_version": str(forward_dataset_version),
                "monitoring_protocol_version": str(monitoring_protocol_version),
                "qualification_hash": str(qualification_hash),
                "risk_config_hash": str(risk_config_hash),
                "strategy_id": str(strategy_id),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical_bytes).hexdigest()
        return cls(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            baseline_replay_report_hash=baseline_replay_report_hash,
            forward_dataset_version=forward_dataset_version,
            forward_dataset_sha256=forward_dataset_sha256,
            execution_policy=execution_policy,
            cost_schedule_hash=cost_schedule_hash,
            risk_config_hash=risk_config_hash,
            monitoring_protocol_version=monitoring_protocol_version,
            regime_hash=digest,
            created_at=created_at,
        )
