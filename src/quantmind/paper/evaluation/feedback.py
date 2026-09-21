"""Research Feedback Service — PRD v4.0 Milestone 7.

Translates authoritative evaluation degradation evidence (DegradationEvent) into
immutable observational ResearchFeedbackRecord entries for future research formulation.

CRITICAL INVARIANTS:
1. Strictly observational: Research feedback is NOT a trial, NOT a qualification record,
   and cannot mutate StrategyRegistry or StrategySpec directly.
2. Complete provenance: Every record references a verified DegradationEvent and MonitoringSnapshot.
3. Cryptographic integrity: All records are digest-verified and stored immutably.
4. Causal consistency: Feedback timestamp must be temporally >= event and snapshot timestamps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from quantmind.paper.evaluation.ledger import EvaluationLedger, EvaluationLedgerIntegrityError
from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringSnapshot,
    PaperEvaluationRegime,
    ResearchFeedbackRecord,
)

if TYPE_CHECKING:
    from quantmind.strategy.registry import StrategyRegistry


class ResearchFeedbackError(Exception):
    """Base exception for research feedback operations."""


class ResearchFeedbackIntegrityError(ResearchFeedbackError):
    """Raised when feedback violates cryptographic or relational integrity."""


class ResearchFeedbackCausalError(ResearchFeedbackError):
    """Raised when feedback violates temporal causality."""


RULE_FAILURE_MODE_MAP: dict[str, str] = {
    "RULE_DD_EXPANSION_CRITICAL": "DRAWDOWN_EXPANSION",
    "RULE_SHARPE_COLLAPSE": "SHARPE_COLLAPSE",
    "RULE_SLIPPAGE_ANOMALY": "SLIPPAGE_ANOMALY",
    "RULE_RISK_REJECTION_SPIKE": "RISK_REJECTION_SPIKE",
    "RULE_TRADE_DROPOUT": "TRADE_DROPOUT",
}


class ResearchFeedbackService:
    """Service for creating and querying observational research feedback from degradation events."""

    def __init__(
        self,
        strategy_registry: StrategyRegistry,
        evaluation_ledger: EvaluationLedger,
    ) -> None:
        self._registry = strategy_registry
        self._ledger = evaluation_ledger

    def create_feedback_from_event(
        self,
        degradation_event: DegradationEvent,
        empirical_notes: str = "",
        created_at: str | None = None,
    ) -> ResearchFeedbackRecord:
        """Create an immutable ResearchFeedbackRecord from an authoritative DegradationEvent.

        Performs 15 strict integrity and causality checks:
        1. Type check on degradation_event
        2. Digest verification on degradation_event
        3. Existence of strategy in StrategyRegistry
        4. Strategy lifecycle state must not be CANDIDATE (must be qualified/monitored)
        5. Qualification hash consistency with registry
        6. Existence of DegradationEvent in EvaluationLedger
        7. Semantic match of DegradationEvent with stored record
        8. Existence of referenced MonitoringSnapshot in EvaluationLedger
        9. Digest verification on MonitoringSnapshot
        10. Strategy ID consistency between snapshot and degradation event
        11. Qualification hash consistency between snapshot and degradation event
        12. Regime integrity if bound to snapshot
        13. Regime strategy and qualification consistency
        14. Temporal causality against degradation event timestamp
        15. Temporal causality against snapshot creation timestamp
        """
        # 1. Type check
        if not isinstance(degradation_event, DegradationEvent):
            raise TypeError(
                f"Expected DegradationEvent, got {type(degradation_event).__name__}"
            )

        # 2. Digest verification
        if not degradation_event.verify_digest():
            raise ResearchFeedbackIntegrityError(
                f"DegradationEvent failed digest verification: stored={degradation_event.event_hash}, "
                f"computed={degradation_event.compute_hash()}"
            )

        # 3. Strategy existence
        try:
            strategy = self._registry.get_strategy(degradation_event.strategy_id)
        except KeyError:
            raise ResearchFeedbackIntegrityError(
                f"Strategy '{degradation_event.strategy_id}' not found in StrategyRegistry"
            )

        # 4. Lifecycle state check
        from quantmind.strategy.registry import StrategyLifecycleState

        if strategy.state in (
            StrategyLifecycleState.IDEA,
            StrategyLifecycleState.RESEARCH,
            StrategyLifecycleState.VALIDATION,
            StrategyLifecycleState.REJECTED,
        ):
            raise ResearchFeedbackIntegrityError(
                f"Strategy '{degradation_event.strategy_id}' is in state '{strategy.state.value}'; "
                "cannot create feedback for an unmonitored strategy"
            )

        # 5. Qualification hash consistency
        if strategy.qualification_hash and strategy.qualification_hash != degradation_event.qualification_hash:
            raise ResearchFeedbackIntegrityError(
                f"Strategy qualification hash '{strategy.qualification_hash}' does not match "
                f"degradation event qualification hash '{degradation_event.qualification_hash}'"
            )

        # 6. Event existence in ledger
        stored_event = self._ledger.get_degradation_event(degradation_event.event_hash)
        if stored_event is None:
            raise ResearchFeedbackIntegrityError(
                f"DegradationEvent '{degradation_event.event_hash}' not found in EvaluationLedger"
            )

        # 7. Semantic match of event
        if stored_event.canonical_dict() != degradation_event.canonical_dict():
            raise ResearchFeedbackIntegrityError(
                f"DegradationEvent '{degradation_event.event_hash}' does not match stored ledger record"
            )

        # 8. Snapshot existence
        snapshot = self._ledger.get_snapshot(degradation_event.snapshot_hash)
        if snapshot is None:
            raise ResearchFeedbackIntegrityError(
                f"MonitoringSnapshot '{degradation_event.snapshot_hash}' not found in EvaluationLedger"
            )

        # 9. Snapshot digest verification
        if not snapshot.verify_digest():
            raise ResearchFeedbackIntegrityError(
                f"MonitoringSnapshot '{snapshot.snapshot_hash}' failed digest verification"
            )

        # 10. Strategy ID consistency
        if snapshot.strategy_id != degradation_event.strategy_id:
            raise ResearchFeedbackIntegrityError(
                f"Snapshot strategy '{snapshot.strategy_id}' does not match event strategy '{degradation_event.strategy_id}'"
            )

        # 11. Qualification hash consistency
        if snapshot.qualification_hash != degradation_event.qualification_hash:
            raise ResearchFeedbackIntegrityError(
                f"Snapshot qualification hash '{snapshot.qualification_hash}' does not match event '{degradation_event.qualification_hash}'"
            )

        # 12 & 13. Regime verification if bound
        if snapshot.regime_hash:
            regime = self._ledger.get_regime(snapshot.regime_hash)
            if regime is None:
                raise ResearchFeedbackIntegrityError(
                    f"Snapshot regime '{snapshot.regime_hash}' not found in EvaluationLedger"
                )
            if not regime.verify_digest():
                raise ResearchFeedbackIntegrityError(
                    f"PaperEvaluationRegime '{regime.regime_hash}' failed digest verification"
                )
            if regime.strategy_id != degradation_event.strategy_id:
                raise ResearchFeedbackIntegrityError(
                    f"Regime strategy '{regime.strategy_id}' does not match event strategy '{degradation_event.strategy_id}'"
                )
            if regime.qualification_hash != degradation_event.qualification_hash:
                raise ResearchFeedbackIntegrityError(
                    f"Regime qualification hash '{regime.qualification_hash}' does not match event '{degradation_event.qualification_hash}'"
                )

        # 14 & 15. Temporal causality checks
        feedback_ts = created_at or datetime.now(timezone.utc).isoformat()
        if feedback_ts < degradation_event.timestamp:
            raise ResearchFeedbackCausalError(
                f"Feedback timestamp '{feedback_ts}' is earlier than degradation event timestamp '{degradation_event.timestamp}'"
            )
        if feedback_ts < snapshot.created_at:
            raise ResearchFeedbackCausalError(
                f"Feedback timestamp '{feedback_ts}' is earlier than snapshot created_at '{snapshot.created_at}'"
            )

        # Determine failure mode strictly from authoritative DegradationEvent rule_name
        eff_failure_mode = RULE_FAILURE_MODE_MAP.get(
            degradation_event.rule_name, degradation_event.rule_name
        )

        # Determine drawdown expansion ratio strictly from authoritative details / snapshot
        details: dict[str, Any] = {}
        try:
            details = json.loads(degradation_event.details_json)
        except Exception:
            details = {}

        base_dd = details.get("baseline_max_dd_bps")
        if base_dd is not None and float(base_dd) > 0:
            eff_dd_ratio = float(snapshot.max_drawdown_bps) / float(base_dd)
        elif (
            degradation_event.rule_name == "RULE_DD_EXPANSION_CRITICAL"
            and degradation_event.threshold_value > 0
        ):
            limit = float(details.get("max_drawdown_expansion_limit", 1.0))
            base_dd_est = degradation_event.threshold_value / limit
            eff_dd_ratio = (
                degradation_event.observed_value / base_dd_est
                if base_dd_est > 0
                else 1.0
            )
        else:
            eff_dd_ratio = 1.0

        # Construct feedback record
        record = ResearchFeedbackRecord.create(
            strategy_id=degradation_event.strategy_id,
            qualification_hash=degradation_event.qualification_hash,
            degradation_event_hash=degradation_event.event_hash,
            dataset_version=snapshot.dataset_version,
            failure_mode=eff_failure_mode,
            realized_sharpe=snapshot.realized_sharpe,
            drawdown_expansion_ratio=eff_dd_ratio,
            realized_slippage_bps=snapshot.realized_slippage_bps,
            empirical_notes=empirical_notes,
            created_at=feedback_ts,
        )

        # Persist into EvaluationLedger (idempotent)
        return self._ledger.record_feedback(record)

    def get_feedback(self, feedback_hash: str) -> ResearchFeedbackRecord | None:
        """Retrieve a ResearchFeedbackRecord by its semantic hash."""
        return self._ledger.get_feedback(feedback_hash)

    def get_feedback_by_id(self, feedback_id: str) -> ResearchFeedbackRecord | None:
        """Retrieve a ResearchFeedbackRecord by its derived ID."""
        return self._ledger.get_feedback_by_id(feedback_id)

    def list_feedback(self, strategy_id: str) -> list[ResearchFeedbackRecord]:
        """List all research feedback records for a given strategy."""
        return self._ledger.list_feedback(strategy_id)
