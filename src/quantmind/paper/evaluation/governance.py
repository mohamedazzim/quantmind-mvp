"""Authoritative Paper Governance Service (PRD v4.0 M6).

Coordinates lifecycle state machine transitions between StrategyRegistry and
EvaluationLedger with strict evidence verification, cryptographic provenance,
causal timestamp enforcement, and fail-closed persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import TYPE_CHECKING, Any

from quantmind.paper.evaluation.ledger import EvaluationLedger, EvaluationLedgerIntegrityError
from quantmind.paper.evaluation.models import DegradationEvent, PaperEvaluationTransition
from quantmind.strategy.registry import (
    StrategyLifecycleState,
    StrategyRegistry,
    StrategyRegistryError,
    StrategyRegistryRecord,
)


class GovernanceIntegrityError(StrategyRegistryError):
    """Raised when governance state machine or evidence verification fails."""


class GovernanceCausalError(GovernanceIntegrityError):
    """Raised when an operation violates causal or temporal ordering."""


class PaperGovernanceService:
    """Authoritative service coordinating paper strategy governance and lifecycle state transitions.

    Guarantees:
    - Pure separation of concerns: Zero metric calculation, zero trade execution.
    - Deterministic transition verification: Requires immutable cryptographic evidence for each transition.
    - Immutable audit trail: All state transitions are recorded into EvaluationLedger (paper_evaluation_transitions).
    - Causal consistency: Timestamps must be causally ordered; future evidence is rejected.
    - Fail-closed atomicity: Failed registry transitions abort and fail closed.
    - Strict position flatness: Strategies cannot transition to RETIRED or RESEARCH with open positions.
    """

    def __init__(
        self,
        strategy_registry: StrategyRegistry,
        evaluation_ledger: EvaluationLedger,
    ) -> None:
        if type(strategy_registry) is not StrategyRegistry:
            raise TypeError(
                f"Expected StrategyRegistry instance, got {type(strategy_registry).__name__}"
            )
        if type(evaluation_ledger) is not EvaluationLedger:
            raise TypeError(
                f"Expected EvaluationLedger instance, got {type(evaluation_ledger).__name__}"
            )
        self._registry = strategy_registry
        self._ledger = evaluation_ledger

    @property
    def registry(self) -> StrategyRegistry:
        return self._registry

    @property
    def ledger(self) -> EvaluationLedger:
        return self._ledger

    def activate_strategy(
        self,
        strategy_id: str,
        *,
        initiator: str = "governance",
        timestamp: str | None = None,
    ) -> PaperEvaluationTransition:
        """Activate a strategy from PAPER_ELIGIBLE to PAPER_ACTIVE.

        Requires:
        - Strategy currently in PAPER_ELIGIBLE state.
        - Authoritative PaperEvaluationBaseline registered in EvaluationLedger.
        """
        record = self._registry.get_strategy(strategy_id)
        if record.state != StrategyLifecycleState.PAPER_ELIGIBLE:
            raise GovernanceIntegrityError(
                f"Cannot activate strategy '{strategy_id}': current state is '{record.state.value}', "
                f"requires '{StrategyLifecycleState.PAPER_ELIGIBLE.value}'"
            )

        if not record.qualification_hash:
            raise GovernanceIntegrityError(
                f"Cannot activate strategy '{strategy_id}': missing qualification_hash in registry"
            )

        # Verify authoritative baseline exists in evaluation ledger
        baseline = self._ledger.get_baseline(strategy_id, record.qualification_hash)
        if baseline is None:
            raise GovernanceIntegrityError(
                f"Cannot activate strategy '{strategy_id}': no authoritative baseline binding found "
                f"in EvaluationLedger for qualification_hash '{record.qualification_hash}'"
            )

        trans_ts = timestamp or datetime.now(timezone.utc).isoformat()
        if trans_ts < record.updated_at:
            raise GovernanceCausalError(
                f"Causal ordering violation: transition timestamp '{trans_ts}' is earlier than updated_at '{record.updated_at}'"
            )

        transition = PaperEvaluationTransition.create(
            strategy_id=strategy_id,
            old_state=StrategyLifecycleState.PAPER_ELIGIBLE.value,
            new_state=StrategyLifecycleState.PAPER_ACTIVE.value,
            initiator=initiator,
            evidence_type="baseline_binding",
            evidence_hash=baseline.binding_hash,
            reason="Authoritative baseline verified; activated for paper execution",
            timestamp=trans_ts,
            qualification_hash=record.qualification_hash,
        )

        # Record in ledger first, then transition state in registry
        self._ledger.record_transition(transition)
        self._registry.transition_state(
            strategy_id,
            StrategyLifecycleState.PAPER_ACTIVE,
            reason=transition.reason,
            transition=transition,
            evaluation_ledger=self._ledger,
            timestamp=trans_ts,
        )
        return transition

    def degrade_strategy(
        self,
        degradation_event: DegradationEvent,
        *,
        initiator: str = "governance",
        timestamp: str | None = None,
    ) -> PaperEvaluationTransition:
        """Transition a strategy from PAPER_ACTIVE (or PAPER_ELIGIBLE) to DEGRADED upon verified degradation breach.

        Requires:
        - Valid, digest-verified DegradationEvent.
        - Strategy in PAPER_ACTIVE (or PAPER_ELIGIBLE) state.
        - Event strategy_id and qualification_hash match the registry.
        - Causal ordering: degradation event timestamp must not be in the future relative to transition timestamp.
        """
        if type(degradation_event) is not DegradationEvent:
            raise TypeError(
                f"Expected DegradationEvent, got {type(degradation_event).__name__}"
            )

        if not degradation_event.verify_digest():
            raise GovernanceIntegrityError(
                f"DegradationEvent failed digest verification (stored: {degradation_event.event_hash}, "
                f"computed: {degradation_event.compute_hash()})"
            )

        strategy_id = degradation_event.strategy_id
        record = self._registry.get_strategy(strategy_id)

        if record.state not in (StrategyLifecycleState.PAPER_ACTIVE, StrategyLifecycleState.PAPER_ELIGIBLE):
            raise GovernanceIntegrityError(
                f"Cannot degrade strategy '{strategy_id}': current state is '{record.state.value}', "
                f"requires PAPER_ACTIVE or PAPER_ELIGIBLE"
            )

        if record.qualification_hash != degradation_event.qualification_hash:
            raise GovernanceIntegrityError(
                f"DegradationEvent qualification_hash '{degradation_event.qualification_hash}' "
                f"does not match registered qualification_hash '{record.qualification_hash}'"
            )

        trans_ts = timestamp or datetime.now(timezone.utc).isoformat()
        if degradation_event.timestamp > trans_ts:
            raise GovernanceCausalError(
                f"DegradationEvent timestamp '{degradation_event.timestamp}' is in the future relative to "
                f"governance transition timestamp '{trans_ts}'"
            )

        if trans_ts < record.updated_at:
            raise GovernanceCausalError(
                f"Causal ordering violation: transition timestamp '{trans_ts}' is earlier than updated_at '{record.updated_at}'"
            )

        # Lookup snapshot in ledger to retrieve regime_hash if available
        snapshot = self._ledger.get_snapshot(degradation_event.snapshot_hash)
        regime_h = snapshot.regime_hash if snapshot is not None else ""

        reason_str = (
            f"Degradation breach: {degradation_event.rule_name} "
            f"(observed={degradation_event.observed_value}, threshold={degradation_event.threshold_value})"
        )

        transition = PaperEvaluationTransition.create(
            strategy_id=strategy_id,
            old_state=record.state.value,
            new_state=StrategyLifecycleState.DEGRADED.value,
            initiator=initiator,
            evidence_type="degradation_event",
            evidence_hash=degradation_event.event_hash,
            reason=reason_str,
            timestamp=trans_ts,
            qualification_hash=record.qualification_hash or "",
            snapshot_hash=degradation_event.snapshot_hash,
            regime_hash=regime_h,
        )

        self._ledger.record_transition(transition)
        self._registry.transition_state(
            strategy_id,
            StrategyLifecycleState.DEGRADED,
            reason=transition.reason,
            transition=transition,
            evaluation_ledger=self._ledger,
            timestamp=trans_ts,
        )
        return transition

    def retire_strategy(
        self,
        strategy_id: str,
        reason: str,
        *,
        position_quantity: float | int = 0.0,
        initiator: str = "governance",
        timestamp: str | None = None,
    ) -> PaperEvaluationTransition:
        """Retire a strategy permanently to RETIRED.

        Requires:
        - Strategy is in PAPER_ELIGIBLE, PAPER_ACTIVE, or DEGRADED state.
        - position_quantity == 0 (flat position required; governance NEVER liquidates positions as a side-effect).
        - Non-empty reason.
        """
        if not reason or not reason.strip():
            raise GovernanceIntegrityError(
                f"Retirement of strategy '{strategy_id}' requires an explicit non-empty reason"
            )

        if abs(position_quantity) > 1e-9:
            raise GovernanceIntegrityError(
                f"Cannot retire strategy '{strategy_id}' with open position ({position_quantity}). "
                "Strategy must be flat (position_quantity == 0) before retirement."
            )

        record = self._registry.get_strategy(strategy_id)
        if record.state not in (
            StrategyLifecycleState.PAPER_ELIGIBLE,
            StrategyLifecycleState.PAPER_ACTIVE,
            StrategyLifecycleState.DEGRADED,
        ):
            raise GovernanceIntegrityError(
                f"Cannot retire strategy '{strategy_id}' from state '{record.state.value}'"
            )

        trans_ts = timestamp or datetime.now(timezone.utc).isoformat()
        if trans_ts < record.updated_at:
            raise GovernanceCausalError(
                f"Causal ordering violation: transition timestamp '{trans_ts}' is earlier than updated_at '{record.updated_at}'"
            )

        evidence_content = f"{strategy_id}:{reason}:{trans_ts}".encode("utf-8")
        evidence_hash = hashlib.sha256(evidence_content).hexdigest()

        transition = PaperEvaluationTransition.create(
            strategy_id=strategy_id,
            old_state=record.state.value,
            new_state=StrategyLifecycleState.RETIRED.value,
            initiator=initiator,
            evidence_type="retirement_decision",
            evidence_hash=evidence_hash,
            reason=reason,
            timestamp=trans_ts,
            qualification_hash=record.qualification_hash or "",
        )

        self._ledger.record_transition(transition)
        self._registry.transition_state(
            strategy_id,
            StrategyLifecycleState.RETIRED,
            reason=reason,
            transition=transition,
            evaluation_ledger=self._ledger,
            position_quantity=position_quantity,
            timestamp=trans_ts,
        )
        return transition

    def re_research_strategy(
        self,
        strategy_id: str,
        reason: str,
        *,
        position_quantity: float | int = 0.0,
        initiator: str = "governance",
        timestamp: str | None = None,
    ) -> PaperEvaluationTransition:
        """Demote a DEGRADED strategy back to RESEARCH for model revision.

        Requires:
        - Strategy currently in DEGRADED state.
        - position_quantity == 0 (flat position required).
        - Invalidates qualification bindings in StrategyRegistry.
        """
        if not reason or not reason.strip():
            raise GovernanceIntegrityError(
                f"Demotion to RESEARCH for strategy '{strategy_id}' requires an explicit non-empty reason"
            )

        if abs(position_quantity) > 1e-9:
            raise GovernanceIntegrityError(
                f"Cannot return strategy '{strategy_id}' to RESEARCH with open position ({position_quantity}). "
                "Strategy must be flat (position_quantity == 0)."
            )

        record = self._registry.get_strategy(strategy_id)
        if record.state != StrategyLifecycleState.DEGRADED:
            raise GovernanceIntegrityError(
                f"Cannot return strategy '{strategy_id}' to RESEARCH: current state is '{record.state.value}', "
                f"requires '{StrategyLifecycleState.DEGRADED.value}'"
            )

        trans_ts = timestamp or datetime.now(timezone.utc).isoformat()
        if trans_ts < record.updated_at:
            raise GovernanceCausalError(
                f"Causal ordering violation: transition timestamp '{trans_ts}' is earlier than updated_at '{record.updated_at}'"
            )

        evidence_content = f"{strategy_id}:{reason}:{trans_ts}".encode("utf-8")
        evidence_hash = hashlib.sha256(evidence_content).hexdigest()

        transition = PaperEvaluationTransition.create(
            strategy_id=strategy_id,
            old_state=StrategyLifecycleState.DEGRADED.value,
            new_state=StrategyLifecycleState.RESEARCH.value,
            initiator=initiator,
            evidence_type="re_research_decision",
            evidence_hash=evidence_hash,
            reason=reason,
            timestamp=trans_ts,
            qualification_hash=record.qualification_hash or "",
        )

        self._ledger.record_transition(transition)
        self._registry.transition_state(
            strategy_id,
            StrategyLifecycleState.RESEARCH,
            reason=reason,
            transition=transition,
            evaluation_ledger=self._ledger,
            position_quantity=position_quantity,
            timestamp=trans_ts,
        )
        return transition
