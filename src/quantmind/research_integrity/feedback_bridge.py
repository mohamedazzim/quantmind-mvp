"""Research Feedback Bridge — PRD v4.0 Milestone 7.

Bridges empirical evaluation feedback (ResearchFeedbackRecord) into structured
research tasks (ResearchFeedbackTask / ResearchTask) for research formulation.

CRITICAL INVARIANTS:
1. Passive Bridge: Translates observational degradation evidence into a structured
   research task. Does NOT generate strategy code, does NOT mutate parameters,
   and does NOT bypass research validation gates.
2. Cryptographic Integrity: Task binds to the verified feedback_hash and feedback provenance.
3. Unconditional Holdout Protection: Research tasks created through this bridge
   cannot access FINAL_HOLDOUT data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from quantmind.paper.evaluation.models import ResearchFeedbackRecord
from quantmind.research_integrity.harness import ResearchTask


class ResearchBridgeError(Exception):
    """Base exception for research bridge operations."""


class ResearchBridgeIntegrityError(ResearchBridgeError):
    """Raised when feedback record or task derivation fails cryptographic validation."""


@dataclass(frozen=True)
class ResearchFeedbackTask(ResearchTask):
    """Structured research task derived deterministically from an authoritative ResearchFeedbackRecord.

    Subclasses ResearchTask so it can be passed directly as `research_task_id=task.task_id`
    to `ResearchHarness.run_trial()`.
    """

    strategy_id: str
    qualification_hash: str
    feedback_hash: str
    degradation_event_hash: str
    dataset_version: str
    failure_mode: str
    drawdown_expansion_ratio: float
    realized_sharpe: float | None
    realized_slippage_bps: float
    empirical_notes: str
    research_protocol_version: str
    created_at: str

    def format_research_context(self) -> str:
        """Produce structured textual research context summarizing empirical failure mode and metrics."""
        sharpe_str = f"{self.realized_sharpe:.4f}" if self.realized_sharpe is not None else "N/A"
        return (
            f"=== RESEARCH HYPOTHESIS FORMULATION CONTEXT ===\n"
            f"Task ID: {self.task_id}\n"
            f"Origin Strategy ID: {self.strategy_id}\n"
            f"Qualification Hash: {self.qualification_hash}\n"
            f"Feedback Hash: {self.feedback_hash}\n"
            f"Dataset Version: {self.dataset_version}\n"
            f"Failure Mode: {self.failure_mode}\n"
            f"Drawdown Expansion Ratio: {self.drawdown_expansion_ratio:.4f}x\n"
            f"Realized Sharpe: {sharpe_str}\n"
            f"Realized Slippage: {self.realized_slippage_bps:.2f} bps\n"
            f"Empirical Notes: {self.empirical_notes}\n"
            f"Research Protocol: {self.research_protocol_version}\n"
            f"================================================="
        )

    def to_dict(self) -> dict[str, Any]:
        """Produce dictionary representation of the research task context."""
        return {
            "task_id": self.task_id,
            "strategy_id": self.strategy_id,
            "qualification_hash": self.qualification_hash,
            "feedback_hash": self.feedback_hash,
            "degradation_event_hash": self.degradation_event_hash,
            "dataset_version": self.dataset_version,
            "failure_mode": self.failure_mode,
            "drawdown_expansion_ratio": self.drawdown_expansion_ratio,
            "realized_sharpe": self.realized_sharpe,
            "realized_slippage_bps": self.realized_slippage_bps,
            "empirical_notes": self.empirical_notes,
            "research_protocol_version": self.research_protocol_version,
            "created_at": self.created_at,
        }


def derive_feedback_task_id(feedback_hash: str, protocol_version: str) -> str:
    """Derive deterministic administrative research task ID from feedback hash and protocol."""
    seed = f"{feedback_hash}:{protocol_version}".encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()[:16]
    return f"TASK-RFB-{digest}"


def create_research_task_from_feedback(
    feedback: ResearchFeedbackRecord,
    research_protocol_version: str = "RP-1.0",
) -> ResearchFeedbackTask:
    """Create a structured ResearchFeedbackTask from an authoritative ResearchFeedbackRecord.

    Validates:
    - Type of feedback record
    - Cryptographic digest verification of feedback_hash
    - Non-empty research_protocol_version
    """
    if not isinstance(feedback, ResearchFeedbackRecord):
        raise TypeError(
            f"Expected ResearchFeedbackRecord, got {type(feedback).__name__}"
        )

    if not feedback.verify_digest():
        raise ResearchBridgeIntegrityError(
            f"ResearchFeedbackRecord '{feedback.feedback_hash}' failed cryptographic digest verification"
        )

    if not research_protocol_version or not research_protocol_version.strip():
        raise ValueError("research_protocol_version cannot be empty")

    task_id = derive_feedback_task_id(feedback.feedback_hash, research_protocol_version)

    return ResearchFeedbackTask(
        task_id=task_id,
        strategy_id=feedback.strategy_id,
        qualification_hash=feedback.qualification_hash,
        feedback_hash=feedback.feedback_hash,
        degradation_event_hash=feedback.degradation_event_hash,
        dataset_version=feedback.dataset_version,
        failure_mode=feedback.failure_mode,
        drawdown_expansion_ratio=feedback.drawdown_expansion_ratio,
        realized_sharpe=feedback.realized_sharpe,
        realized_slippage_bps=feedback.realized_slippage_bps,
        empirical_notes=feedback.empirical_notes,
        research_protocol_version=research_protocol_version,
        created_at=feedback.created_at,
    )
