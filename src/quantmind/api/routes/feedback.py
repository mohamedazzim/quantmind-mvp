"""Research feedback inspection and task derivation routes."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from quantmind.api.dependencies import get_ctx, get_current_user, require_role
from quantmind.app.auth.models import User, UserRole
from quantmind.app.context import AppContext
from quantmind.paper.evaluation.models import ResearchFeedbackRecord
from quantmind.research_integrity.feedback_bridge import (
    create_research_task_from_feedback,
    derive_feedback_task_id,
)

router = APIRouter(prefix="/feedback", tags=["Research Feedback"])


@router.get("", response_model=list[dict[str, Any]])
async def list_feedback_records(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List all post-mortem empirical research feedback packages."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM research_feedback ORDER BY created_at DESC"
        ).fetchall()
        return [
            {
                "feedback_id": r["feedback_id"],
                "strategy_id": r["strategy_id"],
                "qualification_hash": r["qualification_hash"],
                "degradation_event_hash": r["degradation_event_hash"],
                "dataset_version": r["dataset_version"],
                "failure_mode": r["failure_mode"],
                "realized_sharpe": float(r["realized_sharpe"]) if r["realized_sharpe"] is not None else None,
                "drawdown_expansion_ratio": float(r["drawdown_expansion_ratio"]),
                "realized_slippage_bps": float(r["realized_slippage_bps"]),
                "empirical_notes": r["empirical_notes"],
                "created_at": r["created_at"],
                "feedback_hash": r["feedback_hash"],
                "task_id": derive_feedback_task_id(r["feedback_hash"], "RP-1.0"),
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/{feedback_hash}", response_model=dict[str, Any])
async def get_feedback_record(
    feedback_hash: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retrieve detailed research feedback record by its cryptographic hash."""
    rec = ctx.feedback_service.get_feedback(feedback_hash)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Feedback record '{feedback_hash}' not found")

    task = create_research_task_from_feedback(rec)
    return {
        "feedback_id": rec.derived_feedback_id,
        "strategy_id": rec.strategy_id,
        "qualification_hash": rec.qualification_hash,
        "degradation_event_hash": rec.degradation_event_hash,
        "dataset_version": rec.dataset_version,
        "failure_mode": rec.failure_mode,
        "realized_sharpe": rec.realized_sharpe,
        "drawdown_expansion_ratio": rec.drawdown_expansion_ratio,
        "realized_slippage_bps": rec.realized_slippage_bps,
        "empirical_notes": rec.empirical_notes,
        "created_at": rec.created_at,
        "feedback_hash": rec.feedback_hash,
        "task": {
            "task_id": task.task_id,
            "suggested_hypothesis": task.suggested_hypothesis,
            "failure_mode": task.failure_mode,
        },
    }


@router.post("/{feedback_hash}/research-task", dependencies=[Depends(require_role(UserRole.RESEARCHER, UserRole.ADMIN))])
async def create_research_task_endpoint(
    feedback_hash: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Derive a zero-trial ResearchFeedbackTask without mutating historical evidence or trial ledger."""
    rec = ctx.feedback_service.get_feedback(feedback_hash)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Feedback record '{feedback_hash}' not found")

    task = create_research_task_from_feedback(rec)
    ctx.auth_service.log_action(user, "RESEARCH_TASK_DERIVED", {"task_id": task.task_id, "strategy_id": task.strategy_id})

    return {
        "message": "Research task formulated successfully (0 trials created)",
        "task_id": task.task_id,
        "strategy_id": task.strategy_id,
        "failure_mode": task.failure_mode,
        "suggested_hypothesis": task.suggested_hypothesis,
        "created_at": task.created_at,
    }
