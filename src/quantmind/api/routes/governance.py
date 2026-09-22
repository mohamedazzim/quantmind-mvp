"""Governance and state transition inspection routes."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext

router = APIRouter(prefix="/governance", tags=["Governance"])


@router.get("", response_model=list[dict[str, Any]])
async def list_governance_transitions(
    limit: int = 100,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List historical strategy lifecycle transitions."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM paper_evaluation_transitions ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "transition_id": r["transition_id"],
                "strategy_id": r["strategy_id"],
                "old_state": r["old_state"],
                "new_state": r["new_state"],
                "initiator": r["initiator"],
                "evidence_type": r["evidence_type"],
                "evidence_hash": r["evidence_hash"],
                "reason": r["reason"],
                "timestamp": r["timestamp"],
                "transition_hash": r["transition_hash"],
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/{strategy_id}", response_model=list[dict[str, Any]])
async def get_strategy_governance_history(
    strategy_id: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """Retrieve state transition history and evidence hashes for a specific strategy."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM paper_evaluation_transitions WHERE strategy_id = ? ORDER BY timestamp DESC
            """,
            (strategy_id,),
        ).fetchall()
        return [
            {
                "transition_id": r["transition_id"],
                "strategy_id": r["strategy_id"],
                "old_state": r["old_state"],
                "new_state": r["new_state"],
                "initiator": r["initiator"],
                "evidence_type": r["evidence_type"],
                "evidence_hash": r["evidence_hash"],
                "reason": r["reason"],
                "timestamp": r["timestamp"],
                "transition_hash": r["transition_hash"],
            }
            for r in rows
        ]
    finally:
        conn.close()
