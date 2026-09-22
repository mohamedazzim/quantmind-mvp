"""Strategy Qualification inspection routes."""

from __future__ import annotations

import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext

router = APIRouter(prefix="/qualification", tags=["Qualification"])


@router.get("", response_model=list[dict[str, Any]])
async def list_qualifications(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List all certified strategy qualification records."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM strategy_qualifications ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        return [
            {
                "qualification_id": r["qualification_id"],
                "strategy_id": r["strategy_id"],
                "record_hash": r["record_hash"],
                "dataset_version": r["dataset_version"],
                "dataset_sha256": r["dataset_sha256"],
                "research_protocol_version": r["research_protocol_version"],
                "effective_trial_count": int(r["effective_trial_count"]),
                "observed_sharpe": float(r["observed_sharpe"]),
                "dsr": float(r["dsr"]),
                "trade_count": int(r["trade_count"]),
                "holdout_state": r["holdout_state"],
                "robustness_status": r["robustness_status"],
                "final_status": r["final_status"],
                "reasons": json.loads(r["reasons_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/{strategy_id}", response_model=dict[str, Any])
async def get_strategy_qualification(
    strategy_id: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retrieve qualification record for a specific strategy."""
    conn = ctx.get_core_connection()
    try:
        r = conn.execute(
            "SELECT * FROM strategy_qualifications WHERE strategy_id = ? ORDER BY created_at DESC LIMIT 1",
            (strategy_id,),
        ).fetchone()
        if not r:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No qualification record found for strategy '{strategy_id}'",
            )
        return {
            "qualification_id": r["qualification_id"],
            "strategy_id": r["strategy_id"],
            "record_hash": r["record_hash"],
            "dataset_version": r["dataset_version"],
            "dataset_sha256": r["dataset_sha256"],
            "research_protocol_version": r["research_protocol_version"],
            "effective_trial_count": int(r["effective_trial_count"]),
            "observed_sharpe": float(r["observed_sharpe"]),
            "dsr": float(r["dsr"]),
            "trade_count": int(r["trade_count"]),
            "holdout_state": r["holdout_state"],
            "robustness_status": r["robustness_status"],
            "final_status": r["final_status"],
            "reasons": json.loads(r["reasons_json"]),
            "created_at": r["created_at"],
        }
    finally:
        conn.close()
