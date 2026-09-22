"""Trials inspection and multiple-testing population accounting routes."""

from __future__ import annotations

import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext

router = APIRouter(prefix="/trials", tags=["Trials"])


@router.get("", response_model=list[dict[str, Any]])
async def list_trials(
    strategy_id: str | None = Query(None),
    dataset_version: str | None = Query(None),
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List research trials with optional filtering."""
    conn = ctx.get_core_connection()
    try:
        query = "SELECT * FROM trials"
        params: list[Any] = []
        conditions = []
        if strategy_id:
            conditions.append("strategy_id = ?")
            params.append(strategy_id)
        if dataset_version:
            conditions.append("dataset_version = ?")
            params.append(dataset_version)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp_started DESC LIMIT 100"

        rows = conn.execute(query, params).fetchall()
        return [
            {
                "trial_id": r["trial_id"],
                "experiment_id": r["experiment_id"],
                "strategy_id": r["strategy_id"],
                "dataset_version": r["dataset_version"],
                "split_zone": r["split_zone"],
                "research_protocol_version": r["research_protocol_version"],
                "mode": r["mode"],
                "status": r["status"],
                "timestamp_started": r["timestamp_started"],
                "timestamp_completed": r["timestamp_completed"],
                "actual_runtime_minutes": float(r["actual_runtime_minutes"] or r["estimated_runtime_minutes"]),
                "result": json.loads(r["result_json"]) if r["result_json"] else None,
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/{trial_id}", response_model=dict[str, Any])
async def get_trial_detail(
    trial_id: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retrieve complete detail for a specific research trial."""
    conn = ctx.get_core_connection()
    try:
        r = conn.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Trial '{trial_id}' not found")

        # Check for OOS artifact if present
        artifact_rows = conn.execute(
            "SELECT * FROM artifacts WHERE trial_id = ?", (trial_id,)
        ).fetchall()
        artifacts = [
            {
                "artifact_id": a["artifact_id"],
                "sha256": a["sha256"],
                "row_count": a["row_count"],
                "created_at": a["created_at"],
            }
            for a in artifact_rows
        ]

        return {
            "trial_id": r["trial_id"],
            "experiment_id": r["experiment_id"],
            "strategy_id": r["strategy_id"],
            "dataset_version": r["dataset_version"],
            "split_zone": r["split_zone"],
            "research_protocol_version": r["research_protocol_version"],
            "feature_version": r["feature_version"],
            "mode": r["mode"],
            "dataset_kind": r["dataset_kind"],
            "status": r["status"],
            "timestamp_started": r["timestamp_started"],
            "timestamp_completed": r["timestamp_completed"],
            "actual_runtime_minutes": float(r["actual_runtime_minutes"] or r["estimated_runtime_minutes"]),
            "parameters": json.loads(r["parameter_set_json"]),
            "strategy_spec": json.loads(r["strategy_spec_json"]),
            "result": json.loads(r["result_json"]) if r["result_json"] else None,
            "artifacts": artifacts,
        }
    finally:
        conn.close()


@router.get("/population/summary", response_model=dict[str, Any])
async def get_population_summary(
    dataset_version: str = Query("DS-NIFTY-2023-V1"),
    research_protocol_version: str = Query("RP-2"),
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retrieve multiple-testing trial population statistics and budget usage."""
    usage = ctx.trial_ledger.usage(dataset_version, research_protocol_version, mode="PRODUCTION")
    fixture_usage = ctx.trial_ledger.usage(dataset_version, research_protocol_version, mode="FIXTURE")
    return {
        "dataset_version": dataset_version,
        "research_protocol_version": research_protocol_version,
        "production_trials": usage["trials"],
        "production_experiments": usage["experiments"],
        "production_variants": usage["strategy_variants"],
        "fixture_trials": fixture_usage["trials"],
    }
