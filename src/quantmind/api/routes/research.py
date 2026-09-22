"""Research Workspace routes for candidate validation, trials, and feedback tasks (PRD v4.0 Productization)."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from quantmind.api.dependencies import get_ctx, get_current_user, require_role
from quantmind.api.schemas.research import (
    CandidateValidateRequest,
    CandidateValidateResponse,
    ResearchTaskItem,
    TrialSubmitRequest,
)
from quantmind.app.auth.models import User, UserRole
from quantmind.app.context import AppContext
from quantmind.app.jobs.models import JobType
from quantmind.paper.evaluation.models import ResearchFeedbackRecord
from quantmind.research_integrity.feedback_bridge import (
    create_research_task_from_feedback,
    derive_feedback_task_id,
)
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.spec import StrategySpec

router = APIRouter(prefix="/research", tags=["Research Workspace"])


@router.get("/tasks", response_model=list[ResearchTaskItem])
async def list_research_tasks(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[ResearchTaskItem]:
    """List all structured research feedback tasks derived from degradation feedback."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM research_feedback ORDER BY created_at DESC
            """
        ).fetchall()
        tasks = []
        for r in rows:
            record = ResearchFeedbackRecord(
                strategy_id=r["strategy_id"],
                qualification_hash=r["qualification_hash"],
                degradation_event_hash=r["degradation_event_hash"],
                dataset_version=r["dataset_version"],
                failure_mode=r["failure_mode"],
                realized_sharpe=float(r["realized_sharpe"]) if r["realized_sharpe"] is not None else None,
                drawdown_expansion_ratio=float(r["drawdown_expansion_ratio"]),
                realized_slippage_bps=float(r["realized_slippage_bps"]),
                empirical_notes=r["empirical_notes"],
                created_at=r["created_at"],
                feedback_hash=r["feedback_hash"],
            )
            task = create_research_task_from_feedback(record)
            tasks.append(
                ResearchTaskItem(
                    task_id=task.task_id,
                    source_feedback_hash=task.feedback_hash,
                    strategy_id=task.strategy_id,
                    failure_mode=task.failure_mode,
                    drawdown_expansion_ratio=task.drawdown_expansion_ratio,
                    realized_slippage_bps=task.realized_slippage_bps,
                    realized_sharpe=task.realized_sharpe,
                    suggested_hypothesis=f"Mitigate {task.failure_mode} degradation on {task.strategy_id}",
                    empirical_notes=task.empirical_notes,
                    created_at=task.created_at,
                )
            )
        return tasks
    finally:
        conn.close()


@router.get("/tasks/{task_id}", response_model=ResearchTaskItem)
async def get_research_task(
    task_id: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> ResearchTaskItem:
    """Get detail for a specific research feedback task."""
    tasks = await list_research_tasks(user=user, ctx=ctx)
    for t in tasks:
        if t.task_id == task_id:
            return t
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research task '{task_id}' not found")


@router.post("/candidates/validate", response_model=CandidateValidateResponse)
async def validate_candidate(
    req: CandidateValidateRequest,
    user: User = Depends(get_current_user),
) -> CandidateValidateResponse:
    """Validate parameter schema and preview canonical identity of candidate StrategySpec."""
    spec = StrategySpec(
        strategy_version=req.strategy_version,
        feature_version=req.feature_version,
        signal_name=req.signal_name,
        parameters=req.parameters,
    )
    # Compiler normalizes and checks schema against whitelisted signal
    normalized = normalize_strategy_spec(spec)
    strategy_id = derive_strategy_id(normalized)
    canonical_json = normalized.canonical_json()
    spec_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return CandidateValidateResponse(
        strategy_id=strategy_id,
        normalized_parameters=dict(normalized.parameters),
        canonical_spec_json=canonical_json,
        spec_hash=spec_hash,
        is_valid=True,
    )


@router.post("/trials/submit", dependencies=[Depends(require_role(UserRole.RESEARCHER, UserRole.ADMIN))])
async def submit_research_trial(
    req: TrialSubmitRequest,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Submit a research trial for background worker execution via ResearchHarness.
    
    API creates a persistent application job in quantmind_app.db.
    The worker claims the job and invokes ResearchHarness.run_trial().
    The API does NOT directly insert rows into TrialLedger.
    """
    spec = StrategySpec(
        strategy_version=req.strategy_version,
        feature_version=req.feature_version,
        signal_name=req.signal_name,
        parameters=req.parameters,
    )
    normalized = normalize_strategy_spec(spec)
    strat_id = derive_strategy_id(normalized)

    # Persist job record in quantmind_app.db for worker consumption
    job = ctx.job_orchestrator.enqueue_job(
        job_type=JobType.RESEARCH_TRIAL,
        initiator_user_id=user.user_id,
        parameters={
            "strategy_id": strat_id,
            "strategy_spec": normalized.canonical_dict(),
            "signal_name": req.signal_name,
            "dataset_version": req.dataset_version,
            "split_zone": req.split_zone,
            "research_protocol_version": req.research_protocol_version,
            "seed": req.seed,
        },
    )

    ctx.auth_service.log_action(user, "TRIAL_SUBMITTED", {"strategy_id": strat_id, "job_id": job.job_id})

    return {
        "message": "Research trial submitted to job queue",
        "job_id": job.job_id,
        "strategy_id": strat_id,
        "status": job.status.value,
    }
