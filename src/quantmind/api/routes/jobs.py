"""Asynchronous Job orchestration and polling routes."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext
from quantmind.app.jobs.models import Job, JobStatus, JobType

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("", response_model=list[Job])
async def list_jobs(
    limit: int = Query(50, le=100),
    job_type: JobType | None = Query(None),
    status: JobStatus | None = Query(None),
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[Job]:
    """List asynchronous background jobs."""
    return ctx.job_orchestrator.list_jobs(limit=limit, status=status, job_type=job_type)


@router.get("/{job_id}", response_model=Job)
async def get_job_status(
    job_id: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> Job:
    """Retrieve current execution progress and result for a specific background job."""
    job = ctx.job_orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found")
    return job
