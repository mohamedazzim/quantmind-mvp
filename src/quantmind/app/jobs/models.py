"""Application Job System models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states of an asynchronous application job."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobType(str, Enum):
    """Supported asynchronous job task types."""

    RESEARCH_TRIAL = "RESEARCH_TRIAL"
    PAPER_REPLAY = "PAPER_REPLAY"
    EVALUATION_WINDOW = "EVALUATION_WINDOW"
    QUALIFICATION_GATE = "QUALIFICATION_GATE"


class Job(BaseModel):
    """Asynchronous job representation."""

    job_id: str
    job_type: JobType
    status: JobStatus
    progress_pct: int = 0
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    initiator_user_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    result_ref: str | None = None
    error_message: str | None = None
