"""Strategy API schemas."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class StrategyListItem(BaseModel):
    """Summary item for strategy inventory."""

    strategy_id: str
    state: str
    qualification_hash: str | None = None
    created_at: str
    updated_at: str


class StrategyDetail(BaseModel):
    """Detailed view of a strategy including all tabs data."""

    strategy_id: str
    state: str
    qualification_hash: str | None = None
    created_at: str
    updated_at: str
    specification: dict[str, Any] | None = None
    qualification: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    replay_report: dict[str, Any] | None = None
    regime: dict[str, Any] | None = None
    transitions: list[dict[str, Any]] = Field(default_factory=list)
    recent_snapshots: list[dict[str, Any]] = Field(default_factory=list)
    degradation_events: list[dict[str, Any]] = Field(default_factory=list)
    feedback_records: list[dict[str, Any]] = Field(default_factory=list)


class StrategyActivateRequest(BaseModel):
    """Payload to activate a strategy to PAPER_ACTIVE."""

    initiator: str = "UI_USER"


class StrategyRetireRequest(BaseModel):
    """Payload to retire a strategy."""

    reason: str = Field(..., min_length=3)
    initiator: str = "UI_USER"


class StrategyReResearchRequest(BaseModel):
    """Payload to transition degraded strategy back to RESEARCH."""

    reason: str = Field(..., min_length=3)
    initiator: str = "UI_USER"
