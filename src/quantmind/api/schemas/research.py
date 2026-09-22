"""Research Workspace API schemas."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class CandidateValidateRequest(BaseModel):
    """Payload to validate and derive a candidate StrategySpec."""

    strategy_version: str = "v1.0"
    feature_version: str = "f1"
    signal_name: str = Field(..., description="Whitelisted signal name (e.g. current_bar_momentum)")
    parameters: dict[str, Any] = Field(default_factory=dict)


class CandidateValidateResponse(BaseModel):
    """Normalized preview and derived identifier for candidate."""

    strategy_id: str
    normalized_parameters: dict[str, Any]
    canonical_spec_json: str
    spec_hash: str
    is_valid: bool = True


class TrialSubmitRequest(BaseModel):
    """Payload to launch a research trial via ResearchHarness."""

    strategy_version: str = "v1.0"
    feature_version: str = "f1"
    signal_name: str
    parameters: dict[str, Any]
    dataset_version: str
    split_zone: str = "RESEARCH"
    research_protocol_version: str = "RP-2"
    seed: int = 42
    mode: str = "FIXTURE"


class ResearchTaskItem(BaseModel):
    """Structured research feedback task item."""

    task_id: str
    source_feedback_hash: str
    strategy_id: str
    failure_mode: str
    drawdown_expansion_ratio: float
    realized_slippage_bps: float
    realized_sharpe: float | None = None
    suggested_hypothesis: str
    empirical_notes: str
    created_at: str
