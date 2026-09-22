"""Dashboard routes."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.adapters.dashboard import DashboardAdapter
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("", response_model=dict[str, Any])
@router.get("/kpis", response_model=dict[str, Any])
async def get_dashboard(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retrieve system overview KPIs, metrics, and recent activities."""
    adapter = DashboardAdapter(ctx)
    return adapter.get_dashboard_summary()
