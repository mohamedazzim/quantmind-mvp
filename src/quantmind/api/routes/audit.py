"""Audit Provenance Explorer routes."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.adapters.audit import AuditAdapter
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext

router = APIRouter(prefix="/audit", tags=["Audit Explorer"])


@router.get("/graph/{identifier}", response_model=dict[str, Any])
@router.get("/{identifier}", response_model=dict[str, Any])
async def get_audit_provenance(
    identifier: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Resolve and return the complete provenance graph for any entity or hash."""
    adapter = AuditAdapter(ctx)
    return adapter.get_provenance_graph(identifier)

