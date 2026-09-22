"""Monitoring snapshots and degradation detection routes."""

from __future__ import annotations

import json
from typing import Any
from fastapi import APIRouter, Depends, Query

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("", response_model=dict[str, Any])
async def get_monitoring_overview(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retrieve global monitoring snapshots and active degradation breaches."""
    conn = ctx.get_core_connection()
    try:
        deg_rows = conn.execute(
            "SELECT * FROM degradation_events ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
        snap_rows = conn.execute(
            "SELECT * FROM monitoring_snapshots ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

        return {
            "total_degradations": len(deg_rows),
            "total_snapshots": len(snap_rows),
            "recent_degradations": [
                {
                    "event_hash": d["event_hash"],
                    "strategy_id": d["strategy_id"],
                    "rule_name": d["rule_name"],
                    "threshold_value": float(d["threshold_value"]),
                    "observed_value": float(d["observed_value"]),
                    "monitoring_protocol_version": d["monitoring_protocol_version"],
                    "snapshot_hash": d["snapshot_hash"],
                    "timestamp": d["timestamp"],
                }
                for d in deg_rows
            ],
            "recent_snapshots": [
                {
                    "snapshot_id": s["snapshot_id"],
                    "snapshot_hash": s["snapshot_hash"],
                    "strategy_id": s["strategy_id"],
                    "regime_hash": s["regime_hash"],
                    "total_trades": int(s["total_trades"]),
                    "net_pnl": float(s["net_pnl"]),
                    "max_drawdown_bps": float(s["max_drawdown_bps"]),
                    "realized_sharpe": float(s["realized_sharpe"] or 0.0),
                    "realized_slippage_bps": float(s["realized_slippage_bps"]),
                    "created_at": s["created_at"],
                }
                for s in snap_rows
            ],
        }
    finally:
        conn.close()


@router.get("/{strategy_id}", response_model=dict[str, Any])
async def get_strategy_monitoring(
    strategy_id: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retrieve monitoring history and degradation events for a specific strategy."""
    conn = ctx.get_core_connection()
    try:
        deg_rows = conn.execute(
            "SELECT * FROM degradation_events WHERE strategy_id = ? ORDER BY timestamp DESC",
            (strategy_id,),
        ).fetchall()
        snap_rows = conn.execute(
            "SELECT * FROM monitoring_snapshots WHERE strategy_id = ? ORDER BY created_at DESC",
            (strategy_id,),
        ).fetchall()

        return {
            "strategy_id": strategy_id,
            "degradation_events": [
                {
                    "event_hash": d["event_hash"],
                    "rule_name": d["rule_name"],
                    "threshold_value": float(d["threshold_value"]),
                    "observed_value": float(d["observed_value"]),
                    "timestamp": d["timestamp"],
                    "details": json.loads(d["details_json"]),
                }
                for d in deg_rows
            ],
            "snapshots": [
                {
                    "snapshot_id": s["snapshot_id"],
                    "snapshot_hash": s["snapshot_hash"],
                    "max_drawdown_bps": float(s["max_drawdown_bps"]),
                    "realized_sharpe": float(s["realized_sharpe"] or 0.0),
                    "realized_slippage_bps": float(s["realized_slippage_bps"]),
                    "cost_to_turnover_bps": float(s["cost_to_turnover_bps"]),
                    "net_pnl": float(s["net_pnl"]),
                    "total_trades": int(s["total_trades"]),
                    "created_at": s["created_at"],
                }
                for s in snap_rows
            ],
        }
    finally:
        conn.close()
