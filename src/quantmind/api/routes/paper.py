"""Paper execution and deterministic replay inspection routes."""

from __future__ import annotations

import json
from typing import Any
from fastapi import APIRouter, Depends, Query

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext

router = APIRouter(prefix="/paper", tags=["Paper Execution"])


@router.get("/overview", response_model=dict[str, Any])
async def get_paper_overview(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retrieve overview of forward paper execution state and active positions."""
    conn = ctx.get_core_connection()
    try:
        active_strats = conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE state = 'PAPER_ACTIVE'"
        ).fetchone()[0]
        degraded_strats = conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE state = 'DEGRADED'"
        ).fetchone()[0]
        total_orders = conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
        total_fills = conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0]
        total_risk_events = conn.execute("SELECT COUNT(*) FROM paper_risk_events").fetchone()[0]

        reports = conn.execute(
            "SELECT report_json FROM paper_reports ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        total_pnl = sum(
            float(json.loads(r["report_json"]).get("net_pnl", 0.0))
            for r in reports
            if r["report_json"]
        )

        positions = conn.execute(
            "SELECT strategy_id, symbol, quantity, entry_price, realized_pnl, unrealized_pnl FROM paper_positions WHERE ABS(quantity) > 0"
        ).fetchall()

        return {
            "active_strategies": int(active_strats),
            "degraded_strategies": int(degraded_strats),
            "total_orders": int(total_orders),
            "total_fills": int(total_fills),
            "total_risk_events": int(total_risk_events),
            "total_net_pnl": round(total_pnl, 2),
            "open_positions": [
                {
                    "strategy_id": p["strategy_id"],
                    "symbol": p["symbol"],
                    "quantity": float(p["quantity"]),
                    "entry_price": float(p["entry_price"]),
                    "realized_pnl": float(p["realized_pnl"]),
                    "unrealized_pnl": float(p["unrealized_pnl"]),
                }
                for p in positions
            ],
        }
    finally:
        conn.close()


@router.get("/orders", response_model=list[dict[str, Any]])
async def list_orders(
    strategy_id: str | None = Query(None),
    limit: int = 100,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List simulated paper orders."""
    conn = ctx.get_core_connection()
    try:
        query = "SELECT * FROM paper_orders"
        params: list[Any] = []
        if strategy_id:
            query += " WHERE strategy_id = ?"
            params.append(strategy_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [
            {
                "order_id": r["order_id"],
                "strategy_id": r["strategy_id"],
                "symbol": r["symbol"],
                "side": r["side"],
                "quantity": float(r["quantity"]),
                "order_type": r["order_type"],
                "status": r["status"],
                "limit_price": float(r["limit_price"]) if r["limit_price"] is not None else None,
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/fills", response_model=list[dict[str, Any]])
async def list_fills(
    strategy_id: str | None = Query(None),
    limit: int = 100,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List simulated execution fills."""
    conn = ctx.get_core_connection()
    try:
        query = "SELECT * FROM paper_fills"
        params: list[Any] = []
        if strategy_id:
            query += " WHERE strategy_id = ?"
            params.append(strategy_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [
            {
                "fill_id": r["fill_id"],
                "order_id": r["order_id"],
                "strategy_id": r["strategy_id"],
                "symbol": r["symbol"],
                "side": r["side"],
                "quantity": float(r["quantity"]),
                "price": float(r["price"]),
                "cost": float(r["cost"]),
                "slippage": float(r["slippage"]),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/positions", response_model=list[dict[str, Any]])
async def list_positions(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List real-time strategy positions from PaperLedger."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute("SELECT * FROM paper_positions").fetchall()
        return [
            {
                "strategy_id": r["strategy_id"],
                "symbol": r["symbol"],
                "quantity": float(r["quantity"]),
                "entry_price": float(r["entry_price"]),
                "realized_pnl": float(r["realized_pnl"]),
                "unrealized_pnl": float(r["unrealized_pnl"]),
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/risk", response_model=list[dict[str, Any]])
async def list_risk_events(
    limit: int = 100,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List pre-trade risk check rejections."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute("SELECT * FROM paper_risk_events ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [
            {
                "event_id": r["event_id"],
                "strategy_id": r["strategy_id"],
                "order_id": r["order_id"],
                "rule_name": r["rule_name"],
                "reason": r["reason"],
                "timestamp": r["timestamp"],
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/reports", response_model=list[dict[str, Any]])
async def list_replay_reports(
    limit: int = 50,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List authoritative deterministic replay reports with SHA-256 digests."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute("SELECT * FROM paper_reports ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        reports = []
        for r in rows:
            rep_data = json.loads(r["report_json"]) if r["report_json"] else {}
            reports.append({
                "strategy_id": r["strategy_id"],
                "qualification_id": r["qualification_id"],
                "report_hash": r["report_hash"],
                "created_at": r["created_at"],
                "gross_pnl": float(rep_data.get("gross_pnl", 0.0)),
                "net_pnl": float(rep_data.get("net_pnl", 0.0)),
                "costs": float(rep_data.get("costs", 0.0)),
                "slippage": float(rep_data.get("slippage", 0.0)),
                "max_drawdown_bps": float(rep_data.get("max_drawdown_bps", 0.0)),
                "sharpe_ratio": float(rep_data.get("sharpe_ratio", 0.0)),
                "trade_count": int(rep_data.get("trade_count", 0)),
                "win_rate": float(rep_data.get("win_rate", 0.0)),
            })
        return reports
    finally:
        conn.close()



@router.post("/replay/submit")
async def submit_paper_replay(
    payload: dict[str, Any],
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Enqueue a deterministic paper replay job for worker execution."""
    strategy_id = payload.get("strategy_id")
    if not strategy_id:
        raise HTTPException(status_code=400, detail="Missing 'strategy_id'")

    strat = ctx.strategy_registry.get_strategy(strategy_id)
    if strat.state.value in ("REJECTED", "RETIRED"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run replay for strategy in state '{strat.state.value}'",
        )

    from quantmind.app.jobs.models import JobType
    job = ctx.job_orchestrator.enqueue_job(
        job_type=JobType.PAPER_REPLAY,
        initiator_user_id=user.user_id,
        parameters={
            "strategy_id": strategy_id,
            "bars": payload.get("bars", 100),
        },
    )
    ctx.auth_service.log_action(user, "PAPER_REPLAY_SUBMITTED", {"strategy_id": strategy_id, "job_id": job.job_id})
    return {
        "message": "Paper replay job submitted to worker queue",
        "job_id": job.job_id,
        "strategy_id": strategy_id,
        "status": job.status.value,
    }

