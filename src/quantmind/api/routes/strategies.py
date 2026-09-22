"""Strategy management and lifecycle inspection routes."""

from __future__ import annotations

import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from quantmind.api.dependencies import get_ctx, get_current_user, require_role
from quantmind.api.schemas.strategy import (
    StrategyActivateRequest,
    StrategyDetail,
    StrategyListItem,
    StrategyReResearchRequest,
    StrategyRetireRequest,
)
from quantmind.app.auth.models import User, UserRole
from quantmind.app.context import AppContext
from quantmind.strategy.registry import StrategyLifecycleState

from quantmind.strategy.compiler import SUPPORTED_SIGNALS

router = APIRouter(prefix="/strategies", tags=["Strategies"])


@router.get("/spec-schema")
async def get_strategy_spec_schema(
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve schema definitions for valid StrategySpecs derived from domain compiler."""
    return {
        "supported_signals": sorted(list(SUPPORTED_SIGNALS)),
        "default_versions": {
            "strategy_version": "1.0.0",
            "feature_version": "1.0.0",
        },
        "signal_parameters": {
            "current_bar_momentum": {
                "calendar_session_bars": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 100,
                    "description": "Number of bars in calendar session (must be > 1)",
                },
                "session_window": {
                    "type": "range",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": [0.0, 1.0],
                    "description": "Fractional session window [start, end] where 0 <= start < end <= 1",
                },
            },
            "current_bar_mean_reversion": {
                "calendar_session_bars": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 100,
                    "description": "Number of bars in calendar session (must be > 1)",
                },
                "session_window": {
                    "type": "range",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": [0.0, 1.0],
                    "description": "Fractional session window [start, end] where 0 <= start < end <= 1",
                },
            },
        },
    }


@router.get("", response_model=list[StrategyListItem])
async def list_strategies(
    state: StrategyLifecycleState | None = Query(None, description="Filter by lifecycle state"),
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[StrategyListItem]:
    """List strategies across all 8 lifecycle states."""
    conn = ctx.get_core_connection()
    try:
        if state:
            rows = conn.execute(
                "SELECT strategy_id, state, qualification_hash, created_at, updated_at FROM strategies WHERE state = ? ORDER BY updated_at DESC",
                (state.value,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT strategy_id, state, qualification_hash, created_at, updated_at FROM strategies ORDER BY updated_at DESC"
            ).fetchall()

        return [
            StrategyListItem(
                strategy_id=r["strategy_id"],
                state=r["state"],
                qualification_hash=r["qualification_hash"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/{strategy_id}", response_model=StrategyDetail)
async def get_strategy_detail(
    strategy_id: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> StrategyDetail:
    """Retrieve full strategy detail across all tabs."""
    conn = ctx.get_core_connection()
    try:
        s = conn.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,)).fetchone()
        if not s:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy '{strategy_id}' not found")

        # 1. Specification
        spec_dict = None
        try:
            strat_record = ctx.strategy_registry.get_strategy(strategy_id)
            if strat_record.spec:
                spec_dict = {
                    "strategy_version": strat_record.spec.strategy_version,
                    "feature_version": strat_record.spec.feature_version,
                    "signal_name": strat_record.spec.signal_name,
                    "parameters": strat_record.spec.parameters,
                }
        except Exception:
            pass

        # 2. Qualification
        qual_dict = None
        if s["qualification_hash"]:
            qr = conn.execute(
                "SELECT * FROM strategy_qualifications WHERE record_hash = ?", (s["qualification_hash"],)
            ).fetchone()
            if qr:
                qual_dict = {
                    "qualification_id": qr["qualification_id"],
                    "record_hash": qr["record_hash"],
                    "dsr": float(qr["dsr"]),
                    "observed_sharpe": float(qr["observed_sharpe"]),
                    "trade_count": int(qr["trade_count"]),
                    "effective_trial_count": int(qr["effective_trial_count"]),
                    "holdout_state": qr["holdout_state"],
                    "final_status": qr["final_status"],
                    "dataset_version": qr["dataset_version"],
                    "created_at": qr["created_at"],
                }

        # 3. Baseline & Replay Report
        base_dict = None
        rep_dict = None
        base = conn.execute(
            "SELECT * FROM evaluation_baselines WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if base:
            base_dict = {
                "binding_hash": base["binding_hash"],
                "baseline_replay_report_hash": base["baseline_replay_report_hash"],
                "baseline_dataset_version": base["baseline_dataset_version"],
                "created_at": base["created_at"],
            }
            rep = conn.execute(
                "SELECT * FROM paper_reports WHERE report_hash = ?",
                (base["baseline_replay_report_hash"],),
            ).fetchone()
            if rep:
                rep_data = json.loads(rep["report_json"]) if rep["report_json"] else {}
                rep_dict = {
                    "report_hash": rep["report_hash"],
                    "gross_pnl": float(rep_data.get("gross_pnl", 0.0)),
                    "net_pnl": float(rep_data.get("net_pnl", 0.0)),
                    "costs": float(rep_data.get("costs", 0.0)),
                    "slippage": float(rep_data.get("slippage", 0.0)),
                    "max_drawdown_bps": float(rep_data.get("max_drawdown_bps", 0.0)),
                    "sharpe_ratio": float(rep_data.get("sharpe_ratio", 0.0)),
                    "trade_count": int(rep_data.get("trade_count", 0)),
                    "win_rate": float(rep_data.get("win_rate", 0.0)),
                }

        # 4. Regime
        reg_dict = None
        reg = conn.execute(
            "SELECT * FROM paper_evaluation_regimes WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if reg:
            reg_dict = {
                "regime_hash": reg["regime_hash"],
                "monitoring_protocol_version": reg["monitoring_protocol_version"],
                "execution_policy": reg["execution_policy"],
                "forward_dataset_version": reg["forward_dataset_version"],
                "created_at": reg["created_at"],
            }

        # 5. Transitions
        trans_rows = conn.execute(
            "SELECT * FROM paper_evaluation_transitions WHERE strategy_id = ? ORDER BY timestamp DESC",
            (strategy_id,),
        ).fetchall()
        transitions = [
            {
                "transition_id": t["transition_id"],
                "transition_hash": t["transition_hash"],
                "old_state": t["old_state"],
                "new_state": t["new_state"],
                "reason": t["reason"],
                "initiator": t["initiator"],
                "timestamp": t["timestamp"],
            }
            for t in trans_rows
        ]

        # 6. Recent Snapshots
        snap_rows = conn.execute(
            "SELECT * FROM monitoring_snapshots WHERE strategy_id = ? ORDER BY created_at DESC LIMIT 10",
            (strategy_id,),
        ).fetchall()
        snapshots = [
            {
                "snapshot_id": sn["snapshot_id"],
                "snapshot_hash": sn["snapshot_hash"],
                "max_drawdown_bps": float(sn["max_drawdown_bps"]),
                "realized_sharpe": float(sn["realized_sharpe"] or 0.0),
                "total_trades": int(sn["total_trades"]),
                "created_at": sn["created_at"],
            }
            for sn in snap_rows
        ]

        # 7. Degradation Events
        deg_rows = conn.execute(
            "SELECT * FROM degradation_events WHERE strategy_id = ? ORDER BY timestamp DESC",
            (strategy_id,),
        ).fetchall()
        degradations = [
            {
                "event_hash": dg["event_hash"],
                "rule_name": dg["rule_name"],
                "threshold_value": float(dg["threshold_value"]),
                "observed_value": float(dg["observed_value"]),
                "timestamp": dg["timestamp"],
            }
            for dg in deg_rows
        ]

        # 8. Feedback Records
        fb_rows = conn.execute(
            "SELECT * FROM research_feedback WHERE strategy_id = ? ORDER BY created_at DESC",
            (strategy_id,),
        ).fetchall()
        feedback = [
            {
                "feedback_id": fb["feedback_id"],
                "feedback_hash": fb["feedback_hash"],
                "failure_mode": fb["failure_mode"],
                "drawdown_expansion_ratio": float(fb["drawdown_expansion_ratio"]),
                "realized_slippage_bps": float(fb["realized_slippage_bps"]),
                "created_at": fb["created_at"],
            }
            for fb in fb_rows
        ]

        return StrategyDetail(
            strategy_id=strategy_id,
            state=s["state"],
            qualification_hash=s["qualification_hash"],
            created_at=s["created_at"],
            updated_at=s["updated_at"],
            specification=spec_dict,
            qualification=qual_dict,
            baseline=base_dict,
            replay_report=rep_dict,
            regime=reg_dict,
            transitions=transitions,
            recent_snapshots=snapshots,
            degradation_events=degradations,
            feedback_records=feedback,
        )
    finally:
        conn.close()


@router.post("/{strategy_id}/activate", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def activate_strategy(
    strategy_id: str,
    req: StrategyActivateRequest,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Activate strategy from PAPER_ELIGIBLE to PAPER_ACTIVE (Admin only)."""
    trans = ctx.governance_service.activate_strategy(strategy_id, initiator=f"{user.username}:{req.initiator}")
    ctx.auth_service.log_action(user, "STRATEGY_ACTIVATE", {"strategy_id": strategy_id})
    return {
        "message": "Strategy activated successfully",
        "transition_id": trans.transition_hash,
        "transition_hash": trans.transition_hash,
        "state": "PAPER_ACTIVE",
        "new_state": "PAPER_ACTIVE",
    }


@router.post("/{strategy_id}/retire", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def retire_strategy(
    strategy_id: str,
    req: StrategyRetireRequest,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Retire strategy permanently to RETIRED (Admin only). Position must be flat."""
    # Check current position from PaperLedger
    pos_row = ctx.paper_ledger.get_position(strategy_id)
    curr_qty = float(pos_row.quantity) if pos_row else 0.0
    trans = ctx.governance_service.retire_strategy(
        strategy_id=strategy_id,
        reason=req.reason,
        position_quantity=curr_qty,
        initiator=f"{user.username}:{req.initiator}",
    )
    ctx.auth_service.log_action(user, "STRATEGY_RETIRE", {"strategy_id": strategy_id, "reason": req.reason})
    return {
        "message": "Strategy retired successfully",
        "transition_id": trans.transition_hash,
        "transition_hash": trans.transition_hash,
        "state": "RETIRED",
        "new_state": "RETIRED",
    }


@router.post("/{strategy_id}/re-research", dependencies=[Depends(require_role(UserRole.RESEARCHER, UserRole.ADMIN))])
async def re_research_strategy(
    strategy_id: str,
    req: StrategyReResearchRequest,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Transition DEGRADED strategy back to RESEARCH."""
    trans = ctx.governance_service.re_research_strategy(
        strategy_id=strategy_id,
        reason=req.reason,
        initiator=f"{user.username}:{req.initiator}",
    )
    ctx.auth_service.log_action(user, "STRATEGY_RE_RESEARCH", {"strategy_id": strategy_id, "reason": req.reason})
    return {
        "message": "Strategy returned to RESEARCH",
        "transition_id": trans.transition_hash,
        "transition_hash": trans.transition_hash,
        "state": "RESEARCH",
        "new_state": "RESEARCH",
    }
