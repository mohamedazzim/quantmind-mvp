"""Dashboard Read-Model Adapter for high-density overview."""

from __future__ import annotations

import json
from typing import Any
from quantmind.app.context import AppContext
from quantmind.strategy.registry import StrategyLifecycleState


class DashboardAdapter:
    """Aggregates system-wide metrics directly from authoritative ledgers."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Aggregate high-level overview metrics across the platform."""
        conn = self.ctx.get_core_connection()
        try:
            # 1. Strategy counts by state
            strat_rows = conn.execute(
                "SELECT state, COUNT(*) as count FROM strategies GROUP BY state"
            ).fetchall()
            counts_by_state = {state.value: 0 for state in StrategyLifecycleState}
            for r in strat_rows:
                counts_by_state[r["state"]] = int(r["count"])

            total_strategies = sum(counts_by_state.values())
            active_strategies = counts_by_state.get(StrategyLifecycleState.PAPER_ACTIVE.value, 0)
            eligible_strategies = counts_by_state.get(StrategyLifecycleState.PAPER_ELIGIBLE.value, 0)
            degraded_strategies = counts_by_state.get(StrategyLifecycleState.DEGRADED.value, 0)
            retired_strategies = counts_by_state.get(StrategyLifecycleState.RETIRED.value, 0)

            # 2. Paper performance overview from latest replay reports
            report_rows = conn.execute(
                "SELECT report_json FROM paper_reports ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            pnls = []
            sharpes = []
            for r in report_rows:
                if r["report_json"]:
                    data = json.loads(r["report_json"])
                    pnls.append(float(data.get("net_pnl", 0.0)))
                    if data.get("sharpe_ratio") is not None:
                        sharpes.append(float(data["sharpe_ratio"]))
            total_net_pnl = sum(pnls)
            avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0


            # 3. Paper risk events
            risk_count_row = conn.execute("SELECT COUNT(*) FROM paper_risk_events").fetchone()
            risk_events_count = int(risk_count_row[0]) if risk_count_row else 0

            # 4. Trials & budget summary
            trials_count_row = conn.execute("SELECT COUNT(*) FROM trials").fetchone()
            total_trials = int(trials_count_row[0]) if trials_count_row else 0

            # 5. Degradation events count
            deg_count_row = conn.execute("SELECT COUNT(*) FROM degradation_events").fetchone()
            total_degradations = int(deg_count_row[0]) if deg_count_row else 0

            # 6. Research feedback records
            fb_count_row = conn.execute("SELECT COUNT(*) FROM research_feedback").fetchone()
            total_feedback = int(fb_count_row[0]) if fb_count_row else 0

            # 7. Recent state transitions
            trans_rows = conn.execute(
                """
                SELECT transition_id, strategy_id, old_state, new_state, initiator, reason, timestamp
                FROM paper_evaluation_transitions ORDER BY timestamp DESC LIMIT 6
                """
            ).fetchall()
            recent_transitions = [
                {
                    "transition_id": r["transition_id"],
                    "strategy_id": r["strategy_id"],
                    "old_state": r["old_state"],
                    "new_state": r["new_state"],
                    "initiator": r["initiator"],
                    "reason": r["reason"],
                    "timestamp": r["timestamp"],
                }
                for r in trans_rows
            ]

            # 8. Recent degradation events
            event_rows = conn.execute(
                """
                SELECT event_hash, strategy_id, rule_name, threshold_value, observed_value, timestamp
                FROM degradation_events ORDER BY timestamp DESC LIMIT 6
                """
            ).fetchall()
            recent_degradations = [
                {
                    "event_hash": r["event_hash"],
                    "strategy_id": r["strategy_id"],
                    "rule_name": r["rule_name"],
                    "threshold_value": float(r["threshold_value"]),
                    "observed_value": float(r["observed_value"]),
                    "timestamp": r["timestamp"],
                }
                for r in event_rows
            ]

            # 9. Recent strategies
            strat_recent = conn.execute(
                """
                SELECT strategy_id, state, qualification_hash, registered_at AS created_at, updated_at
                FROM strategies ORDER BY updated_at DESC LIMIT 6
                """
            ).fetchall()
            recent_strategies = [
                {
                    "strategy_id": r["strategy_id"],
                    "state": r["state"],
                    "qualification_hash": r["qualification_hash"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in strat_recent
            ]

            summary_data = {
                "total_strategies": total_strategies,
                "active_strategies": active_strategies,
                "eligible_strategies": eligible_strategies,
                "degraded_strategies": degraded_strategies,
                "retired_strategies": retired_strategies,
                "total_trials": total_trials,
                "total_degradations": total_degradations,
                "total_degradation_events": total_degradations,
                "total_feedback": total_feedback,
                "risk_events_count": risk_events_count,
                "total_net_pnl": round(total_net_pnl, 2),
                "average_sharpe": round(avg_sharpe, 2),
                "avg_sharpe_ratio": round(avg_sharpe, 2),
            }

            return {
                **summary_data,
                "summary": summary_data,
                "kpis": summary_data,
                "counts_by_state": counts_by_state,
                "recent_transitions": recent_transitions,
                "recent_degradations": recent_degradations,
                "recent_strategies": recent_strategies,
            }
        finally:
            conn.close()
