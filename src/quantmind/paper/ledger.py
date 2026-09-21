"""Paper Trading Ledger & Audit Trail (PRD v3.9).

Append-only SQLite ledger storing all orders, fills, positions, risk events,
and replay reports generated during deterministic paper trading.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from quantmind.paper.models import (
    PaperFill,
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    PaperRiskEvent,
    ReplayReport,
    ReplaySessionSummary,
)


class PaperLedger:
    """Append-only ledger for all paper replay events."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is not None:
            self._connection = connection
        elif db_path is not None:
            self._connection = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,
            )
        else:
            self._connection = sqlite3.connect(
                ":memory:",
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,
            )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS paper_orders (
                order_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                qualification_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                signal_timestamp TEXT NOT NULL,
                submit_timestamp TEXT NOT NULL,
                requested_price REAL NOT NULL,
                status TEXT NOT NULL,
                rejection_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS paper_fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                fill_timestamp TEXT NOT NULL,
                fill_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                side INTEGER NOT NULL,
                cost REAL NOT NULL,
                slippage REAL NOT NULL,
                FOREIGN KEY(order_id) REFERENCES paper_orders(order_id)
            );

            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                fees_costs REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_risk_events (
                event_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                limit_value REAL NOT NULL,
                requested_value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                reason TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_reports (
                qualification_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                report_hash TEXT PRIMARY KEY,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TRIGGER IF NOT EXISTS paper_orders_no_delete
            BEFORE DELETE ON paper_orders
            BEGIN
                SELECT RAISE(ABORT, 'paper orders are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_orders_no_update
            BEFORE UPDATE ON paper_orders
            BEGIN
                SELECT RAISE(ABORT, 'paper orders are immutable and cannot be updated');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_fills_no_delete
            BEFORE DELETE ON paper_fills
            BEGIN
                SELECT RAISE(ABORT, 'paper fills are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_fills_no_update
            BEFORE UPDATE ON paper_fills
            BEGIN
                SELECT RAISE(ABORT, 'paper fills are immutable and cannot be updated');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_positions_no_delete
            BEFORE DELETE ON paper_positions
            BEGIN
                SELECT RAISE(ABORT, 'paper positions are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_positions_no_update
            BEFORE UPDATE ON paper_positions
            BEGIN
                SELECT RAISE(ABORT, 'paper positions are immutable and cannot be updated');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_risk_no_delete
            BEFORE DELETE ON paper_risk_events
            BEGIN
                SELECT RAISE(ABORT, 'paper risk events are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_risk_no_update
            BEFORE UPDATE ON paper_risk_events
            BEGIN
                SELECT RAISE(ABORT, 'paper risk events are immutable and cannot be updated');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_reports_no_delete
            BEFORE DELETE ON paper_reports
            BEGIN
                SELECT RAISE(ABORT, 'paper reports are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_reports_no_update
            BEFORE UPDATE ON paper_reports
            BEGIN
                SELECT RAISE(ABORT, 'paper reports are immutable and cannot be updated');
            END;
            """
        )

    def record_order(self, order: PaperOrder) -> None:
        self._connection.execute(
            """
            INSERT INTO paper_orders (
                order_id, strategy_id, qualification_id, symbol, side,
                quantity, order_type, signal_timestamp, submit_timestamp,
                requested_price, status, rejection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                order.strategy_id,
                order.qualification_id,
                order.symbol,
                order.side,
                order.quantity,
                order.order_type,
                order.signal_timestamp,
                order.submit_timestamp,
                order.requested_price,
                order.status.value,
                order.rejection_reason,
            ),
        )

    def record_fill(self, fill: PaperFill) -> None:
        self._connection.execute(
            """
            INSERT INTO paper_fills (
                fill_id, order_id, strategy_id, symbol, fill_timestamp,
                fill_price, quantity, side, cost, slippage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.strategy_id,
                fill.symbol,
                fill.fill_timestamp,
                fill.fill_price,
                fill.quantity,
                fill.side,
                fill.cost,
                fill.slippage,
            ),
        )

    def record_risk_event(self, event: PaperRiskEvent) -> None:
        self._connection.execute(
            """
            INSERT INTO paper_risk_events (
                event_id, order_id, strategy_id, rule_name,
                limit_value, requested_value, timestamp, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.order_id,
                event.strategy_id,
                event.rule_name,
                event.limit_value,
                event.requested_value,
                event.timestamp,
                event.reason,
            ),
        )

    def record_position_snapshot(self, timestamp: str, strategy_id: str, pos: PaperPosition) -> None:
        self._connection.execute(
            """
            INSERT INTO paper_positions (
                timestamp, strategy_id, symbol, quantity, entry_price,
                current_price, realized_pnl, unrealized_pnl, fees_costs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                strategy_id,
                pos.symbol,
                pos.quantity,
                pos.entry_price,
                pos.current_price,
                pos.realized_pnl,
                pos.unrealized_pnl,
                pos.fees_costs,
            ),
        )

    def record_report(self, report: ReplayReport) -> None:
        row = self._connection.execute(
            "SELECT report_json FROM paper_reports WHERE report_hash = ?",
            (report.report_hash,),
        ).fetchone()
        if row is not None:
            if row["report_json"] == report.canonical_json():
                return  # Idempotent
            raise sqlite3.IntegrityError(f"Conflicting report exists for report_hash '{report.report_hash}'")

        self._connection.execute(
            """
            INSERT INTO paper_reports (
                qualification_id, strategy_id, report_hash, report_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                report.qualification_id,
                report.strategy_id,
                report.report_hash,
                report.canonical_json(),
                report.created_at,
            ),
        )

    def get_report(self, report_hash: str) -> ReplayReport | None:
        row = self._connection.execute(
            "SELECT report_json, created_at FROM paper_reports WHERE report_hash = ?",
            (report_hash,),
        ).fetchone()
        if row is None:
            return None
        rep_dict = json.loads(row["report_json"])
        sessions = tuple(
            ReplaySessionSummary(
                session_id=s["session_id"],
                start_ts=s["start_ts"],
                end_ts=s["end_ts"],
                trades=s["trades"],
                gross_pnl=s["gross_pnl"],
                net_pnl=s["net_pnl"],
            )
            for s in rep_dict.get("session_breakdown", [])
        )
        return ReplayReport(
            strategy_id=rep_dict["strategy_id"],
            qualification_id=rep_dict["qualification_id"],
            dataset_version=rep_dict["dataset_version"],
            trade_count=rep_dict["trade_count"],
            gross_pnl=rep_dict["gross_pnl"],
            net_pnl=rep_dict["net_pnl"],
            costs=rep_dict["costs"],
            slippage=rep_dict["slippage"],
            max_drawdown_bps=rep_dict["max_drawdown_bps"],
            exposure=rep_dict["exposure"],
            win_rate=rep_dict["win_rate"],
            expectancy=rep_dict["expectancy"],
            sharpe_ratio=rep_dict.get("sharpe_ratio"),
            session_breakdown=sessions,
            report_hash=report_hash,
            created_at=row["created_at"],
            strategy_spec_hash=rep_dict.get("strategy_spec_hash", ""),
            qualification_hash=rep_dict.get("qualification_hash", ""),
            dataset_sha256=rep_dict.get("dataset_sha256", ""),
            research_protocol_version=rep_dict.get("research_protocol_version", ""),
            symbol=rep_dict.get("symbol", "NIFTY_FUT"),
            lot_size=rep_dict.get("lot_size", 50),
            tick_size=rep_dict.get("tick_size", 0.05),
            slippage_bps_per_side=rep_dict.get("slippage_bps_per_side", 0.0),
            cost_schedule_id=rep_dict.get("cost_schedule_id", ""),
            cost_schedule_hash=rep_dict.get("cost_schedule_hash", ""),
            risk_config_hash=rep_dict.get("risk_config_hash", ""),
            execution_policy=rep_dict.get("execution_policy", "next_bar_open_v1"),
            quantity=rep_dict.get("quantity", 1),
            hold_bars=rep_dict.get("hold_bars", 1),
            initial_capital=rep_dict.get("initial_capital", 100_000.0),
            enforce_session_boundaries=rep_dict.get("enforce_session_boundaries", True),
            split_zone=rep_dict.get("split_zone", "FORWARD_PAPER"),
            bars_sha256=rep_dict.get("bars_sha256", ""),
        )

    def get_orders(self, strategy_id: str | None = None) -> list[sqlite3.Row]:
        if strategy_id:
            return self._connection.execute(
                "SELECT * FROM paper_orders WHERE strategy_id = ? ORDER BY submit_timestamp ASC",
                (strategy_id,),
            ).fetchall()
        return self._connection.execute("SELECT * FROM paper_orders ORDER BY submit_timestamp ASC").fetchall()

    def get_fills(self, strategy_id: str | None = None) -> list[sqlite3.Row]:
        if strategy_id:
            return self._connection.execute(
                "SELECT * FROM paper_fills WHERE strategy_id = ? ORDER BY fill_timestamp ASC",
                (strategy_id,),
            ).fetchall()
        return self._connection.execute("SELECT * FROM paper_fills ORDER BY fill_timestamp ASC").fetchall()

    def get_risk_events(self, strategy_id: str | None = None) -> list[sqlite3.Row]:
        if strategy_id:
            return self._connection.execute(
                "SELECT * FROM paper_risk_events WHERE strategy_id = ? ORDER BY timestamp ASC",
                (strategy_id,),
            ).fetchall()
        return self._connection.execute("SELECT * FROM paper_risk_events ORDER BY timestamp ASC").fetchall()
