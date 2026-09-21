"""Evaluation Ledger & SQLite Append-Only Evidence (PRD v4.0 Milestone 2).

This module implements the immutable evaluation ledger persisting:
- monitoring_snapshots: windowed evaluation snapshots with snapshot_hash
- degradation_events: threshold breach events with event_hash
- paper_evaluation_transitions: lifecycle transition audit records with transition_hash

All tables are protected against tampering via SQLite BEFORE UPDATE and BEFORE DELETE triggers.
Idempotent insertion ensures identical repeated evaluations return existing records without
error or corruption, while duplicate hashes with conflicting content raise integrity errors.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringSnapshot,
    PaperEvaluationBaseline,
    PaperEvaluationRegime,
    PaperEvaluationTransition,
)
from quantmind.paper.models import ReplayReport, ReplaySessionSummary


class EvaluationLedgerError(RuntimeError):
    """Raised when an evaluation ledger operation fails."""


class EvaluationLedgerIntegrityError(EvaluationLedgerError):
    """Raised when an integrity check, digest verification, or immutability violation occurs."""


class EvaluationLedger:
    """Authoritative append-only SQLite ledger for paper evaluation evidence."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is not None:
            self._connection = connection
            self._owned = False
        elif db_path is not None:
            self._connection = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,
            )
            self._owned = True
        else:
            self._connection = sqlite3.connect(
                ":memory:",
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,
            )
            self._owned = True

        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        """Create tables, foreign keys, and immutability triggers deterministically."""
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS monitoring_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                qualification_hash TEXT NOT NULL,
                replay_report_hash TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                split_zone TEXT NOT NULL,
                monitoring_protocol_version TEXT NOT NULL,
                monitoring_config_hash TEXT NOT NULL,
                window_start_ts TEXT NOT NULL,
                window_end_ts TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                net_pnl REAL NOT NULL,
                max_drawdown_bps REAL NOT NULL,
                realized_sharpe REAL,
                realized_slippage_bps REAL NOT NULL,
                cost_to_turnover_bps REAL NOT NULL,
                risk_event_count INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                snapshot_hash TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS degradation_events (
                event_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                qualification_hash TEXT NOT NULL,
                snapshot_hash TEXT NOT NULL,
                baseline_replay_report_hash TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                observed_value REAL NOT NULL,
                monitoring_protocol_version TEXT NOT NULL,
                monitoring_config_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                details_json TEXT NOT NULL,
                event_hash TEXT UNIQUE NOT NULL,
                FOREIGN KEY(snapshot_hash) REFERENCES monitoring_snapshots(snapshot_hash)
            );

            CREATE TABLE IF NOT EXISTS paper_evaluation_transitions (
                transition_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                old_state TEXT NOT NULL,
                new_state TEXT NOT NULL,
                initiator TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                transition_hash TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_baselines (
                strategy_id TEXT NOT NULL,
                qualification_hash TEXT NOT NULL,
                baseline_replay_report_hash TEXT NOT NULL,
                baseline_dataset_version TEXT NOT NULL,
                baseline_dataset_sha256 TEXT NOT NULL,
                baseline_split_zone TEXT NOT NULL,
                baseline_execution_policy TEXT NOT NULL,
                baseline_cost_schedule_hash TEXT NOT NULL,
                baseline_risk_config_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                binding_hash TEXT PRIMARY KEY,
                UNIQUE(strategy_id, qualification_hash)
            );

            CREATE TABLE IF NOT EXISTS paper_evaluation_regimes (
                regime_hash TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                qualification_hash TEXT NOT NULL,
                baseline_replay_report_hash TEXT NOT NULL,
                forward_dataset_version TEXT NOT NULL,
                forward_dataset_sha256 TEXT NOT NULL,
                execution_policy TEXT NOT NULL,
                cost_schedule_hash TEXT NOT NULL,
                risk_config_hash TEXT NOT NULL,
                monitoring_protocol_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- IMMUTABILITY TRIGGERS: NO UPDATE, NO DELETE

            CREATE TRIGGER IF NOT EXISTS monitoring_snapshots_no_delete
            BEFORE DELETE ON monitoring_snapshots
            BEGIN
                SELECT RAISE(ABORT, 'monitoring snapshots are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS monitoring_snapshots_no_update
            BEFORE UPDATE ON monitoring_snapshots
            BEGIN
                SELECT RAISE(ABORT, 'monitoring snapshots are immutable and cannot be updated');
            END;

            CREATE TRIGGER IF NOT EXISTS degradation_events_no_delete
            BEFORE DELETE ON degradation_events
            BEGIN
                SELECT RAISE(ABORT, 'degradation events are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS degradation_events_no_update
            BEFORE UPDATE ON degradation_events
            BEGIN
                SELECT RAISE(ABORT, 'degradation events are immutable and cannot be updated');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_evaluation_transitions_no_delete
            BEFORE DELETE ON paper_evaluation_transitions
            BEGIN
                SELECT RAISE(ABORT, 'paper evaluation transitions are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_evaluation_transitions_no_update
            BEFORE UPDATE ON paper_evaluation_transitions
            BEGIN
                SELECT RAISE(ABORT, 'paper evaluation transitions are immutable and cannot be updated');
            END;

            CREATE TRIGGER IF NOT EXISTS evaluation_baselines_no_delete
            BEFORE DELETE ON evaluation_baselines
            BEGIN
                SELECT RAISE(ABORT, 'evaluation baselines are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS evaluation_baselines_no_update
            BEFORE UPDATE ON evaluation_baselines
            BEGIN
                SELECT RAISE(ABORT, 'evaluation baselines are immutable and cannot be updated');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_evaluation_regimes_no_delete
            BEFORE DELETE ON paper_evaluation_regimes
            BEGIN
                SELECT RAISE(ABORT, 'paper evaluation regimes are permanent and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS paper_evaluation_regimes_no_update
            BEFORE UPDATE ON paper_evaluation_regimes
            BEGIN
                SELECT RAISE(ABORT, 'paper evaluation regimes are immutable and cannot be updated');
            END;
            """
        )

    # -----------------------------------------------------------------------
    # Row Deserializers
    # -----------------------------------------------------------------------

    def _row_to_snapshot(self, row: sqlite3.Row) -> MonitoringSnapshot:
        return MonitoringSnapshot(
            strategy_id=row["strategy_id"],
            qualification_hash=row["qualification_hash"],
            replay_report_hash=row["replay_report_hash"],
            dataset_version=row["dataset_version"],
            dataset_sha256=row["dataset_sha256"],
            split_zone=row["split_zone"],
            monitoring_protocol_version=row["monitoring_protocol_version"],
            monitoring_config_hash=row["monitoring_config_hash"],
            window_start_ts=row["window_start_ts"],
            window_end_ts=row["window_end_ts"],
            total_trades=row["total_trades"],
            net_pnl=row["net_pnl"],
            max_drawdown_bps=row["max_drawdown_bps"],
            realized_sharpe=row["realized_sharpe"],
            realized_slippage_bps=row["realized_slippage_bps"],
            cost_to_turnover_bps=row["cost_to_turnover_bps"],
            risk_event_count=row["risk_event_count"],
            metrics_json=row["metrics_json"],
            created_at=row["created_at"],
            snapshot_hash=row["snapshot_hash"],
        )

    def _row_to_degradation_event(self, row: sqlite3.Row) -> DegradationEvent:
        return DegradationEvent(
            strategy_id=row["strategy_id"],
            qualification_hash=row["qualification_hash"],
            snapshot_hash=row["snapshot_hash"],
            rule_name=row["rule_name"],
            threshold_value=row["threshold_value"],
            observed_value=row["observed_value"],
            monitoring_protocol_version=row["monitoring_protocol_version"],
            monitoring_config_hash=row["monitoring_config_hash"],
            timestamp=row["timestamp"],
            details_json=row["details_json"],
            event_hash=row["event_hash"],
            baseline_replay_report_hash=row["baseline_replay_report_hash"],
        )

    def _row_to_baseline(self, row: sqlite3.Row) -> PaperEvaluationBaseline:
        return PaperEvaluationBaseline(
            strategy_id=row["strategy_id"],
            qualification_hash=row["qualification_hash"],
            baseline_replay_report_hash=row["baseline_replay_report_hash"],
            baseline_dataset_version=row["baseline_dataset_version"],
            baseline_dataset_sha256=row["baseline_dataset_sha256"],
            baseline_split_zone=row["baseline_split_zone"],
            baseline_execution_policy=row["baseline_execution_policy"],
            baseline_cost_schedule_hash=row["baseline_cost_schedule_hash"],
            baseline_risk_config_hash=row["baseline_risk_config_hash"],
            created_at=row["created_at"],
            binding_hash=row["binding_hash"],
        )

    def _row_to_transition(self, row: sqlite3.Row) -> PaperEvaluationTransition:
        return PaperEvaluationTransition(
            strategy_id=row["strategy_id"],
            old_state=row["old_state"],
            new_state=row["new_state"],
            initiator=row["initiator"],
            evidence_type=row["evidence_type"],
            evidence_hash=row["evidence_hash"],
            reason=row["reason"],
            timestamp=row["timestamp"],
            transition_hash=row["transition_hash"],
        )

    def _row_to_regime(self, row: sqlite3.Row) -> PaperEvaluationRegime:
        return PaperEvaluationRegime(
            strategy_id=row["strategy_id"],
            qualification_hash=row["qualification_hash"],
            baseline_replay_report_hash=row["baseline_replay_report_hash"],
            forward_dataset_version=row["forward_dataset_version"],
            forward_dataset_sha256=row["forward_dataset_sha256"],
            execution_policy=row["execution_policy"],
            cost_schedule_hash=row["cost_schedule_hash"],
            risk_config_hash=row["risk_config_hash"],
            monitoring_protocol_version=row["monitoring_protocol_version"],
            regime_hash=row["regime_hash"],
            created_at=row["created_at"],
        )

    # -----------------------------------------------------------------------
    # Append APIs
    # -----------------------------------------------------------------------

    def get_or_insert_snapshot(self, snapshot: MonitoringSnapshot) -> MonitoringSnapshot:
        """Idempotently insert a MonitoringSnapshot or return the existing identical record.

        Raises EvaluationLedgerIntegrityError if:
        - The snapshot fails cryptographic digest verification.
        - A snapshot with identical hash exists but has conflicting semantic content.
        - The deterministic snapshot_id collides with an existing record under a different hash.
        """
        if type(snapshot) is not MonitoringSnapshot:
            raise TypeError(f"Expected MonitoringSnapshot, got {type(snapshot).__name__}")

        if not snapshot.verify_digest():
            raise EvaluationLedgerIntegrityError(
                f"MonitoringSnapshot failed digest verification (stored: {snapshot.snapshot_hash}, "
                f"computed: {snapshot.compute_hash()})"
            )

        # 1. Query by snapshot_hash (semantic identity)
        row = self._connection.execute(
            "SELECT * FROM monitoring_snapshots WHERE snapshot_hash = ?",
            (snapshot.snapshot_hash,),
        ).fetchone()

        if row is not None:
            existing = self._row_to_snapshot(row)
            if existing.canonical_dict() != snapshot.canonical_dict():
                raise EvaluationLedgerIntegrityError(
                    f"Snapshot hash collision with conflicting semantic content for hash '{snapshot.snapshot_hash}'"
                )
            return existing

        # 2. Verify deterministic ID does not collide with a different hash
        id_collision = self._connection.execute(
            "SELECT snapshot_hash FROM monitoring_snapshots WHERE snapshot_id = ?",
            (snapshot.derived_snapshot_id,),
        ).fetchone()
        if id_collision is not None:
            raise EvaluationLedgerIntegrityError(
                f"Snapshot ID collision for '{snapshot.derived_snapshot_id}' with different hash "
                f"'{id_collision['snapshot_hash']}'"
            )

        # 3. Insert new record
        try:
            self._connection.execute(
                """
                INSERT INTO monitoring_snapshots (
                    snapshot_id, strategy_id, qualification_hash, replay_report_hash,
                    dataset_version, dataset_sha256, split_zone, monitoring_protocol_version,
                    monitoring_config_hash, window_start_ts, window_end_ts, total_trades,
                    net_pnl, max_drawdown_bps, realized_sharpe, realized_slippage_bps,
                    cost_to_turnover_bps, risk_event_count, metrics_json, created_at, snapshot_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.derived_snapshot_id,
                    snapshot.strategy_id,
                    snapshot.qualification_hash,
                    snapshot.replay_report_hash,
                    snapshot.dataset_version,
                    snapshot.dataset_sha256,
                    snapshot.split_zone,
                    snapshot.monitoring_protocol_version,
                    snapshot.monitoring_config_hash,
                    snapshot.window_start_ts,
                    snapshot.window_end_ts,
                    snapshot.total_trades,
                    snapshot.net_pnl,
                    snapshot.max_drawdown_bps,
                    snapshot.realized_sharpe,
                    snapshot.realized_slippage_bps,
                    snapshot.cost_to_turnover_bps,
                    snapshot.risk_event_count,
                    snapshot.metrics_json,
                    snapshot.created_at,
                    snapshot.snapshot_hash,
                ),
            )
        except sqlite3.IntegrityError as e:
            raise EvaluationLedgerIntegrityError(f"Failed to insert monitoring snapshot: {e}") from e

        return snapshot

    def record_degradation_event(self, event: DegradationEvent) -> DegradationEvent:
        """Idempotently insert a DegradationEvent or return the existing identical record.

        Enforces that event.snapshot_hash references an existing monitoring snapshot.
        """
        if type(event) is not DegradationEvent:
            raise TypeError(f"Expected DegradationEvent, got {type(event).__name__}")

        if not event.verify_digest():
            raise EvaluationLedgerIntegrityError(
                f"DegradationEvent failed digest verification (stored: {event.event_hash}, "
                f"computed: {event.compute_hash()})"
            )

        # 1. Query by event_hash
        row = self._connection.execute(
            "SELECT * FROM degradation_events WHERE event_hash = ?",
            (event.event_hash,),
        ).fetchone()

        if row is not None:
            existing = self._row_to_degradation_event(row)
            if existing.canonical_dict() != event.canonical_dict():
                raise EvaluationLedgerIntegrityError(
                    f"Degradation event hash collision with conflicting semantic content for hash '{event.event_hash}'"
                )
            return existing

        # 2. Check foreign key reference explicitly before insert
        snap_ref = self._connection.execute(
            "SELECT 1 FROM monitoring_snapshots WHERE snapshot_hash = ?",
            (event.snapshot_hash,),
        ).fetchone()
        if snap_ref is None:
            raise EvaluationLedgerIntegrityError(
                f"DegradationEvent references non-existent snapshot_hash '{event.snapshot_hash}'"
            )

        # 3. Check ID collision
        id_collision = self._connection.execute(
            "SELECT event_hash FROM degradation_events WHERE event_id = ?",
            (event.derived_event_id,),
        ).fetchone()
        if id_collision is not None:
            raise EvaluationLedgerIntegrityError(
                f"Event ID collision for '{event.derived_event_id}' with different hash "
                f"'{id_collision['event_hash']}'"
            )

        # 4. Insert
        try:
            self._connection.execute(
                """
                INSERT INTO degradation_events (
                    event_id, strategy_id, qualification_hash, snapshot_hash,
                    baseline_replay_report_hash, rule_name, threshold_value,
                    observed_value, monitoring_protocol_version,
                    monitoring_config_hash, timestamp, details_json, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.derived_event_id,
                    event.strategy_id,
                    event.qualification_hash,
                    event.snapshot_hash,
                    event.baseline_replay_report_hash,
                    event.rule_name,
                    event.threshold_value,
                    event.observed_value,
                    event.monitoring_protocol_version,
                    event.monitoring_config_hash,
                    event.timestamp,
                    event.details_json,
                    event.event_hash,
                ),
            )
        except sqlite3.IntegrityError as e:
            raise EvaluationLedgerIntegrityError(f"Failed to insert degradation event: {e}") from e

        return event

    def record_transition(self, transition: PaperEvaluationTransition) -> PaperEvaluationTransition:
        """Idempotently insert a PaperEvaluationTransition or return the existing identical record."""
        if type(transition) is not PaperEvaluationTransition:
            raise TypeError(f"Expected PaperEvaluationTransition, got {type(transition).__name__}")

        if not transition.verify_digest():
            raise EvaluationLedgerIntegrityError(
                f"PaperEvaluationTransition failed digest verification (stored: {transition.transition_hash}, "
                f"computed: {transition.compute_hash()})"
            )

        # 1. Query by transition_hash
        row = self._connection.execute(
            "SELECT * FROM paper_evaluation_transitions WHERE transition_hash = ?",
            (transition.transition_hash,),
        ).fetchone()

        if row is not None:
            existing = self._row_to_transition(row)
            if existing.canonical_dict() != transition.canonical_dict():
                raise EvaluationLedgerIntegrityError(
                    f"Transition hash collision with conflicting semantic content for hash '{transition.transition_hash}'"
                )
            return existing

        # 2. Check ID collision
        id_collision = self._connection.execute(
            "SELECT transition_hash FROM paper_evaluation_transitions WHERE transition_id = ?",
            (transition.derived_transition_id,),
        ).fetchone()
        if id_collision is not None:
            raise EvaluationLedgerIntegrityError(
                f"Transition ID collision for '{transition.derived_transition_id}' with different hash "
                f"'{id_collision['transition_hash']}'"
            )

        # 3. Insert
        try:
            self._connection.execute(
                """
                INSERT INTO paper_evaluation_transitions (
                    transition_id, strategy_id, old_state, new_state, initiator,
                    evidence_type, evidence_hash, reason, timestamp, transition_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition.derived_transition_id,
                    transition.strategy_id,
                    transition.old_state,
                    transition.new_state,
                    transition.initiator,
                    transition.evidence_type,
                    transition.evidence_hash,
                    transition.reason,
                    transition.timestamp,
                    transition.transition_hash,
                ),
            )
        except sqlite3.IntegrityError as e:
            raise EvaluationLedgerIntegrityError(f"Failed to insert evaluation transition: {e}") from e

        return transition

    def register_baseline(
        self,
        baseline: PaperEvaluationBaseline,
        replay_report: ReplayReport | None = None,
        paper_ledger: Any | None = None,
    ) -> PaperEvaluationBaseline:
        """Register an authoritative PaperEvaluationBaseline binding into the ledger.

        Enforces:
        1. Exact type check
        2. Digest integrity verification
        3. Independent authenticity verification of the bound ReplayReport
        4. Exactly ONE active immutable baseline per strategy_id + qualification_hash
        5. Idempotency: re-registering an identical baseline returns the existing record
        6. Conflicting second baseline raises EvaluationLedgerIntegrityError
        """
        if type(baseline) is not PaperEvaluationBaseline:
            raise TypeError(f"Expected PaperEvaluationBaseline, got {type(baseline).__name__}")

        if not baseline.verify_digest():
            raise EvaluationLedgerIntegrityError(
                f"PaperEvaluationBaseline failed digest verification (stored: {baseline.binding_hash}, "
                f"computed: {baseline.compute_hash()})"
            )

        if not baseline.strategy_id:
            raise EvaluationLedgerIntegrityError("baseline strategy_id cannot be empty")
        if not baseline.qualification_hash:
            raise EvaluationLedgerIntegrityError("baseline qualification_hash cannot be empty")
        if not baseline.baseline_replay_report_hash:
            raise EvaluationLedgerIntegrityError("baseline baseline_replay_report_hash cannot be empty")

        # 1. Resolve and independently verify authoritative bound ReplayReport
        rep = replay_report
        if rep is None and paper_ledger is not None:
            if hasattr(paper_ledger, "get_report"):
                rep = paper_ledger.get_report(baseline.baseline_replay_report_hash)
            elif hasattr(paper_ledger, "_connection"):
                row = paper_ledger._connection.execute(
                    "SELECT report_json, created_at FROM paper_reports WHERE report_hash = ?",
                    (baseline.baseline_replay_report_hash,),
                ).fetchone()
                if row is not None:
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
                    rep = ReplayReport(
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
                        report_hash=baseline.baseline_replay_report_hash,
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

        if rep is None:
            raise EvaluationLedgerIntegrityError(
                "Authoritative ReplayReport is required to register baseline binding (cannot register unverified metadata-only baseline)"
            )

        if type(rep) is not ReplayReport:
            raise TypeError(f"Expected ReplayReport, got {type(rep).__name__}")

        if rep.report_hash != rep.compute_report_hash():
            raise EvaluationLedgerIntegrityError(
                "Bound ReplayReport failed digest verification (tampered replay report)"
            )

        if rep.report_hash != baseline.baseline_replay_report_hash:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport hash mismatch: expected {baseline.baseline_replay_report_hash}, got {rep.report_hash}"
            )

        if rep.strategy_id != baseline.strategy_id:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport strategy_id mismatch: expected {baseline.strategy_id}, got {rep.strategy_id}"
            )

        if rep.qualification_hash != baseline.qualification_hash:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport qualification_hash mismatch: expected {baseline.qualification_hash}, got {rep.qualification_hash}"
            )

        if rep.dataset_version != baseline.baseline_dataset_version:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport dataset_version mismatch: expected {baseline.baseline_dataset_version}, got {rep.dataset_version}"
            )

        if rep.dataset_sha256 != baseline.baseline_dataset_sha256:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport dataset_sha256 mismatch: expected {baseline.baseline_dataset_sha256}, got {rep.dataset_sha256}"
            )

        if rep.split_zone != baseline.baseline_split_zone:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport split_zone mismatch: expected {baseline.baseline_split_zone}, got {rep.split_zone}"
            )

        if rep.execution_policy != baseline.baseline_execution_policy:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport execution_policy mismatch: expected {baseline.baseline_execution_policy}, got {rep.execution_policy}"
            )

        if rep.cost_schedule_hash != baseline.baseline_cost_schedule_hash:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport cost_schedule_hash mismatch: expected {baseline.baseline_cost_schedule_hash}, got {rep.cost_schedule_hash}"
            )

        if rep.risk_config_hash != baseline.baseline_risk_config_hash:
            raise EvaluationLedgerIntegrityError(
                f"ReplayReport risk_config_hash mismatch: expected {baseline.baseline_risk_config_hash}, got {rep.risk_config_hash}"
            )

        # 2. Check existing baseline for (strategy_id, qualification_hash)
        row = self._connection.execute(
            "SELECT * FROM evaluation_baselines WHERE strategy_id = ? AND qualification_hash = ?",
            (baseline.strategy_id, baseline.qualification_hash),
        ).fetchone()

        if row is not None:
            existing = self._row_to_baseline(row)
            if existing.binding_hash == baseline.binding_hash and existing.canonical_dict() == baseline.canonical_dict():
                return existing  # Idempotent re-registration returns existing record
            raise EvaluationLedgerIntegrityError(
                f"Strategy '{baseline.strategy_id}' with qualification '{baseline.qualification_hash}' "
                f"already has an authoritative baseline registered ({existing.binding_hash}). "
                f"Cannot register conflicting baseline ({baseline.binding_hash})."
            )

        # 3. Check binding_hash collision
        hash_row = self._connection.execute(
            "SELECT * FROM evaluation_baselines WHERE binding_hash = ?",
            (baseline.binding_hash,),
        ).fetchone()
        if hash_row is not None:
            raise EvaluationLedgerIntegrityError(
                f"Baseline binding hash collision for '{baseline.binding_hash}'"
            )

        # 4. Insert
        try:
            self._connection.execute(
                """
                INSERT INTO evaluation_baselines (
                    strategy_id, qualification_hash, baseline_replay_report_hash,
                    baseline_dataset_version, baseline_dataset_sha256, baseline_split_zone,
                    baseline_execution_policy, baseline_cost_schedule_hash, baseline_risk_config_hash,
                    created_at, binding_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    baseline.strategy_id,
                    baseline.qualification_hash,
                    baseline.baseline_replay_report_hash,
                    baseline.baseline_dataset_version,
                    baseline.baseline_dataset_sha256,
                    baseline.baseline_split_zone,
                    baseline.baseline_execution_policy,
                    baseline.baseline_cost_schedule_hash,
                    baseline.baseline_risk_config_hash,
                    baseline.created_at,
                    baseline.binding_hash,
                ),
            )
        except sqlite3.IntegrityError as e:
            raise EvaluationLedgerIntegrityError(f"Failed to insert evaluation baseline: {e}") from e

        return baseline

    # -----------------------------------------------------------------------
    # Read APIs (Strictly Deterministic with Explicit ORDER BY)
    # -----------------------------------------------------------------------

    def get_snapshot(self, snapshot_hash: str) -> MonitoringSnapshot | None:
        """Retrieve a MonitoringSnapshot by its semantic SHA-256 digest."""
        row = self._connection.execute(
            "SELECT * FROM monitoring_snapshots WHERE snapshot_hash = ?",
            (snapshot_hash,),
        ).fetchone()
        return self._row_to_snapshot(row) if row is not None else None

    def list_snapshots(
        self,
        strategy_id: str,
        window_start_ts: str | None = None,
        window_end_ts: str | None = None,
    ) -> list[MonitoringSnapshot]:
        """List snapshots for a strategy with optional ISO 8601 UTC window bounds.

        Results are strictly ordered by window_start_ts ASC, window_end_ts ASC.
        """
        query = "SELECT * FROM monitoring_snapshots WHERE strategy_id = ?"
        params: list[Any] = [strategy_id]

        if window_start_ts is not None:
            query += " AND window_start_ts >= ?"
            params.append(window_start_ts)

        if window_end_ts is not None:
            query += " AND window_end_ts <= ?"
            params.append(window_end_ts)

        query += " ORDER BY window_start_ts ASC, window_end_ts ASC"

        rows = self._connection.execute(query, params).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    def get_degradation_event(self, event_hash: str) -> DegradationEvent | None:
        """Retrieve a DegradationEvent by its semantic SHA-256 digest."""
        row = self._connection.execute(
            "SELECT * FROM degradation_events WHERE event_hash = ?",
            (event_hash,),
        ).fetchone()
        return self._row_to_degradation_event(row) if row is not None else None

    def list_degradation_events(
        self,
        strategy_id: str,
        timestamp_from: str | None = None,
        timestamp_to: str | None = None,
    ) -> list[DegradationEvent]:
        """List degradation events for a strategy with optional timestamp bounds.

        Results are strictly ordered by timestamp ASC.
        """
        query = "SELECT * FROM degradation_events WHERE strategy_id = ?"
        params: list[Any] = [strategy_id]

        if timestamp_from is not None:
            query += " AND timestamp >= ?"
            params.append(timestamp_from)

        if timestamp_to is not None:
            query += " AND timestamp <= ?"
            params.append(timestamp_to)

        query += " ORDER BY timestamp ASC"

        rows = self._connection.execute(query, params).fetchall()
        return [self._row_to_degradation_event(r) for r in rows]

    def get_transition(self, transition_hash: str) -> PaperEvaluationTransition | None:
        """Retrieve a PaperEvaluationTransition by its semantic SHA-256 digest."""
        row = self._connection.execute(
            "SELECT * FROM paper_evaluation_transitions WHERE transition_hash = ?",
            (transition_hash,),
        ).fetchone()
        return self._row_to_transition(row) if row is not None else None

    def list_transitions(
        self,
        strategy_id: str,
        timestamp_from: str | None = None,
        timestamp_to: str | None = None,
    ) -> list[PaperEvaluationTransition]:
        """List evaluation transitions for a strategy with optional timestamp bounds.

        Results are strictly ordered by timestamp ASC.
        """
        query = "SELECT * FROM paper_evaluation_transitions WHERE strategy_id = ?"
        params: list[Any] = [strategy_id]

        if timestamp_from is not None:
            query += " AND timestamp >= ?"
            params.append(timestamp_from)

        if timestamp_to is not None:
            query += " AND timestamp <= ?"
            params.append(timestamp_to)

        query += " ORDER BY timestamp ASC"

        rows = self._connection.execute(query, params).fetchall()
        return [self._row_to_transition(r) for r in rows]

    def get_baseline(self, strategy_id: str, qualification_hash: str) -> PaperEvaluationBaseline | None:
        """Return the single authoritative PaperEvaluationBaseline for a strategy + qualification, or None."""
        row = self._connection.execute(
            "SELECT * FROM evaluation_baselines WHERE strategy_id = ? AND qualification_hash = ?",
            (strategy_id, qualification_hash),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_baseline(row)

    def get_baseline_by_hash(self, binding_hash: str) -> PaperEvaluationBaseline | None:
        """Return a PaperEvaluationBaseline by its binding_hash, or None."""
        row = self._connection.execute(
            "SELECT * FROM evaluation_baselines WHERE binding_hash = ?",
            (binding_hash,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_baseline(row)

    def register_regime(
        self,
        regime: PaperEvaluationRegime,
    ) -> PaperEvaluationRegime:
        """Register an authoritative PaperEvaluationRegime into the ledger.

        Enforces:
        1. Exact type check
        2. Digest integrity verification
        3. Idempotency: re-registering an identical regime returns the existing record
        4. Conflicting regime with same regime_hash raises EvaluationLedgerIntegrityError
        """
        if type(regime) is not PaperEvaluationRegime:
            raise TypeError(f"Expected PaperEvaluationRegime, got {type(regime).__name__}")

        if not regime.verify_digest():
            raise EvaluationLedgerIntegrityError(
                f"PaperEvaluationRegime failed digest verification (stored: {regime.regime_hash}, "
                f"computed: {regime.compute_regime_hash()})"
            )

        if not regime.strategy_id:
            raise EvaluationLedgerIntegrityError("regime strategy_id cannot be empty")
        if not regime.qualification_hash:
            raise EvaluationLedgerIntegrityError("regime qualification_hash cannot be empty")
        if not regime.regime_hash:
            raise EvaluationLedgerIntegrityError("regime regime_hash cannot be empty")

        # 1. Query by regime_hash
        row = self._connection.execute(
            "SELECT * FROM paper_evaluation_regimes WHERE regime_hash = ?",
            (regime.regime_hash,),
        ).fetchone()

        if row is not None:
            existing = self._row_to_regime(row)
            if existing.canonical_dict() != regime.canonical_dict():
                raise EvaluationLedgerIntegrityError(
                    f"Regime hash collision with conflicting semantic content for hash '{regime.regime_hash}'"
                )
            return existing

        created_at = regime.created_at or datetime.now(timezone.utc).isoformat()

        # 2. Insert
        try:
            self._connection.execute(
                """
                INSERT INTO paper_evaluation_regimes (
                    regime_hash, strategy_id, qualification_hash,
                    baseline_replay_report_hash, forward_dataset_version,
                    forward_dataset_sha256, execution_policy, cost_schedule_hash,
                    risk_config_hash, monitoring_protocol_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    regime.regime_hash,
                    regime.strategy_id,
                    regime.qualification_hash,
                    regime.baseline_replay_report_hash,
                    regime.forward_dataset_version,
                    regime.forward_dataset_sha256,
                    regime.execution_policy,
                    regime.cost_schedule_hash,
                    regime.risk_config_hash,
                    regime.monitoring_protocol_version,
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as e:
            raise EvaluationLedgerIntegrityError(f"Failed to insert evaluation regime: {e}") from e

        if not regime.created_at:
            return PaperEvaluationRegime(
                strategy_id=regime.strategy_id,
                qualification_hash=regime.qualification_hash,
                baseline_replay_report_hash=regime.baseline_replay_report_hash,
                forward_dataset_version=regime.forward_dataset_version,
                forward_dataset_sha256=regime.forward_dataset_sha256,
                execution_policy=regime.execution_policy,
                cost_schedule_hash=regime.cost_schedule_hash,
                risk_config_hash=regime.risk_config_hash,
                monitoring_protocol_version=regime.monitoring_protocol_version,
                regime_hash=regime.regime_hash,
                created_at=created_at,
            )
        return regime

    def get_regime(self, regime_hash: str) -> PaperEvaluationRegime | None:
        """Retrieve a PaperEvaluationRegime by its semantic SHA-256 digest."""
        row = self._connection.execute(
            "SELECT * FROM paper_evaluation_regimes WHERE regime_hash = ?",
            (regime_hash,),
        ).fetchone()
        return self._row_to_regime(row) if row is not None else None

    def list_regimes(self, strategy_id: str) -> list[PaperEvaluationRegime]:
        """List evaluation regimes for a strategy, ordered by created_at ASC."""
        rows = self._connection.execute(
            "SELECT * FROM paper_evaluation_regimes WHERE strategy_id = ? ORDER BY created_at ASC",
            (strategy_id,),
        ).fetchall()
        return [self._row_to_regime(r) for r in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection if owned."""
        if self._owned and self._connection is not None:
            self._connection.close()
