from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping


_ALLOWED_TERMINAL = {"COMPLETED", "FAILED", "REJECTED_NONCAUSAL", "ABANDONED"}
_ALLOWED_STATUS = {"RUNNING", *_ALLOWED_TERMINAL}
_ALLOWED_MODES = {"PRODUCTION", "FIXTURE"}
_ALLOWED_DATASET_KINDS = {"SYNTHETIC", "LICENSED"}


@dataclass(frozen=True)
class ResearchBudget:
    """Cumulative limits for one dataset_version + research_protocol_version population."""

    max_trials: int = 10_000
    max_experiments: int = 100
    max_strategy_variants: int = 1_000
    max_runtime_minutes: float = 120.0
    max_llm_cost: float = 5.0


@dataclass(frozen=True)
class TrialContext:
    """Immutable metadata created by ResearchHarness; callers must not choose IDs."""

    trial_id: str
    experiment_id: str
    strategy_id: str
    dataset_version: str
    research_protocol_version: str
    feature_version: str
    parameter_set: Mapping[str, Any]
    seed: int
    execution_model: str
    cost_model: str
    slippage_model: str
    estimated_runtime_minutes: float
    estimated_llm_cost: float = 0.0
    strategy_spec_json: str = "{}"
    mode: str = "PRODUCTION"
    dataset_kind: str = "LICENSED"


class TrialBudgetExceeded(RuntimeError):
    """Raised when a cumulative research population would exceed its budget."""


class TrialLedger:
    """Authoritative append-only trial ledger.

    Reservation and completion are intended to be driven by ResearchHarness.
    SQLite BEGIN IMMEDIATE makes budget check + reservation atomic across processes.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=30.0,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                research_protocol_version TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                parameter_set_json TEXT NOT NULL,
                strategy_spec_json TEXT NOT NULL,
                seed INTEGER NOT NULL,
                execution_model TEXT NOT NULL,
                cost_model TEXT NOT NULL,
                slippage_model TEXT NOT NULL,
                timestamp_started TEXT NOT NULL,
                timestamp_completed TEXT,
                estimated_runtime_minutes REAL NOT NULL,
                actual_runtime_minutes REAL,
                estimated_llm_cost REAL NOT NULL DEFAULT 0,
                actual_llm_cost REAL,
                result_json TEXT,
                mode TEXT NOT NULL CHECK (mode IN ('PRODUCTION','FIXTURE')),
                dataset_kind TEXT NOT NULL CHECK (dataset_kind IN ('SYNTHETIC','LICENSED')),
                status TEXT NOT NULL CHECK (
                    status IN ('RUNNING','COMPLETED','FAILED','REJECTED_NONCAUSAL','ABANDONED')
                )
            );

            CREATE INDEX IF NOT EXISTS idx_trials_scope
            ON trials(dataset_version, research_protocol_version);

            CREATE TRIGGER IF NOT EXISTS trials_no_delete
            BEFORE DELETE ON trials
            BEGIN
                SELECT RAISE(ABORT, 'trials are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS trials_running_metadata_immutable
            BEFORE UPDATE OF
                trial_id, experiment_id, strategy_id, dataset_version, research_protocol_version,
                feature_version, parameter_set_json, strategy_spec_json, seed, execution_model,
                cost_model, slippage_model, timestamp_started, estimated_runtime_minutes,
                estimated_llm_cost, mode, dataset_kind
            ON trials
            WHEN OLD.status = 'RUNNING'
            BEGIN
                SELECT RAISE(ABORT, 'running trial metadata is immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS trials_frozen
            BEFORE UPDATE ON trials
            WHEN OLD.status <> 'RUNNING'
            BEGIN
                SELECT RAISE(ABORT, 'completed trials are immutable');
            END;
            """
        )
        # Existing pre-v3.2 databases may not have the status CHECK. A fresh MVP ledger
        # is the supported deployment; fail loudly rather than silently migrating mutable history.

    @staticmethod
    def _scope(ctx: TrialContext) -> tuple[str, str]:
        return ctx.dataset_version, ctx.research_protocol_version

    def _usage_unlocked(self, dataset_version: str, research_protocol_version: str, mode: str = "PRODUCTION") -> dict[str, float | int]:
        if mode not in _ALLOWED_MODES:
            raise ValueError(f"invalid trial mode: {mode}")
        row = self._connection.execute(
            """
            SELECT
                COUNT(*) AS trials,
                COUNT(DISTINCT experiment_id) AS experiments,
                COUNT(DISTINCT strategy_id) AS strategy_variants,
                COALESCE(SUM(COALESCE(actual_runtime_minutes, estimated_runtime_minutes)), 0.0) AS runtime_minutes,
                COALESCE(SUM(COALESCE(actual_llm_cost, estimated_llm_cost)), 0.0) AS llm_cost
            FROM trials
            WHERE dataset_version = ? AND research_protocol_version = ? AND mode = ?
            """,
            (dataset_version, research_protocol_version, mode),
        ).fetchone()
        return {
            "trials": int(row["trials"]),
            "experiments": int(row["experiments"]),
            "strategy_variants": int(row["strategy_variants"]),
            "runtime_minutes": float(row["runtime_minutes"]),
            "llm_cost": float(row["llm_cost"]),
        }

    def usage(self, dataset_version: str, research_protocol_version: str, *, mode: str = "PRODUCTION") -> dict[str, float | int]:
        return self._usage_unlocked(dataset_version, research_protocol_version, mode)

    def reserve(self, context: TrialContext, budget: ResearchBudget) -> None:
        if context.estimated_runtime_minutes <= 0:
            raise ValueError("estimated_runtime_minutes must be > 0")
        if context.estimated_llm_cost < 0:
            raise ValueError("estimated_llm_cost must be >= 0")
        if context.mode not in _ALLOWED_MODES:
            raise ValueError(f"invalid trial mode: {context.mode}")
        if context.dataset_kind not in _ALLOWED_DATASET_KINDS:
            raise ValueError(f"invalid dataset kind: {context.dataset_kind}")

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            usage = self._usage_unlocked(*self._scope(context), context.mode)
            is_new_experiment = self._count_distinct_unlocked(
                "experiment_id", context.dataset_version, context.research_protocol_version, context.experiment_id, context.mode
            ) == 0
            is_new_strategy = self._count_distinct_unlocked(
                "strategy_id", context.dataset_version, context.research_protocol_version, context.strategy_id, context.mode
            ) == 0

            projected = {
                "trials": int(usage["trials"]) + 1,
                "experiments": int(usage["experiments"]) + int(is_new_experiment),
                "strategy_variants": int(usage["strategy_variants"]) + int(is_new_strategy),
                "runtime_minutes": float(usage["runtime_minutes"]) + context.estimated_runtime_minutes,
                "llm_cost": float(usage["llm_cost"]) + context.estimated_llm_cost,
            }
            violations: list[str] = []
            if projected["trials"] > budget.max_trials:
                violations.append(f"max_trials={budget.max_trials}")
            if projected["experiments"] > budget.max_experiments:
                violations.append(f"max_experiments={budget.max_experiments}")
            if projected["strategy_variants"] > budget.max_strategy_variants:
                violations.append(f"max_strategy_variants={budget.max_strategy_variants}")
            if projected["runtime_minutes"] > budget.max_runtime_minutes:
                violations.append(f"max_runtime_minutes={budget.max_runtime_minutes}")
            if projected["llm_cost"] > budget.max_llm_cost:
                violations.append(f"max_llm_cost={budget.max_llm_cost}")
            if violations:
                raise TrialBudgetExceeded(
                    "cumulative research budget exceeded for "
                    f"dataset={context.dataset_version}, protocol={context.research_protocol_version}: "
                    + ", ".join(violations)
                )

            now = datetime.now(timezone.utc).isoformat()
            self._connection.execute(
                """
                INSERT INTO trials (
                    trial_id, experiment_id, strategy_id, dataset_version,
                    research_protocol_version, feature_version, parameter_set_json, strategy_spec_json,
                    seed, execution_model, cost_model, slippage_model,
                    timestamp_started, estimated_runtime_minutes,
                    estimated_llm_cost, mode, dataset_kind, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'RUNNING')
                """,
                (
                    context.trial_id,
                    context.experiment_id,
                    context.strategy_id,
                    context.dataset_version,
                    context.research_protocol_version,
                    context.feature_version,
                    json.dumps(context.parameter_set, sort_keys=True, separators=(",", ":")),
                    context.strategy_spec_json,
                    context.seed,
                    context.execution_model,
                    context.cost_model,
                    context.slippage_model,
                    now,
                    context.estimated_runtime_minutes,
                    context.estimated_llm_cost,
                    context.mode,
                    context.dataset_kind,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def complete(
        self,
        trial_id: str,
        *,
        result: Mapping[str, Any],
        actual_runtime_minutes: float,
        actual_llm_cost: float = 0.0,
        status: str = "COMPLETED",
    ) -> None:
        if status not in _ALLOWED_TERMINAL - {"ABANDONED"}:
            raise ValueError(f"invalid completion status: {status}")
        if actual_runtime_minutes < 0 or actual_llm_cost < 0:
            raise ValueError("actual runtime/cost must be >= 0")
        updated = self._connection.execute(
            """
            UPDATE trials
            SET timestamp_completed = ?, actual_runtime_minutes = ?,
                actual_llm_cost = ?, result_json = ?, status = ?
            WHERE trial_id = ? AND status = 'RUNNING'
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                actual_runtime_minutes,
                actual_llm_cost,
                json.dumps(result, default=str, sort_keys=True, separators=(",", ":")),
                status,
                trial_id,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"trial is not RUNNING or does not exist: {trial_id}")

    def reap_stale(self, *, max_age_minutes: float) -> int:
        if max_age_minutes <= 0:
            raise ValueError("max_age_minutes must be > 0")
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_minutes * 60.0
        rows = self._connection.execute(
            "SELECT trial_id, timestamp_started, estimated_runtime_minutes, estimated_llm_cost FROM trials WHERE status='RUNNING'"
        ).fetchall()
        reaped = 0
        for row in rows:
            started = datetime.fromisoformat(row["timestamp_started"]).timestamp()
            if started <= cutoff:
                self._connection.execute(
                    """
                    UPDATE trials
                    SET timestamp_completed = ?,
                        actual_runtime_minutes = estimated_runtime_minutes,
                        actual_llm_cost = estimated_llm_cost,
                        result_json = ?,
                        status = 'ABANDONED'
                    WHERE trial_id = ? AND status = 'RUNNING'
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps({"reason": "stale_running_trial_reaped"}, sort_keys=True),
                        row["trial_id"],
                    ),
                )
                reaped += 1
        return reaped

    def get(self, trial_id: str) -> sqlite3.Row:
        row = self._connection.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown trial_id: {trial_id}")
        return row

    def _count_distinct_unlocked(self, column: str, dataset_version: str, protocol: str, value: str, mode: str = "PRODUCTION") -> int:
        if column not in {"experiment_id", "strategy_id"}:
            raise ValueError("unsupported distinct-count column")
        row = self._connection.execute(
            f"""
            SELECT COUNT(*) AS n FROM (
                SELECT {column}
                FROM trials
                WHERE dataset_version = ? AND research_protocol_version = ? AND mode = ? AND {column} = ?
                GROUP BY {column}
            )
            """,
            (dataset_version, protocol, mode, value),
        ).fetchone()
        return int(row["n"])
