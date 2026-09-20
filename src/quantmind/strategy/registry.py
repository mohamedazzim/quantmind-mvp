"""Strategy Registry & Lifecycle State Management (PRD v3.8).

This module manages the full strategy lifecycle from IDEA to RETIRED,
enforcing that candidates cannot transition to PAPER_ELIGIBLE without an
authoritative, cryptographically verified StrategyQualificationRecord.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from .compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.research_integrity.holdout import HoldoutState
from quantmind.research_integrity.qualification import (
    StrategyQualificationRecord,
    ValidationStatus,
)
from quantmind.strategy.spec import StrategySpec




class StrategyRegistryError(RuntimeError):
    """Raised when an illegal strategy lifecycle operation or state transition occurs."""


class StrategyLifecycleState(str, Enum):
    """Authoritative lifecycle states for strategies."""

    IDEA = "IDEA"
    RESEARCH = "RESEARCH"
    VALIDATION = "VALIDATION"
    REJECTED = "REJECTED"
    PAPER_ELIGIBLE = "PAPER_ELIGIBLE"
    PAPER_ACTIVE = "PAPER_ACTIVE"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"


ALLOWED_STRATEGY_TRANSITIONS: Mapping[StrategyLifecycleState, frozenset[StrategyLifecycleState]] = {
    StrategyLifecycleState.IDEA: frozenset(
        {StrategyLifecycleState.RESEARCH, StrategyLifecycleState.REJECTED}
    ),
    StrategyLifecycleState.RESEARCH: frozenset(
        {StrategyLifecycleState.VALIDATION, StrategyLifecycleState.REJECTED}
    ),
    StrategyLifecycleState.VALIDATION: frozenset(
        {StrategyLifecycleState.PAPER_ELIGIBLE, StrategyLifecycleState.REJECTED}
    ),
    StrategyLifecycleState.PAPER_ELIGIBLE: frozenset(
        {
            StrategyLifecycleState.PAPER_ACTIVE,
            StrategyLifecycleState.DEGRADED,
            StrategyLifecycleState.RETIRED,
            StrategyLifecycleState.REJECTED,
        }
    ),
    StrategyLifecycleState.PAPER_ACTIVE: frozenset(
        {StrategyLifecycleState.DEGRADED, StrategyLifecycleState.RETIRED}
    ),
    StrategyLifecycleState.DEGRADED: frozenset(
        {StrategyLifecycleState.RETIRED, StrategyLifecycleState.RESEARCH}
    ),
    StrategyLifecycleState.REJECTED: frozenset(),
    StrategyLifecycleState.RETIRED: frozenset(),
}


@dataclass(frozen=True)
class StrategyRegistryRecord:
    """Immutable representation of a registered strategy and its lifecycle status."""

    strategy_id: str
    strategy_spec: StrategySpec
    state: StrategyLifecycleState
    qualification_id: str | None
    qualification_hash: str | None
    registered_at: str
    updated_at: str
    history: tuple[tuple[str, str, str], ...]  # (timestamp, state, reason)


class StrategyRegistry:
    """Authoritative registry managing strategy identity and lifecycle state transitions."""

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
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                state TEXT NOT NULL,
                qualification_id TEXT,
                qualification_hash TEXT,
                registered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                history_json TEXT NOT NULL
            );
            """
        )

    def register_strategy(
        self,
        strategy_spec: StrategySpec,
        initial_state: StrategyLifecycleState = StrategyLifecycleState.IDEA,
    ) -> str:
        """Register a new strategy spec under its derived strategy_id."""
        norm_spec = normalize_strategy_spec(strategy_spec)
        strategy_id = derive_strategy_id(norm_spec)
        now = datetime.now(timezone.utc).isoformat()

        row = self._connection.execute(
            "SELECT spec_json, state FROM strategies WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()

        if row is not None:
            existing_spec = json.loads(row["spec_json"])
            if existing_spec == norm_spec.canonical_dict():
                return strategy_id  # Idempotent re-registration
            raise StrategyRegistryError(
                f"Strategy ID conflict: strategy '{strategy_id}' is already registered with different spec"
            )

        history = [(now, initial_state.value, "initial_registration")]
        self._connection.execute(
            """
            INSERT INTO strategies (
                strategy_id, spec_json, state, qualification_id, qualification_hash,
                registered_at, updated_at, history_json
            ) VALUES (?, ?, ?, NULL, NULL, ?, ?, ?)
            """,
            (
                strategy_id,
                norm_spec.canonical_json(),
                initial_state.value,
                now,
                now,
                json.dumps(history),
            ),
        )
        return strategy_id

    def get_strategy(self, strategy_id: str) -> StrategyRegistryRecord:
        """Retrieve strategy registry record by ID."""
        row = self._connection.execute(
            "SELECT * FROM strategies WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Strategy '{strategy_id}' not found in registry")

        spec_data = json.loads(row["spec_json"])
        strategy_spec = StrategySpec(
            strategy_version=spec_data["strategy_version"],
            feature_version=spec_data["feature_version"],
            signal_name=spec_data["signal_name"],
            parameters=spec_data["parameters"],
        )
        history_raw = json.loads(row["history_json"])
        history_tuples = tuple((h[0], h[1], h[2]) for h in history_raw)

        return StrategyRegistryRecord(
            strategy_id=row["strategy_id"],
            strategy_spec=strategy_spec,
            state=StrategyLifecycleState(row["state"]),
            qualification_id=row["qualification_id"],
            qualification_hash=row["qualification_hash"],
            registered_at=row["registered_at"],
            updated_at=row["updated_at"],
            history=history_tuples,
        )

    def transition_state(
        self,
        strategy_id: str,
        target_state: StrategyLifecycleState,
        *,
        reason: str = "",
        qualification_record: StrategyQualificationRecord | None = None,
    ) -> StrategyRegistryRecord:
        """Transition a strategy to a new lifecycle state with strict state machine validation."""
        record = self.get_strategy(strategy_id)

        # 1. State machine transition validity check
        allowed = ALLOWED_STRATEGY_TRANSITIONS.get(record.state, frozenset())
        if target_state not in allowed:
            raise StrategyRegistryError(
                f"Illegal lifecycle transition for '{strategy_id}' from {record.state.value} to {target_state.value}. "
                f"Allowed targets: {sorted(s.value for s in allowed) if allowed else 'None (terminal)'}"
            )

        qual_id = record.qualification_id
        qual_hash = record.qualification_hash

        # 2. Strict qualification gate for PAPER_ELIGIBLE
        if target_state == StrategyLifecycleState.PAPER_ELIGIBLE:
            if qualification_record is None:
                raise StrategyRegistryError(
                    f"Transition to PAPER_ELIGIBLE for strategy '{strategy_id}' requires "
                    "an authoritative StrategyQualificationRecord"
                )

            if not qualification_record.verify_digest():
                raise StrategyRegistryError(
                    f"Qualification record for '{strategy_id}' failed cryptographic digest verification "
                    "(record hash mismatch)"
                )

            if qualification_record.strategy_id != strategy_id:
                raise StrategyRegistryError(
                    f"Qualification record strategy_id '{qualification_record.strategy_id}' "
                    f"does not match registered strategy '{strategy_id}'"
                )

            norm_spec = normalize_strategy_spec(record.strategy_spec)
            expected_spec_hash = hashlib.sha256(norm_spec.canonical_json().encode("utf-8")).hexdigest()
            if qualification_record.strategy_spec_hash != expected_spec_hash:
                raise StrategyRegistryError(
                    f"Qualification record strategy_spec_hash '{qualification_record.strategy_spec_hash}' "
                    f"does not match registered strategy spec hash '{expected_spec_hash}'"
                )

            if qualification_record.final_status != ValidationStatus.PAPER_ELIGIBLE:
                raise StrategyRegistryError(
                    f"Qualification record final_status is '{qualification_record.final_status.value}'; "
                    "must be 'PAPER_ELIGIBLE'"
                )

            if qualification_record.holdout_state != HoldoutState.PASSED.value:
                raise StrategyRegistryError(
                    f"Qualification record holdout_state is '{qualification_record.holdout_state}'; "
                    f"must be '{HoldoutState.PASSED.value}'"
                )

            qual_id = qualification_record.qualification_id
            qual_hash = qualification_record.record_hash

        now = datetime.now(timezone.utc).isoformat()
        new_history = list(record.history)
        new_history.append((now, target_state.value, reason))

        self._connection.execute(
            """
            UPDATE strategies
            SET state = ?, qualification_id = ?, qualification_hash = ?, updated_at = ?, history_json = ?
            WHERE strategy_id = ?
            """,
            (
                target_state.value,
                qual_id,
                qual_hash,
                now,
                json.dumps(new_history),
                strategy_id,
            ),
        )

        return self.get_strategy(strategy_id)

    def list_strategies(
        self, state: StrategyLifecycleState | None = None
    ) -> list[StrategyRegistryRecord]:
        """List registered strategies, optionally filtered by lifecycle state."""
        if state is not None:
            rows = self._connection.execute(
                "SELECT strategy_id FROM strategies WHERE state = ? ORDER BY registered_at ASC",
                (state.value,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT strategy_id FROM strategies ORDER BY registered_at ASC"
            ).fetchall()

        return [self.get_strategy(r["strategy_id"]) for r in rows]
