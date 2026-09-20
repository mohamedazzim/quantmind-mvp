"""Immutable OOS return artifact storage with SHA-256 integrity verification.

Every PRODUCTION trial that completes successfully writes a Parquet artifact
containing its complete OOS trade-level return stream.  The artifact is
registered in a SQLite ``trial_artifacts`` table and is **immutable** once
status = 'COMPLETED'.

Schema guarantees
-----------------
- Parquet file uses a fixed pyarrow schema (deterministic column order/types).
- SHA-256 is computed on the raw bytes *after* writing.
- The same (trial_id, trades, dataset, seed) always produces the same hash.
- Re-writing or deleting a completed artifact raises ``ArtifactImmutabilityError``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantmind.backtest.engine import BacktestResult, BacktestTrade


# ---------------------------------------------------------------------------
# Parquet schema — DO NOT change column names / types without a version bump
# ---------------------------------------------------------------------------

_OOS_RETURNS_SCHEMA = pa.schema(
    [
        pa.field("signal_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("entry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("exit_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("side", pa.int8()),
        pa.field("gross_return_bps", pa.float64()),
        pa.field("cost_bps", pa.float64()),
        pa.field("net_return_bps", pa.float64()),
    ]
)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class ArtifactType(str, Enum):
    OOS_RETURNS = "OOS_RETURNS"


class ArtifactStatus(str, Enum):
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class ArtifactRecord:
    """Immutable summary of a stored artifact as returned by ArtifactRegistry."""

    artifact_id: str
    trial_id: str
    artifact_type: str           # ArtifactType value
    dataset_version: str
    sha256: str
    format: str                  # "parquet"
    row_count: int
    start_timestamp: str         # ISO-8601 UTC
    end_timestamp: str           # ISO-8601 UTC
    created_at: str              # ISO-8601 UTC
    status: str                  # ArtifactStatus value
    artifact_path: str           # absolute path on disk


class ArtifactImmutabilityError(RuntimeError):
    """Raised when an attempt is made to overwrite or delete a completed artifact."""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_utc_us(ts: pd.Timestamp) -> pd.Timestamp:
    """Coerce a pandas Timestamp to UTC with microsecond precision."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.round("us")


def _trades_to_arrow(trades: Sequence[BacktestTrade]) -> pa.Table:
    """Serialize a sequence of BacktestTrade to a deterministic pyarrow Table."""
    if not trades:
        return pa.table(
            {field.name: pa.array([], type=field.type) for field in _OOS_RETURNS_SCHEMA},
            schema=_OOS_RETURNS_SCHEMA,
        )

    signal_ts = pa.array(
        [_to_utc_us(t.signal_timestamp).value // 1000 for t in trades],
        type=pa.timestamp("us", tz="UTC"),
    )
    entry_ts = pa.array(
        [_to_utc_us(t.entry_timestamp).value // 1000 for t in trades],
        type=pa.timestamp("us", tz="UTC"),
    )
    exit_ts = pa.array(
        [_to_utc_us(t.exit_timestamp).value // 1000 for t in trades],
        type=pa.timestamp("us", tz="UTC"),
    )
    side = pa.array([t.side for t in trades], type=pa.int8())
    gross = pa.array([t.gross_return_bps for t in trades], type=pa.float64())
    cost = pa.array([t.cost_bps for t in trades], type=pa.float64())
    net = pa.array([t.net_return_bps for t in trades], type=pa.float64())

    return pa.table(
        {
            "signal_timestamp": signal_ts,
            "entry_timestamp": entry_ts,
            "exit_timestamp": exit_ts,
            "side": side,
            "gross_return_bps": gross,
            "cost_bps": cost,
            "net_return_bps": net,
        },
        schema=_OOS_RETURNS_SCHEMA,
    )


def write_oos_returns_artifact(
    result: BacktestResult,
    *,
    trial_id: str,
    dataset_version: str,
    artifact_dir: Path,
) -> tuple[Path, str, int, str, str]:
    """Serialize OOS trade stream to Parquet and return (path, sha256, row_count,
    start_timestamp_iso, end_timestamp_iso).

    The Parquet file is written with snappy compression and dictionary encoding
    disabled so that hash is stable across platforms/PyArrow versions for the
    same logical data.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    table = _trades_to_arrow(result.trades)
    row_count = len(table)

    fname = f"{trial_id}_oos_returns.parquet"
    path = artifact_dir / fname

    if path.exists():
        raise ArtifactImmutabilityError(
            f"Artifact file already exists and must not be overwritten: {path}"
        )

    pq.write_table(
        table,
        path,
        compression="snappy",
        use_dictionary=False,
        write_statistics=False,
        store_schema=True,
    )

    digest = _sha256_file(path)

    if row_count == 0:
        start_ts = end_ts = datetime.now(timezone.utc).isoformat()
    else:
        starts = table.column("signal_timestamp").to_pylist()
        exits = table.column("exit_timestamp").to_pylist()
        start_ts = pd.Timestamp(starts[0]).isoformat()
        end_ts = pd.Timestamp(exits[-1]).isoformat()

    return path, digest, row_count, start_ts, end_ts


# ---------------------------------------------------------------------------
# Artifact Registry
# ---------------------------------------------------------------------------

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS trial_artifacts (
    artifact_id     TEXT PRIMARY KEY,
    trial_id        TEXT NOT NULL,
    artifact_type   TEXT NOT NULL CHECK (artifact_type IN ('OOS_RETURNS')),
    dataset_version TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    format          TEXT NOT NULL CHECK (format IN ('parquet')),
    row_count       INTEGER NOT NULL,
    start_timestamp TEXT NOT NULL,
    end_timestamp   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('COMPLETED')),
    artifact_path   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_trial
ON trial_artifacts(trial_id);

CREATE INDEX IF NOT EXISTS idx_artifacts_scope
ON trial_artifacts(dataset_version, artifact_type);

CREATE TRIGGER IF NOT EXISTS artifacts_no_delete
BEFORE DELETE ON trial_artifacts
BEGIN
    SELECT RAISE(ABORT, 'trial_artifacts are immutable and append-only');
END;

CREATE TRIGGER IF NOT EXISTS artifacts_no_update
BEFORE UPDATE ON trial_artifacts
BEGIN
    SELECT RAISE(ABORT, 'trial_artifacts are immutable once written');
END;
"""


class ArtifactRegistry:
    """SQLite-backed registry of immutable trial artifacts.

    Shares the same database connection as the TrialLedger when passed one,
    or opens its own SQLite file when given a path.
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
        self._connection.executescript(_CREATE_SCHEMA)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        trial_id: str,
        artifact_type: ArtifactType,
        dataset_version: str,
        sha256: str,
        format: str,
        row_count: int,
        start_timestamp: str,
        end_timestamp: str,
        artifact_path: str | Path,
    ) -> ArtifactRecord:
        """Insert a completed artifact record; raises if a record for *trial_id*
        and *artifact_type* already exists (immutability guarantee)."""
        existing = self._connection.execute(
            "SELECT artifact_id FROM trial_artifacts WHERE trial_id = ? AND artifact_type = ?",
            (trial_id, artifact_type.value),
        ).fetchone()
        if existing is not None:
            raise ArtifactImmutabilityError(
                f"Artifact of type {artifact_type.value} already registered for trial {trial_id}"
            )

        artifact_id = "ART-" + uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()

        self._connection.execute(
            """
            INSERT INTO trial_artifacts (
                artifact_id, trial_id, artifact_type, dataset_version,
                sha256, format, row_count,
                start_timestamp, end_timestamp,
                created_at, status, artifact_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED', ?)
            """,
            (
                artifact_id,
                trial_id,
                artifact_type.value,
                dataset_version,
                sha256,
                format,
                row_count,
                start_timestamp,
                end_timestamp,
                now,
                str(artifact_path),
            ),
        )

        return ArtifactRecord(
            artifact_id=artifact_id,
            trial_id=trial_id,
            artifact_type=artifact_type.value,
            dataset_version=dataset_version,
            sha256=sha256,
            format=format,
            row_count=row_count,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            created_at=now,
            status=ArtifactStatus.COMPLETED.value,
            artifact_path=str(artifact_path),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, artifact_id: str) -> ArtifactRecord:
        row = self._connection.execute(
            "SELECT * FROM trial_artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact_id: {artifact_id}")
        return _row_to_record(row)

    def get_for_trial(self, trial_id: str, artifact_type: ArtifactType) -> ArtifactRecord | None:
        """Return the artifact record for a trial+type, or None if not registered."""
        row = self._connection.execute(
            "SELECT * FROM trial_artifacts WHERE trial_id = ? AND artifact_type = ?",
            (trial_id, artifact_type.value),
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list_for_dataset(
        self,
        dataset_version: str,
        artifact_type: ArtifactType = ArtifactType.OOS_RETURNS,
    ) -> list[ArtifactRecord]:
        """Return all artifact records for a dataset_version + artifact_type."""
        rows = self._connection.execute(
            "SELECT * FROM trial_artifacts WHERE dataset_version = ? AND artifact_type = ? ORDER BY created_at ASC",
            (dataset_version, artifact_type.value),
        ).fetchall()
        return [_row_to_record(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=row["artifact_id"],
        trial_id=row["trial_id"],
        artifact_type=row["artifact_type"],
        dataset_version=row["dataset_version"],
        sha256=row["sha256"],
        format=row["format"],
        row_count=row["row_count"],
        start_timestamp=row["start_timestamp"],
        end_timestamp=row["end_timestamp"],
        created_at=row["created_at"],
        status=row["status"],
        artifact_path=row["artifact_path"],
    )


# ---------------------------------------------------------------------------
# Parquet reader with SHA-256 verification
# ---------------------------------------------------------------------------


def load_oos_returns_verified(record: ArtifactRecord) -> pa.Table:
    """Load a Parquet artifact, verifying SHA-256 before returning the table.

    Raises ``ValueError`` if the digest does not match (tampered file).
    Raises ``FileNotFoundError`` if the file is missing from disk.
    """
    path = Path(record.artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found on disk: {path}")

    actual = _sha256_file(path)
    if actual != record.sha256:
        raise ValueError(
            f"SHA-256 mismatch for artifact {record.artifact_id}: "
            f"expected {record.sha256}, got {actual}"
        )

    table = pq.read_table(path, schema=_OOS_RETURNS_SCHEMA)
    return table


def load_net_returns_array(record: ArtifactRecord) -> np.ndarray:
    """Return net_return_bps as a 1-D float64 NumPy array (SHA-256 verified)."""
    table = load_oos_returns_verified(record)
    col = table.column("net_return_bps")
    return np.asarray(col, dtype=np.float64)


def load_signal_timestamps_array(record: ArtifactRecord) -> np.ndarray:
    """Return signal_timestamp as a 1-D datetime64[us] NumPy array (SHA-256 verified)."""
    table = load_oos_returns_verified(record)
    col = table.column("signal_timestamp")
    # PyArrow timestamp → pandas Series → numpy datetime64[us]
    return col.to_pandas().values.astype("datetime64[us]")
