from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Mapping

import pandas as pd


class DatasetRegistryError(RuntimeError):
    """Raised when a dataset cannot be resolved or verified."""


class DatasetKind(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    LICENSED = "LICENSED"


@dataclass(frozen=True)
class DatasetZone:
    name: str
    start: str | None
    end: str | None


@dataclass(frozen=True)
class DatasetRecord:
    version: str
    kind: DatasetKind
    path: str
    format: str
    sha256: str
    timestamp_column: str
    zones: tuple[DatasetZone, ...]
    metadata: Mapping[str, object]


class DatasetRegistry:
    """Versioned dataset registry with checksum-verified zone loading."""

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
            CREATE TABLE IF NOT EXISTS datasets (
                version TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('SYNTHETIC','LICENSED')),
                path TEXT NOT NULL,
                format TEXT NOT NULL CHECK (format IN ('csv','parquet')),
                sha256 TEXT NOT NULL,
                timestamp_column TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS dataset_zones (
                dataset_version TEXT NOT NULL,
                zone_name TEXT NOT NULL,
                start_ts TEXT,
                end_ts TEXT,
                PRIMARY KEY (dataset_version, zone_name),
                FOREIGN KEY(dataset_version) REFERENCES datasets(version)
            );
            """
        )

    @staticmethod
    def sha256_file(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def register_file(
        self,
        *,
        version: str,
        kind: DatasetKind,
        path: str | Path,
        timestamp_column: str = "timestamp",
        zones: Mapping[str, DatasetZone | tuple[str | None, str | None]] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> DatasetRecord:
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise DatasetRegistryError(f"dataset file not found: {file_path}")
        suffix = file_path.suffix.lower().lstrip(".")
        if suffix not in {"csv", "parquet"}:
            raise DatasetRegistryError(f"unsupported dataset format: {suffix!r}")
        if not version.strip():
            raise DatasetRegistryError("dataset version must not be empty")

        sha256 = self.sha256_file(file_path)
        metadata_json = json.dumps(dict(metadata or {}), sort_keys=True, separators=(",", ":"))
        now_zones: list[DatasetZone] = []
        for name, raw_zone in (zones or {}).items():
            if isinstance(raw_zone, DatasetZone):
                zone = raw_zone
            else:
                start, end = raw_zone
                zone = DatasetZone(name=name, start=start, end=end)
            now_zones.append(zone)

        existing = self._connection.execute(
            "SELECT version, kind, path, format, sha256, timestamp_column, metadata_json FROM datasets WHERE version = ?",
            (version,),
        ).fetchone()
        if existing is not None:
            existing_zones = tuple(
                DatasetZone(name=z["zone_name"], start=z["start_ts"], end=z["end_ts"])
                for z in self._connection.execute(
                    "SELECT zone_name, start_ts, end_ts FROM dataset_zones WHERE dataset_version = ? ORDER BY zone_name",
                    (version,),
                ).fetchall()
            )
            if (
                existing["kind"] != kind.value
                or existing["path"] != str(file_path)
                or existing["format"] != suffix
                or existing["sha256"] != sha256
                or existing["timestamp_column"] != timestamp_column
                or existing["metadata_json"] != metadata_json
                or existing_zones != tuple(sorted(now_zones, key=lambda z: z.name))
            ):
                raise DatasetRegistryError(
                    f"dataset_version={version!r} is immutable; register a new version for changed data or zones"
                )
            return self.get(version)

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                """
                INSERT INTO datasets(version, kind, path, format, sha256, timestamp_column, metadata_json)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version,
                    kind.value,
                    str(file_path),
                    suffix,
                    sha256,
                    timestamp_column,
                    metadata_json,
                ),
            )
            self._connection.execute("DELETE FROM dataset_zones WHERE dataset_version = ?", (version,))
            for zone in now_zones:
                self._validate_zone(zone)
                self._connection.execute(
                    "INSERT INTO dataset_zones(dataset_version, zone_name, start_ts, end_ts) VALUES (?, ?, ?, ?)",
                    (version, zone.name, zone.start, zone.end),
                )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return self.get(version)

    @staticmethod
    def _validate_zone(zone: DatasetZone) -> None:
        if zone.start and zone.end:
            start = pd.Timestamp(zone.start)
            end = pd.Timestamp(zone.end)
            if not start < end:
                raise DatasetRegistryError(f"zone {zone.name!r} must satisfy start < end")

    def get(self, version: str) -> DatasetRecord:
        row = self._connection.execute(
            "SELECT * FROM datasets WHERE version = ?", (version,)
        ).fetchone()
        if row is None:
            raise DatasetRegistryError(f"unknown dataset_version={version!r}")
        zone_rows = self._connection.execute(
            "SELECT zone_name, start_ts, end_ts FROM dataset_zones WHERE dataset_version = ? ORDER BY zone_name",
            (version,),
        ).fetchall()
        return DatasetRecord(
            version=row["version"],
            kind=DatasetKind(row["kind"]),
            path=row["path"],
            format=row["format"],
            sha256=row["sha256"],
            timestamp_column=row["timestamp_column"],
            zones=tuple(
                DatasetZone(name=z["zone_name"], start=z["start_ts"], end=z["end_ts"])
                for z in zone_rows
            ),
            metadata=json.loads(row["metadata_json"]),
        )

    def verify(self, version: str) -> DatasetRecord:
        record = self.get(version)
        actual = self.sha256_file(record.path)
        if actual != record.sha256:
            raise DatasetRegistryError(
                f"dataset checksum mismatch for {version}: expected {record.sha256}, got {actual}"
            )
        return record

    def load_zone(
        self,
        version: str,
        zone_name: str,
        *,
        allowed_kinds: set[DatasetKind] | frozenset[DatasetKind] | None = None,
    ) -> pd.DataFrame:
        record = self.verify(version)
        if allowed_kinds is not None and record.kind not in allowed_kinds:
            raise DatasetRegistryError(
                f"dataset {version} has kind={record.kind.value}; allowed={sorted(k.value for k in allowed_kinds)}"
            )
        zone = next((z for z in record.zones if z.name == zone_name), None)
        if zone is None:
            raise DatasetRegistryError(f"dataset {version} has no zone {zone_name!r}")

        if record.format == "csv":
            frame = pd.read_csv(record.path)
        else:
            frame = pd.read_parquet(record.path)

        if record.timestamp_column not in frame.columns:
            raise DatasetRegistryError(
                f"dataset {version} missing timestamp column {record.timestamp_column!r}"
            )
        timestamps = pd.to_datetime(frame[record.timestamp_column])
        start = pd.Timestamp(zone.start) if zone.start else None
        end = pd.Timestamp(zone.end) if zone.end else None
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= timestamps >= start
        if end is not None:
            mask &= timestamps < end
        selected = frame.loc[mask].copy().reset_index(drop=True)
        if selected.empty:
            raise DatasetRegistryError(f"dataset {version} zone {zone_name!r} is empty")
        return selected
