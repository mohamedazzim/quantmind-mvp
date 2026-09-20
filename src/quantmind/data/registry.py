from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Mapping

import numpy as np
import pandas as pd

from .splits import SplitManifest, SplitManifestError, SplitZone


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
    split_manifest: SplitManifest | None = None


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

            CREATE TABLE IF NOT EXISTS dataset_split_manifests (
                dataset_version TEXT PRIMARY KEY,
                manifest_version TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
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
        split_manifest: SplitManifest | None = None,
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
            existing_manifest_row = self._connection.execute(
                "SELECT manifest_version, manifest_hash, manifest_json FROM dataset_split_manifests WHERE dataset_version = ?",
                (version,),
            ).fetchone()

            manifest_mismatch = False
            if split_manifest is not None:
                if existing_manifest_row is None or existing_manifest_row["manifest_hash"] != split_manifest.manifest_hash():
                    manifest_mismatch = True
            elif existing_manifest_row is not None:
                # Registered without manifest before, or now without manifest
                pass

            if (
                existing["kind"] != kind.value
                or existing["path"] != str(file_path)
                or existing["format"] != suffix
                or existing["sha256"] != sha256
                or existing["timestamp_column"] != timestamp_column
                or existing["metadata_json"] != metadata_json
                or existing_zones != tuple(sorted(now_zones, key=lambda z: z.name))
                or manifest_mismatch
            ):
                raise DatasetRegistryError(
                    f"dataset_version={version!r} is immutable; register a new version for changed data, zones, or split manifest"
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
            if split_manifest is not None:
                if split_manifest.dataset_version != version:
                    raise DatasetRegistryError(
                        f"split_manifest dataset_version ({split_manifest.dataset_version}) does not match registration version ({version})"
                    )
                self._connection.execute(
                    """
                    INSERT INTO dataset_split_manifests(dataset_version, manifest_version, manifest_hash, manifest_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        version,
                        split_manifest.manifest_version,
                        split_manifest.manifest_hash(),
                        split_manifest.canonical_json(),
                    ),
                )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return self.get(version)

    def register_split_manifest(self, manifest: SplitManifest) -> None:
        """Register an immutable split manifest for an existing dataset."""
        record = self.get(manifest.dataset_version)
        existing = self._connection.execute(
            "SELECT manifest_version, manifest_hash, manifest_json FROM dataset_split_manifests WHERE dataset_version = ?",
            (manifest.dataset_version,),
        ).fetchone()
        if existing is not None:
            if existing["manifest_hash"] != manifest.manifest_hash():
                raise DatasetRegistryError(
                    f"dataset_version={manifest.dataset_version!r} is immutable; split manifest cannot be modified"
                )
            return
        self._connection.execute(
            """
            INSERT INTO dataset_split_manifests(dataset_version, manifest_version, manifest_hash, manifest_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                manifest.dataset_version,
                manifest.manifest_version,
                manifest.manifest_hash(),
                manifest.canonical_json(),
            ),
        )

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
        manifest_row = self._connection.execute(
            "SELECT manifest_json FROM dataset_split_manifests WHERE dataset_version = ?",
            (version,),
        ).fetchone()
        manifest = (
            SplitManifest.from_json(manifest_row["manifest_json"])
            if manifest_row is not None
            else None
        )
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
            split_manifest=manifest,
        )

    def verify(self, version: str) -> DatasetRecord:
        record = self.get(version)
        actual = self.sha256_file(record.path)
        if actual != record.sha256:
            raise DatasetRegistryError(
                f"dataset checksum mismatch for {version}: expected {record.sha256}, got {actual}"
            )
        return record

    def _read_file(self, path: str, fmt: str, expected_sha256: str) -> pd.DataFrame:
        if not hasattr(self, "_data_cache"):
            self._data_cache: dict[str, tuple[str, pd.DataFrame]] = {}
        if path in self._data_cache:
            cached_sha256, cached_frame = self._data_cache[path]
            if cached_sha256 == expected_sha256:
                return cached_frame
        if fmt == "csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_parquet(path)
        self._data_cache[path] = (expected_sha256, frame)
        return frame

    def load_zone(
        self,
        version: str,
        zone_name: SplitZone | str,
        *,
        allowed_kinds: set[DatasetKind] | frozenset[DatasetKind] | None = None,
        allow_holdout: bool = False,
    ) -> pd.DataFrame:
        record = self.verify(version)
        if allowed_kinds is not None and record.kind not in allowed_kinds:
            raise DatasetRegistryError(
                f"dataset {version} has kind={record.kind.value}; allowed={sorted(k.value for k in allowed_kinds)}"
            )

        zone_str = zone_name.value if isinstance(zone_name, SplitZone) else str(zone_name).upper()

        if zone_str == SplitZone.FINAL_HOLDOUT.value and not allow_holdout:
            raise DatasetRegistryError(
                "FINAL_HOLDOUT zone is sealed; access is restricted to the controlled final_evaluate path."
            )

        start: str | None = None
        end: str | None = None

        if record.split_manifest is not None:
            try:
                start, end = record.split_manifest.get_zone_bounds(zone_str)
            except SplitManifestError as exc:
                zone = next((z for z in record.zones if z.name == zone_str), None)
                if zone is None:
                    raise DatasetRegistryError(f"dataset {version} has no zone {zone_str!r}") from exc
                start, end = zone.start, zone.end
        else:
            zone = next((z for z in record.zones if z.name == zone_str), None)
            if zone is None:
                raise DatasetRegistryError(f"dataset {version} has no zone {zone_str!r}")
            start, end = zone.start, zone.end

        frame = self._read_file(record.path, record.format, record.sha256)

        if record.timestamp_column not in frame.columns:
            raise DatasetRegistryError(
                f"dataset {version} missing timestamp column {record.timestamp_column!r}"
            )

        timestamps = pd.to_datetime(frame[record.timestamp_column])
        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None

        mask = np.ones(len(frame), dtype=bool)
        if start_ts is not None:
            mask &= (timestamps >= start_ts).to_numpy()
        if end_ts is not None:
            mask &= (timestamps < end_ts).to_numpy()

        selected = frame.loc[mask].copy().reset_index(drop=True)
        if selected.empty:
            raise DatasetRegistryError(f"dataset {version} zone {zone_str!r} is empty")
        return selected
