"""Datasets and split partitions route."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from quantmind.api.dependencies import get_ctx, get_current_user
from quantmind.app.auth.models import User
from quantmind.app.context import AppContext

from quantmind.data.registry import DatasetKind
from quantmind.data.splits import SplitZone

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("/enums")
async def get_dataset_enums(
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve authoritative SplitZone and DatasetKind enums from backend source."""
    return {
        "split_zones": [z.value for z in SplitZone],
        "dataset_kinds": [k.value for k in DatasetKind],
    }


@router.get("", response_model=list[dict[str, Any]])
async def list_datasets(
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> list[dict[str, Any]]:
    """List all registered datasets and their split zones."""
    conn = ctx.get_core_connection()
    try:
        rows = conn.execute(
            """
            SELECT d.version, d.kind, d.path, d.format, d.sha256, d.timestamp_column, d.metadata_json,
                   COUNT(z.zone_name) as zone_count
            FROM datasets d
            LEFT JOIN dataset_zones z ON d.version = z.dataset_version
            GROUP BY d.version
            ORDER BY d.version ASC
            """
        ).fetchall()
        result = []
        for r in rows:
            z_rows = conn.execute(
                "SELECT zone_name, start_ts, end_ts FROM dataset_zones WHERE dataset_version = ?",
                (r["version"],),
            ).fetchall()
            zones = [
                {"zone_name": z["zone_name"], "start_ts": z["start_ts"], "end_ts": z["end_ts"]}
                for z in z_rows
            ]
            result.append({
                "version": r["version"],
                "kind": r["kind"],
                "path": r["path"],
                "format": r["format"],
                "sha256": r["sha256"],
                "timestamp_column": r["timestamp_column"],
                "zones": zones,
            })
        return result
    finally:
        conn.close()


@router.get("/{version}", response_model=dict[str, Any])
async def get_dataset_detail(
    version: str,
    user: User = Depends(get_current_user),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Get detailed information for a specific dataset version."""
    conn = ctx.get_core_connection()
    try:
        row = conn.execute("SELECT * FROM datasets WHERE version = ?", (version,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset {version} not found")

        zones = conn.execute(
            "SELECT zone_name, start_ts, end_ts FROM dataset_zones WHERE dataset_version = ?",
            (version,),
        ).fetchall()
        manifest = conn.execute(
            "SELECT manifest_version, manifest_hash, manifest_json FROM dataset_split_manifests WHERE dataset_version = ?",
            (version,),
        ).fetchone()

        return {
            "version": row["version"],
            "kind": row["kind"],
            "path": row["path"],
            "format": row["format"],
            "sha256": row["sha256"],
            "timestamp_column": row["timestamp_column"],
            "zones": [{"zone_name": z["zone_name"], "start_ts": z["start_ts"], "end_ts": z["end_ts"]} for z in zones],
            "manifest": dict(manifest) if manifest else None,
        }
    finally:
        conn.close()
