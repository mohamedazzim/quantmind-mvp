"""Asynchronous Job Orchestrator for QuantMind (PRD v4.0 Productization)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Any

from quantmind.app.jobs.models import Job, JobStatus, JobType


class JobOrchestrator:
    """Orchestrates persistent asynchronous tasks backed by quantmind_app.db."""

    def __init__(self, app_db_path: Path) -> None:
        self.db_path = Path(app_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_pct INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    initiator_user_id TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    result_json TEXT,
                    result_ref TEXT,
                    error_message TEXT,
                    worker_id TEXT
                )
                """
            )

    def recover_stale_jobs(self) -> int:
        """Mark any RUNNING job as FAILED upon worker process boot."""
        now_ts = datetime.now(timezone.utc).isoformat()
        reason = "Worker process terminated unexpectedly; stale job recovered on startup"
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, error_message = ?
                WHERE status = ?
                """,
                (JobStatus.FAILED.value, now_ts, reason, JobStatus.RUNNING.value),
            )
            return cur.rowcount

    def enqueue_job(
        self,
        job_type: JobType,
        initiator_user_id: str,
        parameters: dict[str, Any],
    ) -> Job:
        """Enqueue a background task to quantmind_app.db."""
        job_id = f"JOB-{secrets.token_hex(8)}"
        now_ts = datetime.now(timezone.utc).isoformat()
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            progress_pct=0,
            created_at=now_ts,
            initiator_user_id=initiator_user_id,
            parameters=parameters,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, status, progress_pct, created_at,
                    initiator_user_id, parameters_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.job_type.value,
                    job.status.value,
                    job.progress_pct,
                    job.created_at,
                    job.initiator_user_id,
                    json.dumps(job.parameters, sort_keys=True),
                ),
            )
        return job

    def claim_next_job(self, worker_id: str) -> Job | None:
        """Atomically claim the oldest QUEUED job for execution."""
        now_ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (JobStatus.QUEUED.value,),
            ).fetchone()

            if not row:
                conn.execute("COMMIT")
                return None

            job_id = row["job_id"]
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?, worker_id = ?
                WHERE job_id = ? AND status = ?
                """,
                (JobStatus.RUNNING.value, now_ts, worker_id, job_id, JobStatus.QUEUED.value),
            )
            conn.execute("COMMIT")
            return self.get_job(job_id)

    def update_progress(self, job_id: str, progress_pct: int) -> None:
        """Update job progress percentage (0-100)."""
        clamped = max(0, min(100, progress_pct))
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET progress_pct = ? WHERE job_id = ?",
                (clamped, job_id),
            )

    def complete_job(
        self,
        job_id: str,
        result: dict[str, Any],
        result_ref: str | None = None,
    ) -> None:
        """Mark job as COMPLETED with structured results and reference key."""
        now_ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, progress_pct = 100, completed_at = ?,
                    result_json = ?, result_ref = ?
                WHERE job_id = ?
                """,
                (
                    JobStatus.COMPLETED.value,
                    now_ts,
                    json.dumps(result, sort_keys=True, default=str),
                    result_ref,
                    job_id,
                ),
            )

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as FAILED with captured error diagnostics."""
        now_ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, error_message = ?
                WHERE job_id = ?
                """,
                (JobStatus.FAILED.value, now_ts, error_message, job_id),
            )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a QUEUED job."""
        now_ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, error_message = 'Cancelled by user'
                WHERE job_id = ? AND status = ?
                """,
                (JobStatus.CANCELLED.value, now_ts, job_id, JobStatus.QUEUED.value),
            )
            return cur.rowcount > 0

    def get_job(self, job_id: str) -> Job | None:
        """Retrieve job state by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return self._row_to_job(row)

    def list_jobs(
        self,
        limit: int = 50,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
    ) -> list[Job]:
        """List recent jobs with optional status and job_type filters."""
        with self._get_connection() as conn:
            query = "SELECT * FROM jobs"
            params: list[Any] = []
            clauses = []
            if status:
                clauses.append("status = ?")
                params.append(status.value)
            if job_type:
                clauses.append("job_type = ?")
                params.append(job_type.value)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_job(r) for r in rows]


    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        params = json.loads(row["parameters_json"]) if row["parameters_json"] else {}
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return Job(
            job_id=row["job_id"],
            job_type=JobType(row["job_type"]),
            status=JobStatus(row["status"]),
            progress_pct=row["progress_pct"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            initiator_user_id=row["initiator_user_id"],
            parameters=params,
            result=result,
            result_ref=row["result_ref"],
            error_message=row["error_message"],
        )
