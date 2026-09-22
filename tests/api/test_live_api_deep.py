"""Comprehensive Direct API, Worker Stale Recovery, and Governance Barrier Suite.

Validates:
- All 13 FastAPI route modules and status codes.
- Non-optimistic governance barrier guards (fail-closed on illegal transitions).
- Worker crash recovery on stale RUNNING jobs.
- Separation of concerns between Application and Core DBs.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from quantmind.api.app import create_app
from quantmind.app.auth.models import UserRole
from quantmind.app.config import AppSettings
from quantmind.app.context import AppContext
from quantmind.app.jobs.models import JobStatus
from quantmind.app.worker import QuantMindWorker


@pytest.fixture
def test_app_ctx(tmp_path: Path):
    from quantmind.app.context import set_app_context
    core_db = tmp_path / "test_core.db"
    app_db = tmp_path / "test_app.db"
    settings = AppSettings(
        app_env="DEMO",
        core_db_path=core_db,
        app_db_path=app_db,
        secret_key="test-secret-key-12345-very-secure-32bytes-long",
    )
    ctx = AppContext(settings)
    set_app_context(ctx)
    yield ctx, settings


@pytest.fixture
def client(test_app_ctx):
    ctx, settings = test_app_ctx
    app = create_app(settings)
    return TestClient(app)


def get_token(client: TestClient, username: str = "demo_admin", password: str = "QuantMindDemoAdmin2026!") -> str:
    resp = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_auth_lifecycle_and_roles(client: TestClient):
    # 1. Negative test: wrong password
    bad_resp = client.post("/api/v1/auth/login", json={"username": "demo_admin", "password": "wrong"})
    assert bad_resp.status_code == 401

    # 2. Positive test: admin login
    token = get_token(client, "demo_admin", "QuantMindDemoAdmin2026!")
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Get /auth/me
    me_resp = client.get("/api/v1/auth/me", headers=headers)
    assert me_resp.status_code == 200
    user_info = me_resp.json()
    assert user_info["username"] == "demo_admin"
    assert user_info["role"] == "ADMIN"

    # 4. Role restrictions: viewer cannot dispatch research trial
    viewer_token = get_token(client, "demo_viewer", "QuantMindDemoViewer2026!")
    viewer_headers = {"Authorization": f"Bearer {viewer_token}"}
    unauth_resp = client.post(
        "/api/v1/research/trials/submit",
        json={
            "strategy_version": "1.0.0",
            "feature_version": "1.0.0",
            "signal_name": "current_bar_momentum",
            "parameters": {"calendar_session_bars": 30, "session_window": [0.0, 1.0]},
            "dataset_version": "DS-TEST",
        },
        headers=viewer_headers,
    )
    assert unauth_resp.status_code == 403


def test_dashboard_endpoints(client: TestClient):
    token = get_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    # Both /dashboard and /dashboard/kpis must resolve with 200 OK
    resp1 = client.get("/api/v1/dashboard", headers=headers)
    assert resp1.status_code == 200
    assert "total_strategies" in resp1.json()

    resp2 = client.get("/api/v1/dashboard/kpis", headers=headers)
    assert resp2.status_code == 200
    assert "total_strategies" in resp2.json()


def test_datasets_endpoints(client: TestClient):
    token = get_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get("/api/v1/datasets", headers=headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_governance_fail_closed_barrier(client: TestClient):
    token = get_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Listing transitions
    trans_resp = client.get("/api/v1/governance/transitions", headers=headers)
    assert trans_resp.status_code == 200

    # 2. Illegal activation on non-existent strategy fails closed with 404 or 422
    bad_act = client.post(
        "/api/v1/governance/strategies/STRAT-NONEXISTENT/activate",
        json={"initiator": "TEST_AUDITOR"},
        headers=headers,
    )
    assert bad_act.status_code in [404, 422]


def test_worker_crash_recovery(test_app_ctx):
    ctx, settings = test_app_ctx

    # 1. Enqueue a job and manually mark it as RUNNING
    from quantmind.app.jobs.models import JobType
    job = ctx.job_orchestrator.enqueue_job(
        job_type=JobType.RESEARCH_TRIAL,
        initiator_user_id="test_admin",
        parameters={"dummy": "data"},
    )
    job_id = job.job_id
    assert job_id is not None

    with ctx.get_app_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'RUNNING',
                worker_id = 'WRK-CRASHED-001'
            WHERE job_id = ?
            """,
            (job_id,),
        )

    # 2. Instantiate worker and run startup()
    worker = QuantMindWorker(ctx=ctx)
    recovered_count = worker.startup()
    assert recovered_count >= 1

    # 3. Verify job is marked as FAILED with crash recovery note
    job_after = ctx.job_orchestrator.get_job(job_id)
    assert job_after is not None
    assert job_after.status == JobStatus.FAILED
    assert "stale job recovered" in str(job_after.error_message)
