"""Automated tests for QuantMind FastAPI API and worker subsystem (PRD v4.0 Productization)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from quantmind.api.app import create_app
from quantmind.app.auth.models import UserRole
from quantmind.app.config import AppSettings
from quantmind.app.context import AppContext, set_app_context
from quantmind.app.jobs.models import JobStatus, JobType
from quantmind.app.worker import QuantMindWorker
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.spec import StrategySpec


@pytest.fixture
def test_ctx(tmp_path: Path) -> AppContext:
    """Creates an isolated AppContext with temp databases for API tests."""
    data_dir = tmp_path / "data"
    core_db = data_dir / "test_core.db"
    app_db = data_dir / "test_app.db"

    settings = AppSettings(
        APP_ENV="DEVELOPMENT",
        data_dir=data_dir,
        core_db_path=core_db,
        app_db_path=app_db,
        AUTH_SECRET_KEY="test-secret-key-for-unit-tests-only",
    )
    ctx = AppContext(settings)
    set_app_context(ctx)

    # Bootstrap initial test admin
    ctx.auth_service.bootstrap_admin(
        username="admin_test",
        email="admin@test.local",
        password="TestAdminSecurePassword123!",
        force_password_change=False,
    )
    return ctx


@pytest.fixture
def client(test_ctx: AppContext) -> TestClient:
    app = create_app()
    return TestClient(app)


def test_health_and_ready(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    assert resp_ready.json()["status"] == "ready"


def test_auth_workflow(client: TestClient, test_ctx: AppContext) -> None:
    # 1. Invalid login
    bad_login = client.post(
        "/api/v1/auth/login",
        json={"username": "admin_test", "password": "WrongPassword!"},
    )
    assert bad_login.status_code == 401

    # 2. Valid login
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "admin_test", "password": "TestAdminSecurePassword123!"},
    )
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    token = token_data["access_token"]
    assert token_data["user"]["role"] == "ADMIN"

    # 3. GET /me with token
    headers = {"Authorization": f"Bearer {token}"}
    me_resp = client.get("/api/v1/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "admin_test"

    # 4. Change password
    change_resp = client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={
            "old_password": "TestAdminSecurePassword123!",
            "new_password": "NewSuperAdminSecurePassword456!",
        },
    )
    assert change_resp.status_code == 200

    # 5. Verify new password works
    relogin_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "admin_test", "password": "NewSuperAdminSecurePassword456!"},
    )
    assert relogin_resp.status_code == 200


def test_spec_schema_and_dataset_enums(client: TestClient, test_ctx: AppContext) -> None:
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "admin_test", "password": "TestAdminSecurePassword123!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Spec schema
    schema_resp = client.get("/api/v1/strategies/spec-schema", headers=headers)
    assert schema_resp.status_code == 200
    schema_data = schema_resp.json()
    assert "current_bar_momentum" in schema_data["supported_signals"]
    assert "current_bar_mean_reversion" in schema_data["supported_signals"]
    assert "calendar_session_bars" in schema_data["signal_parameters"]["current_bar_momentum"]

    # Datasets enums
    enums_resp = client.get("/api/v1/datasets/enums", headers=headers)
    assert enums_resp.status_code == 200
    enums_data = enums_resp.json()
    assert "RESEARCH" in enums_data["split_zones"]
    assert "VALIDATION" in enums_data["split_zones"]
    assert "FINAL_HOLDOUT" in enums_data["split_zones"]
    assert "FORWARD_PAPER" in enums_data["split_zones"]
    assert "SYNTHETIC" in enums_data["dataset_kinds"]
    assert "LICENSED" in enums_data["dataset_kinds"]
    assert "DISCOVERY" not in enums_data["split_zones"]


def test_candidate_validation_and_job_execution(client: TestClient, test_ctx: AppContext) -> None:
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "admin_test", "password": "TestAdminSecurePassword123!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Validate Candidate Spec
    cand_resp = client.post(
        "/api/v1/research/candidates/validate",
        headers=headers,
        json={
            "strategy_version": "1.0.0",
            "feature_version": "1.0.0",
            "signal_name": "current_bar_momentum",
            "parameters": {
                "calendar_session_bars": 100,
                "session_window": [0.0, 1.0],
            },
        },
    )
    assert cand_resp.status_code == 200
    cand_data = cand_resp.json()
    assert cand_data["is_valid"] is True
    strat_id = cand_data["strategy_id"]
    assert strat_id.startswith("STRAT-")

    # 2. Submit trial to queue
    submit_resp = client.post(
        "/api/v1/research/trials/submit",
        headers=headers,
        json={
            "strategy_version": "1.0.0",
            "feature_version": "1.0.0",
            "signal_name": "current_bar_momentum",
            "parameters": {
                "calendar_session_bars": 100,
                "session_window": [0.0, 1.0],
            },
            "dataset_version": "ds-synth-v1",
            "split_zone": "RESEARCH",
            "research_protocol_version": "proto-v1",
            "seed": 42,
        },
    )
    assert submit_resp.status_code == 200
    submit_data = submit_resp.json()
    job_id = submit_data["job_id"]
    assert submit_data["status"] == "QUEUED"

    # 3. Verify Job in QUEUED state via GET /api/v1/jobs/{job_id}
    job_resp = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] == "QUEUED"

    # 4. Worker executes the job
    worker = QuantMindWorker(ctx=test_ctx, worker_id="test-worker-1")
    recovered = worker.startup()
    assert recovered == 0

    # Claim and process job
    processed = worker.process_one_job()
    assert processed is True

    # 5. Verify Job is now COMPLETED or FAILED
    job_done = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    assert job_done.status_code == 200
    job_done_data = job_done.json()
    # It has run through worker!
    assert job_done_data["status"] in ("COMPLETED", "FAILED")


def test_worker_stale_job_recovery(test_ctx: AppContext) -> None:
    # Manually insert a RUNNING job simulating a worker crash
    job = test_ctx.job_orchestrator.enqueue_job(
        job_type=JobType.PAPER_REPLAY,
        initiator_user_id="test-user",
        parameters={"strategy_id": "STRAT-fake"},
    )
    # Claim it to mark RUNNING
    claimed = test_ctx.job_orchestrator.claim_next_job("crashed-worker")
    assert claimed is not None
    assert claimed.status == JobStatus.RUNNING

    # Worker boots up
    worker = QuantMindWorker(ctx=test_ctx, worker_id="recovery-worker")
    recovered_count = worker.startup()
    assert recovered_count == 1

    # Check that the job is marked FAILED with recovery message
    recovered_job = test_ctx.job_orchestrator.get_job(job.job_id)
    assert recovered_job is not None
    assert recovered_job.status == JobStatus.FAILED
    assert "Worker process terminated unexpectedly" in (recovered_job.error_message or "")


def test_governance_non_optimistic_activation_rejection(client: TestClient, test_ctx: AppContext) -> None:
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "admin_test", "password": "TestAdminSecurePassword123!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Register a raw strategy directly into StrategyRegistry
    spec = StrategySpec(
        strategy_version="1.0.0",
        feature_version="1.0.0",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 100, "session_window": [0.0, 1.0]},
    )
    strat_id = test_ctx.strategy_registry.register_strategy(spec)

    # Attempt to activate a strategy that is not PAPER_ELIGIBLE and has no baseline
    activate_resp = client.post(
        f"/api/v1/strategies/{strat_id}/activate",
        headers=headers,
        json={"initiator": "governance-test"},
    )
    # Must fail closed with 422 Unprocessable Entity
    assert activate_resp.status_code == 422
    assert activate_resp.json()["code"] == "GOVERNANCE_INTEGRITY_VIOLATION"


def test_audit_explorer_broken_provenance(client: TestClient, test_ctx: AppContext) -> None:
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "admin_test", "password": "TestAdminSecurePassword123!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Query audit graph for non-existent entity
    graph_resp = client.get("/api/v1/audit/graph/NON_EXISTENT_ID", headers=headers)
    assert graph_resp.status_code == 200
    graph_data = graph_resp.json()
    assert graph_data["nodes"] == []
    assert "No provenance record found" in graph_data.get("error", "")
