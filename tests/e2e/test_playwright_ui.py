"""Playwright Headless Browser UI Test Suite (PRD v4.0 Productization).

Verifies the Next.js frontend running against FastAPI backend:
- Login page authentication flow.
- Navigation across Dashboard, Strategies, Datasets, Research, Paper, Monitoring, Governance, and Audit.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
import pytest
from playwright.sync_api import sync_playwright, Page, Browser

from quantmind.app.config import get_settings


@pytest.fixture(scope="module")
def app_servers():
    """Starts FastAPI API server and Next.js Web server for Playwright E2E tests."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    web_dir = repo_root / "web"

    # Set environment for demo mode
    env = os.environ.copy()
    env["APP_ENV"] = "DEMO"
    env["PYTHONPATH"] = str(repo_root / "src")
    env["PORT"] = "3000"

    # 1. Start FastAPI server
    api_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "quantmind.api.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 2. Start Next.js production server
    web_proc = subprocess.Popen(
        ["npm.cmd" if os.name == "nt" else "npm", "start", "--", "-p", "3000"],
        cwd=str(web_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for servers to be up (poll health endpoint)
    import urllib.request
    api_ready = False
    for _ in range(30):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as resp:
                if resp.status == 200:
                    api_ready = True
                    break
        except Exception:
            time.sleep(0.5)

    web_ready = False
    for _ in range(30):
        try:
            with urllib.request.urlopen("http://127.0.0.1:3000", timeout=1) as resp:
                if resp.status in (200, 307, 308):
                    web_ready = True
                    break
        except Exception:
            time.sleep(0.5)

    if not api_ready or not web_ready:
        api_proc.terminate()
        web_proc.terminate()
        pytest.skip(f"Servers not ready (API: {api_ready}, Web: {web_ready})")

    yield {"api_url": "http://127.0.0.1:8000", "web_url": "http://127.0.0.1:3000"}

    # Teardown
    api_proc.terminate()
    web_proc.terminate()
    api_proc.wait(timeout=5)
    web_proc.wait(timeout=5)


def test_playwright_full_ui_navigation(app_servers):
    """Launches Playwright headless Chromium and tests login + navigation across all routes."""
    web_url = app_servers["web_url"]

    with sync_playwright() as p:
        browser: Browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page: Page = context.new_page()

        # 1. Visit Login Page
        page.goto(f"{web_url}/login")
        page.wait_for_selector("text=Sign In")
        assert "QuantMind" in page.content()

        # 2. Login as Demo Admin
        page.fill('input[type="text"]', "demo_admin")
        page.fill('input[type="password"]', "QuantMindDemoAdmin2026!")
        page.click('button[type="submit"]')

        # 3. Wait for Dashboard Navigation
        page.wait_for_url("**/dashboard", timeout=15000)
        page.wait_for_selector("text=System Overview")

        # Verify Dashboard Metric Cards
        assert "Active Paper Strategies" in page.content()
        assert "Paper Replay Net PnL" in page.content()

        # 4. Visit Strategies Page
        page.goto(f"{web_url}/strategies")
        page.wait_for_selector("text=Strategy Registry")

        # 5. Visit Datasets Page
        page.goto(f"{web_url}/datasets")
        page.wait_for_selector("text=Dataset Registry")

        # 6. Visit Research Page
        page.goto(f"{web_url}/research")
        page.wait_for_selector("text=Research Workspace")

        # 7. Visit Paper Page
        page.goto(f"{web_url}/paper")
        page.wait_for_selector("text=Paper Execution & Replay")

        # 8. Visit Monitoring Page
        page.goto(f"{web_url}/monitoring")
        page.wait_for_selector("text=Performance Monitoring & Degradation Detection")

        # 9. Visit Governance Page
        page.goto(f"{web_url}/governance")
        page.wait_for_selector("text=Governance Audit Trail")

        # 10. Visit Audit Page
        page.goto(f"{web_url}/audit")
        page.wait_for_selector("text=Audit Provenance Explorer")

        context.close()
        browser.close()
