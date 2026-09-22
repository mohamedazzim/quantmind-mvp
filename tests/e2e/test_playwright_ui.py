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

    import urllib.request
    # If servers are already running locally, use them directly
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as r1:
            with urllib.request.urlopen("http://127.0.0.1:3000", timeout=1) as r2:
                if r1.status == 200 and r2.status in (200, 307, 308):
                    yield {"api_url": "http://127.0.0.1:8000", "web_url": "http://127.0.0.1:3000"}
                    return
    except Exception:
        pass

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


def test_playwright_extended_workflows(app_servers):
    """Executes end-to-end interactive workflows in real browser (Phase 29)."""
    web_url = app_servers["web_url"]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Step 1: Login invalid and valid
        page.goto(f"{web_url}/login")
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("button[type='submit']")
        page.fill('input#username', 'invalid_user')
        page.fill('input#password', 'wrong')
        page.click('button[type="submit"]')
        page.wait_for_timeout(1000)
        assert page.locator("div.border-danger\\/40, div:has-text('Invalid')").count() > 0 or "Invalid" in page.content()

        page.fill('input#username', 'demo_admin')
        page.fill('input#password', 'QuantMindDemoAdmin2026!')
        page.click('button[type="submit"]')
        page.wait_for_url("**/dashboard", timeout=10000)

        # Step 2: Dashboard
        assert page.locator("text=System Overview").count() > 0

        # Step 3: Strategy Navigation
        page.goto(f"{web_url}/strategies")
        page.wait_for_selector("text=Strategy Registry")
        page.wait_for_selector("text=STRAT-a161a6c1070d45d0c1d89b8d", timeout=8000)
        page.click("text=STRAT-a161a6c1070d45d0c1d89b8d")
        page.wait_for_selector("text=Back to Strategies", timeout=8000)
        for tab_txt in ["Overview & Specification", "Qualification Evidence", "Governance Commands"]:
            t = page.locator(f"button:has-text('{tab_txt}')")
            if t.count() > 0:
                t.first.click()
                page.wait_for_timeout(200)

        # Step 4 & 5: Research candidate validation & job dispatch
        page.goto(f"{web_url}/research")
        page.wait_for_selector("text=Research Workspace")
        val_btn = page.locator("button:has-text('Preview & Validate Strategy Identity')")
        if val_btn.count() > 0:
            val_btn.first.click()
            page.wait_for_timeout(800)
            sub_btn = page.locator("button:has-text('Confirm & Submit Trial to Worker Queue')")
            if sub_btn.count() > 0:
                sub_btn.first.click()
                page.wait_for_timeout(2000)

        # Step 6: Trials
        page.goto(f"{web_url}/trials")
        page.wait_for_selector("text=Trial Ledger")

        # Step 7: Qualification
        page.goto(f"{web_url}/qualification")
        page.wait_for_selector("text=Strategy Qualification")

        # Step 8: Paper
        page.goto(f"{web_url}/paper")
        page.wait_for_selector("text=Paper Execution & Replay")

        # Step 9: Monitoring
        page.goto(f"{web_url}/monitoring")
        page.wait_for_selector("text=Performance Monitoring & Degradation Detection")

        # Step 10: Governance
        page.goto(f"{web_url}/governance")
        page.wait_for_selector("text=Governance Audit Trail")

        # Step 11: Feedback
        page.goto(f"{web_url}/feedback")
        page.wait_for_selector("text=Research Feedback Bridge")

        # Step 12: Audit Provenance Trace
        page.goto(f"{web_url}/audit")
        page.wait_for_selector("text=Audit Provenance Explorer")
        page.fill("input[placeholder*='Search by strategy_id']", "STRAT-a161a6c1070d45d0c1d89b8d")
        page.click("button:has-text('Trace Provenance')")
        page.wait_for_timeout(1000)
        assert page.locator("text=Type A").count() > 0

        # Step 13: Sign Out
        page.click("button[title='Sign out']")
        page.wait_for_url("**/login", timeout=5000)
        assert page.locator("text=Sign In").count() > 0

        context.close()
        browser.close()
