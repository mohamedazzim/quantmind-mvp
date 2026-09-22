# Local Development & Deployment Guide

This guide walks through setting up, running, testing, and deploying the complete QuantMind application stack.

---

## 1. Prerequisites

- **Python**: `3.10.x` or higher (e.g. `Python 3.10.11`)
- **Node.js**: `v18.x` or `v20.x` or higher (e.g. `v20.x`, `v24.x`)
- **npm**: `v9.x` or higher
- **Docker & Docker Compose** (Optional, for containerized deployment)

---

## 2. Quick Setup

### Python Environment
```bash
# Clone or navigate to the repository
cd quantmind-working

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies in editable mode
pip install -e .
```

### Next.js Frontend Environment
```bash
cd web
npm install
npm run build   # Verifies TypeScript and Next.js App Router compilation
cd ..
```

---

## 3. Configuration & Environment Modes

QuantMind supports three operational modes via the `APP_ENV` environment variable:

| Mode | Description | Credentials / Demo Seeding |
| :--- | :--- | :--- |
| `PRODUCTION` | Strict institutional execution. | Zero default users. Admin MUST be bootstrapped via CLI. |
| `DEVELOPMENT`| Local developer mode. | Permissive CORS, verbose logging. |
| `DEMO` | Platform demonstration & evaluation. | Pre-seeds demo users and realistic market data. |

---

## 4. Bootstrapping Admin Credentials

In `PRODUCTION` or `DEVELOPMENT`, initialize the primary administrator account using the secure interactive CLI:

```bash
# Interactive prompt (password hidden)
python -m quantmind.app.cli bootstrap-admin --username admin --email admin@firm.com

# Non-interactive / CI (minimum 12 characters)
python -m quantmind.app.cli bootstrap-admin --username admin --email admin@firm.com --password "InstitutionalSecure2026!"
```

---

## 5. Seeding Demo Environment

To populate realistic market datasets, a qualified strategy (`PAPER_ACTIVE`), baseline reports, simulated orders, and monitoring snapshots:

```bash
# Run the authoritative demo seed script
python scripts/seed_demo_data.py --force
```

### Pre-Seeded Demo Credentials (`APP_ENV=DEMO`)
- **Admin**: `demo_admin` / `QuantMindDemoAdmin2026!`
- **Researcher**: `demo_researcher` / `QuantMindDemoResearch2026!`
- **Viewer**: `demo_viewer` / `QuantMindDemoViewer2026!`

---

## 6. Running Services Locally

Running the complete application stack requires three processes:

### Terminal 1: FastAPI Gateway
```bash
uvicorn quantmind.api.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000/api/v1`
- Swagger Docs: `http://127.0.0.1:8000/docs`

### Terminal 2: Dedicated Background Worker
```bash
python -m quantmind.app.worker
```
- Processes research trials, qualification gates, and forward paper replays.
- Automatically performs crash recovery of stale jobs on startup.

### Terminal 3: Next.js Frontend
```bash
cd web
npm run dev
```
- Web UI: `http://localhost:3000`

---

## 7. Containerized Deployment (Docker Compose)

Run the complete multi-tier stack with persistent volume storage in a single command:

```bash
# Build and run API, Worker, and Web UI
docker-compose up --build -d

# Check service logs
docker-compose logs -f

# Shut down
docker-compose down
```

---

## 8. Running Automated Tests

```bash
# Run complete test suite (740+ tests across core kernel, API, and E2E)
pytest

# Run FastAPI and Worker API test suite
pytest tests/api/test_api_endpoints.py -v

# Run 15-step End-to-End lifecycle test
pytest tests/e2e/test_e2e_workflow.py -v

# Run Playwright headless browser test
pytest tests/e2e/test_playwright_ui.py -v
```
