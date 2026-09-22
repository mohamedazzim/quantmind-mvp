# QuantMind MVP

**Futures-first AI-assisted quantitative research laboratory.**

QuantMind is an institutional-grade platform for deterministic quantitative research, backtesting, paper (simulated) trading, and lifecycle governance. It pairs a closed, deterministic research kernel with a productized FastAPI + Next.js application layer, enforcing cryptographic provenance, immutable append-only ledgers, and strict split/embargo discipline across the entire strategy lifecycle — from idea to paper-eligible deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Technology Stack](#technology-stack)
5. [Quickstart](#quickstart)
6. [Configuration](#configuration)
7. [Application Modes](#application-modes)
8. [Authentication & RBAC](#authentication--rbac)
9. [Data Storage Model](#data-storage-model)
10. [Research Integrity Pipeline](#research-integrity-pipeline)
11. [Paper Trading & Evaluation](#paper-trading--evaluation)
12. [API Reference](#api-reference)
13. [Testing](#testing)
14. [Deployment](#deployment)
15. [Security Posture](#security-posture)
16. [Documentation Index](#documentation-index)
17. [Roadmap & Known Scope](#roadmap--known-scope)

---

## Overview

QuantMind addresses the core failure modes of overfit quantitative research by making every step of the strategy lifecycle **deterministic, auditable, and gated**:

- **Deterministic research kernel** — reproducible backtests with a next-bar-open execution model, purge/embargo split boundaries, and a sealed final holdout that no research agent can touch.
- **Cryptographic provenance** — every strategy, qualification record, baseline, report, and feedback record carries a SHA-256 digest over its canonical representation, forming a verifiable directed acyclic graph (DAG).
- **Append-only ledgers** — SQLite tables protected by database triggers that abort any `DELETE` or `UPDATE`, so results cannot be retroactively edited.
- **Statistical rigor** — effective trial count via hierarchical correlation clustering (EICT-CORR-1) and Deflated Sharpe Ratio (Bailey & López de Prado, 2014) to correct for multiple-testing bias.
- **Strict no-live-trading boundary** — execution is entirely simulated; there are no broker endpoints, credentials, or live order routing anywhere in the codebase.

The current build stage is **PRD v4.0 — Full Four-Quadrant Research, Evaluation, Governance & Feedback Architecture**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Next.js 14 App Router (Web UI)                      │
│      Dark Institutional Interface • TypeScript • Tailwind CSS • React   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP / REST (JWT Auth)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application Gateway                        │
│         OpenAPI Documentation • Dependency Injection • RBAC             │
└─────────────┬─────────────────────────────────────────────┬─────────────┘
              │                                             │
              ▼                                             ▼
┌───────────────────────────┐                 ┌───────────────────────────┐
│  Application Services     │                 │   Job Orchestrator        │
│  AuthService • Audit      │                 │   Persistent SQLite Queue │
│  Dashboard Aggregator     │                 └─────────────┬─────────────┘
└─────────────┬─────────────┘                               │
              │                                             ▼
              │                               ┌───────────────────────────┐
              │                               │ Dedicated Background      │
              │                               │ Worker Daemon             │
              ▼                               └─────────────┬─────────────┘
┌───────────────────────────────────────────────────────────┴─────────────┐
│                      QuantMind v4.0 Core Domain Kernel                  │
│   ResearchHarness • StrategyValidationGate • PaperReplayEngine           │
│   EvaluationLedger • PaperGovernanceService • FeedbackBridge             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Separated Storage Architecture                       │
│                                                                         │
│   data/quantmind_core.db (Authoritative Ledger — 20 domain tables)      │
│   data/quantmind_app.db  (Application Metadata — users, jobs, audits)   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Five Immutable Boundary Principles

1. **Deterministic core invariance** — the API and frontend never execute backtests in the HTTP request cycle; trials and replays flow through the `JobOrchestrator` and are executed by the dedicated `QuantMindWorker`.
2. **Direct ledger-write prohibition** — no HTTP handler performs direct SQL updates/deletes on authoritative core tables; state transitions are owned exclusively by domain services.
3. **Database segregation** — credentials, tokens, job status, and user audit logs live only in `quantmind_app.db`; all mathematical results, specs, qualifications, baselines, and ledgers live only in `quantmind_core.db`.
4. **Non-optimistic governance barrier** — a strategy cannot reach `PAPER_ACTIVE` without a cryptographically bound evaluation baseline; attempts fail closed with HTTP 422 (`GOVERNANCE_INTEGRITY_VIOLATION`).
5. **Multi-milestone provenance** — all entities form a connected DAG verifiable across five cryptographic edge types (`Type A`–`Type E`).

---

## Repository Structure

```
quantmind-working/
├── configs/                        # Versioned research-protocol YAML (RP-1, RP-2)
├── docs/                           # PRD v3.1 → v4.0, architecture, API, dev guide, QA report
├── scripts/                        # Demo seeding + live-QA / domain-barrier verification
├── src/quantmind/
│   ├── domain/                     # Core dataclasses: Bar, Order, Fill, Trial, FuturesContract
│   ├── backtest/                   # Deterministic backtest engine + causal-signal guards
│   ├── strategy/                   # StrategySpec, compiler, registry + lifecycle state machine
│   ├── data/                       # Dataset registry, split manifests, purge/embargo zones
│   ├── synthetic/                  # Synthetic futures-bar generator + adversarial canaries
│   ├── research_integrity/         # Harness, holdout, population, EICT/DSR, qualification gate
│   ├── paper/                      # Paper replay engine, risk, ledger, evaluation subsystem
│   ├── testing/                     # Fixture-only research harness (never imported in production)
│   ├── app/                        # FastAPI services: auth, jobs, worker, adapters, config, CLI
│   └── api/                        # FastAPI routers + Pydantic schemas + error handlers
├── tests/                          # unit / integration / api / e2e (38 test files)
├── web/                            # Next.js 14 App Router frontend
├── Dockerfile.api                  # FastAPI service image
├── Dockerfile.worker               # Dedicated worker image
├── Dockerfile.web                  # Multi-stage Next.js image
├── docker-compose.yml              # Full three-tier stack
└── pyproject.toml                  # Python package + dependencies + pytest config
```

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.10+ |
| API | FastAPI (0.110+), Uvicorn, Pydantic v2, pydantic-settings |
| Frontend | Next.js 14 (App Router), React 18, TypeScript 5, Tailwind CSS 3 |
| Data | SQLite (dual-DB separation), NumPy, pandas, PyArrow, SciPy |
| Auth | PyJWT (HS256), PBKDF2-SHA256 (passlib-style, 100k iterations) |
| Async jobs | Persistent SQLite queue + dedicated polling worker daemon |
| Testing | pytest 8+, pytest-cov, Playwright (browser E2E) |
| Deployment | Docker + Docker Compose |
| Linting | Ruff |

---

## Quickstart

### Prerequisites

- Python `3.10.x` or higher
- Node.js `v18.x` / `v20.x` or higher, npm `v9.x`+
- Docker & Docker Compose (optional, for containerized deployment)

### Local development (three processes)

```bash
# 1. Install Python dependencies (editable)
pip install -e .

# 2. Install frontend dependencies
cd web && npm install && cd ..

# 3. Seed the demo environment (APP_ENV=DEMO)
python scripts/seed_demo_data.py --force

# 4. Start the API gateway (Terminal 1)
uvicorn quantmind.api.app:create_app --factory --reload --host 127.0.0.1 --port 8000

# 5. Start the worker daemon (Terminal 2)
python -m quantmind.app.worker

# 6. Start the web console (Terminal 3)
cd web && npm run dev
```

Visit `http://localhost:3000`. Swagger UI is at `http://127.0.0.1:8000/docs`.

### Docker Compose (single command)

```bash
docker-compose up --build -d
docker-compose logs -f      # follow logs
docker-compose down         # stop
```

---

## Configuration

Configuration is centralized in `src/quantmind/app/config.py` (`AppSettings`, a `pydantic-settings` model). Values are read from environment variables or a `.env` file.

| Environment variable | Default | Purpose |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | Operational mode: `PRODUCTION` \| `DEVELOPMENT` \| `DEMO` |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API bind port |
| `QUANTMIND_DATA_DIR` | `data` | Root data directory |
| `QUANTMIND_CORE_DB_PATH` | `data/quantmind_core.db` | Authoritative core-ledger SQLite path |
| `QUANTMIND_APP_DB_PATH` | `data/quantmind_app.db` | Application-metadata SQLite path |
| `AUTH_SECRET_KEY` | *(insecure dev default)* | JWT signing key — **must be overridden in production** |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000/api/v1` | Client-side API base URL (web build arg) |
| `INTERNAL_API_URL` | `http://api:8000` | Server-side proxy destination (Docker service name) |

> ⚠️ The default `AUTH_SECRET_KEY` is a documented-insecure placeholder. Always set `AUTH_SECRET_KEY` to a strong secret in `PRODUCTION`.

---

## Application Modes

| Mode | Description | Credentials |
| :--- | :--- | :--- |
| `PRODUCTION` | Strict institutional execution. | Zero default users — admin bootstrapped via CLI. |
| `DEVELOPMENT` | Local developer mode. | Permissive CORS, verbose logging. |
| `DEMO` | Platform demonstration & evaluation. | Pre-seeded demo users + realistic market data. |

### Bootstrapping the admin account

```bash
# Interactive (password hidden)
python -m quantmind.app.cli bootstrap-admin --username admin --email admin@firm.com

# Non-interactive / CI (min 12 chars)
python -m quantmind.app.cli bootstrap-admin --username admin --email admin@firm.com \
  --password "InstitutionalSecure2026!"
```

### Demo credentials (`APP_ENV=DEMO` only)

| Role | Username | Password |
| :--- | :--- | :--- |
| Admin | `demo_admin` | `QuantMindDemoAdmin2026!` |
| Researcher | `demo_researcher` | `QuantMindDemoResearch2026!` |
| Viewer | `demo_viewer` | `QuantMindDemoViewer2026!` |

---

## Authentication & RBAC

- **Password hashing** — PBKDF2 with SHA-256, 100,000 iterations, unique 16-byte random salt per user.
- **Session tokens** — signed JWT (`HS256`), 24-hour expiration, carrying user ID, username, and role.
- **Roles**:
  - `ADMIN` — full access: lifecycle activation, retirement, user creation, worker control.
  - `RESEARCHER` — spec validation, trial submission, qualification, re-research transitions.
  - `VIEWER` — read-only access to dashboards, ledgers, reports, and provenance graphs.
- **Credential hygiene** — zero hardcoded production passwords; demo accounts are strictly gated to `APP_ENV=DEMO`.

---

## Data Storage Model

Two strictly separated SQLite databases:

### Authoritative core ledger — `quantmind_core.db` (20 tables)

| Concern | Tables |
| :--- | :--- |
| Datasets & splits | `datasets`, `dataset_zones`, `dataset_split_manifests` |
| Strategies | `strategies`, `strategy_qualifications` |
| Trials | `trials`, `trial_artifacts` |
| Paper execution | `paper_orders`, `paper_fills`, `paper_positions`, `paper_risk_events`, `paper_reports` |
| Evaluation & governance | `evaluation_baselines`, `paper_evaluation_regimes`, `monitoring_snapshots`, `degradation_events`, `paper_evaluation_transitions`, `research_feedback` |
| Holdout security | `holdout_burns`, `holdout_evaluations` |

### Application metadata — `quantmind_app.db` (3 tables)

- `users` — identity, PBKDF2 salt/hash, role, active state, password-rotation flag.
- `jobs` — background queue (`JOB-<hex>`, type, status, progress %, result payload/ref).
- `app_audit_logs` — user-initiated operations (`STRATEGY_ACTIVATE`, `JOB_SUBMIT`, …).

Immutability is enforced with SQLite triggers (e.g. `paper_orders_no_delete`, `paper_orders_no_update`) that raise integrity errors on any modification attempt.

---

## Research Integrity Pipeline

The production research path is strictly sequential; callers never supply arbitrary DataFrames or file paths:

```
DatasetRegistry (checksum-verified, immutable)
  → SplitManager / SplitManifest (purged & embargoed boundaries)
  → ResearchHarness (LICENSED dataset, non-holdout zone, budget enforcement)
  → StrategyCompiler (whitelist compilation of declarative StrategySpec)
  → CausalityPreflight (causal signals across cut points)
  → BacktestEngine (deterministic next-bar-open execution)
  → ArtifactRegistry (immutable OOS returns Parquet + SHA-256 digest)
  → TrialLedger (append-only, immutable trials)
  → ProductionPopulationQuery (authoritative production trial filter)
  → EictCorr1Calculator (hierarchical average-linkage clustering)
  → DeflatedSharpeCalculator (Bailey & López de Prado DSR)
  → StrategyValidationGate (protocol sample-size + holdout verification)
  → QualificationLedger (append-only, delete/update-abort triggers)
  → StrategyRegistry (state machine; PAPER_ELIGIBLE requires qualification)
  → PaperReplayEngine (PAPER_ELIGIBLE + PASSED holdout)
  → ReplayFeed (normalized, vectorized streaming)
  → PaperRiskEngine (7 deterministic pre-trade risk rules)
  → PaperLedger (append-only orders/fills/positions/risk events)
  → ReplayReport (deterministic, sealed with SHA-256 report_hash)
```

### Split & holdout discipline

- **Split zones**: `RESEARCH`, `VALIDATION`, `FINAL_HOLDOUT`, `FORWARD_PAPER`.
- **Purge & embargo** windows derived from temporal dependencies prevent label overlap and serial-correlation leakage.
- **Sealed `FINAL_HOLDOUT`** — standard research APIs cannot request holdout data; candidates may only evaluate via the dedicated `final_evaluate()` gateway after passing earlier gates.
- **Holdout state machine**: `UNTOUCHED` → `EVALUATED` → `PASSED` / `FAILED` / `BURNED`; failed candidates are permanently blocked from re-tuning against that holdout.

### Statistical validation

- **EICT-CORR-1** — effective independent trial count via hierarchical clustering with average linkage at distance threshold 0.6325; sparse trials (< 100 common observations) isolated as singleton clusters.
- **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado (2014) formulation adjusting for selection bias, effective trial count, non-normality, and sample length.

### Strategy lifecycle state machine

```
IDEA → RESEARCH → VALIDATION → PAPER_ELIGIBLE → PAPER_ACTIVE → DEGRADED → RETIRED
                      │              │              │
                      ▼              ▼              ▼
                  REJECTED       REJECTED       REJECTED
```

`PAPER_ELIGIBLE` strictly requires a cryptographically verified `StrategyQualificationRecord` with `final_status == PAPER_ELIGIBLE` and `holdout_state == PASSED`. Illegal shortcuts (e.g. `REJECTED → PAPER_ELIGIBLE`) are permanently barred.

---

## Paper Trading & Evaluation

The paper subsystem simulates execution over the closed kernel and tracks post-deployment health:

- **`PaperReplayEngine`** — deterministic forward replay with qualification gating, provenance verification (spec hash, strategy ID, dataset checksum, protocol version), and synthetic/fixture dataset barring.
- **`ReplayFeed`** — normalized `MarketDataFeed` interface with pre-extracted NumPy arrays; strictly prohibits access to sealed `FINAL_HOLDOUT` partitions.
- **`PaperRiskEngine`** — 7 deterministic pre-trade risk rules: `kill_switch`, `max_order_quantity`, `max_position`, `max_trades_per_session`, `max_daily_loss`, `max_strategy_drawdown`, `max_exposure`.
- **`PaperLedger`** — append-only tables for orders, fills, positions, risk events, and reports.
- **`EvaluationLedger`** — baselines, regimes, monitoring snapshots, and degradation events.
- **`PaperGovernanceService`** — lifecycle state machine with causal ordering and flat-retirement guards.
- **`ResearchFeedbackBridge`** — zero-trial post-mortem records and derived research tasks.

**Strict no-live-trading boundary** — execution is entirely simulated; there are zero broker endpoints or credentials in the codebase.

---

## API Reference

All routes are mounted under `/api/v1` and documented at `http://localhost:8000/docs` (Swagger) and `/redoc`.

| Resource | Representative endpoints |
| :--- | :--- |
| Auth | `POST /auth/login`, `GET /auth/me`, `POST /auth/logout`, `POST /auth/change-password`, `GET /auth/audit-logs` |
| Dashboard | `GET /dashboard/kpis` |
| Datasets | `GET /datasets`, `GET /datasets/enums`, `GET /datasets/{version}` |
| Strategies | `GET /strategies`, `GET /strategies/spec-schema`, `GET /strategies/{id}`, `POST /strategies/{id}/activate` (ADMIN), `/retire` (ADMIN), `/re-research` |
| Research | `POST /research/candidates/validate`, `POST /research/trials/submit`, `GET /research/tasks`, `GET /research/tasks/{id}` |
| Trials | `GET /trials`, `GET /trials/{id}`, `GET /trials/population/summary` |
| Qualification | `GET /qualification`, `GET /qualification/{strategy_id}` |
| Paper | `GET /paper/overview|orders|fills|positions|risk|reports`, `POST /paper/replay/submit` |
| Monitoring | `GET /monitoring`, `GET /monitoring/{strategy_id}` |
| Governance | `GET /governance/transitions`, `GET /governance/{strategy_id}` |
| Feedback | `GET /feedback`, `GET /feedback/{hash}`, `POST /feedback/{hash}/research-task` |
| Audit | `GET /audit/graph/{identifier}`, `GET /audit/{identifier}` |
| Jobs | `GET /jobs`, `GET /jobs/{job_id}` |

Health/readiness: `GET /health`, `GET /ready`.

See [docs/API.md](docs/API.md) for the complete reference.

---

## Testing

```bash
pip install -e ".[dev]"     # install pytest, pytest-cov, ruff

pytest                                        # full suite (745 tests)
pytest tests/e2e/test_e2e_workflow.py -v      # 15-step quantitative lifecycle
pytest tests/e2e/test_playwright_ui.py -v     # browser navigation (requires playwright + chromium)
pytest tests/api/test_api_endpoints.py -v     # FastAPI + worker endpoints
```

The suite spans **38 test files** across unit, integration, API, and E2E layers, with comprehensive adversarial integrity tests covering: holdout security, purge/embargo boundaries, look-ahead canaries, ledger immutability, provenance tamper-resistance, statistical validation bypass guards, and cross-milestone governance.

> **Verified:** `745 tests passed` (excludes the 2 Playwright browser tests, which require `pip install playwright && python -m playwright install chromium`).

---

## Deployment

### Docker Compose (recommended)

Three services with persistent volume storage:

| Service | Image | Port | Role |
| :--- | :--- | :--- | :--- |
| `api` | `Dockerfile.api` | 8000 | FastAPI gateway (healthcheck via `/health`) |
| `worker` | `Dockerfile.worker` | — | Background job daemon (crash recovery on startup) |
| `web` | `Dockerfile.web` | 3000 | Next.js production build (multi-stage, non-root user) |

The web container proxies `/api/v1/*` server-side through the App Router catch-all (`web/src/app/api/v1/[...path]/route.ts`) reading `INTERNAL_API_URL` at request time, avoiding Next.js build-time rewrite limitations.

```bash
docker-compose up --build -d
```

### Production hardening checklist

1. Set a strong `AUTH_SECRET_KEY`.
2. Run `APP_ENV=PRODUCTION` (zero default users).
3. Bootstrap admin via `python -m quantmind.app.cli bootstrap-admin`.
4. Mount `quantmind_data` to durable storage.

---

## Security Posture

- **Role-based access control** on state-changing endpoints (activation, retirement, trial submission, feedback-task creation).
- **Append-only ledgers** with SQLite triggers aborting deletes/updates on authoritative tables.
- **Cryptographic provenance** — SHA-256 digests binding every entity to its ancestors.
- **Holdout burn protection** — failed candidates cannot re-tune; burned holdouts block all research/tuning.
- **No live-trading surface** — no broker endpoints, credentials, or order routing exist in the codebase.
- **PBKDF2 password hashing** and signed JWT sessions with configurable expiration.

> In-process Python trust limitations (e.g. direct private-method invocation or SQLite file permissions) are documented and slated for future hardening via PostgreSQL role-level security.

---

## Documentation Index

| Document | Purpose |
| :--- | :--- |
| [docs/PRD_v4.0.md](docs/PRD_v4.0.md) | PRD v4.0 architectural specification & closure record |
| [docs/APP_ARCHITECTURE.md](docs/APP_ARCHITECTURE.md) | Productization architecture, storage segregation, provenance DAG |
| [docs/API.md](docs/API.md) | REST API endpoint reference |
| [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md) | Setup, seeding, and deployment guide |
| [docs/LIVE_APPLICATION_QA_REPORT.md](docs/LIVE_APPLICATION_QA_REPORT.md) | Live-application QA findings |
| [configs/research_protocol_v2.yaml](configs/research_protocol_v2.yaml) | Current research protocol (RP-2) |

---

## Roadmap & Known Scope

The backtester is currently a deterministic **one-bar-hold regression harness** (not yet the complete multi-bar execution engine). Research trials enter only through `ResearchHarness.run_trial()`. Explicitly **out of scope / later work**:

- Position-state management, stop/target handling, intrabar adverse-first resolution, and futures roll handling.
- PostgreSQL role-level security (currently SQLite + in-process trust).
- `EVALUATION_WINDOW` job type (declared but not yet implemented by the worker).

Synthetic fixtures live in `quantmind.testing` and are recorded with `mode=FIXTURE`; production trials accept only named `LICENSED` dataset versions and split zones.

---

**Stage:** PRD v4.0 complete · **Version:** 4.0.0 · **License:** not specified in-repo
