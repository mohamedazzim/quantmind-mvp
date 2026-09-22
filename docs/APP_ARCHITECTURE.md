# QuantMind Application Architecture & Productization Specification

## 1. Architectural Philosophy & Invariants

QuantMind is an institutional-grade quantitative research, backtesting, paper evaluation, and governance platform. The core quantitative kernel (PRD v1.0 through v4.0) is **closed, authoritative, and deterministic**.

The productization layer turns this closed core into a usable end-user application without mutating or bypassing domain invariants:

```text
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
│   data/quantmind_core.db (Authoritative Ledger)                         │
│   - 20 Domain Tables • 16 Immutability Triggers • Append-Only Ledgers   │
│                                                                         │
│   data/quantmind_app.db (Application Metadata)                          │
│   - users • sessions • jobs • app_audit_logs                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Five Immutable Boundary Principles

1. **Deterministic Core Invariance**:
   The API and frontend never execute backtests directly in the HTTP request cycle. Computational trials and simulated replays pass through the asynchronous `JobOrchestrator` and are executed by the dedicated `QuantMindWorker`.
2. **Direct Ledger Write Prohibition**:
   No HTTP handler or application service performs direct SQL updates or deletes on authoritative core tables. State transitions are exclusively managed by domain services (`StrategyRegistry`, `StrategyValidationGate`, `PaperGovernanceService`, `PaperLedger`).
3. **Database Segregation**:
   All user credentials, authentication tokens, job statuses, and user audit logs reside strictly in `quantmind_app.db`. All mathematical backtests, strategy specifications, qualification certificates, baseline bindings, and execution ledgers reside in `quantmind_core.db`.
4. **Non-Optimistic Governance Barrier**:
   A strategy cannot be promoted to `PAPER_ACTIVE` without an established, cryptographically bound evaluation baseline in `evaluation_baselines`. Any attempt to force activation fails closed with HTTP 422 (`GOVERNANCE_INTEGRITY_VIOLATION`).
5. **Multi-Milestone Provenance Guarantee**:
   All entities (Strategy, Spec, Qualification Record, Baseline, Regime, Replay Report, Snapshot, Degradation Event, Governance Transition, Feedback Record, Research Task) form a connected Directed Acyclic Graph (DAG) verifiable across 5 cryptographic edge types.

---

## 3. Storage & Ledger Separation

### Authoritative Ledger (`data/quantmind_core.db`)
Encapsulates 20 authoritative domain tables with 16 append-only SQLite triggers preventing deletion or modification:
- `datasets`, `dataset_zones`, `dataset_split_manifests`
- `strategies`, `strategy_qualifications`
- `trials`, `trial_artifacts`
- `paper_orders`, `paper_fills`, `paper_positions`, `paper_risk_events`, `paper_reports`
- `evaluation_baselines`, `paper_evaluation_regimes`, `monitoring_snapshots`, `degradation_events`, `paper_evaluation_transitions`, `research_feedback`

### Application Database (`data/quantmind_app.db`)
Encapsulates user session management, operational queue state, and application audit history:
- `users`: User identity, PBKDF2 salt and hash, role (`ADMIN`, `RESEARCHER`, `VIEWER`), active state, password rotation requirement.
- `jobs`: Background job queue (`JOB-<hex>`, type, status: `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, progress %, result payload, result reference).
- `app_audit_logs`: Audit log of user-initiated operations (`STRATEGY_ACTIVATE`, `JOB_SUBMIT`, etc.).

---

## 4. Authentication & Role-Based Access Control (RBAC)

- **Password Hashing**: PBKDF2 with SHA-256, 100,000 iterations, unique 16-byte random salt per user.
- **Session Tokens**: Cryptographically signed JWT tokens with 24-hour expiration, containing user ID, username, and role.
- **Credential Hygiene**:
  - Zero hardcoded production passwords in source code or configuration defaults.
  - Production admin bootstrap is conducted via interactive CLI (`quantmind.app.cli bootstrap-admin`) or environment variables.
  - Optional deterministic demo accounts (`demo_admin`, `demo_researcher`, `demo_viewer`) are strictly restricted to `APP_ENV=DEMO`.
- **Role Permissions**:
  - `ADMIN`: Full access (Lifecycle activation, strategy retirement, user creation, worker control).
  - `RESEARCHER`: Spec validation, trial submission, candidate qualification, re-research transitions.
  - `VIEWER`: Read-only access to dashboard, ledger views, reports, and provenance graphs.

---

## 5. Job Orchestration & Worker Architecture

Long-running quantitative workflows run asynchronously through a decoupled architecture:

```text
FastAPI Endpoint                     SQLite App DB                    Worker Process
       │                                   │                                 │
       ├──── Enqueue Job (QUEUED) ────────>│                                 │
       │     Return job_id immediately     │                                 │
       │                                   │<── Claim Next Job (RUNNING) ────┤
       │                                   │                                 ├─ Execute Domain Task
       │                                   │<── Update Progress (40%, 90%) ──┤
       │                                   │                                 ├─ Save Result / Trial ID
       │                                   │<── Complete Job (COMPLETED) ────┤
       │                                   │                                 │
       ├──── GET /api/v1/jobs/{id} ───────>│                                 │
       │     Returns status & result_ref   │                                 │
```

### Stale Job Recovery
If a worker daemon terminates abruptly (kill, power loss, OOM):
1. Upon restarting, `QuantMindWorker.startup()` executes an atomic recovery query.
2. All jobs left in `RUNNING` assigned to inactive workers are transitioned to `FAILED`.
3. Error messages are updated to `"Worker process terminated unexpectedly; job failed safely during recovery"`.

---

## 6. Five-Type Cryptographic Provenance DAG

The `AuditAdapter` inspects and stitches all domain evidence into a connected DAG:

```text
[Dataset Version] ──(Type E)──> [Split Manifest] ──(Type E)──> [Split Zone: FORWARD_PAPER]
                                                                        │
[Declarative Spec] ──(Type B)──> [Strategy ID]                         │ (Type A)
                                       │                                │
                                  (Type D)                              ▼
                                       ▼                        [Replay Report]
                       [Qualification Record (DSR/EICT)]                │
                                       │                                │ (Type A)
                                  (Type A)                              ▼
                                       └───────────────────────> [Evaluation Baseline]
                                                                        │
                                                                   (Type A)
                                                                        ▼
                                                             [Forward Regime]
                                                                        │
                                                                   (Type A)
                                                                        ▼
                                                             [Monitoring Snapshot]
                                                                        │
                                                                   (Type A)
                                                                        ▼
                                                             [Degradation Event]
                                                                        │
                                                                   (Type A)
                                                                        ▼
                                                             [Feedback Record]
                                                                        │
                                                                   (Type C)
                                                                        ▼
                                                             [Zero-Trial Research Task]
```

### Edge Classifications
- **Type A (Cryptographic Digest / Hash Binding)**: The parent entity's SHA-256 digest is embedded into the child record's payload before computing the child's digest.
- **Type B (Foreign Key / Entity Association)**: Direct database key linkage (e.g., Order to Strategy, Fill to Order).
- **Type C (Zero-Trial Semantic Derivation)**: Deterministic function deriving task identity from feedback without executing additional backtests.
- **Type D (Lifecycle State Precondition)**: State machine transition constraint requiring certified qualification records before promotion.
- **Type E (Dataset Zone Isolation)**: Temporal split boundaries ensuring research, validation, and holdout data remain non-overlapping.
