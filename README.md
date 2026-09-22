# QuantMind MVP

Futures-first AI-assisted quantitative research laboratory.

## Current build stage

**Stage: QuantMind PRD v4.0 — Full Four-Quadrant Research, Evaluation, Governance & Feedback Architecture**

Current verification: **732 tests passing** across synthetic market generation, real gap/wick semantics, cross-session surrogates, directional intraday nulls, recomputable causal positive controls, independent wick preservation, paired edge recovery, one-sided confidence-bound gates, look-ahead canary, checksum-verified Dataset Registry, immutable SplitManifest, purge and embargo boundaries, sealed final holdout security, holdout state machine, comprehensive adversarial integrity tests, immutable OOS return Parquet artifacts, SHA-256 byte verification, production population accounting, EICT-CORR-1 hierarchical clustering, Deflated Sharpe Ratio multiple-testing adjustment (Bailey & López de Prado 2014), permutation-invariant clustering, multiple-testing anti-bypass guards, deterministic Strategy Validation Gate, immutable Strategy Qualification Records with canonical SHA-256 digests, append-only SQLite qualification ledger triggers, Strategy Registry lifecycle state machine, normalized MarketDataFeed streaming, deterministic PaperReplayEngine execution, pre-trade risk controls, append-only PaperLedger, deterministic SHA-256 ReplayReport verification, append-only EvaluationLedger (6 tables with trigger immutability), PaperEvaluationBaseline & Regime registration, periodic MonitoringSnapshots, DegradationEvent detection, PaperGovernanceService lifecycle state machine with causal ordering and flat retirement guards, ResearchFeedbackRecord post-mortem capture, zero-trial ResearchFeedbackBridge, and full 26-scenario cross-milestone adversarial verification.

See [docs/PRD_v4.0.md](docs/PRD_v4.0.md) for the complete PRD v4.0 architectural specification and closure record.

## Milestones (PRD v4.0 Complete)

1. Core domain models
2. Synthetic futures-bar generator
3. Cross-session structure-preserving surrogate
4. Hypothesis-specific directional null
5. Observable symmetric positive-control injector
6. Gap-only look-ahead canary
7. Confidence-bound acquisition gate sourced from YAML
8. Deterministic next-bar-open futures backtest regression harness
9. ResearchHarness + append-only trial ledger + cumulative research budget
10. Declarative StrategySpec + typed whitelist compiler; arbitrary agent signal code is excluded from production trials
11. Checksum-verified Dataset Registry + named split-zone loader; production harness accepts LICENSED data only
12. Purged/embargoed split manager and sealed holdout (v3.5)
13. Research population accounting + trial return artifacts + EICT-CORR-1 / DSR inputs (v3.6)
14. Statistical validation: EICT-CORR-1 + Deflated Sharpe Ratio (v3.7 / v3.7.1)
15. Strategy Validation Gate + Immutable Qualification Record (v3.8)
16. Deterministic Paper Replay Engine + Normalized Replay Feed + Pre-Trade Risk Controls (v3.9)
17. Paper Evaluation Ledger, Baselines, Regimes & Monitoring Snapshots (PRD v4.0 M5.3)
18. Paper Evaluation State Machine, Lifecycle Execution Gating & Governance (PRD v4.0 M6)
19. Research Feedback Bridge & Observational Post-Mortem Records (PRD v4.0 M7)
20. Final Documentation & Comprehensive Cross-Milestone Adversarial Verification (PRD v4.0 M8)


## Production Research Architecture

The production research path is strictly sequential and callers never supply arbitrary DataFrames or file paths:

```text
DatasetRegistry (checksum-verified, immutable)
    ↓
SplitManager / SplitManifest (purged & embargoed boundaries)
    ↓
ResearchHarness (enforces LICENSED dataset, non-holdout split zone, budget)
    ↓
StrategyCompiler (whitelist compilation of declarative StrategySpec)
    ↓
CausalityPreflight (asserts causal signals across cut points)
    ↓
BacktestEngine (deterministic next-bar-open execution)
    ↓
ArtifactRegistry (persists immutable OOS returns Parquet with SHA-256 digest)
    ↓
TrialLedger (append-only, immutable completed trials with split_zone & artifact hash)
    ↓
ProductionPopulationQuery (authoritative production trial filter)
    ↓
EictCorr1Calculator (hierarchical average-linkage clustering, singleton isolation)
    ↓
DeflatedSharpeCalculator (Bailey & López de Prado DSR formulation)
    ↓
StrategyValidationGate (authoritative gate, protocol sample size requirements, holdout verification)
    ↓
QualificationLedger (append-only SQLite table with delete/update abort triggers)
    ↓
StrategyRegistry (lifecycle state machine enforcing qualification record for PAPER_ELIGIBLE)
    ↓
PaperReplayEngine (accepts PAPER_ELIGIBLE + PASSED holdout, checks spec & dataset hash)
    ↓
ReplayFeed (normalized MarketDataFeed interface, fast vectorized streaming)
    ↓
PaperRiskEngine (evaluates 7 deterministic pre-trade risk rules)
    ↓
PaperLedger (append-only SQLite ledger with triggers: orders, fills, positions, risk events)
    ↓
ReplayReport (deterministic observational report sealed with SHA-256 report_hash)
```

### Deterministic Paper Replay Engine, Replay Feed, & Risk Controls (v3.9)

- **`MarketDataFeed` & `ReplayFeed`**: Normalized market data feed interface with high-performance zero-overhead streaming via pre-extracted NumPy arrays. Tracks session boundaries, supports pause/resume/reset, and strictly prohibits access to sealed `FINAL_HOLDOUT` partitions (`MarketFeedSecurityError`).
- **Qualification Gating & Provenance Verification**: Replay strictly requires an authoritative `StrategyQualificationRecord` with `final_status == ValidationStatus.PAPER_ELIGIBLE` and `holdout_state == "PASSED"`. Verifies cryptographic record digest, strategy specification hash match, strategy ID match, dataset checksum match, and protocol version compatibility. Synthetic/fixture datasets are barred from production paper replay.
- **Deterministic Execution Model (`next_bar_open_v1`)**: Causal execution executing bar $t$ signal at bar $t+1$ open with realistic slippage modeling (`slippage_bps_per_side`), price quantization to instrument tick size (e.g. 0.05 for NIFTY futures), date-effective cost schedule resolution (`CostSchedule`), and multi-bar holding duration.
- **Pre-Trade Deterministic Risk Engine (`PaperRiskEngine`)**: Evaluates 7 deterministic pre-trade risk rules (`kill_switch`, `max_order_quantity`, `max_position`, `max_trades_per_session`, `max_daily_loss`, `max_strategy_drawdown`, `max_exposure`). Rejections generate immutable `PaperRiskEvent` audit records and set order status to `REJECTED`.
- **Append-Only Paper Ledger (`PaperLedger`)**: Backed by SQLite tables (`paper_orders`, `paper_fills`, `paper_positions`, `paper_risk_events`, `replay_reports`) protected by database triggers preventing deletions or updates.
- **Deterministic Observational Report (`ReplayReport`)**: Produces observational performance summaries (gross/net P&L, costs, slippage, max drawdown, exposure, win rate, expectancy, Sharpe) sealed with a deterministic SHA-256 `report_hash` over its canonical JSON representation.
- **Strict No-Live-Trading Boundary**: Zero broker network endpoints, credentials, or live order routing. Execution is entirely simulated.

### Strategy Validation Gate & Immutable Qualification Record (v3.8)

- **`StrategyValidationGate`**: Deterministic gate consuming authoritative system evidence only (`DatasetRegistry`, `SplitManifest`, `HoldoutManager`, `TrialLedger`, `ProductionPopulationQuery`, `EICT-CORR-1`, `DSR`, and `StrategySpec`). Rejects any manual statistical overrides (`manual_sharpe`, `manual_dsr`, `manual_trials`). Dynamically evaluates protocol-defined sample size requirements (`minimum_trades_validation`, `minimum_effective_outcome_observations`).
- **`StrategyQualificationRecord`**: Frozen dataclass containing complete provenance, observed Sharpe, DSR, EICT, trade count, holdout state, robustness status, and validation status, sealed by a SHA-256 digest over its canonical JSON.
- **`QualificationLedger`**: Append-only SQLite ledger with triggers raising SQL integrity errors on any `DELETE` or `UPDATE` attempt.
- **Validation State Machine**: Enforces legal transitions (`CANDIDATE` -> `UNDERPOWERED` / `VALIDATION` / `REJECTED` / `HOLDOUT_REQUIRED` -> `HOLDOUT_PASSED` -> `PAPER_ELIGIBLE`). Illegal shortcuts (e.g. `REJECTED` -> `PAPER_ELIGIBLE` or `UNDERPOWERED` -> `PAPER_ELIGIBLE`) are permanently barred.
- **`StrategyRegistry`**: Tracks 8 authoritative lifecycle states (`IDEA`, `RESEARCH`, `VALIDATION`, `REJECTED`, `PAPER_ELIGIBLE`, `PAPER_ACTIVE`, `DEGRADED`, `RETIRED`). Transition to `PAPER_ELIGIBLE` strictly requires an authoritative, cryptographically verified `StrategyQualificationRecord` with `final_status == ValidationStatus.PAPER_ELIGIBLE` and `holdout_state == "PASSED"`.
- **`PaperReplayEligibility`**: Cryptographically verifies qualifications to authorize deterministic forward replay testing without starting live broker trading.



### Statistical Validation: EICT-CORR-1 + Deflated Sharpe Ratio (v3.7)

- **LAYER A — EICT-CORR-1**: Computes effective independent trial count via hierarchical clustering with average linkage at distance threshold $0.6325$ ($\sqrt{2(1 - 0.80)}$). Sparse trials ($< 100$ common observations) are isolated as singleton clusters. Outputs deterministic cluster mappings and cryptographic `population_hash`.
- **LAYER B — Deflated Sharpe Ratio (DSR)**: Implements Bailey & López de Prado (2014) formulation adjusting for selection bias, effective trial count ($N = \text{effective\_trial\_count}$), non-normality (skewness, Fisher excess kurtosis), and sample length $T$.
- **Multiple-Testing Guard**: Rigid architectural enforcement barring manual overrides of `observed_sharpe` or `effective_trial_count`. All statistical evaluations must anchor to an authoritative `ProductionPopulationQuery`.
- **Integrated Pipeline**: `StatisticalValidationPipeline` automates population loading, EICT clustering, and candidate DSR evaluation.

### Research Population Accounting & OOS Return Artifacts (v3.6)

- **OOS Return Artifacts**: Every completed `PRODUCTION` trial records its complete trade stream into an immutable Parquet file matching `_OOS_RETURNS_SCHEMA`.
- **Cryptographic Auditability**: Each artifact has a SHA-256 digest verified at load time. Files and database registry records cannot be updated or deleted (guaranteed by SQLite triggers and `ArtifactImmutabilityError`).
- **Production Population Eligibility**: Only terminal `COMPLETED` production trials within the scope `(dataset_version, research_protocol_version)` with verified artifacts enter the research population. Fixture, failed, non-causal, abandoned, or holdout-rejected trials are permanently barred.
- **EICT-CORR-1 Inputs**: Aligns common timestamps across pairs, computes Pearson correlation and distance $d = \sqrt{2(1-r)}$, and isolates singleton clusters when common observations fall below 100. Protocol constants are verified and protected against runtime mutation.
- **DSR Inputs**: Generates return distribution summaries (mean, std, skewness, Fisher kurtosis, observed Sharpe, effective observations) as the input layer for future Deflated Sharpe Ratio calculation.

### Split Manager & Sealed Holdout (v3.5)

- **`SplitManifest`**: Defines `RESEARCH`, `VALIDATION`, `FINAL_HOLDOUT`, and `FORWARD_PAPER` zones with versioned, immutable boundaries.
- **Purge & Embargo**: Purge windows are derived from temporal dependencies (`feature_lookback + prediction_horizon + holding_period + forward_dependency`), preventing overlapping label windows. Embargo buffers prevent serial correlation leakage.
- **High-Performance Zone Loading**: Extracted via numpy boolean masks (no `.iloc` loops) with in-memory caching and SHA-256 verification.
- **Sealed `FINAL_HOLDOUT`**: Standard research APIs and agents (`load_zone`, `ResearchHarness.run_trial`) are prohibited from requesting holdout data.
- **Dedicated `final_evaluate()` Gateway**: Candidates can only evaluate holdout if they have passed earlier required gates (completed `RESEARCH` or `VALIDATION` trial).
- **Holdout State Machine**: Tracks `UNTOUCHED`, `EVALUATED`, `PASSED`, `FAILED`, and `BURNED`. Failed candidates are permanently blocked from retuning or re-evaluating against that holdout dataset. Burned holdouts cannot be evaluated or tuned against.
- **Trust Boundary**: The production boundary is enforced through `ResearchHarness` + `DatasetRegistry` + `StrategyCompiler` + `TrialLedger`. In-process Python trust limitations (e.g. direct private method invocation or SQLite file permissions) will be hardened in future releases via PostgreSQL role-level security.

## Productized Application Layer (FastAPI + Next.js + Dedicated Worker)

The QuantMind application builds an interactive institutional platform around the closed v4.0 research kernel:

- **Next.js 14 Web Interface**: Modern, dark quantitative console (`web/`) with 16 App Router views:
  - `/dashboard`: Real-time portfolio KPIs, active paper strategies, recent degradations, and governance feed.
  - `/strategies`: Strategy registry, canonical specification inspection, and 8-state lifecycle manager.
  - `/datasets`: Versioned licensed and synthetic dataset catalog, temporal split boundaries (`RESEARCH`, `VALIDATION`, `FINAL_HOLDOUT`, `FORWARD_PAPER`).
  - `/research`: Declarative strategy candidate builder with compiler validation and asynchronous trial dispatch.
  - `/trials`: Immutable trial ledger explorer with population eligibility indicators and backtest metric charts.
  - `/qualification`: Statistical qualification gate reports with Deflated Sharpe Ratio (DSR) and EICT-CORR-1 clustering.
  - `/paper`: Forward paper trading cockpit, simulated orders, execution fills, real-time positions, and replay reports.
  - `/monitoring`: Rolling evaluation windows, max drawdown breach tracking, and degradation alerts.
  - `/governance`: Cryptographically sealed governance transition audit trail.
  - `/feedback`: Structured empirical failure mode records and zero-trial research feedback bridge tasks.
  - `/audit`: Interactive 5-type cryptographic provenance DAG explorer (`Type A`, `Type B`, `Type C`, `Type D`, `Type E`).
- **FastAPI Gateway**: High-performance REST API (`src/quantmind/api/`) enforcing RBAC (`ADMIN`, `RESEARCHER`, `VIEWER`), OpenAPI schema introspection, and non-optimistic governance barriers.
- **Dedicated Background Worker**: Independent daemon (`python -m quantmind.app.worker`) consuming persistent SQLite queues, tracking live job progress, and performing automatic crash recovery on startup.
- **Documentation**:
  - [docs/APP_ARCHITECTURE.md](docs/APP_ARCHITECTURE.md): Architectural design, storage segregation, and 5-type provenance DAG.
  - [docs/API.md](docs/API.md): Comprehensive REST API endpoint reference.
  - [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md): Local development, demo seeding, and deployment guide.

## Quickstart & Local Execution

```bash
# 1. Install dependencies
pip install -e .
cd web && npm install && cd ..

# 2. Seed Demo Environment
python scripts/seed_demo_data.py --force

# 3. Run Backend Gateway (Terminal 1)
uvicorn quantmind.api.app:create_app --factory --reload --port 8000

# 4. Run Dedicated Worker Daemon (Terminal 2)
python -m quantmind.app.worker

# 5. Run Web Console (Terminal 3)
cd web && npm run dev
```

Visit `http://localhost:3000` and sign in with demo credentials:
- `demo_admin` / `QuantMindDemoAdmin2026!`

Or run with Docker Compose:
```bash
docker-compose up --build -d
```

## Running the Automated Test Suite

```bash
# Run full suite (741+ passed across unit, integration, API, and E2E)
pytest

# Run 15-step end-to-end quantitative workflow
pytest tests/e2e/test_e2e_workflow.py -v

# Run Playwright headless browser navigation test
pytest tests/e2e/test_playwright_ui.py -v
```

## Backtester scope

The current engine is a deterministic **one-bar-hold regression harness**, not yet the complete Section 33 execution engine. Research trials must enter through `ResearchHarness.run_trial()`, which owns checksum-verified dataset resolution, mandatory causality preflight, typed StrategySpec compilation, harness-generated trial/strategy/experiment IDs, cumulative budget reservation, failure recording, and the internal engine call. Production research trials accept only named LICENSED dataset versions and split zones; caller-supplied DataFrames, precomputed signal columns, and arbitrary signal callables are excluded from the production API. Synthetic fixtures live in `quantmind.testing` and are recorded with `mode=FIXTURE`. Position state, stop/target handling, intrabar adverse-first resolution, futures roll handling, and risk integration remain later work.
