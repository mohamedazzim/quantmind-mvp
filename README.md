# QuantMind MVP

Futures-first AI-assisted quantitative research laboratory.

## Current build stage

**Stage 1: deterministic backtester + synthetic research-integrity harness + split/holdout manager + statistical validation (EICT + DSR) + strategy validation gate (v3.8)**

Current verification: **243 tests passing** across synthetic market generation, real gap/wick semantics, cross-session surrogates, directional intraday nulls, recomputable causal positive controls, independent wick preservation, paired edge recovery, one-sided confidence-bound gates, look-ahead canary, checksum-verified Dataset Registry, immutable SplitManifest, purge and embargo boundaries, sealed final holdout security, holdout state machine, comprehensive adversarial integrity tests, immutable OOS return Parquet artifacts, SHA-256 byte verification, production population accounting, EICT-CORR-1 hierarchical clustering, Deflated Sharpe Ratio multiple-testing adjustment (Bailey & López de Prado 2014), permutation-invariant clustering, multiple-testing anti-bypass guards, deterministic Strategy Validation Gate, immutable Strategy Qualification Records with canonical SHA-256 digests, append-only SQLite qualification ledger triggers, Strategy Registry lifecycle state machine, and Paper Replay eligibility verification.

The synthetic harness is intentionally a research fixture, not a market model. Its purpose is to make the deterministic backtest and research-integrity layers falsifiable before any paid historical market data is purchased or frozen.

## First milestones

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
16. Bounded research agents


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
PaperReplayEligibility (bridge certifying readiness for deterministic forward replay)
```

### Strategy Validation Gate & Immutable Qualification Record (v3.8)

- **`StrategyValidationGate`**: Deterministic gate consuming authoritative system evidence only (`DatasetRegistry`, `SplitManifest`, `HoldoutManager`, `TrialLedger`, `ProductionPopulationQuery`, `EICT-CORR-1`, `DSR`, and `StrategySpec`). Rejects any manual statistical overrides (`manual_sharpe`, `manual_dsr`, `manual_trials`). Dynamically evaluates protocol-defined sample size requirements (`minimum_trades_validation`, `minimum_effective_outcome_observations`).
- **`StrategyQualificationRecord`**: Frozen dataclass containing complete provenance, observed Sharpe, DSR, EICT, trade count, holdout state, robustness status, and validation status, sealed by a SHA-256 digest over its canonical JSON.
- **`QualificationLedger`**: Append-only SQLite ledger with triggers raising SQL integrity errors on any `DELETE` or `UPDATE` attempt.
- **Validation State Machine**: Enforces legal transitions (`CANDIDATE` -> `UNDERPOWERED` / `VALIDATION` / `REJECTED` / `HOLDOUT_REQUIRED` -> `HOLDOUT_PASSED` -> `PAPER_ELIGIBLE`). Illegal shortcuts (e.g. `REJECTED` -> `PAPER_ELIGIBLE` or `UNDERPOWERED` -> `PAPER_ELIGIBLE`) are permanently barred.
- **`StrategyRegistry`**: Tracks lifecycle states (`IDEA` -> `RESEARCH` -> `VALIDATION` -> `PAPER_ELIGIBLE` -> `PAPER_ACTIVE` -> `DEGRADED` -> `RETIRED`). Transition to `PAPER_ELIGIBLE` strictly requires an authoritative, cryptographically verified `StrategyQualificationRecord` with `final_status == ValidationStatus.PAPER_ELIGIBLE` and `holdout_state == "PASSED"`.
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

## Run tests

```bash
python -m pip install -e '.[dev]'
pytest -q
```

## Backtester scope

The current engine is a deterministic **one-bar-hold regression harness**, not yet the complete Section 33 execution engine. Research trials must enter through `ResearchHarness.run_trial()`, which owns checksum-verified dataset resolution, mandatory causality preflight, typed StrategySpec compilation, harness-generated trial/strategy/experiment IDs, cumulative budget reservation, failure recording, and the internal engine call. Production research trials accept only named LICENSED dataset versions and split zones; caller-supplied DataFrames, precomputed signal columns, and arbitrary signal callables are excluded from the production API. Synthetic fixtures live in `quantmind.testing` and are recorded with `mode=FIXTURE`. Position state, stop/target handling, intrabar adverse-first resolution, futures roll handling, and risk integration remain later work.
