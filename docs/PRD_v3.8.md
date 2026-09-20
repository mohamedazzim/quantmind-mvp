# QuantMind — Product Requirements Document v3.8
## Futures-First MVP: Strategy Validation Gate & Immutable Qualification Record

**Version:** 3.8  
**Status:** Locked MVP baseline for strategy qualification and paper eligibility  

**Scope:** MVP only  
**Primary market:** Indian index futures research, beginning with NIFTY 50 futures  
**Initial data granularity:** NIFTY 50 futures 1-minute historical data  
**Initial execution mode:** Backtest + deterministic historical replay paper trading  
**Architecture:** Modular monolith  
**Primary objective:** Build a reproducible, auditable research system that can generate and test hypotheses without fooling itself.

**Synthetic research protocol:** RP-2

---

## Changelog

### v3.8 (Strategy Validation Gate & Immutable Qualification Record)
- **Deterministic Strategy Validation Gate (`StrategyValidationGate`)**:
  - Implemented in `quantmind.research_integrity.qualification`.
  - Strictly consumes authoritative evidence: `DatasetRegistry`, `SplitManifest`, `HoldoutManager`, `TrialLedger`, `ProductionPopulationQuery`, `EICT-CORR-1`, `DSR`, and `StrategySpec`.
  - Rejects any manual caller-supplied overrides of statistical evidence (`manual_sharpe`, `manual_dsr`, `manual_trials`, etc.).
  - Reads sample size gates and holdout requirements dynamically from versioned research protocols (`configs/research_protocol_v2.yaml`), avoiding hardcoded thresholds.
  - Enforces `mode == "PRODUCTION"` and `split_zone == "VALIDATION"`, rejecting `FIXTURE` or `RESEARCH` zone trials.
- **Immutable Strategy Qualification Record (`StrategyQualificationRecord`)**:
  - Frozen dataclass with canonical JSON serialization and SHA-256 digest (`record_hash`).
  - Contains complete audit trail: strategy spec hash, dataset version/hash, split manifest, protocol version, population hash, effective trial count, observed Sharpe, DSR, trade count, holdout state, robustness status, final status, creation timestamp, and evaluation reasons.
  - Cryptographic verification method `verify_digest()` detecting any tampering with recorded statistics.
- **Append-Only Qualification Ledger (`QualificationLedger`)**:
  - Backed by SQLite table `strategy_qualifications`.
  - Protected by database triggers `qualifications_no_delete` and `qualifications_no_update`, raising SQL integrity errors on any mutation or deletion attempt.
  - Verifies cryptographic digest before insertion; idempotent on identical record writes; rejects conflicting records with same ID.
- **Validation State Machine (`ValidationStatus`)**:
  - Explicit statuses: `CANDIDATE`, `UNDERPOWERED`, `VALIDATION`, `REJECTED`, `REJECTED_FINAL_HOLDOUT`, `HOLDOUT_REQUIRED`, `HOLDOUT_PASSED`, `PAPER_ELIGIBLE`.
  - Transition validator (`validate_transition`) enforcing legal pathways only (e.g. `REJECTED -> PAPER_ELIGIBLE` and `UNDERPOWERED -> PAPER_ELIGIBLE` are strictly blocked with `InvalidStateTransitionError`).
- **Strategy Registry & Lifecycle State Machine (`StrategyRegistry`)**:
  - Implemented in `quantmind.strategy.registry`.
  - Manages full lifecycle states: `IDEA`, `RESEARCH`, `VALIDATION`, `REJECTED`, `PAPER_ELIGIBLE`, `PAPER_ACTIVE`, `DEGRADED`, `RETIRED`.
  - Transition to `PAPER_ELIGIBLE` strictly requires an authoritative, cryptographically verified `StrategyQualificationRecord` with `final_status == ValidationStatus.PAPER_ELIGIBLE`, matching `strategy_id` and `strategy_spec_hash`, and `holdout_state == "PASSED"`.
- **Paper Replay Eligibility Bridge (`PaperReplayEligibility`)**:
  - Interface and verification helper `check_paper_replay_eligibility` certifying readiness for deterministic paper replay without initiating live broker trading or execution.
- **Validation Report (`ValidationReport`)**:
  - Structured human-readable summary of checks passed, failed, and warned. Strictly excludes relative ranking, leaderboards, or "best strategy" claims.
- **Adversarial & Exploit Test Suite**:
  - Proved rejection of fake Sharpe, fake DSR, fake trial counts, fixture modes, holdout burn bypasses, and unauthorized state transitions.
  - Total test count expanded from **206 passed** to **243 passed** (0 failures, 0 warnings).

---

# 1. Executive Summary

QuantMind is an AI-assisted quantitative research laboratory.

The MVP is intentionally smaller than the long-term vision. It will not attempt to build an autonomous multi-agent trading organization, RL platform, or live-trading broker execution. The first tradable instrument is NIFTY 50 index futures, so the research and paper-execution layers have an actual tradable contract, lot size, expiry, and roll lifecycle.

The MVP proves one thing:
> Can a single, auditable system use AI to formulate and evaluate quantitative hypotheses while a deterministic research kernel prevents data leakage, overfitting, invalid execution assumptions, and uncontrolled experimentation?

The system lifecycle is:

```text
Research Objective
        ↓
Data Quality
        ↓
Hypothesis & StrategySpec
        ↓
Deterministic Backtest (RESEARCH / VALIDATION)
        ↓
Research Integrity Layer (TrialLedger + ArtifactRegistry + EICT + DSR)
        ↓
Strategy Validation Gate
        ↓
Sealed Holdout Gate (HoldoutManager)
        ↓
Immutable Qualification Record (QualificationLedger)
        ↓
Strategy Registry (PAPER_ELIGIBLE)
        ↓
Paper Replay Testing
```

---

# 2. Strategy Validation Gate Specification

### 2.1 Evidence Consumption Invariants
The `StrategyValidationGate` accepts only authoritative system artifacts:
1. **Validation Trial Verification**:
   - `mode` must be `PRODUCTION`. `FIXTURE` trials are strictly rejected.
   - `split_zone` must be `VALIDATION`.
   - `status` must be `COMPLETED`.
   - `strategy_id` must match `derive_strategy_id(normalize_strategy_spec(strategy_spec))`.
   - `dataset_version` and `research_protocol_version` must match the registered dataset and protocol.
2. **Protocol-Driven Sample Size Requirements**:
   - Sample size thresholds are read dynamically from `configs/research_protocol_v2.yaml`.
   - If `trade_count < minimum_trades_validation` (default 75) or `sample_length < minimum_effective_outcome_observations` (default 100), status is resolved to `UNDERPOWERED`.
3. **Statistical Multiplicity Adjustment**:
   - The gate loads the cumulative production population via `ProductionPopulationQuery`.
   - Computes Effective Independent Trial Count via `EictCorr1Calculator`.
   - Computes Deflated Sharpe Ratio via `DeflatedSharpeCalculator`.
   - Manual overrides (`manual_sharpe`, `manual_dsr`, `manual_trials`, etc.) raise `ValueError`.
4. **Holdout Protection**:
   - Consults `HoldoutManager` for candidate state:
     - `BURNED` -> `REJECTED`.
     - `FAILED` -> `REJECTED_FINAL_HOLDOUT`.
     - `UNTOUCHED` (when holdout required) -> `HOLDOUT_REQUIRED`.
     - `PASSED` -> candidate eligible for `PAPER_ELIGIBLE`.

### 2.2 Validation Status State Machine
```text
                  ┌──────────────┐
                  │  CANDIDATE   │
                  └──────┬───────┘
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │UNDERPOWERED│ │VALIDATION │  │ REJECTED  │
    └─────┬─────┘  └─────┬─────┘  └───────────┘
          │              │              ▲
          └──────┬───────┘              │
                 ↓                      │
        ┌─────────────────┐             │
        │ HOLDOUT_REQUIRED├─────────────┤
        └────────┬────────┘             │
                 │                      │
        ┌────────┴────────┐             │
        ↓                 ↓             │
  ┌────────────┐  ┌──────────────────┐  │
  │HOLDOUT_PASSED││REJECTED_FINAL_    │  │
  └─────┬──────┘  │HOLDOUT           │  │
        │         └──────────────────┘  │
        ↓                               │
  ┌──────────────┐                      │
  │PAPER_ELIGIBLE├──────────────────────┘
  └──────────────┘
```

---

# 3. Immutable Strategy Qualification Record

Every strategy evaluation creates a `StrategyQualificationRecord`:
```json
{
  "created_at": "2026-09-20T12:00:00+00:00",
  "dataset_sha256": "...",
  "dataset_version": "DS-GATE-V1",
  "dsr": 0.965,
  "effective_trial_count": 14.0,
  "final_status": "PAPER_ELIGIBLE",
  "holdout_state": "PASSED",
  "observed_sharpe": 1.85,
  "population_hash": "...",
  "qualification_id": "QUAL-...",
  "reasons": ["All validation checks passed"],
  "research_protocol_version": "RP-2",
  "robustness_status": "PASSED",
  "split_manifest_version": "m1",
  "strategy_id": "STRAT-...",
  "strategy_spec_hash": "...",
  "trade_count": 120,
  "record_hash": "<sha256-digest-over-canonical-json>"
}
```

### 3.1 Security & Immutability Guarantees
1. **Frozen Dataclass**: Python runtime attributes cannot be modified post-creation.
2. **Deterministic Canonical JSON**: Canonical representation with sorted keys and normalized floats.
3. **Cryptographic SHA-256 Digest**: Validated via `verify_digest()`.
4. **SQLite Triggers**:
   ```sql
   CREATE TRIGGER qualifications_no_delete
   BEFORE DELETE ON strategy_qualifications
   BEGIN
       SELECT RAISE(ABORT, 'qualification records are permanent and append-only');
   END;

   CREATE TRIGGER qualifications_no_update
   BEFORE UPDATE ON strategy_qualifications
   BEGIN
       SELECT RAISE(ABORT, 'qualification records are permanent and immutable');
   END;
   ```

---

# 4. Strategy Registry & Lifecycle Bridge

The `StrategyRegistry` enforces lifecycle discipline:
- States: `IDEA` -> `RESEARCH` -> `VALIDATION` -> `PAPER_ELIGIBLE` -> `PAPER_ACTIVE` -> `DEGRADED` -> `RETIRED`.
- Transition to `PAPER_ELIGIBLE` requires:
  1. A valid, uncorrupted `StrategyQualificationRecord` (`verify_digest() == True`).
  2. Matching `strategy_id` and normalized `strategy_spec_hash`.
  3. `record.final_status == ValidationStatus.PAPER_ELIGIBLE`.
  4. `record.holdout_state == "PASSED"`.
- Paper trading is limited strictly to deterministic forward replay without live broker execution.

---

# 5. Verification Metrics

- Total Passing Tests: **243**
- Compilation: `python -m compileall src tests` passes cleanly.
- Adversarial Tests: 12 dedicated exploit test cases verifying tamper detection, override rejection, and boundary isolation.
- Git Branch: `feat/strategy-validation-gate`
