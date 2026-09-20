# QuantMind — Product Requirements Document v3.6
## Futures-First MVP: AI-Assisted Quantitative Research & Replay Paper Trading

**Version:** 3.6  
**Status:** Locked MVP baseline for implementation  

**Scope:** MVP only  
**Primary market:** Indian index futures research, beginning with NIFTY 50 futures  
**Initial data granularity:** NIFTY 50 futures 1-minute historical data  
**Initial execution mode:** Backtest + deterministic historical replay paper trading  
**Architecture:** Modular monolith  
**Primary objective:** Build a reproducible, auditable research system that can generate and test hypotheses without fooling itself.

**Synthetic research protocol:** RP-2

---

## Changelog

### v3.6
- **OOS Return Artifacts**: Every completed `PRODUCTION` trial serializes its full trade-level evaluation stream (`signal_timestamp`, `entry_timestamp`, `exit_timestamp`, `side`, `gross_return_bps`, `cost_bps`, `net_return_bps`) to a deterministic Parquet file using a fixed PyArrow schema (`_OOS_RETURNS_SCHEMA`).
- **Artifact Registry**: Implemented `trial_artifacts` table in SQLite with strict database triggers (`artifacts_no_delete`, `artifacts_no_update`) enforcing append-only immutability.
- **SHA-256 Byte Verification**: Post-write digest computation and mandatory checksum verification upon loading. Corrupted or tampered files fail validation immediately.
- **Production Population Accounting**: Implemented `ProductionPopulationQuery` ensuring that only valid `PRODUCTION` trials with status `COMPLETED` and valid OOS artifacts enter the population for a given `(dataset_version, research_protocol_version)` scope. Fixture, failed, abandoned, rejected, or running trials are strictly excluded.
- **Common Timestamp Alignment**: `align_pair` computes exact timestamp intersections across trial pairs for synchronous correlation analysis without forward-fill lookahead.
- **EICT-CORR-1 Input Preparation**: Implemented `EictCorr1InputBuilder` to prepare pairwise Pearson correlations, EICT distances ($d = \sqrt{2(1 - r)}$), correlation/distance matrices, and identify singleton clusters when overlap falls below the protocol threshold (100 observations). Protocol parameters (`correlation_threshold=0.80`, `distance_threshold=0.6325`, `linkage=average`, `insufficient_overlap_policy=singleton_cluster`) are immutable.
- **DSR Input Preparation**: Implemented `compute_return_distribution` and `build_dsr_inputs` computing sample length, mean return, standard deviation, observed Sharpe, Fisher excess kurtosis, and skewness for each eligible trial as the evidence layer for Deflated Sharpe Ratio calculation.
- **Performance & Zero-Copy**: Read pipeline uses PyArrow zero-copy and NumPy arrays; row-by-row pandas iteration is eliminated.

### v3.5
- Implemented `SplitManifest` model and deterministic boundary calculator (`compute_split_manifest`) supporting RESEARCH, VALIDATION, FINAL_HOLDOUT, and FORWARD_PAPER zones.
- Bound purge windows to explicit temporal dependencies: `feature_lookback_bars + prediction_horizon_bars + holding_period_bars + forward_dependency_bars`.
- Added embargo buffers following boundaries to eliminate serial correlation leakage.
- Enhanced `DatasetRegistry` with split manifest immutability: identical manifest registration is idempotent; modified manifests for existing dataset versions are rejected.
- Implemented high-performance, vectorized zone loading via boolean masks (no `.iloc` loops) with in-memory caching and dataset checksum verification.
- Sealed `FINAL_HOLDOUT`: standard research APIs and agents (`load_zone`, `ResearchHarness.run_trial`) are prohibited from requesting or loading holdout data.
- Added dedicated `final_evaluate()` gateway in `HoldoutManager` with strict gate prerequisites (completed RESEARCH/VALIDATION trial required).
- Implemented explicit holdout lifecycle state machine (`UNTOUCHED`, `EVALUATED`, `PASSED`, `FAILED`, `BURNED`) with immutable database records.
- Enforced retuning prevention: candidates failing final holdout are permanently blocked from re-evaluation or retuning against that dataset.
- Added `split_zone` tracking to `TrialLedger` (`trials` table) with immutable running-trial triggers.
- Documented in-process trust boundaries (internal method bypass vs harness boundary).

### v3.4
- Moved synthetic fixture execution out of `quantmind.research_integrity.ResearchHarness` into `quantmind.testing` so production agents cannot reach fixture inputs through the research API.
- Added trial `mode` (`PRODUCTION` / `FIXTURE`) and dataset-kind provenance (`LICENSED` / `SYNTHETIC`); fixture rows are excluded from production budgets and populations.
- Added a checksum-verified Dataset Registry and named split-zone loader. `ResearchHarness.run_trial()` now resolves data internally by `dataset_version + split_zone` and accepts only `LICENSED` datasets.
- Added typed StrategySpec schemas and compile-time range validation before budget reservation. Invalid specs consume no trial budget.
- Normalized strategy parameters before identity derivation and made variant identity depend on `feature_version + signal_name + normalized parameters`; `strategy_version` remains metadata.
- Documented the SQLite in-process trust boundary and future PostgreSQL role enforcement.

### v3.3
- Made `ResearchHarness.run_trial()` accept only declarative `StrategySpec` objects. Arbitrary signal functions and precomputed signal columns are excluded from the production research path.
- Added a whitelist `StrategyCompiler`; agent-generated strategy specifications are compiled only into registered feature/signal implementations.
- Made `strategy_id` the SHA-256 identity of the complete canonical strategy specification, including strategy version, feature version, signal name, and parameters.
- Added an explicit test-fixture path for synthetic signal columns/callables so regression fixtures cannot be mistaken for production research inputs.

### v3.2
- Promoted the synthetic control protocol from RP-1 to RP-2 after validating the prior-bar-sign control could recursively amplify through injected returns.
- Replaced the dense 10% random/price-direction condition with a sparse, observable `prior_volume_tail_oi_direction` condition using a versioned 99.3% prior-session volume tail, prior OI-change direction, and configured session window.
- Locked a target trigger frequency of 1–3 per session and a maximum lag-1 return autocorrelation of 0.05 for the strong-control fixture.
- Required the control runner to read `condition`, `signal_window`, `tail_quantile`, and `lookback_sessions` from the research protocol rather than hardcoding them.
- Kept dense controls as smoke tests only; the positive-control ladder is evaluated through the research pipeline once the backtester and integrity gates are available.

### v3.1
- Locked execution timing: close(t) signal is earliest fillable at open(t+1), with adverse-first intrabar ambiguity handling.
- Added date-effective, versioned cost schedules bound to dataset and experiment versions.
- Made multiplicity and research budgets cumulative across `dataset_version + research_protocol_version`.
- Locked `EICT-CORR-1` sparse-trial handling as singleton clusters and fixed the 0.6325 distance cut.
- Added explicit historical holdout pass criteria and forward-data update requirements.
- Added boundary-specific purge/embargo requirements and split-feasibility checks.
- Added synthetic end-to-end holdout testing and clarified that zero real-data survivors is a valid outcome.
- Added a synthetic acquisition gate: at least 200 surrogate runs, with a one-sided 95% upper confidence bound on full-gate false-pass rate at or below 5%.
- Added a strong positive-control gate: at least 100 runs, with a one-sided 95% lower confidence bound on detection rate at or above 80%.
- Defined the strong positive-control injection as a versioned net edge in basis points after the synthetic cost schedule.
- Added end-to-end null false-pass measurement including the final holdout gate.

---

# 1. Executive Summary

QuantMind is an AI-assisted quantitative research laboratory.

The MVP is intentionally smaller than the long-term vision. It will not attempt to build a 16-agent autonomous trading organization, distributed research cluster, RL platform, or live-trading infrastructure. The first tradable instrument is NIFTY 50 index futures, not the cash index, so the research and paper-execution layers have an actual tradable contract, lot size, expiry and roll lifecycle.

The MVP will prove one thing:

> Can a single, auditable system use AI to formulate and evaluate quantitative hypotheses while a deterministic research kernel prevents data leakage, overfitting, invalid execution assumptions, and uncontrolled experimentation?

The system will support this lifecycle:

```text
Research Objective
        ↓
Supervisor
        ↓
Data Quality
        ↓
Hypothesis
        ↓
Strategy Builder
        ↓
Deterministic Backtest
        ↓
Research Integrity Layer (Population Accounting + OOS Artifacts + EICT/DSR Inputs)
        ↓
Validation Gate
        ↓
Strategy Registry
        ↓
Paper Trading
        ↓
Monitoring
```

---

# 2. Research Population Accounting & Trial Return Artifacts

### 2.1 OOS Return Stream Storage
Every completed `PRODUCTION` trial evaluates its trading decisions on out-of-sample data. To support defensible multiple-testing adjustment (Deflated Sharpe Ratio and Effective Independent Trial Count), the full trade-level evaluation stream is persisted to disk in Parquet format.

Each record conforms strictly to `_OOS_RETURNS_SCHEMA`:
- `signal_timestamp`: UTC microsecond timestamp of the signal generation bar.
- `entry_timestamp`: UTC microsecond timestamp of the order entry bar.
- `exit_timestamp`: UTC microsecond timestamp of the order exit bar.
- `side`: Trade direction (+1 for Long, -1 for Short).
- `gross_return_bps`: Gross return in basis points.
- `cost_bps`: Total slippage and commission costs in basis points.
- `net_return_bps`: Net return after all friction.

### 2.2 Immutability and Audit Integrity
1. **Hash Verification**: A SHA-256 digest is computed from the written Parquet bytes and stored in `trial_artifacts`. Any subsequent read verifies this digest before returning the data.
2. **Database Integrity**: The `trial_artifacts` table is protected by SQLite triggers `artifacts_no_delete` and `artifacts_no_update`, preventing modification or deletion.
3. **Ledger Correlation**: The trial's completion record in `TrialLedger` stores the `artifact_sha256` in its result summary.

### 2.3 Eligibility & Population Boundaries
A trial belongs to the statistical population of a research objective if and only if:
- `mode == 'PRODUCTION'` (fixtures and synthetic tests are permanently excluded)
- `status == 'COMPLETED'` (failed, rejected non-causal, abandoned, or running trials are excluded)
- It matches the exact scope: `(dataset_version, research_protocol_version)`
- Its OOS return artifact exists on disk and passes SHA-256 verification

### 2.4 EICT-CORR-1 Input Preparation Layer
The Effective Independent Trial Count (EICT-CORR-1) method clusters trials by their return correlation to estimate the effective number of independent strategies tested:
- **Timestamp Alignment**: Common observations between pairs of trials are identified via inner set intersection without lookahead or extrapolation.
- **Metric**: Pearson correlation $r$ of net return streams over common timestamps.
- **Distance Formula**: $d = \sqrt{2(1 - r)}$.
- **Sparse Trial Policy**: Trials with fewer than 100 common observations are isolated as singleton clusters per protocol specification (`insufficient_overlap_policy: singleton_cluster`).
- **Protocol Values**:
  - `correlation_threshold = 0.80`
  - `distance_threshold = 0.6325`
  - `linkage = average`
  - `minimum_overlap_observations = 100`

### 2.5 DSR Input Preparation Layer
For each eligible trial in the population, the following summary statistics are computed from its immutable OOS return stream:
- `mean_return`: Mean net return (bps).
- `std_return`: Sample standard deviation of net return (bps).
- `observed_sharpe`: Sample Sharpe ratio ($\mu / \sigma$).
- `skewness`: Fisher-Pearson coefficient of skewness.
- `kurtosis`: Fisher excess kurtosis.
- `trade_count`: Total observation count.
- `effective_observations`: Effective number of independent observations.
- `effective_trial_count`: Population trial count after EICT clustering.

*Note: The actual DSR threshold testing and statistical hypothesis decisions will be executed in the subsequent statistical gate milestone.*
