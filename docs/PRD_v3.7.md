# QuantMind — Product Requirements Document v3.7
## Futures-First MVP: AI-Assisted Quantitative Research & Replay Paper Trading

**Version:** 3.7  
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

### v3.7
- **Statistical Validation Layer**: Implemented decoupled, auditable statistical evaluation in `quantmind.research_integrity.statistical_validation`:
  - **LAYER A — EICT-CORR-1 Hierarchical Clustering**: Computes effective independent trial count via hierarchical clustering with average linkage at distance cut $0.6325$ ($\sqrt{2(1 - 0.80)}$). Sparse trials with $< 100$ common observations are strictly isolated as singleton clusters. Outputs `raw_trial_count`, `effective_trial_count`, deterministic cluster memberships, and cryptographic `population_hash`.
  - **LAYER B — Deflated Sharpe Ratio (DSR)**: Implemented Bailey & López de Prado (2014) formulation adjusting for selection bias, backtest multiplicity ($N = \text{effective\_trial\_count}$), return distribution non-normality (skewness, Fisher kurtosis), and sample length $T$. Simplifies to Probabilistic Sharpe Ratio (PSR) when $N = 1$.
  - **Multiple-Testing Guard**: Rigid architectural enforcement barring manual overrides of `observed_sharpe` or `effective_trial_count`. All statistical evaluations must anchor to an authoritative `ProductionPopulationQuery`.
  - **Integrated Pipeline**: Added `StatisticalValidationPipeline` unifying population querying, EICT clustering, and candidate DSR evaluation.
- Current test verification: **178 tests passing** (0 failures, 0 regressions).

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
Research Integrity Layer (Population Accounting + OOS Artifacts + EICT + DSR)
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

# 2. Statistical Validation Layer: EICT-CORR-1 + Deflated Sharpe Ratio

### 2.1 EICT-CORR-1 Methodology
The Effective Independent Trial Count adjusts the total trial count $K$ downward to an effective count $N \le K$ by identifying clusters of correlated strategies:
1. **Distance Metric**: $d_{i,j} = \sqrt{2(1 - r_{i,j})}$, where $r_{i,j}$ is the Pearson correlation of trade net returns over common timestamps.
2. **Linkage**: Average linkage hierarchical clustering (`scipy.cluster.hierarchy.linkage(..., method='average')`).
3. **Distance Cut**: $t = 0.6325$ corresponds to $r = 0.80$. Pairs with correlation $\ge 0.80$ cluster together.
4. **Sparse Trial Policy**: Trials with fewer than 100 common observations are isolated as singleton clusters. Each singleton contributes $1$ to the effective trial count.
5. **Output**: Exposes `raw_trial_count`, `effective_trial_count`, `clusters`, `cluster_memberships`, `singleton_clusters`, and a canonical `population_hash`.

### 2.2 Deflated Sharpe Ratio (DSR) Formulation
Following Bailey & López de Prado (2014):
1. **Expected Maximum Sharpe under Null ($H_0: SR = 0$)**:
   $$E[\max_{n=1 \dots N} \{SR_n\}] \approx \sigma_{SR} \left( (1 - \gamma) Z^{-1}\left(1 - \frac{1}{N}\right) + \gamma Z^{-1}\left(1 - \frac{1}{N e}\right) \right)$$
   where:
   - $N$ is the `effective_trial_count` from EICT-CORR-1
   - $\sigma_{SR}$ is the standard deviation of observed Sharpes across the eligible population
   - $\gamma \approx 0.5772156649$ is the Euler-Mascheroni constant
   - $Z^{-1}$ is the inverse standard normal CDF (`norm.ppf`)
   - If $N = 1$ or $\sigma_{SR} = 0$, $E[\max] = 0$.

2. **Sharpe Standard Error with Non-Normality Adjustment (Mertens, 2002; Lo, 2002)**:
   $$\sigma_{\hat{SR}} = \sqrt{ \frac{1}{T - 1} \left( 1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \hat{SR}^2 \right) }$$
   where $\hat{\gamma}_3$ is return skewness, $\hat{\gamma}_4$ is raw kurtosis ($= k_{\text{excess}} + 3$), and $T$ is the number of out-of-sample observations.

3. **Test Statistic and DSR**:
   $$z = \frac{\hat{SR} - E[\max\{SR\}]}{\sigma_{\hat{SR}}}$$
   $$\text{DSR} = \Phi(z) = \text{scipy.stats.norm.cdf}(z)$$
   A candidate strategy passes the DSR statistical validation gate if $\text{DSR} \ge 0.95$ (5% significance level).
