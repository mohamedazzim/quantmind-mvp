# QuantMind — Product Requirements Document v3.1
## Futures-First MVP: AI-Assisted Quantitative Research & Replay Paper Trading

**Version:** 3.1  
**Status:** Locked MVP baseline for implementation  

**Scope:** MVP only  
**Primary market:** Indian index futures research, beginning with NIFTY 50 futures  
**Initial data granularity:** NIFTY 50 futures 1-minute historical data  
**Initial execution mode:** Backtest + deterministic historical replay paper trading  
**Architecture:** Modular monolith  
**Primary objective:** Build a reproducible, auditable research system that can generate and test hypotheses without fooling itself.

## Changelog

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
Research Integrity Layer
        ↓
Validation Gate
        ↓
Strategy Registry
        ↓
Paper Trading
        ↓
Monitoring
```

The MVP deliberately excludes advanced pattern-discovery, autonomous strategy evolution, reinforcement learning, distributed messaging, cloud-first deployment, and real-money execution.

---

# 2. Product Vision

Build a trustworthy AI-assisted research system that behaves less like a trading chatbot and more like a disciplined quantitative research assistant.

The system should help answer:

```text
"What hypotheses should we investigate?"

"What experiment would test them?"

"Did the strategy actually work?"

"How many attempts were made before this result appeared?"

"Could this result be caused by leakage or overfitting?"

"Does it survive unseen data and realistic execution assumptions?"

"Can the result be reproduced?"
```

---

# 3. The Three Success Levels

QuantMind success is evaluated at three distinct levels.

## Level 1 — Engineering Success

The platform can reliably execute:

```text
Data
→ Hypothesis
→ Strategy
→ Backtest
→ Validation
→ Registry
→ Paper Trading
```

with reproducible results.

## Level 2 — Research Integrity Success

The platform can:

- Detect or prevent look-ahead bias.
- Detect or prevent temporal leakage.
- Enforce experiment boundaries.
- Count trials automatically.
- Protect the final holdout.
- Correct for multiple testing.
- Distinguish real synthetic signal from null data.
- Reject invalid strategies.
- Reproduce experiments.
- Preserve failed experiments.

## Level 3 — Trading Research Success

Strategies that survive Level 2 are evaluated under realistic paper execution.

Profitability is an empirical research result, not a product guarantee.

---

# 4. Product Principles

## P1 — AI proposes; deterministic code computes

Agents may generate hypotheses, strategy definitions, and research plans.

The quantitative kernel computes:

- Features
- Signals
- Orders
- Fills
- P&L
- Costs
- Slippage
- Metrics
- Statistical tests

## P2 — Risk is deterministic

The risk engine is not controlled by an LLM.

## P3 — Experiments are immutable

Historical experiment results cannot be silently overwritten.

## P4 — Holdout data is protected

The final holdout is unavailable to research agents until the final evaluation gate.

## P5 — Every trial counts

The backtest harness—not the agent—records every experiment and trial.

## P6 — Failure is useful

Rejected strategies remain in the research record.

## P7 — Profitability is not the primary validation criterion

A strong research system that correctly rejects false discoveries is more valuable than one that finds attractive backtests by overfitting.

---

# 5. MVP Scope

## Included

### Task 0 — Data Acquisition & Licensing

Before implementation work that depends on historical data begins, the project must:

- Select the exact historical NIFTY 50 futures dataset.
- Record the provider/source.
- Record the coverage period and timestamps.
- Verify 1-minute granularity and required fields.
- Verify contract identifiers, expiry dates, lot-size history and trading sessions.
- Record the license/usage terms.
- Verify that the intended use (private research, backtesting and replay paper trading) is permitted.
- Store the agreement/reference in the project documentation.
- Record whether redistribution is prohibited.
- Record the source's correction/revision policy.

The preferred source of record is a properly licensed NSE Data & Analytics historical-data product or a third-party vendor whose agreement explicitly permits the intended research use. NSE currently offers historical, EOD, snapshot and real-time market-data products, and its data-usage policy states that usage, handling and redistribution are governed by the relevant subscriber agreement. The final provider cannot be considered selected until those terms are reviewed and recorded.

### Agent layer

- Supervisor Agent
- Data Quality Agent
- Hypothesis Agent
- Strategy Builder Agent

### Deterministic quant layer

- Data ingestion
- Dataset versioning
- Feature calculation
- Strategy interface
- Event-driven backtester
- Transaction cost model
- Slippage model
- Portfolio accounting
- Risk engine

### Research integrity

- Trial ledger
- Dataset locking
- Purged/embargoed time splits
- Final holdout
- Null/surrogate testing
- Positive-control testing
- Multiple-testing metadata
- Deflated Sharpe support
- Probability of Backtest Overfitting support
- Robustness tests

### Platform

- Strategy Registry
- Experiment Registry
- FastAPI
- PostgreSQL
- DuckDB
- Parquet
- Docker Compose
- Basic web dashboard
- Audit log
- Paper trading

---

# 6. Explicitly Deferred

The following are NOT MVP requirements:

- Autonomous pattern-discovery agent
- Autonomous strategy-evolution agent
- RL agent
- Deep multi-agent research tournament
- Market-depth research
- Tick research
- 1-second research
- Distributed backtesting
- Kafka
- NATS
- Kubernetes
- AWS-first deployment
- Multi-region architecture
- Real-money execution
- Broker-native execution
- Portfolio optimization across multiple markets
- Advanced order-book simulation
- Fully autonomous live trading

These may be introduced after MVP validation.

---

# 6A. Data Acquisition Baseline

## Source of record

The MVP must not assume that historical data is already available.

Task 0 is explicit data acquisition and licensing.

The project must create a **Data Acquisition Record** containing:

```text
provider
product_name
exchange
segment
instrument
date_range
granularity
delivery_format
update_cadence
historical_backfill_rights
future_update_rights
retention_period
fields
license_type
permitted_use
redistribution_rights
storage_rights
correction_policy
download_date
dataset_version
```

## Preferred source

Use a licensed source for NSE market data. The first choice is an official NSE Data & Analytics historical-data product if its product scope, granularity, coverage and terms match the project.

NSE currently lists historical, EOD, real-time and snapshot market-data products, and its market-data policy states that subscribers operate under relevant agreements governing use, handling and redistribution.

A third-party source is acceptable only when the contract/terms explicitly permit the intended private research and paper-replay use.

## Forward-data requirement

Because MVP paper trading uses replay, the eventual forward-evaluation window requires new market-data deliveries after the research protocol is frozen.

The Data Acquisition Record must therefore document:

```text
update cadence
new-data availability timing
historical backfill rights
future-date delivery rights
retention period
revision/correction notification
```

If the provider does not permit continued data updates for the intended use, the forward-evaluation phase is not available under that provider.

## Licensing gate

No research run may begin until:

```text
Provider selected
+
Coverage verified
+
1-minute futures data verified
+
Contract metadata verified
+
Usage rights recorded
```

If the provider permits research use but prohibits redistribution, the data remains private and is never committed to the public repository.

## Correction policy

The provider's historical-data correction/revision policy must be recorded.

If a dataset changes:

```text
Old dataset
    ↓
New dataset version
    ↓
Dataset hash changes
    ↓
Affected experiments are marked stale
```

Past experiment results remain immutable.

# 7. MVP Architecture

```text
                         ┌──────────────────────┐
                         │     USER / RESEARCHER│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   WEB APP / API      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   SUPERVISOR AGENT   │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
       ┌────────────────┐ ┌──────────────────┐ ┌────────────────┐
       │ DATA QUALITY   │ │ HYPOTHESIS AGENT │ │ RISK ENGINE    │
       │ AGENT          │ └────────┬─────────┘ └───────┬────────┘
       └───────┬────────┘          │                   │
               │                   ▼                   │
               │          ┌──────────────────┐         │
               │          │ STRATEGY BUILDER │         │
               │          └────────┬─────────┘         │
               │                   │                   │
               └───────────────────┼───────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ DETERMINISTIC QUANT │
                         │ KERNEL               │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ BACKTEST HARNESS     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ RESEARCH INTEGRITY   │
                         │ LAYER                │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                 ┌──────────────┐      ┌───────────────┐
                 │ REJECT       │      │ VALIDATION    │
                 │              │      │ GATE          │
                 └──────────────┘      └───────┬───────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ STRATEGY        │
                                      │ REGISTRY        │
                                      └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ PAPER TRADING   │
                                      └─────────────────┘
```

---

# 8. Architectural Decision: Modular Monolith

The MVP will use a modular monolith.

## Runtime

```text
FastAPI
    +
Python modules
    +
PostgreSQL
    +
DuckDB
    +
Parquet
    +
Docker Compose
```

## Why

The system is initially built and operated by one developer.

The modular monolith provides:

- Simple deployment
- Easy debugging
- Lower operational overhead
- Low network complexity
- Strong module boundaries
- Easy extraction into services later

No Kafka, NATS, Kubernetes, or cloud orchestration is required for MVP.

---

# 9. Core Domain Boundaries

The codebase must maintain strict internal boundaries.

```text
quantmind/
├── api/
├── agents/
├── data/
├── quant/
├── backtest/
├── research_integrity/
├── risk/
├── paper_trading/
├── registry/
├── experiments/
├── storage/
├── audit/
└── web/
```

Each module should expose narrow interfaces.

---

# 10. MVP Agents

## 10.1 Supervisor Agent

### Responsibilities

- Parse user research objectives.
- Create research tasks.
- Select allowed experiment templates.
- Dispatch deterministic tools.
- Coordinate agents.
- Produce research summaries.

### Non-responsibilities

- Direct broker access.
- Direct order execution.
- Risk override.
- Editing final holdout.
- Editing trial counts.

---

## 10.2 Data Quality Agent

### Responsibilities

- Validate dataset.
- Produce quality report.
- Identify gaps and anomalies.
- Block bad datasets.

### Mandatory checks

- Schema
- Timestamp ordering
- Duplicate records
- Missing data
- Invalid OHLC
- Invalid volume
- Out-of-session records
- Timezone consistency
- NaN/inf
- Dataset boundaries

---

## 10.3 Hypothesis Agent

### Responsibilities

Convert a research objective into explicit, testable hypotheses.

Each hypothesis must define:

```yaml
hypothesis:
  id:
  question:
  market:
  timeframe:
  observation_window:
  prediction_horizon:
  independent_variables:
  dependent_variable:
  entry_conditions:
  exit_conditions:
  null_hypothesis:
  alternative_hypothesis:
  evaluation_metrics:
```

The agent may only use approved datasets and features.

---

## 10.4 Strategy Builder Agent

### Responsibilities

Convert a hypothesis into an executable strategy specification.

### Required output

```yaml
strategy:
  id:
  version:
  hypothesis_id:
  universe:
  timeframe:

  entry:
    conditions: []

  exit:
    conditions: []

  position_sizing:
    model:
    max_position:

  costs:
    model:

  slippage:
    model:

  execution:
    order_type:

  risk:
    stop_loss:
    max_holding_period:
    max_trades_per_session:
```

The builder creates specifications, not arbitrary execution code.

---

# 11. Deterministic Quantitative Kernel

The quant kernel is the core of the platform.

It must remain independent of the LLM framework.

Responsibilities:

- Market data iteration
- Feature computation
- Signal evaluation
- Order generation
- Portfolio state
- Position accounting
- P&L
- Costs
- Slippage
- Risk
- Backtest event handling

The same kernel should be usable by:

```text
Backtest
Paper Trading
Future Live Adapter
```

---

# 12. Custom Backtest Engine

For MVP, build a small deterministic event-driven engine behind our own narrow interface.

Do not build a second engine merely to benchmark it against another framework.

A future integration with a third-party engine can happen only if a real limitation is identified.

## Interface

```python
class BacktestEngine:
    def run(
        self,
        dataset,
        strategy,
        execution_model,
        cost_model,
        slippage_model,
        risk_model,
        config,
    ):
        ...
```

The interface must not expose internal storage implementation.

---

# 13. Event Model

Core events:

```text
MarketBar
FeatureSnapshot
SignalGenerated
OrderSubmitted
OrderAccepted
OrderRejected
OrderFilled
PositionChanged
RiskEvent
SessionClosed
BacktestCompleted
```

Every event contains:

```text
event_id
timestamp
run_id
source
sequence
payload
```

---

# 14. Initial Market Scope

## Primary instrument

**NIFTY 50 index futures (FUTIDX), traded on NSE.**

This replaces the index-bar-first design.

The cash index may still be retained as a reference series for research features, but it is not the MVP execution instrument.

NSE identifies NIFTY 50 futures as FUTIDX contracts on the NIFTY underlying, with a three-month trading cycle. Contract expiry and lot size are exchange-defined and must be read from the versioned contract master rather than hardcoded in strategy code.

## Contract-level representation

Historical data must retain individual contract identity:

```text
symbol
expiry
contract_id
timestamp
open
high
low
close
volume
open_interest (when available)
lot_size_effective
```

## Continuous research series

A derived continuous series may be created for indicator research, but:

- The transformation method must be versioned.
- Roll dates must be deterministic.
- Roll rules must be defined before research.
- The continuous series must never be used as if it were an actual tradable contract for P&L.
- Actual paper/backtest orders must reference real dated futures contracts.

## MVP roll policy

The initial roll policy is **calendar-based**:

> Roll the research/execution contract at the start of the session that is five exchange trading sessions before the near contract expiry.

This is intentionally simple and deterministic for MVP.

The roll schedule must be generated from the official contract calendar and stored with the dataset version.

The system must later support volume/OI-based roll rules, but those are deferred until after the MVP.

## Why futures are required

A cash-index series is not itself the traded instrument. Using it as if it were directly executable would omit futures basis, actual contract pricing, lot size, expiry and roll mechanics.

NIFTY futures therefore provide the first MVP execution instrument. The current NSE contract specification confirms that NIFTY 50 futures are FUTIDX contracts on NIFTY with a three-month trading cycle; the current contract lot size is date-effective and must be sourced from the exchange's permitted-lot-size information rather than hardcoded.

## Initial data scope

The initial research dataset should target approximately four years of NIFTY futures 1-minute history, subject to actual availability and licensing.

The exact start/end dates are recorded in the locked dataset manifest after acquisition.

# 15. Dataset Lifecycle

```text
RAW
 ↓
NORMALIZED
 ↓
VALIDATED
 ↓
VERSIONED
 ↓
RESEARCH-ELIGIBLE
 ↓
LOCKED
```

A dataset version is immutable after publication.

---

# 16. Historical Data Split

The dataset must be split before broad strategy research begins.

The MVP uses four temporal zones separated by two explicit purge/embargo boundaries:

```text
Historical Dataset
│
├── Research / Development Set
│
├── Purge + Embargo Boundary
│
├── Validation / OOS Set
│
├── Purge + Embargo Boundary
│
├── Locked Historical Holdout
│
└── Future Paper Evaluation Window
```

The **Future Paper Evaluation Window** is not part of the historical dataset used for model/strategy selection. It begins only after the hypothesis templates, strategy templates and research configuration are frozen.

This matters because a historical holdout can still be indirectly contaminated by prior knowledge encoded in the research process.

The first real dataset run must use a versioned split manifest containing exact dates. No agent may modify those dates.

The split manager must persist:

```text
research_start
research_end

research_validation_purge_start
research_validation_purge_end
research_validation_embargo_start
research_validation_embargo_end

validation_start
validation_end

validation_holdout_purge_start
validation_holdout_purge_end
validation_holdout_embargo_start
validation_holdout_embargo_end

holdout_start
holdout_end

paper_forward_start
```

If the licensed dataset cannot support the required minimum sample size, the research run is blocked rather than silently widening the windows.

---

# 17. Purging and Embargoing

The system must support purged and embargoed temporal splits.

## Purging

Remove observations whose label or outcome window overlaps the boundary between datasets.

## Embargo

Insert an exclusion interval after a training or validation boundary so information can not leak through overlapping or temporally dependent observations.

Both mechanisms are mandatory whenever the prediction horizon, feature lookback, or label construction can create overlap.

---


## Boundary requirement

Purging and/or embargoing must be applied at **every temporal boundary where information could cross**:

```text
Research → Validation
Validation → Historical Holdout
Historical Holdout → Forward Paper Evaluation
```

The exact purge and embargo durations are determined from:

- maximum feature lookback
- label horizon
- execution holding period
- any other temporal dependency

The boundary configuration must be stored in the split manifest.

# 18. Final Holdout

The final historical holdout is a one-shot evaluation set.

## Rules

The holdout:

- Is created before broad research begins.
- Is stored separately.
- Is inaccessible to research agents.
- Is inaccessible to optimization code.
- Is not used for parameter selection.
- Is not used to choose features.
- Is not used to tune prompts.
- Is evaluated only through the controlled final-evaluation path.
- Is the **last historical statistical gate before paper-trading approval**.

## Failure rule

If a candidate fails the final holdout:

```text
Candidate
   ↓
FINAL HOLDOUT
   ↓
FAIL
   ↓
PERMANENTLY REJECTED
```

The candidate cannot be tuned and re-run against the same holdout.

If the holdout is used to make a tuning decision, its status is burned and a new locked holdout must be created from later unseen data.

## Synthetic holdout exercise

The end-to-end test suite must exercise the final-evaluation path using a dedicated synthetic dataset whose holdout is secret to the test runner.

This verifies the holdout-lock mechanism before any real historical holdout is ever exposed.

---

# 19. Minimum Sample Size Planning

The MVP uses **locked minimum thresholds** for the first research protocol.

These numbers are not claims of universal statistical sufficiency; they are explicit gates chosen so the first implementation is deterministic and auditable.

```yaml
sample_requirements:
  minimum_trades_total: 300
  minimum_trades_validation: 75
  minimum_trades_final_holdout: 50
  minimum_sessions_total: 250
  minimum_sessions_oos: 60
  minimum_distinct_regimes: 3
  minimum_effective_outcome_observations: 100
```

## Interpretation

- `minimum_trades_total`: at least 300 completed trades across the eligible historical research/validation path.
- `minimum_trades_validation`: at least 75 completed trades in the validation/OOS period.
- `minimum_trades_final_holdout`: at least 50 completed trades in the historical final holdout.
- `minimum_sessions_total`: evidence across at least 250 trading sessions.
- `minimum_sessions_oos`: at least 60 trading sessions in the OOS period.
- `minimum_distinct_regimes`: evidence across at least 3 predefined regime categories where the protocol provides regime labels.
- `minimum_effective_outcome_observations`: at least 100 effective outcome observations after the protocol's overlap/autocorrelation adjustment.

## Blocking rule

Any candidate failing a minimum threshold is:

```text
UNDERPOWERED
```

and cannot be promoted.

The thresholds are versioned in the research protocol and may not be changed inside an experiment.

A future research protocol may define different thresholds, but that becomes a new protocol version rather than a silent parameter change.

# 20. Research Integrity Layer

This is a first-class subsystem.

```text
research_integrity/
├── split_manager/
├── holdout_manager/
├── trial_ledger/
├── leakage_detector/
├── surrogate_tests/
├── positive_controls/
├── multiple_testing/
├── dsr/
├── pbo/
├── robustness/
└── research_budget/
```

---

# 21. Trial Ledger

The trial ledger is the authoritative record of experimentation.

The count must come from the backtest harness itself.

Agents must not self-report trial counts.

## Every trial records

```text
trial_id
experiment_id
strategy_id
dataset_version
feature_version
parameter_set
seed
execution_model
cost_model
slippage_model
timestamp_started
timestamp_completed
result
status
```

---

# 22. What Counts as a Trial

A trial is created whenever a configured backtest evaluation is executed.

Examples:

```text
One parameter set = one trial.

One strategy variant = one trial.

One feature-set variant = one trial.

One model configuration = one trial.
```

The exact definition must be centralized in the backtest harness.

---

# 23. Effective Independent Trial Count

Raw trial count is not equivalent to independent trial count.

Highly correlated parameter sweeps can produce thousands of related results.

Therefore the research integrity layer must store sufficient metadata to estimate:

```text
raw trials
correlation structure
strategy-family grouping
parameter similarity
feature similarity
effective independent trials
```

The Deflated Sharpe Ratio implementation must use an appropriate estimate of effective multiplicity rather than blindly assuming every raw trial is independent.

---

# 23A. Effective Independent Trial Count — MVP Method

The MVP uses a **versioned correlation-clustering method** identified as:

```text
EICT-CORR-1
```

## Scope

Multiplicity is cumulative across the full research population:

```text
dataset_version
+
research_protocol_version
```

It is **not** reset by:

- Research task
- Agent run
- Prompt run
- Strategy family
- Experiment
- Session
- UI job

This prevents researchers or agents from resetting the multiplicity burden by splitting work into multiple tasks on the same dataset and protocol.

## Method

1. Collect the out-of-sample return stream for every completed trial that belongs to the same `dataset_version + research_protocol_version` population.
2. Align trial return series on the common evaluation timestamps.
3. Compute pairwise Pearson correlations.
4. Convert correlation to distance:

```text
distance(i,j) = sqrt(2 * (1 - correlation(i,j)))
```

At the locked correlation threshold of `0.80`:

```text
distance = sqrt(2 * (1 - 0.80))
        ≈ 0.6325
```

5. Perform hierarchical clustering using average linkage.
6. Cut the tree at the locked distance threshold:

```text
distance_threshold = 0.6325
```

7. Each resulting cluster counts as one effective trial.

## Sparse / short trials

A trial with fewer than the minimum overlap required for pairwise correlation is **not dropped** from the effective count.

Instead:

```text
insufficient-overlap trial
        ↓
singleton cluster
        ↓
counts as 1 effective trial
```

This prevents rarely-trading strategies from artificially reducing the multiplicity burden.

The trial remains in the raw trial ledger and in the effective-trial accounting.

## Protocol parameters

```yaml
effective_trial_count_method:
  id: EICT-CORR-1
  population_scope:
    keys:
      - dataset_version
      - research_protocol_version
  correlation_measure: pearson
  distance_formula: sqrt_2_1_minus_corr
  linkage: average
  correlation_threshold: 0.80
  distance_threshold: 0.6325
  minimum_overlap_observations: 100
  insufficient_overlap_policy: singleton_cluster
```

A future protocol may replace this method, but the method ID must change. Results from different methods must not be compared as though they used the same multiplicity model.

## Cumulative ledger requirement

The backtest harness must query the authoritative trial population before calculating effective trial count. The count must not depend on self-reported experiment scope.

# 24. Research Budget

The system must enforce experiment budgets.

Budgets are scoped to:

```text
dataset_version
+
research_protocol_version
```

They accumulate across all research tasks using that population.

Example:

```yaml
research_budget:
  max_trials: 10000
  max_experiments: 100
  max_strategy_variants: 1000
  max_runtime_minutes: 120
  max_llm_cost: 5.00
```

A research job exceeding its available cumulative budget is automatically stopped.

Starting a new task does not reset the remaining budget.

The budget ledger is maintained by the platform, not by agents.


---

# 25. Multiple-Testing Controls

The research integrity layer must track:

- Number of tested hypotheses
- Number of strategy variants
- Number of parameter combinations
- Number of feature variants
- Number of model configurations
- Number of experiments

The platform should support statistical corrections and diagnostics appropriate to the selected research protocol.

---

# 26. Deflated Sharpe Ratio

The platform should implement or integrate a Deflated Sharpe Ratio calculation.

Inputs should include, as applicable:

- Observed Sharpe
- Number of trials
- Effective number of trials
- Return distribution characteristics
- Skewness
- Kurtosis
- Sample length

The raw Sharpe ratio must never be interpreted alone.

---

# 27. Probability of Backtest Overfitting

The platform should support a Probability of Backtest Overfitting methodology or an equivalent multiple-testing diagnostic.

The system must record:

- Candidate count
- Partitioning approach
- Performance distributions
- Selection process
- Validation outcomes

---

# 28. Null / Surrogate Testing

The MVP must not rely on naive row-by-row random shuffling as its only negative control.

Naive shuffling can destroy important temporal structure.

Instead the platform must support structure-preserving nulls where practical.

Examples include:

- Block bootstrap
- Stationary bootstrap
- Circular/block resampling where appropriate
- Surrogate series preserving selected dependence properties

The selected null must preserve the market properties relevant to the research question.

Potential properties to preserve:

- Volatility clustering
- Autocorrelation structure
- Time-of-day seasonality
- Session boundaries
- Distributional characteristics

---

# 29. Positive Controls

A validator that rejects everything is not a successful validator.

Therefore the test framework must include synthetic positive controls.

## Positive-control requirement

Inject a known synthetic edge into otherwise null/surrogate data.

Example:

```text
Randomized market process
+
Known deterministic signal
=
Synthetic market with known edge
```

The research pipeline should:

1. Detect the synthetic edge.
2. Rank it appropriately.
3. Produce evidence.
4. Pass validation when the signal is sufficiently strong.
5. Fail to detect it when deliberately weakened below the predefined detection threshold.

This verifies that the system has both:

- Specificity against false discoveries
- Sensitivity to a real known signal

---

# 29A. Synthetic Acquisition Gate

Before real historical data is purchased or frozen for research, the synthetic integrity suite must satisfy this gate:

```yaml
synthetic_acquisition_gate:
  surrogate_runs: 200
  false_pass_confidence: 0.95
  false_pass_upper_bound: 0.05
  false_pass_max_count_at_200: 4

  positive_control_runs: 100
  positive_control_confidence: 0.95
  positive_control_lower_bound: 0.80
  positive_control_min_detected_at_100: 87

  strong_positive_control:
    gross_edge_bps_per_triggered_trade: 12.0
    synthetic_round_trip_cost_bps: 2.0
    net_edge_bps_per_triggered_trade: 10.0
    condition_probability: 0.10
```

## Statistical criterion — false positives

The full-gate null test uses a **one-sided 95% Clopper–Pearson upper confidence bound**.

A false-pass gate is satisfied only when:

```text
Upper95(full_gate_false_pass_rate) <= 0.05
```

For exactly 200 surrogate runs, this requires **4 or fewer full-gate false passes**. Ten or fewer is not sufficient to certify a 5% rate.

A **full-gate false pass** is a surrogate run whose selected candidate passes every promotion gate, including the historical final-holdout gate.

The false-pass rate is:

```text
full_gate_false_pass_rate
= surrogate runs with a full-gate survivor / total surrogate runs
```

The measurement must include the complete gate sequence rather than stopping at an earlier validation stage.

## Statistical criterion — positive controls

The strong positive-control test uses a **one-sided 95% Clopper–Pearson lower confidence bound**.

A positive-control gate is satisfied only when:

```text
Lower95(detection_rate) >= 0.80
```

For exactly 100 positive-control runs, this requires **at least 87 detections**.

The detection rate is:

```text
detected strong positive controls / total strong positive-control runs
```

## Strong positive-control definition

The strong control is not defined only by a qualitative label. It is a versioned injected edge measured in net basis points per triggered trade.

For MVP protocol `RP-1`:

```text
Gross injected edge:       +12 bps / triggered trade
Synthetic round-trip cost:   2 bps / triggered trade
Net injected edge:          +10 bps / triggered trade
Condition probability:       10%
Execution timing:            close(t) -> open(t+1)
```

The injected edge must be detectable by the oracle strategy under the same next-bar-open execution rule used by the deterministic backtest engine.

If the synthetic cost schedule changes, the strong-control definition changes version and must not be compared directly with results from the previous protocol.

## Why the confidence-bound gate is required

A point estimate alone is too noisy at small sample sizes. The confidence-bound rule prevents a lucky 200-run result from being treated as proof that the false-positive process is at or below the target rate.

# 30. Research Integrity Test Matrix

The validator must be tested against at least:

| Dataset | Expected Result |
|---|---|
| Pure null/surrogate | Reject |
| Structure-preserving null | Reject |
| Weak synthetic edge | Possibly reject / low evidence |
| Strong synthetic edge | Detect |
| Real data with intentionally leaked feature | Reject |
| Real data with look-ahead | Reject |
| Stable synthetic edge with parameter perturbation | Survive |
| Strategy with randomized labels | Reject |
| Strategy with shuffled outcomes | Reject |

---

# 31. Look-Ahead Detection

The system must test for:

- Future bars referenced in features
- Future labels used as features
- Centered rolling windows
- Incorrect resampling
- Incorrect joins
- Future information in normalization
- Leakage through scaling
- Leakage through feature selection

Feature functions should declare required lookback windows.

---

# 32. Feature Contract

Each feature must define:

```yaml
feature:
  id:
  name:
  version:
  lookback:
  warmup:
  timestamp_semantics:
  required_columns:
  future_data_allowed: false
```

The engine must reject feature definitions that violate temporal contracts.

---

# 33. Execution Model

The MVP uses deterministic, conservative bar-based execution semantics.

## Signal timing

For a strategy evaluated on 1-minute bars:

```text
Bar t:
  open
  high
  low
  close
      ↓
Signal is evaluated at close(t)
      ↓
Earliest executable event:
open(t+1)
```

A signal generated from information available at the close of bar `t` **must not** fill at the close of bar `t`.

The earliest fillable price is therefore the open of bar `t+1`, adjusted by the configured slippage model.

## Entry timing rule

```text
signal_timestamp = close(t)
earliest_fill_timestamp = open(t+1)
```

The same rule applies to exits generated from bar-close information.

## Intrabar stop/target ambiguity

If, during one bar, both a stop-loss and take-profit could have been hit but the ordering cannot be established from the available bar data, the engine must assume the **adverse event occurs first**.

For a long position:

```text
stop touched
before
target touched
```

For a short position:

```text
stop touched
before
target touched
```

This conservative rule is mandatory for MVP and must be recorded in every backtest configuration.

## Gap handling

If an order or protective stop is triggered but the next available executable price gaps beyond the requested level, the fill occurs at the executable market price under the configured gap/slippage policy.

The engine must not manufacture a fill at an unreachable historical price.

## Same-bar order interaction

A strategy may not create a same-bar close-to-close fill unless an explicitly timestamped intrabar event exists in the dataset.

## Execution assumptions are versioned

Every run records:

```text
execution_model_id
signal_timing_rule
fill_timing_rule
intrabar_priority_rule
gap_rule
slippage_model_id
cost_schedule_version
```

# 34. Cost Model

Costs are not a single global configuration.

The MVP uses a **date-effective, versioned cost schedule** tied to the dataset and experiment protocol.

## Cost schedule

```yaml
cost_schedule:
  schedule_id: NSE_NIFTY_FUTURES_<version>
  effective_from:
  effective_to:
  brokerage:
  exchange_charges:
  transaction_taxes:
  regulatory_charges:
  gst_or_applicable_tax:
  other_configured_costs:
  source_reference:
```

A schedule may contain multiple effective periods:

```text
Period A → cost schedule v1
Period B → cost schedule v2
Period C → cost schedule v3
```

The correct schedule is selected by the contract/trade timestamp.

## Dataset binding

Every dataset version records the cost-schedule package available to it.

Every experiment records:

```text
cost_schedule_package_id
cost_schedule_version
```

## No silent updates

Changing a historical cost schedule creates a new version.

Existing experiment results remain tied to the original schedule version.

A backtest that crosses a fee/tax change must automatically apply the relevant dated schedule to each trade.

## Required cost components

Where applicable, the model must support:

- Brokerage
- Exchange charges
- Applicable transaction taxes
- Regulatory/statutory charges
- GST or other applicable tax
- Configured miscellaneous charges
- Bid/ask/spread costs when separately modeled

The actual rates are data, not hardcoded business logic.

# 35. Slippage Model

MVP support:

```text
Fixed
Percentage
Volatility-scaled
Spread-scaled
```

The selected model must be stored in the experiment metadata.

---

# 36. Strategy Interface

```python
class Strategy:
    def on_bar(self, market_state):
        ...

    def on_fill(self, fill):
        ...

    def on_session_end(self, session):
        ...
```

Strategies should not access databases, LLM APIs, or arbitrary network resources.

---

# 37. Risk Engine

The risk engine is deterministic and independent.

## MVP limits

- Max position size
- Max trades/session
- Max daily loss
- Max strategy drawdown
- Max exposure
- Max order quantity
- Max consecutive losses
- Kill switch

Risk rejection must be recorded.

---

# 38. Paper Trading

## MVP definition

MVP paper trading is **historical replay**, not live-market trading.

Recorded NIFTY futures bars are streamed through the same market-data interface that a future live feed will use.

```text
Recorded Market Data
       ↓
Replay Feed
       ↓
Strategy
       ↓
Risk Engine
       ↓
Paper Execution Simulator
       ↓
Fill
       ↓
Portfolio
       ↓
P&L
```

## Replay requirements

The replay engine must support:

- Real-time-like bar pacing
- Accelerated replay
- Pause/resume
- Session boundaries
- Contract expiry
- Contract roll
- Order lifecycle
- Costs
- Slippage
- Lot size
- Position limits

## No live credentials in MVP

The MVP must not require a broker API key.

No broker credentials are stored.

A future live-data phase may introduce a market-data-only API credential through a secrets manager, but that is outside MVP.

# 39. Strategy Registry

Every candidate receives a versioned identity.

```text
STRAT-NIFTY-000001
STRAT-NIFTY-000001-v2
```

Store:

- Hypothesis
- Strategy definition
- Features
- Dataset
- Parameters
- Trial history
- Backtest results
- Integrity results
- OOS results
- Robustness results
- Paper results
- Status
- Retirement/rejection reason

---

# 40. Experiment Registry

Experiment records:

```text
experiment_id
research_task_id
dataset_version
split_version
hypothesis_version
strategy_version
feature_versions
trial_count
effective_trial_count
random_seed
execution_model
cost_model
slippage_model
results
integrity_results
decision
```

---

# 41. Audit Log

Every consequential action must be recorded.

Examples:

- Dataset publication
- Dataset locking
- Experiment creation
- Trial execution
- Holdout evaluation
- Strategy creation
- Strategy rejection
- Strategy approval
- Paper deployment
- Risk rejection
- Kill switch activation

Audit records are append-only.

---

# 42. Paper Trading Acceptance

A strategy can enter paper trading only after it has passed:

```text
Data Quality
    ↓
Leakage Checks
    ↓
Backtest
    ↓
Cost/Slippage
    ↓
Validation / OOS
    ↓
Purged/Embargoed Validation
    ↓
Robustness
    ↓
Multiple-Testing Review
    ↓
Historical Final Holdout
    ↓
Research Integrity Gate
    ↓
Paper Replay
```

The historical final holdout is the **last historical statistical gate** before paper deployment.

## Locked MVP holdout gate

A candidate passes the historical final holdout only when all protocol conditions are satisfied:

1. `minimum_trades_final_holdout >= 50`
2. Net-of-cost mean trade return is strictly greater than zero.
3. The final-holdout mean trade return is not materially worse than the locked validation estimate beyond the protocol tolerance.

### MVP tolerance

The first protocol uses:

```yaml
final_holdout_gate:
  minimum_trades: 50
  min_net_mean_trade_return: 0
  max_relative_degradation_vs_validation: 0.50
```

Interpretation:

```text
holdout_mean >= 0
AND
holdout_mean >= 50% of validation_mean
```

This is a **consistency gate**, not a claim of statistical significance or future profitability.

If the validation mean is non-positive, the strategy cannot pass the holdout consistency gate.

No strategy is allowed to tune its threshold after seeing the holdout.

## Failure rule

If a candidate fails the final holdout:

```text
Candidate
   ↓
FINAL HOLDOUT
   ↓
FAIL
   ↓
PERMANENTLY REJECTED
```

The candidate cannot be tuned and re-run against the same holdout.

If the holdout is used to make a tuning decision, its status is burned and a new locked holdout must be created from later unseen data.

# 42A. Forward Paper Evaluation

Historical holdout is necessary but not sufficient because an LLM may have prior knowledge of public historical market outcomes.

The long-run validation sequence is:

```text
Historical Final Holdout
        ↓
Forward Paper Evaluation
```

## MVP scope

The MVP only proves the **boundary and isolation mechanism**.

It must demonstrate in tests that:

```text
research runner
    ✕
forward data

paper-evaluation path
    ✓
forward data
```

The MVP does **not** require waiting months to accumulate a real 50-trade / 60-session forward sample.

The actual forward performance result is **post-MVP**.

## Post-MVP forward evaluation

Once the protocol is frozen, new vendor-delivered data becomes eligible for forward evaluation.

The future paper window must be:

- Timestamped after protocol freeze.
- Unavailable to historical research jobs.
- Unavailable to optimization jobs.
- Unavailable to prompt-tuning jobs.
- Consumed only by the paper-evaluation path.

The Data Acquisition Record must support continued data updates for this purpose.

# 43. Robustness Testing

At minimum:

## Parameter perturbation

Nearby values.

## Cost stress

Higher costs.

## Slippage stress

Higher slippage.

## Time-window stability

Test across different historical intervals.

## Regime stability

Where available, test across distinct market conditions.

---

# 44. Monitoring

MVP monitoring must cover:

- Data freshness
- Backtest job status
- Paper order status
- Paper P&L
- Drawdown
- Risk limits
- System health

Advanced autonomous monitoring agents are deferred.

---

# 45. Reflection

MVP reflection is not a separate autonomous agent.

Instead, the Supervisor may generate structured post-run analysis from recorded results.

Store:

```text
Expected
Observed
Deviation
Likely causes
Evidence
Follow-up hypothesis
```

A dedicated Reflection Agent can be introduced later.

---

# 46. Data Architecture

## Storage

```text
Parquet
+
DuckDB
+
PostgreSQL
```

### Parquet

Historical market data.

### DuckDB

Research queries and analytical joins.

### PostgreSQL

Metadata, registries, users, jobs and audit records.

No requirement to load an entire historical dataset into pandas memory.

---

# 47. Initial Data Model

## Market Bar

```text
timestamp
symbol
open
high
low
close
volume
```

## Dataset

```text
dataset_id
version
symbol
timeframe
start_time
end_time
schema_version
row_count
checksum
status
```

---

# 48. Data Quality Report

```yaml
quality_report:
  dataset_id:
  version:
  status:
  row_count:
  duplicates:
  missing_rows:
  invalid_prices:
  invalid_volume:
  timestamp_issues:
  session_issues:
  checksum:
  blocking_issues:
```

---

# 49. API

MVP API:

```text
GET    /api/v1/datasets
GET    /api/v1/datasets/{id}
POST   /api/v1/research
GET    /api/v1/research/{id}
POST   /api/v1/hypotheses
GET    /api/v1/hypotheses/{id}
POST   /api/v1/strategies
GET    /api/v1/strategies
GET    /api/v1/strategies/{id}
POST   /api/v1/backtests
GET    /api/v1/backtests/{id}
GET    /api/v1/experiments
GET    /api/v1/trials
GET    /api/v1/paper/orders
GET    /api/v1/paper/positions
GET    /api/v1/paper/pnl
GET    /api/v1/risk/status
GET    /api/v1/audit
```

---

# 50. Web Application

MVP pages:

```text
Dashboard
Datasets
Research
Hypotheses
Strategies
Backtests
Experiments
Trials
Paper Trading
Risk
Audit
```

No advanced agent-visualization requirement for MVP.

---

# 51. Dashboard

Show:

- Current research jobs
- Dataset status
- Trials executed
- Strategy candidates
- Validation results
- Rejected candidates
- Paper P&L
- Risk state
- System state

---

# 52. Research Page

Show:

```text
Research objective
Dataset
Hypotheses
Experiments
Trial count
Candidate strategies
Validation outcomes
Research integrity report
```

---

# 53. Strategy Page

Show:

```text
Strategy ID
Hypothesis
Logic
Features
Parameters
Backtest
OOS
Robustness
Integrity
Paper result
Status
```

---

# 54. Trial Page

Show:

```text
Trial ID
Experiment
Strategy
Dataset
Parameters
Seed
Cost model
Slippage model
Result
Status
Timestamp
```

This makes the trial ledger inspectable.

---

# 55. Technology Stack

## Backend

- Python
- FastAPI
- Pydantic

## Data

- Parquet
- DuckDB
- Polars
- NumPy
- SciPy

## ML

- Scikit-learn
- XGBoost/LightGBM where required

## Agent

- LangGraph or equivalent workflow framework
- LLM API
- Structured tool calling

## Database

- PostgreSQL

## Deployment

- Docker Compose

## Frontend

- Next.js
- React
- TypeScript

The agent framework is replaceable. The quant kernel must not depend on it.

---

# 56. Repository Structure

```text
quantmind/
│
├── apps/
│   ├── api/
│   └── web/
│
├── agents/
│   ├── supervisor/
│   ├── data_quality/
│   ├── hypothesis/
│   └── strategy_builder/
│
├── data/
│   ├── ingestion/
│   ├── validation/
│   ├── schemas/
│   ├── catalog/
│   └── transforms/
│
├── quant/
│   ├── features/
│   ├── indicators/
│   ├── strategies/
│   ├── portfolio/
│   ├── execution/
│   └── risk/
│
├── backtest/
│   ├── engine/
│   ├── events/
│   ├── costs/
│   ├── slippage/
│   └── metrics/
│
├── research_integrity/
│   ├── splits/
│   ├── holdout/
│   ├── trial_ledger/
│   ├── leakage/
│   ├── nulls/
│   ├── positive_controls/
│   ├── multiple_testing/
│   ├── dsr/
│   ├── pbo/
│   └── robustness/
│
├── paper_trading/
│
├── registry/
│
├── experiments/
│
├── audit/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── research_integrity/
│   └── end_to_end/
│
├── docs/
│
├── configs/
│
├── scripts/
│
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# 57. Testing Strategy

## Unit Tests

Test:

- Indicators
- Features
- P&L
- Costs
- Slippage
- Position accounting
- Risk rules
- Split manager
- Purging
- Embargo
- Trial counting

## Integration Tests

Test:

```text
Dataset
→ Feature
→ Strategy
→ Backtest
→ Integrity
→ Registry
```

## Agent Tests

Validate structured outputs and tool permissions.

## End-to-End Test

Execute:

```text
Research Request
→ Supervisor
→ Hypothesis
→ Strategy
→ Backtest
→ Integrity
→ Registry
→ Paper Trading
```

---

# 58. Research Integrity Test Suite

This is a mandatory test suite.

## Test A — Pure null

Expected:

```text
No reliable strategy survives.
```

## Test B — Structure-preserving null

Expected:

```text
No reliable strategy survives beyond the predefined false-discovery threshold.
```

### Full-gate null criterion

Across exactly 200 surrogate runs, including the final-holdout gate:

```text
Clopper–Pearson upper95(false_pass_rate) <= 5%
```

For `n=200`, this means:

```text
full-gate false passes <= 4
```

## Test C — Strong positive control

Expected:

```text
Strong synthetic edge is detected with sufficient statistical confidence.
```

Across exactly 100 strong-control runs:

```text
Clopper–Pearson lower95(detection_rate) >= 80%
```

For `n=100`, this means at least 87 detections.

## Test D — Weak positive control

Expected:

```text
Detection degrades according to predefined threshold.
```

## Test E — Look-ahead strategy

Expected:

```text
Leakage detector blocks it.
```

## Test F — Future-feature strategy

Expected:

```text
Feature contract rejects it.
```

## Test G — Over-optimized strategy family

Expected:

```text
Multiple-testing diagnostics reduce confidence.
Effective trial count is populated.
```

## Test H — Holdout access attempt

Expected:

```text
Blocked.
```

## Test I — Trial counter bypass attempt

Expected:

```text
Impossible through public research API.
```

## Test J — Final-evaluation path

Expected:

```text
Synthetic candidate passes all earlier gates.
Final evaluation reads only the sealed synthetic holdout.
A passing candidate proceeds to the paper gate.
A failing candidate is permanently rejected.
```

## Test K — Post-failure re-tuning attempt

Expected:

```text
The failed candidate cannot access the same final holdout again.
```

## Test L — Forward paper boundary

Expected:

```text
Paper-forward data is unavailable to the historical research runner.
Only the paper-evaluation path can consume it.
```

---

# 59. Holdout Security Test

The system must include an automated test that attempts to access the final holdout through:

- API
- Agent tool
- Backtest runner
- Data query
- Database query
- File path
- Configuration

All must fail except the controlled final-evaluation path.

---

# 60. Trial-Ledger Security Test

The trial count cannot be incremented manually by agents.

The ledger is written by the backtest harness.

The system must reject attempts to:

- Modify previous trial counts
- Delete trial records
- Rewrite trial results
- Mark a trial as skipped after execution

---

# 61. Reproducibility

A research run must be reproducible using:

```text
Dataset version
Code commit
Feature version
Strategy version
Seed
Execution model
Cost model
Slippage model
Configuration
```

The system should be able to rerun the experiment and compare results.

---

# 62. Strategy Selection Rules

The system must not rank strategies by return alone.

The selection report should consider:

- OOS performance
- Robustness
- Drawdown
- Transaction costs
- Slippage
- Trade count
- Statistical significance
- Multiple-testing burden
- Parameter sensitivity
- Regime dependence

The platform should present the evidence rather than declaring a universally "best" strategy.

---

# 63. Minimum Research Output

Every completed research task must produce:

```text
Research Summary
Dataset Summary
Hypotheses
Trial Statistics
Candidate Strategies
Rejected Strategies
Validation Results
Integrity Results
Holdout Status
Paper Candidates
Limitations
Next Research Questions
```

---

# 64. Example Research Run

## User request

> Investigate NIFTY 50 futures 1-minute intraday momentum behavior.

## Supervisor

Creates research task.

## Data Quality

Checks dataset.

## Hypothesis Agent

Generates predefined, bounded hypotheses.

Examples:

```text
H1:
Short-term return persistence after abnormal futures volume.

H2:
Momentum conditioned on time of day.

H3:
Momentum conditioned on realized volatility.
```

## Strategy Builder

Converts hypotheses into executable strategies.

## Backtest

Runs deterministic trials.

## Trial Ledger

Records every execution.

## Integrity Layer

Runs:

- Leakage tests
- OOS
- Purged/embargoed validation
- Robustness
- Multiple-testing diagnostics
- Surrogate tests

## Registry

Stores surviving candidates and rejected candidates.

## Paper Trading

Only candidates satisfying approval criteria enter paper trading.

---

# 65. Example Null Experiment

The system receives the same research request but applies a structure-preserving surrogate process.

The exact same strategy-generation budget is used.

The result should not be judged by:

> "At least one strategy was profitable."

Instead the relevant comparison is:

```text
Real Data
vs
Null Data
```

across the distribution of maximum observed performance after accounting for the research process.

---

# 66. Positive Control Design

A known edge is embedded into surrogate data.

Example:

```text
Base process:
Structure-preserving null

Injected signal:
When condition X occurs,
future return receives a known deterministic component.
```

The experiment records:

```text
Signal strength
Detection rate
False positive rate
Rank among candidates
Validation outcome
```

The purpose is to prove the research system can detect real structure without simply being permissive.

---

# 67. Sample-Size Policy

The first MVP research protocol uses the locked thresholds defined in Section 19.

```yaml
sample_requirements:
  minimum_trades_total: 300
  minimum_trades_validation: 75
  minimum_trades_final_holdout: 50
  minimum_sessions_total: 250
  minimum_sessions_oos: 60
  minimum_distinct_regimes: 3
  minimum_effective_outcome_observations: 100
```

These are **MVP protocol gates**, not universal claims of statistical sufficiency.

The research protocol must record:

```text
sample_protocol_id
sample_protocol_version
thresholds
effective-observation method
research objective
prediction horizon
strategy family
```

Changing a threshold creates a new research protocol version and invalidates direct comparison with runs performed under the previous protocol.

A candidate below any required threshold is classified as `UNDERPOWERED` and cannot be promoted to the final historical holdout or paper-replay stages.

---


## Pre-run feasibility check

Before purchasing or freezing the real dataset, the split manager must perform a feasibility simulation using the proposed calendar split.

The check must answer:

```text
How many sessions are in the proposed final holdout?
How many trades would the minimum-trade threshold require per session?
How much historical trading frequency would a candidate need?
```

Example feasibility calculation:

```text
required_trades = 50
holdout_sessions = 150

required_average_frequency
= 50 / 150
≈ 0.333 trades/session
≈ 1 trade every 3 sessions
```

This is a feasibility test only. It does not assume that a real strategy will meet that frequency.

If the planned split cannot plausibly support the locked minimum observation thresholds, the split configuration must be revised **before** the real dataset is purchased/frozen.

The feasibility result becomes part of the dataset-acquisition decision record.
# 68. Acceptance Criteria

MVP cannot be declared complete unless all are true.

## Engineering

- Data ingestion works.
- Dataset versioning works.
- Backtests are deterministic.
- Paper trading works.
- Registry works.
- API works.
- UI works.

## Research Integrity

- Trial counter is harness-enforced.
- Holdout is protected.
- Purging works.
- Embargoing works.
- Leakage detection works.
- Null tests work.
- Positive controls work.
- Multiple-testing metadata is recorded.
- DSR is implemented or integrated.
- PBO methodology is implemented or integrated.
- Failed experiments remain auditable.

## Research Quality

- Real research requests produce reproducible experiments.
- The complete 200-run surrogate suite satisfies the Section 29A confidence-bound false-pass gate.
- The complete 100-run strong-positive-control suite satisfies the Section 29A confidence-bound detection gate.
- Null datasets do not routinely produce apparently strong results beyond the predefined false-discovery threshold.
- Strong synthetic controls are detectable under the locked execution model.

## Paper Trading

- Orders are simulated.
- Costs are modeled.
- Slippage is modeled.
- Risk checks are enforced.
- P&L is reproducible.

---

# 69. MVP Definition of Done

The MVP is done when the following scenario passes end-to-end.

## Synthetic end-to-end test

The complete autonomous research path is exercised on controlled synthetic data:

```text
Research Request
→ Supervisor
→ Hypothesis
→ Strategy
→ Backtest
→ Integrity
→ Final Holdout Evaluation
→ Registry
→ Paper Replay
```

The synthetic test must exercise both:

```text
passing synthetic control
```

and:

```text
failing/rejecting synthetic control
```

The real historical dataset is **not** required to produce a surviving strategy for the MVP.

A real-data run that produces zero survivors is a valid and expected result if every candidate fails the locked research-integrity gates.

## Required conditions

1. Data ingestion works.
2. Dataset versioning works.
3. Backtests are deterministic.
4. Paper replay works.
5. Registry works.
6. API works.
7. UI works.
8. Trial counter is harness-enforced.
9. Cumulative research budgets are enforced.
10. Effective independent trial count uses `EICT-CORR-1`.
11. Holdout is **sealed to agents until final evaluation**.
12. Purging and embargoing apply at every relevant dataset boundary.
13. Cost schedules are date-effective and versioned.
14. Fill timing follows the locked next-bar-open rule.
15. Intrabar stop/target ambiguity uses adverse-first ordering.
16. The 200-run surrogate suite satisfies the one-sided 95% upper-bound false-pass gate.
17. The 100-run strong-control suite satisfies the one-sided 95% lower-bound detection gate.
18. The injected edge is the versioned +10 bps net-per-triggered-trade control specified by the protocol.
19. Final-evaluation path is exercised on a synthetic sealed holdout.
20. A holdout failure permanently rejects the candidate for that protocol.
21. Audit logs are complete.
22. Research runs are reproducible.
23. Paper replay uses the same normalized market-data interface intended for future live data.

# 70. Four-to-Six-Month Part-Time Implementation Roadmap

The schedule is designed for a solo developer working part-time alongside studies/internship commitments.

The schedule may move. The MVP scope does not expand.

**Data acquisition/licensing runs in parallel with synthetic development.** It is not a prerequisite for implementing the kernel or research-integrity test suite. Real-data purchase/freeze occurs only after the synthetic integrity suite satisfies Section 29A: 200 surrogate runs, false-pass one-sided 95% upper bound <=5%, and 100 strong-positive-control runs with detection one-sided 95% lower bound >=80%.

## Task 0 — Data Acquisition & Licensing

Run this workstream in parallel with Months 1–4; do not block synthetic development on the real-data purchase.

Before real-data-dependent research:

```text
Select source
Verify terms
Verify coverage
Acquire sample
Validate fields
Record contract metadata
Freeze data-acquisition record
```

**Exit gate:** a licensed NIFTY futures 1-minute sample can be loaded and contract metadata is available.

## Month 1 — Foundation + Data

Deliver:

```text
Repository
Docker Compose
FastAPI
PostgreSQL
Configuration
Logging
Parquet
DuckDB
Dataset registry
Data ingestion
Data quality
Contract metadata
```

## Month 2 — Futures Quant Kernel

Deliver:

```text
NIFTY futures contract model
Lot-size history
Expiry calendar
Roll schedule
Feature interface
Basic futures features
Strategy interface
Portfolio accounting
P&L
Risk engine
```

## Month 3 — Deterministic Backtest

Deliver:

```text
Event loop
Orders
Fills
Futures roll handling
Costs
Slippage
Metrics
Determinism
Replay feed
```

## Month 4 — Research Integrity

Deliver:

```text
Temporal split manager
Purging
Embargo
Final holdout
Holdout access controls
Trial ledger
Trial budgets
EICT-CORR-1
Leakage tests
Null/surrogate tests
Positive controls
DSR
PBO
Robustness
```

## Month 5 — AI Research Layer

Deliver:

```text
Supervisor
Hypothesis Agent
Strategy Builder Agent
Structured tool contracts
Bounded hypothesis templates
Research budgets
Experiment registry
Strategy registry
```

## Month 6 — Paper Replay + UI + Final Verification

Deliver:

```text
Paper replay
Risk enforcement
Dashboard
Research reports
Audit log
End-to-end synthetic holdout test
Null/positive-control suite
Reproducibility suite
Failure injection
Performance validation
```

## Final MVP Gate

The project does not exit MVP until:

```text
Synthetic integrity suite PASSES
+
Real licensed dataset passes data-quality gate
+
Backtest reproducibility PASS
+
Final holdout control path PASS
+
Paper replay PASS
+
Auditability PASS
```

# 71. Post-MVP Roadmap

Only after the MVP passes all research-integrity acceptance criteria:

## Phase 2

- Pattern Discovery Agent
- Reflection Agent
- Regime Detection
- Advanced market features
- NIFTY options research
- Market-depth research
- Live market-data-only feed
- Additional futures instruments

## Phase 3

- Strategy Evolution
- Genetic programming
- Bayesian optimization
- Larger research budgets

## Phase 4

- Reinforcement learning
- Advanced simulation environments
- Distributed research

## Phase 5

- Broker adapters
- Controlled live execution
- Additional operational and regulatory controls

---

# 72. Explicit Non-Goals for Live Trading

The MVP must not:

- Send orders to a real broker.
- Store live broker credentials.
- Allow LLM-generated broker commands.
- Automatically transition paper strategies to live.
- Present backtest performance as expected future return.

A future live phase requires its own architecture and approval process.

---

# 73. Future Live Boundary

When eventually introduced:

```text
AI Research
      ↓
Strategy Artifact
      ↓
Validation Gate
      ↓
Human / Operational Approval
      ↓
Risk Gateway
      ↓
Broker Adapter
      ↓
Broker
```

The LLM remains outside direct execution authority.

---

# 74. Observability Requirements

MVP observability:

- Structured logs
- Run IDs
- Experiment IDs
- Trial IDs
- Error tracking
- Basic health checks
- Backtest duration
- Job status

Prometheus/Grafana/OpenTelemetry are deferred unless a concrete need appears.

---

# 75. Security Requirements

- Secrets outside source code.
- Agent tools use allowlists.
- Strategy execution is sandboxed where generated code is involved.
- Holdout data is access-controlled.
- Audit logs are append-only.
- API authentication is required for protected operations.
- Paper trading and research credentials are isolated.

---

# 76. Failure Handling

The MVP must safely handle:

- Dataset corruption
- Missing data
- Backtest failure
- Agent timeout
- Invalid structured output
- Database failure
- Duplicate job submission
- Partial paper execution
- Risk rejection

No failure should silently produce a valid-looking research result.

---

# 77. Idempotency

The following operations must be idempotent or explicitly deduplicated:

- Dataset registration
- Experiment creation
- Trial submission
- Paper order submission
- Fill creation
- Strategy activation
- Risk action

---

# 78. Research Artifact Storage

Each research run should produce immutable artifacts:

```text
research_report.md
experiment.json
trial_manifest.json
strategy.yaml
metrics.json
integrity_report.json
paper_deployment.json
```

Artifacts should reference exact dataset and code versions.

---

# 79. Research Report Template

```text
# Research Report

## Objective

## Dataset

## Split Design

## Hypotheses

## Experiment Budget

## Trials

## Candidate Strategies

## Backtest Results

## OOS Results

## Purging / Embargo

## Robustness

## Null / Surrogate Tests

## Positive Controls

## Multiple-Testing Analysis

## Final Holdout

## Paper Candidates

## Limitations

## Rejected Hypotheses

## Next Research Questions
```

---

# 80. Long-Term Architecture Compatibility

The MVP must be designed so later capabilities can be added without rewriting the deterministic kernel.

Future adapters:

```text
PatternDiscoveryAgent
EvolutionAgent
ReflectionAgent
RegimeAgent
RLAgent
NautilusAdapter
BrokerAdapter
DistributedBacktestAdapter
CloudExecutionAdapter
```

The MVP should therefore define stable interfaces rather than premature microservices.

---

# 81. Final Product Definition

QuantMind MVP is:

```text
An AI-assisted, deterministic, reproducible quantitative
research laboratory for NIFTY 50 futures with historical replay paper-trading capability.
```

It is NOT:

```text
A guaranteed-profit trading AI.
A black-box trading bot.
A single LLM making BUY/SELL decisions.
A fully autonomous live-trading system.
```

---

# 81A. LLM Knowledge Contamination Control

A historical holdout does not fully protect against an LLM knowing public historical outcomes before hypothesis generation.

Therefore:

## MVP mitigation

1. Use a bounded library of predefined hypothesis templates for the first research protocol.
2. Limit free-form hypothesis generation.
3. Require each hypothesis to be serialized before running experiments.
4. Record the hypothesis timestamp and protocol version.
5. Do not expose future paper-evaluation data to the research agent.
6. Do not allow prompts to contain future performance results.
7. Use the forward paper-evaluation window as a separate generalization test.

## Research record

Store:

```text
hypothesis_version
prompt_version
model_id
generation_timestamp
research_protocol_version
template_id
```

## Interpretation

Historical holdout:

```text
necessary
```

Forward paper evaluation:

```text
additional contamination/generalization check
```

Neither is treated as proof of future profitability.

# 82. Final Engineering Principle

The first version should optimize for:

```text
TRUST
    >
COMPLEXITY
```

and:

```text
REPRODUCIBILITY
    >
APPARENT PROFITABILITY
```

and:

```text
RESEARCH INTEGRITY
    >
NUMBER OF AGENTS
```

The most valuable first milestone is not:

> "The AI discovered a profitable strategy."

It is:

> "The system can generate a candidate, prove exactly how it was found, count every attempt in the harness, protect unseen data, detect leakage, distinguish null structure from a known synthetic edge, enforce locked sample-size and multiplicity protocols, replay the candidate against a real tradable futures contract, and reproduce the result."

Once that foundation is proven, more sophisticated agents can be added without turning QuantMind into an untrustworthy black box.
