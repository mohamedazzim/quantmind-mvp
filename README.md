# QuantMind MVP

Futures-first AI-assisted quantitative research laboratory.

## Current build stage

**Stage 1: deterministic backtester + synthetic research-integrity harness**

Current verification: **51 tests passing** across synthetic market generation, real gap/wick semantics, cross-session surrogates, directional intraday nulls, recomputable causal positive controls, independent wick preservation, paired edge recovery, one-sided confidence-bound gates, and the look-ahead canary.

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
12. Purged/embargoed split manager and sealed holdout
13. DSR/PBO research-integrity layer
14. Bounded research agents

## Run tests

```bash
python -m pip install -e '.[dev]'
pytest -q
```

## Backtester scope

The current engine is a deterministic **one-bar-hold regression harness**, not yet the complete Section 33 execution engine. Research trials must enter through `ResearchHarness.run_trial()`, which owns checksum-verified dataset resolution, mandatory causality preflight, typed StrategySpec compilation, harness-generated trial/strategy/experiment IDs, cumulative budget reservation, failure recording, and the internal engine call. Production research trials accept only named LICENSED dataset versions and split zones; caller-supplied DataFrames, precomputed signal columns, and arbitrary signal callables are excluded from the production API. Synthetic fixtures live in `quantmind.testing` and are recorded with `mode=FIXTURE`. Position state, stop/target handling, intrabar adverse-first resolution, futures roll handling, and risk integration remain later work.
