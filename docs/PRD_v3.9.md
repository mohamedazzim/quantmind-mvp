# QuantMind — Product Requirements Document v3.9
## Futures-First MVP: Deterministic Paper Replay Engine, Market Data Replay Feed, & Pre-Trade Risk Controls

**Version:** 3.9  
**Status:** Locked MVP baseline for deterministic paper replay execution  

**Scope:** MVP only  
**Primary market:** Indian index futures research, beginning with NIFTY 50 futures  
**Initial data granularity:** NIFTY 50 futures 1-minute historical data  
**Execution mode:** Deterministic historical & forward recorded-data replay (strictly non-live)  
**Architecture:** Modular monolith  
**Primary objective:** Build an auditable, strictly deterministic paper replay engine and market data streaming interface that enforces qualification invariants, pre-trade risk controls, date-effective transaction costs, and multi-session position accounting without live broker execution.

**Synthetic research protocol:** RP-2  

---

## Changelog

### v3.9 (Deterministic Paper Replay Engine & Normalized Replay Feed)
- **Normalized Market Data Interface (`MarketDataFeed`, `ReplayFeed`, `ReplayBar`)**:
  - Implemented in `quantmind.paper.feed`.
  - Abstract base class `MarketDataFeed` defining the contract for both recorded replay and future streaming feeds (`symbol`, `lot_size`, `tick_size`, `stream_bars`).
  - High-performance, zero-overhead streaming via columnar NumPy arrays pre-extracted upon initialization (eliminates costly per-bar Pandas `.iloc` indexing).
  - Explicit multi-session boundary tracking (`session_id` derived per calendar session).
  - Playback control methods: `pause()`, `resume()`, and `reset()`.
  - Prohibits direct access to sealed `FINAL_HOLDOUT` partitions (`MarketFeedSecurityError`).
  - Authoritative dataset loader: `ReplayFeed.from_dataset_registry` enforcing checksum verification and prohibiting synthetic/fixture datasets from production feeds.
- **Strict Qualification Gate & Provenance Boundary**:
  - Implemented in `PaperReplayEngine.run_replay`.
  - Rejects raw or uncertified strategy specifications; strictly accepts only authoritative `StrategyQualificationRecord` artifacts.
  - Requires `final_status == ValidationStatus.PAPER_ELIGIBLE` and `holdout_state == "PASSED"`.
  - Cryptographic verification via `record.verify_digest()` rejecting any tampered records.
  - Enforces bitwise consistency between the provided `StrategySpec` and the qualification record's `strategy_id` and `strategy_spec_hash`.
  - Enforces dataset version and dataset sha256 consistency between `ReplayFeed`, `StrategyQualificationRecord`, and `DatasetRegistry`.
  - Enforces research protocol version match (`expected_protocol_version`).
  - Explicitly bars `FINAL_HOLDOUT` partition access during paper replay.
- **Deterministic Execution & Position Model (`next_bar_open_v1`)**:
  - Implemented in `quantmind.paper.engine`.
  - Causal execution: Signal generated on completed bar $t$ (using close) executes strictly at the open of bar $t+1$.
  - Realistic slippage modeling: configurable bps per side (`slippage_bps_per_side`) applied against bar open price.
  - Price quantization: fills strictly rounded to instrument `tick_size` (e.g., 0.05 for NIFTY futures).
  - Date-effective transaction costs: integrates `CostSchedule` to resolve historical round-trip bps for the specific bar date, applied per side on entry and exit.
  - Position accounting: contract lot size multiplier (`lot_size`), multi-bar holding duration (`hold_bars`), session-boundary intraday liquidation (`enforce_session_boundaries`), and accurate realized/unrealized P&L tracking.
- **Pre-Trade Deterministic Risk Engine (`PaperRiskConfig`, `PaperRiskEngine`)**:
  - Implemented in `quantmind.paper.risk`.
  - Evaluates each order before submission against 7 deterministic risk controls:
    1. `kill_switch`: Immediate emergency halt blocking all order submissions.
    2. `max_order_quantity`: Cap on individual order size.
    3. `max_position`: Cap on cumulative net position post-fill.
    4. `max_trades_per_session`: Maximum trade executions allowed per calendar session.
    5. `max_daily_loss`: Cumulative realized loss limit per session/day.
    6. `max_strategy_drawdown`: Strategy peak-to-trough drawdown threshold.
    7. `max_exposure`: Cap on gross portfolio exposure ($P \times Q \times \text{lot\_size}$).
  - Violations generate immutable `PaperRiskEvent` audit records and immediately set order status to `REJECTED`.
- **Append-Only Paper Ledger (`PaperLedger`)**:
  - Implemented in `quantmind.paper.ledger`.
  - SQLite database backing tables: `paper_orders`, `paper_fills`, `paper_positions`, `paper_risk_events`, `replay_reports`.
  - Protected by SQLite triggers (`paper_orders_no_delete`, `paper_fills_no_delete`, `paper_risk_no_delete`, `replay_reports_no_delete`) preventing any audit tampering or deletions.
- **Deterministic Observational Replay Report (`ReplayReport`)**:
  - Implemented in `quantmind.paper.models`.
  - Summarizes trade count, gross P&L, net P&L, transaction costs, slippage, max drawdown, exposure, win rate, expectancy, annualized Sharpe, and per-session breakdowns (`ReplaySessionSummary`).
  - Deterministic SHA-256 `report_hash` computed over canonical JSON representation.
  - Strictly observational: does not make predictive claims or compute relative strategy rankings.
- **Explicit No-Live-Trading Security Boundary**:
  - All paper engine components are strictly simulated.
  - Codebase contains zero broker credentials (API keys, secret keys, access tokens, webhook URLs, external sockets, or live routing mechanisms).
- **Test Suite Expansion**:
  - 65 new tests added (adversarial suite covering qualification tampering, execution causality, ledger SQL triggers, risk limits, feed validation, and complete provenance closure).
  - Total test suite expanded from **283 passed** to **348 passed** (0 failures, 0 warnings, clean compileall).

---

# 1. System Lifecycle Architecture

QuantMind enforces an unbroken chain of custody from hypothesis generation to deterministic paper replay:

```text
Research Objective
        ↓
Data Quality (DatasetRegistry + SplitManifest)
        ↓
Hypothesis & StrategySpec (Whitelist Compiler)
        ↓
Deterministic Backtest (RESEARCH / VALIDATION)
        ↓
Research Integrity Layer (TrialLedger + ArtifactRegistry + EICT-CORR-1 + DSR)
        ↓
Strategy Validation Gate
        ↓
Sealed Final Holdout Evaluation (HoldoutManager)
        ↓
Immutable Strategy Qualification Record (QualificationLedger: PAPER_ELIGIBLE)
        ↓
Strategy Registry Promotion (StrategyLifecycleState.PAPER_ELIGIBLE)
        ↓
Deterministic Paper Replay Engine (ReplayFeed + PaperRiskEngine + PaperLedger)
        ↓
Deterministic ReplayReport (SHA-256 report_hash)
```

---

# 2. Market Data Replay Feed Specification

### 2.1 Contract & Interface
The abstract class `MarketDataFeed` establishes the unified contract:
- `symbol`: Trading instrument ticker (e.g. `NIFTY_FUT`).
- `lot_size`: Multiplier per contract (e.g. 50).
- `tick_size`: Minimum price movement increment (e.g. 0.05).
- `stream_bars()`: Chronological iterator producing immutable `ReplayBar` records.

### 2.2 ReplayFeed Implementation
- **Fast Vectorized Streaming**: Upon initialization, DataFrame columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `open_interest`) are copied into contiguous 1D NumPy arrays. Bar iteration yields lightweight `ReplayBar` objects without runtime DataFrame slicing.
- **Session ID Resolution**: Each bar timestamp is normalized to calendar date strings (`YYYY-MM-DD`), enabling intra-session and cross-session tracking.
- **Feed Controls**:
  - `pause()`: Pauses bar consumption stream.
  - `resume()`: Resumes bar consumption.
  - `reset()`: Rewinds cursor to bar index 0.
- **Partition Security**:
  - Direct initialization on `SplitZone.FINAL_HOLDOUT` is rejected with `MarketFeedSecurityError`.
  - `ReplayFeed.from_dataset_registry` validates dataset existence, verifies sha256 checksums, enforces `DatasetKind.LICENSED` for production replays, and loads authorized partitions (`FORWARD_PAPER`).

---

# 3. Execution & Risk Engine Specification

### 3.1 Order Lifecycle & Fill Timing (`next_bar_open_v1`)
Paper replay strictly enforces causal order execution:
1. **Signal at Bar Close**: Strategy evaluates bar data up to bar $t$. If signal $\neq 0$, a pending `PaperOrder` is created with timestamp $t_{\text{close}}$.
2. **Pre-Trade Risk Check**: At the beginning of bar $t+1$ (prior to fill generation), the pending order is evaluated by `PaperRiskEngine`.
   - If risk limit breached: Order status set to `REJECTED`, `PaperRiskEvent` logged, order terminated.
   - If approved: Order proceeds to fill generation.
3. **Execution at Bar Open**: Order executes at bar $t+1$ open price ($P_{\text{open}}$).
4. **Slippage Application**:
   $$\text{Raw Fill} = P_{\text{open}} \times (1 + \text{side} \times \text{slippage\_bps} / 10000)$$
5. **Tick Size Rounding**: Fill price is rounded to nearest valid tick increment:
   $$\text{Fill Price} = \text{round}(\text{round}(\text{Raw Fill} / \text{tick\_size}) \times \text{tick\_size}, 4)$$
6. **Cost Schedule**: Date-effective round-trip bps obtained from `CostSchedule.round_trip_bps_at(timestamp)`. Entry and exit each incur half the round-trip bps.

### 3.2 Pre-Trade Risk Controls
The `PaperRiskEngine` maintains deterministic intra-session portfolio state:
- Realized loss resets on calendar session change.
- Peak equity tracked continuously across all bars.
- Order submission is evaluated synchronously before any fill is generated.
- Any breach generates a signed `PaperRiskEvent` and prevents position accumulation.

### 3.3 Position & P&L Accounting
- Fills update `PaperPosition` state (signed quantity, weighted entry price, realized P&L, fees).
- Intra-day positions held for `hold_bars` or until session close (when `enforce_session_boundaries=True`).
- Position snapshots recorded at the close of every bar in `PaperLedger`.

---

# 4. Audit & Verification Guarantees

### 4.1 Strict Replay Determinism
- Any two executions of `PaperReplayEngine` with identical `StrategyQualificationRecord`, `StrategySpec`, and `ReplayFeed` produce bitwise identical orders, fills, trade lists, and `report_hash`.
- `ReplayReport.canonical_json()` provides a reproducible, order-invariant representation for SHA-256 verification.

### 4.2 Tamper Resistance, Provenance Closure & Immutability
- `PaperLedger` rejects both `DELETE` and `UPDATE` SQL queries via SQLite database triggers across all 5 tables: `paper_orders`, `paper_fills`, `paper_risk_events`, `paper_positions`, and `paper_reports`.
- `paper_reports` enforces primary key uniqueness on `report_hash`, preventing report overwriting or state mutation.
- `ReplayReport` cryptographically binds all 19 result-affecting parameters into its canonical representation and `report_hash`:
  1. `strategy_spec_hash`: SHA-256 of normalized strategy spec
  2. `qualification_hash`: SHA-256 audit digest of `StrategyQualificationRecord`
  3. `dataset_version`: Authoritative dataset version string
  4. `dataset_sha256`: SHA-256 disk checksum of market dataset
  5. `research_protocol_version`: Governed research protocol version
  6. `symbol`: Instrument symbol
  7. `lot_size`: Contract lot multiplier
  8. `tick_size`: Instrument minimum price increment
  9. `slippage_bps_per_side`: Slippage applied per side in bps
  10. `cost_schedule_id`: Identifier of applied transaction fee schedule
  11. `cost_schedule_hash`: SHA-256 over canonical schedule periods and rates
  12. `risk_config_hash`: SHA-256 over canonical risk limit parameters
  13. `execution_policy`: Locked execution model identifier (`next_bar_open_v1`)
  14. `quantity`: Order size in contracts
  15. `hold_bars`: Trade holding horizon in bars
  16. `initial_capital`: Replay capital base in currency units
  17. `enforce_session_boundaries`: Boolean flag controlling intra-day liquidation
  18. `split_zone`: Authoritative dataset split partition (`FORWARD_PAPER`)
  19. `bars_sha256`: SHA-256 digest of immutable contiguous raw bar buffers
- Replay engine rejects any qualification record where `record.verify_digest()` fails.
- `ReplayFeed` enforces strict preflight validation: rejects empty datasets, missing columns, NaNs, infinities, non-monotonic or duplicate timestamps, non-positive prices, and invalid OHLC bounds (`high < low`, `high < open`, etc.). Disk checksums are verified against the registry via SHA-256 before loading.
- `PaperRiskEngine` scales exposure calculations by contract `lot_size` ($Q \times P \times \text{lot\_size}$) and guarantees risk-reducing liquidation orders cannot be trapped if drawdown or loss thresholds are breached.

### 4.3 Security Boundaries
- Live broker execution is completely excluded by design.
- No network APIs, credentials, or live order routing endpoints exist.

### 4.4 Replay Boundary Isolation & Anti-Bypass Hardening
- **Exact Type Enforcement**: `PaperReplayEngine.run_replay` rejects subclasses of `StrategyQualificationRecord`, `StrategySpec`, and `ReplayFeed` (`type(feed) is not ReplayFeed`), closing class-inheritance hijacking and method override vulnerabilities.
- **In-Memory Buffer Immutability**: All 8 columnar NumPy arrays (`_timestamps`, `_opens`, `_highs`, `_lows`, `_closes`, `_volumes`, `_open_interests`, `_session_ids`) in `ReplayFeed` are frozen (`flags.writeable = False`). In-place price modifications or TOCTOU mutations are prevented at the memory level.
- **Authoritative Registry Feed Verification**: Replay feeds created from `DatasetRegistry.from_dataset_registry` are authenticated (`feed.is_authoritative = True`) and bind `feed.dataset_sha256`. Production replay against registered datasets requires authoritative feeds, rejecting unverified direct constructor feeds.
- **Sealed Partition Isolation**: Access to `SplitZone.FINAL_HOLDOUT` is systematically prohibited across `ReplayFeed.__init__`, `ReplayFeed.from_dataset_registry`, `DatasetRegistry.load_zone`, and `PaperReplayEngine.run_replay`.
