# QUANTMIND PRD v4.0 ARCHITECTURAL SPECIFICATION & CLOSURE RECORD

**Version:** 4.0.0-final  
**Status:** FULLY CLOSED & VERIFIED  
**Baseline Test Suite:** 732 passed  
**Branch:** `feat/paper-evaluation-monitoring`  
**Head Commit Baseline:** `63451b2` -> `docs(v4): finalize QuantMind PRD v4.0 architecture and adversarial closure`  
**Main Branch State:** `72af8823af2aa207a4d37e81ed206927813ffd8f` (untouched, nothing pushed)

---

## 1. Executive Summary & Core Principles

QuantMind v4.0 is a deterministic quantitative research and paper evaluation laboratory for systematic trading strategies. The core architecture guarantees scientific rigor through cryptographic immutability, strict separation of concerns across four decoupled quadrants, deterministic causal execution, and automated statistical false-discovery controls.

```text
+-----------------------------------------------------------------------------------+
|                            QUANTMIND FOUR-QUADRANT ARCHITECTURE                  |
+-----------------------------------------------------------------------------------+
|  [ RESEARCH QUADRANT ]                               [ GOVERNANCE QUADRANT ]       |
|  - StrategyCompiler / StrategySpec                   - StrategyRegistry            |
|  - DatasetRegistry (LICENSED / SYNTHETIC)            - PaperGovernanceService      |
|  - SplitManager / Sealed Holdout                     - Lifecycle Transitions       |
|  - TrialLedger / Population Accounting               - State Gating Evidence       |
|  - EICT-CORR-1 / DSR Calculators                     - Non-flat Retirement Guard   |
|  - StrategyValidationGate / QualificationLedger      - Causal Timestamp Ordering   |
+------------------------------------------------------+-----------------------------+
|                                    | (Authorizes)                 | (Governs State)|
|                                    v                              v                |
+------------------------------------------------------+-----------------------------+
|  [ EXECUTION QUADRANT ]                              | [ OBSERVATION QUADRANT ]    |
|  - PaperReplayEngine / PaperExecutionLifecycleContext| - EvaluationLedger (6 tables)|
|  - Normalized ReplayFeed (FORWARD_PAPER only)        - PaperEvaluationBaseline     |
|  - Pre-Trade PaperRiskEngine (7 rules)               - PaperEvaluationRegime       |
|  - PaperLedger (Orders, Fills, Positions)            - MonitoringSnapshot          |
|  - ReplayReport (Deterministic SHA-256 Digest)       - DegradationEvent            |
|  - Strict Zero-Live-Trading Boundary                 - ResearchFeedbackRecord      |
|                                                      - ResearchFeedbackBridge      |
+-----------------------------------------------------------------------------------+
```

### Core Invariants:
1. **Four-Quadrant Strict Isolation**:
   - **Execution** never modifies governance lifecycle states or mutates evidence.
   - **Observation** passively measures replay outputs against registered baselines; it has no execution capabilities and cannot order trades.
   - **Governance** authorizes execution transitions based strictly on verified cryptographic evidence; it never places orders or liquidates positions.
   - **Research** explores hypotheses bounded by strict cumulative budgets; it has zero direct access to forward paper execution.
2. **Deterministic Replay Authorization**: Replay requires explicit `PaperExecutionLifecycleContext`. Unbound, `None`, `RETIRED`, or `REJECTED` contexts immediately halt execution (`PaperReplaySecurityError`).
3. **Immutability by Database Triggers**: All ledgers (`EvaluationLedger`, `QualificationLedger`, `PaperLedger`, `TrialLedger`, `DatasetRegistry`) enforce SQLite triggers aborting `UPDATE` and `DELETE` operations on historical evidence.
4. **Causal Monotonicity**: Evidence and governance timestamps enforce monotonic causality: `event.timestamp >= snapshot.created_at >= baseline.created_at`. Future timestamps or transitions dated prior to prior state updates are strictly rejected (`GovernanceCausalError`).
5. **No Automatic Strategy Mutation**: The Research Feedback Bridge creates structured `ResearchFeedbackTask` contexts without mutating existing strategies, historical evidence, or creating unauthorized trials.

---

## 2. Canonical Terminology Reconciliation (M8-A & M8-B)

Across the implementation of Milestones 1 through 7, naming and identifiers are unified as follows:

| Conceptual Entity | Authoritative Python Class | Authoritative Database Table | Primary Identifier / Digest | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Strategy Specification** | `StrategySpec` | Memory / Normalized JSON | `derive_strategy_id(spec)` | Deterministic SHA-256 over normalized canonical JSON. |
| **Dataset Registration** | `DatasetRecord` | `datasets`, `dataset_zones`, `dataset_files` | `version` (PK), `sha256` | Checksum-verified physical data files with typed split zones. |
| **Split Manifest** | `SplitManifest` | `split_manifests`, `holdout_audits` | `manifest_version` (PK), `manifest_sha256` | Purged and embargoed temporal boundaries with sealed holdouts. |
| **Trial Context** | `TrialContext` | `trials` | `trial_id` (PK) | Append-only trial ledger tracking population for EICT/DSR. |
| **Qualification Record** | `StrategyQualificationRecord` | `qualification_records` | `qualification_id` (PK), `record_hash` | Cryptographic seal binding spec, dataset, split, DSR, and holdout. |
| **Paper Evaluation Baseline** | `PaperEvaluationBaseline` | `paper_evaluation_baselines` | `binding_id` (PK), `binding_hash` | Binds strategy qualification to authoritative replay report hash. |
| **Paper Evaluation Regime** | `PaperEvaluationRegime` | `paper_evaluation_regimes` | `regime_hash` (PK/Hash) | Frozen forward evaluation parameters and protocol versions. |
| **Monitoring Snapshot** | `MonitoringSnapshot` | `monitoring_snapshots` | `snapshot_id` (PK), `snapshot_hash` | Periodic empirical measurement window over forward execution. |
| **Degradation Event** | `DegradationEvent` | `degradation_events` | `derived_event_id`, `event_hash` | Breached rule evidence binding snapshot, baseline, and thresholds. |
| **Governance Transition** | `PaperEvaluationTransition` | `paper_evaluation_transitions` | `transition_id` (PK), `transition_hash` | State transition audit binding old state, new state, and evidence hash. |
| **Research Feedback** | `ResearchFeedbackRecord` | `research_feedback` | `derived_feedback_id`, `feedback_hash` | Post-mortem observational package informing future research. |
| **Research Feedback Task** | `ResearchFeedbackTask` | Memory (Pydantic / dataclass) | `task_id` (`RFBT-...`) | Structured context for researcher/agent hypothesis reformulation. |

*Reconciliation Note:* The regime table in `EvaluationLedger` is named `paper_evaluation_regimes` with primary key `regime_hash`. The Python dataclass is `PaperEvaluationRegime` (residing in `src/quantmind/paper/evaluation/models.py`), whose semantic digest is `regime_hash`.

---

## 3. Authoritative Table Inventory (M8-C)

The QuantMind v4.0 architecture comprises exactly **20 relational tables** across 8 subsystems:

| Subsystem / Ledger | Table Name | Primary Key | Key Foreign Keys | Immutability Triggers | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DatasetRegistry** | `datasets` | `version` | None | `datasets_no_delete`, `datasets_no_update` | Registered dataset versions and metadata |
| | `dataset_zones` | `(version, zone_name)` | `version -> datasets(version)` | `zones_no_delete`, `zones_no_update` | Named temporal partitions (DISCOVERY, VALIDATION, etc.) |
| | `dataset_files` | `(version, relative_path)` | `version -> datasets(version)` | `files_no_delete`, `files_no_update` | File paths and SHA-256 byte digests |
| **TrialLedger** | `trials` | `trial_id` | None | `trials_no_delete`, `trials_frozen` | Research trials with budget and parameter accounting |
| **ArtifactRegistry** | `artifacts` | `artifact_id` | None | `artifacts_no_delete`, `artifacts_no_update` | SHA-256 verified Parquet return series |
| **HoldoutManager** | `split_manifests` | `manifest_version` | None | `split_manifests_no_delete` | Purge and embargo boundary specifications |
| | `holdout_audits` | `audit_id` | None | `holdout_audits_no_delete` | Access log and state tracking for sealed holdout partitions |
| **QualificationLedger** | `qualification_records` | `qualification_id` | None | `qualification_records_no_delete`, `qualification_records_no_update` | Certified statistical qualifications (EICT + DSR + Holdout) |
| **StrategyRegistry** | `strategies` | `strategy_id` | None | State machine validation hooks | Strategy lifecycle state and qualification binding |
| **PaperLedger** | `paper_orders` | `order_id` | None | `paper_orders_no_delete`, `paper_orders_no_update` | Replay simulated order audit log |
| | `paper_fills` | `fill_id` | `order_id -> paper_orders(order_id)` | `paper_fills_no_delete`, `paper_fills_no_update` | Replay fill executions with cost and slippage |
| | `paper_positions` | `strategy_id` | None | `paper_positions_no_delete` | Real-time position tracking per strategy |
| | `paper_risk_events` | `event_id` | None | `paper_risk_events_no_delete`, `paper_risk_events_no_update` | Pre-trade risk rejection audit records |
| | `replay_reports` | `report_id` | None | `replay_reports_no_delete`, `replay_reports_no_update` | Deterministic replay summary reports with SHA-256 |
| **EvaluationLedger** | `paper_evaluation_baselines` | `binding_id` | None | `baselines_no_delete`, `baselines_no_update` | Baseline binding connecting strategy to replay report |
| | `paper_evaluation_regimes` | `regime_hash` | None | `regimes_no_delete`, `regimes_no_update` | Forward evaluation regimes and monitoring configs |
| | `monitoring_snapshots` | `snapshot_id` | `regime_hash -> paper_evaluation_regimes` | `snapshots_no_delete`, `snapshots_no_update` | Periodic performance measurement windows |
| | `degradation_events` | `event_hash` | `snapshot_hash -> monitoring_snapshots` | `degradations_no_delete`, `degradations_no_update` | Breached monitoring rule records |
| | `paper_evaluation_transitions`| `transition_id`| None | `transitions_no_delete`, `transitions_no_update` | State machine transition evidence log |
| | `research_feedback` | `feedback_id` | `degradation_event_hash -> degradation_events`| `research_feedback_no_delete`, `research_feedback_no_update`| Post-mortem feedback records informing research |

---

## 4. Cryptographic Provenance & Graph Classifications (M8-D & M8-F)

The QuantMind provenance chain guarantees that no strategy can execute, degrade, or inform feedback without an unbroken cryptographic chain back to registered data.

```text
DatasetRecord (sha256) + SplitManifest (manifest_sha256)
     │
     ▼ [Edge 1: Type A]
StrategyQualificationRecord (record_hash)
     │
     ├─────────────────────────────────────────┐
     ▼ [Edge 2: Type A]                        ▼ [Edge 3: Type A]
StrategyRegistry (PAPER_ELIGIBLE)         PaperReplayEngine (Execution Authorization)
     │                                         │
     ▼ [Edge 4: Type A]                        ▼ [Edge 5: Type A]
PaperEvaluationBaseline (binding_hash)    ReplayReport (report_hash)
     │                                         │
     ├─────────────────────────────────────────┘
     ▼ [Edge 6: Type A]
PaperEvaluationRegime (regime_hash)
     │
     ▼ [Edge 7: Type A]
MonitoringSnapshot (snapshot_hash)
     │
     ▼ [Edge 8: Type A]
DegradationEvent (event_hash)
     │
     ├─────────────────────────────────────────┐
     ▼ [Edge 9: Type A]                        ▼ [Edge 10: Type A]
PaperGovernanceService.degrade_strategy   ResearchFeedbackService (Feedback Record)
     │                                         │
     ▼ [Edge 11: Type A]                       ▼ [Edge 12: Type A]
PaperEvaluationTransition (DEGRADED)      ResearchFeedbackTask (Zero-Trial Context)
```

### Provenance Edge Classification Table:

| Edge # | Source Entity | Target Entity | Edge Type | Classification Rationale & Integrity Rule |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `DatasetRecord` + `SplitManifest` | `StrategyQualificationRecord` | **Type A** | `qualification_hash` embeds `dataset_sha256` and `split_manifest_sha256`. Bitwise verification. |
| **2** | `StrategyQualificationRecord` | `StrategyRegistry` | **Type A** | Registry checks `record.verify_digest()` and matches `record.strategy_id` before entering `PAPER_ELIGIBLE`. |
| **3** | `StrategyQualificationRecord` | `PaperReplayEngine` | **Type A** | Replay engine asserts `check_paper_replay_eligibility(record)`. Rejects forged or mismatched records. |
| **4** | `StrategyQualificationRecord` | `PaperEvaluationBaseline` | **Type A** | `binding_hash` incorporates `qualification_hash` and `strategy_id`. |
| **5** | `PaperReplayEngine` | `ReplayReport` | **Type A** | Replay report computes SHA-256 digest over normalized session trade log and P&L metrics. |
| **6** | `ReplayReport` | `PaperEvaluationBaseline` | **Type A** | Baseline binds authoritative `baseline_replay_report_hash`. Must exist in ledger. |
| **7** | `PaperEvaluationRegime` | `MonitoringSnapshot` | **Type A** | Snapshot embeds `regime_hash`. Ledger verifies regime existence and context match. |
| **8** | `MonitoringSnapshot` | `DegradationEvent` | **Type A** | Event embeds `snapshot_hash` and `baseline_replay_report_hash`. Verified via `event.verify_digest()`. |
| **9** | `DegradationEvent` | `PaperEvaluationTransition` | **Type A** | Governance transition incorporates `event_hash` as `evidence_hash`. |
| **10** | `DegradationEvent` | `ResearchFeedbackRecord` | **Type A** | Feedback record embeds `degradation_event_hash` and `qualification_hash`. |
| **11** | `PaperEvaluationTransition` | `StrategyRegistry` | **Type B** | Foreign key / transition audit binding target state in registry to ledger transition record. |
| **12** | `ResearchFeedbackRecord` | `ResearchFeedbackTask` | **Type C** | Task deterministically derives `task_id` from `feedback_hash` (`derive_feedback_task_id`). |

---

## 5. Authoritative Lifecycle State Machine (M8-G)

QuantMind enforces an explicit, directional lifecycle state machine in `StrategyRegistry`:

```text
[ IDEA ]
    │
    ▼ (Research exploration under budget)
[ RESEARCH ]
    │
    ▼ (Causal preflight & backtest completion)
[ VALIDATION ]
    │
    ▼ (StrategyValidationGate certified: EICT + DSR + PASSED holdout)
[ PAPER_ELIGIBLE ]
    │
    ▼ (Authoritative PaperEvaluationBaseline registered in EvaluationLedger)
[ PAPER_ACTIVE ] ─── (Degradation Event Breached) ───► [ DEGRADED ]
    │                                                       │
    │ (Orderly flat retirement)                             │ (Flat retirement)
    ├───────────────────────────────────────────────────────┴────────► [ RETIRED ]
    │
    ▼ (Failure or operator veto)
[ REJECTED ]
```

### Evidence Requirements for State Transitions:
1. **`IDEA -> RESEARCH`**: Initial strategy specification registered; parameter schema validated.
2. **`RESEARCH -> VALIDATION`**: Completed trials in `TrialLedger`; causal preflight verified.
3. **`VALIDATION -> PAPER_ELIGIBLE`**: Strictly requires an authoritative `StrategyQualificationRecord`:
   - Valid SHA-256 digest (`record.verify_digest() == True`).
   - `final_status == ValidationStatus.PAPER_ELIGIBLE`.
   - `holdout_state == HoldoutState.PASSED`.
   - Protocol statistical sample size satisfied (`effective_trial_count`, `observed_sharpe`, `dsr`).
4. **`PAPER_ELIGIBLE -> PAPER_ACTIVE`**: Authoritative `PaperEvaluationBaseline` registered in `EvaluationLedger` bound to valid `ReplayReport`.
5. **`PAPER_ACTIVE -> DEGRADED`**: Authoritative `DegradationEvent` with valid digest, matching strategy ID and qualification hash, referencing existing `MonitoringSnapshot`.
6. **`DEGRADED -> RETIRED` or `PAPER_ACTIVE -> RETIRED`**: Strategy position quantity must be zero (`abs(position_quantity) < 1e-9`). Non-flat retirement is strictly rejected (`GovernanceIntegrityError`).

---

## 6. Execution Invariants & Deterministic Replay (M8-H)

The `PaperReplayEngine` adheres to strict operational boundaries:

1. **Explicit Lifecycle Context**: Every call to `run_replay()` or `run_fixture_replay()` requires a non-null `PaperExecutionLifecycleContext`. Passing `None` immediately raises `PaperReplaySecurityError`.
2. **State Gating Rules**:
   - `PAPER_ACTIVE`: Allows full execution (entries and exits).
   - `DEGRADED`: Strictly blocks new order entries (`allows_new_entries == False`). Only risk-reducing position exits are processed.
   - `PAPER_ELIGIBLE`: Execution prohibited until formal activation.
   - `RETIRED` / `REJECTED` / `IDEA` / `RESEARCH` / `VALIDATION`: Execution completely prohibited (`is_prohibited == True`).
3. **Execution Model (`next_bar_open_v1`)**:
   - Bar $t$ signals generate orders executed at Bar $t+1$ open.
   - Tick-size quantization (0.05 for NIFTY futures).
   - Date-effective fee schedules and configurable slippage models.
4. **Zero-Live-Trading Boundary**: The engine contains no network sockets, broker API clients, or live order routing. All operations are simulated.

---

## 7. Dataset & Holdout Security Models (M8-I)

1. **Dataset Integrity**:
   - Every file registered in `DatasetRegistry` is verified against its SHA-256 digest.
   - Production research accepts only `DatasetKind.LICENSED`. Synthetic data is strictly barred from production qualification.
2. **Purged and Embargoed Partitions**:
   - `SplitManager` enforces temporal purge windows and post-validation embargoes to prevent autocorrelation leakage.
3. **Sealed Final Holdout**:
   - The `FINAL_HOLDOUT` zone is locked and tamper-evident.
   - Any attempt to load `FINAL_HOLDOUT` data during research or replay raises `MarketFeedSecurityError` or `HoldoutSecurityError`.
   - Access is permitted exactly once during final validation gating via `HoldoutManager.evaluate_holdout()`. Re-evaluation is permanently barred.

---

## 8. Multiple-Testing Population Accounting & False-Discovery Controls (M8-J)

1. **Comprehensive Population Accounting**:
   - Every research trial is logged in `TrialLedger` (append-only).
   - Deleting, overwriting, or modifying trials is blocked by SQLite triggers (`trials_no_delete`, `trials_frozen`).
2. **EICT-CORR-1 Formulation**:
   - Hierarchical average-linkage clustering over out-of-sample return series.
   - Correlation distance threshold $d = 1 - \rho$.
   - Singleton clusters represent independent strategy bets.
3. **Deflated Sharpe Ratio (DSR)**:
   - Adjusts observed Sharpe ratio for multiple testing based on Bailey & López de Prado (2014).
   - Computes expected maximum Sharpe ratio under the null hypothesis given the effective number of independent trials ($N$).
   - Rejects strategies failing the protocol DSR confidence threshold (e.g. $DSR < 0.95$).

---

## 9. Research Feedback Bridge & Loop (M8-K)

1. **Controlled Feedback Ingestion**:
   - Degradation events generate structured `ResearchFeedbackRecord` instances in `EvaluationLedger`.
   - Records capture empirical failure modes (e.g., `DRAWDOWN_EXPANSION`, `SLIPPAGE_DIVERGENCE`), realized Sharpe ratios, and slippage divergence.
2. **Strict Non-Mutation Invariant**:
   - Feedback records are strictly observational.
   - They contain no trial ID, create no trials in `TrialLedger`, and do not adjust multiple-testing counts.
   - They cannot directly promote a strategy or modify historical evaluation evidence.
3. **Task Formulation**:
   - `create_research_task_from_feedback()` produces a `ResearchFeedbackTask` containing failure analysis for researchers or future LLM agents to formulate new hypotheses.

---

## 10. Adversarial Verification Matrix (26 Scenarios) (M8-M & M8-L)

All 26 adversarial scenarios are implemented and verified in `tests/integration/test_m8_cross_milestone_adversarial.py`:

| # | Adversarial Attack / Boundary Scenario | Target Module / Boundary | Enforced Defense / Invariant | Test Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Forged qualification record (altered Sharpe / corrupted digest) | `StrategyValidationGate` | Digest verification fails; execution blocked | **PASSED** |
| **2** | Failed holdout qualification promoted to PAPER_ELIGIBLE | `StrategyRegistry` | `StrategyRegistryError` raised on transition attempt | **PASSED** |
| **3** | Holdout re-evaluation attempt | `HoldoutManager` | `HoldoutSecurityError` raised; single-eval invariant | **PASSED** |
| **4** | Unregistered dataset used in research / replay | `DatasetRegistry` | `DatasetRegistryError` raised; unverified data blocked | **PASSED** |
| **5** | Checksum mismatch on dataset file | `DatasetRegistry` | Bitwise SHA-256 verification fails; load rejected | **PASSED** |
| **6** | Synthetic data evaluated as production paper strategy | `StrategyValidationGate` | `DatasetKind.SYNTHETIC` rejected for production gate | **PASSED** |
| **7** | Sealed `FINAL_HOLDOUT` data accessed during replay | `ReplayFeed` | `MarketFeedSecurityError` raised; holdout isolation | **PASSED** |
| **8** | Strategy in `PAPER_ELIGIBLE` attempts order entry | `PaperReplayEngine` | Context `allows_new_entries == False`; orders blocked | **PASSED** |
| **9** | Strategy in `DEGRADED` attempts new position entry | `PaperReplayEngine` | Context blocks entry orders; order rejected | **PASSED** |
| **10** | Strategy in `DEGRADED` attempts risk-reducing exit | `PaperReplayEngine` | Context permits exit orders; fills processed | **PASSED** |
| **11** | Strategy in `RETIRED` attempts replay execution | `PaperReplayEngine` | `PaperReplaySecurityError` raised; execution prohibited | **PASSED** |
| **12** | Strategy in `REJECTED` attempts replay execution | `PaperReplayEngine` | `PaperReplaySecurityError` raised; execution prohibited | **PASSED** |
| **13** | Forged degradation event (bad digest / fabricated fields) | `DegradationEvent` | `verify_digest()` returns False; governance rejects | **PASSED** |
| **14** | Degradation event referencing non-existent / mismatched strategy | `PaperGovernanceService` | `KeyError` / `StrategyRegistryError` raised | **PASSED** |
| **15** | Cross-regime / mismatched context snapshot | `EvaluationLedger` | `EvaluationLedgerIntegrityError` raised at ledger | **PASSED** |
| **16** | Transition timestamp earlier than degradation event timestamp | `PaperGovernanceService` | `GovernanceCausalError` raised; non-causal transition blocked | **PASSED** |
| **17** | Research feedback for mismatched or non-existent strategy | `ResearchFeedbackService` | `ResearchFeedbackIntegrityError` raised | **PASSED** |
| **18** | Duplicate research feedback registration | `ResearchFeedbackService` | Idempotently returns existing record; zero duplicates | **PASSED** |
| **19** | Feedback task creation impacts `TrialLedger` | `ResearchFeedbackBridge` | `TrialLedger` trial count remains unchanged (zero trials) | **PASSED** |
| **20** | Feedback task attempts direct state promotion to `PAPER_ACTIVE` | `StrategyRegistry` | `StrategyRegistryError` raised; bypass impossible | **PASSED** |
| **21** | Lifecycle state mutation during replay run | `PaperReplayEngine` | Execution results deterministic; context frozen | **PASSED** |
| **22** | Direct SQL `UPDATE` / `DELETE` on evaluation evidence | `EvaluationLedger` | SQLite triggers abort queries; evidence permanent | **PASSED** |
| **23** | Retirement attempted with open non-flat position | `PaperGovernanceService` | `GovernanceIntegrityError` raised; flat position required | **PASSED** |
| **24** | Illegal reverse lifecycle state transition | `StrategyRegistry` | `StrategyRegistryError` raised; illegal transitions blocked | **PASSED** |
| **25** | Direct SQL `DELETE` on trial population | `TrialLedger` | SQLite trigger `trials_no_delete` aborts deletion | **PASSED** |
| **26** | Strategy identity mutation through parameter modification | `StrategyCompiler` | Semantic hash over canonical spec prevents tampering | **PASSED** |

---

## 11. PRD v4.0 Requirement-by-Requirement Alignment Table (M8-N)

| Requirement ID | Specification Requirement | Realizing Component / File | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **PRD-4.0-REQ-01** | Checksum-verified physical dataset management with typed split zones | `DatasetRegistry` (`src/quantmind/data/registry.py`) | Checksums verified; 3 tables; tamper-evident |
| **PRD-4.0-REQ-02** | Purged and embargoed temporal partitions and sealed final holdout | `SplitManager`, `HoldoutManager` (`src/quantmind/research_integrity/holdout.py`) | Zero-leakage temporal boundaries; single-eval holdout |
| **PRD-4.0-REQ-03** | Cumulative research budget & append-only population accounting | `TrialLedger` (`src/quantmind/research_integrity/trial_ledger.py`) | Budget caps enforced; SQL triggers prevent deletion |
| **PRD-4.0-REQ-04** | Out-of-sample Parquet return series storage with byte verification | `ArtifactRegistry` (`src/quantmind/research_integrity/artifacts.py`) | SHA-256 byte verification over persisted series |
| **PRD-4.0-REQ-05** | Multiple-testing false discovery adjustment via EICT-CORR-1 & DSR | `EictCorr1Calculator`, `DeflatedSharpeCalculator` | Bailey & López de Prado (2014) formulation |
| **PRD-4.0-REQ-06** | Authoritative Strategy Validation Gate & Qualification Records | `StrategyValidationGate`, `QualificationLedger` | Immutable qualification record with SHA-256 digest |
| **PRD-4.0-REQ-07** | Directional Strategy Lifecycle State Machine | `StrategyRegistry` (`src/quantmind/strategy/registry.py`) | 7 lifecycle states; qualification gating enforced |
| **PRD-4.0-REQ-08** | Normalized high-performance Market Data Replay Feed | `ReplayFeed` (`src/quantmind/paper/feed.py`) | Vectorized NumPy streaming; holdout access barred |
| **PRD-4.0-REQ-09** | Pre-trade deterministic risk controls (7 rules) | `PaperRiskEngine` (`src/quantmind/paper/risk.py`) | 7 pre-trade rules; immutable risk event audit |
| **PRD-4.0-REQ-08** | Deterministic causal paper execution model (`next_bar_open_v1`) | `PaperReplayEngine` (`src/quantmind/paper/engine.py`) | Next-bar-open execution; tick quantization; fee schedules |
| **PRD-4.0-REQ-11** | Append-only paper execution ledger | `PaperLedger` (`src/quantmind/paper/ledger.py`) | 5 tables; SQLite triggers abort delete/update |
| **PRD-4.0-REQ-12** | Deterministic Replay Summary Reports | `ReplayReport` (`src/quantmind/paper/models.py`) | SHA-256 digest over canonical execution metrics |
| **PRD-4.0-REQ-13** | Paper Evaluation Baseline & Regime Registration | `EvaluationLedger` (`src/quantmind/paper/evaluation/ledger.py`) | Binds qualification to authoritative baseline report |
| **PRD-4.0-REQ-14** | Forward Monitoring Snapshots & Degradation Detection | `PaperEvaluationService`, `DegradationDetector` | Rolling metric windows; threshold breach events |
| **PRD-4.0-REQ-15** | Governance Lifecycle Degradation & Safe Flat Retirement | `PaperGovernanceService` (`src/quantmind/paper/evaluation/governance.py`) | Causal timestamps; non-flat position retirement blocked |
| **PRD-4.0-REQ-16** | Observational Research Feedback Bridge | `ResearchFeedbackService`, `ResearchFeedbackBridge` | Non-mutating feedback records; zero-trial tasks |
| **PRD-4.0-REQ-17** | Complete Database Immutability via SQL Triggers | All 8 Ledgers (20 Tables) | 16 BEFORE UPDATE/DELETE triggers abort tampering |
| **PRD-4.0-REQ-18** | End-to-End Cryptographic Provenance Chain | Full Architecture (Type A Provenance Edges) | Cryptographic digests verify end-to-end auditability |
