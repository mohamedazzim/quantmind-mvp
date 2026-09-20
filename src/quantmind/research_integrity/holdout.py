from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Mapping
import uuid

import pandas as pd

from quantmind.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from quantmind.backtest.strategies import NonCausalSignalError, assert_causal_signal
from quantmind.data import DatasetKind, DatasetRegistry
from quantmind.data.splits import SplitZone
from quantmind.strategy import StrategySpec, compile_strategy_spec, normalize_strategy_spec
from quantmind.strategy.compiler import derive_strategy_id
from .trial_ledger import ResearchBudget, TrialContext, TrialLedger, derive_experiment_id



class HoldoutSecurityError(RuntimeError):
    """Raised when holdout isolation rules or security constraints are violated."""


class HoldoutState(str, Enum):
    UNTOUCHED = "UNTOUCHED"
    EVALUATED = "EVALUATED"
    PASSED = "PASSED"
    FAILED = "FAILED"
    BURNED = "BURNED"


@dataclass(frozen=True)
class FinalEvaluationResult:
    candidate_strategy_id: str
    dataset_version: str
    research_protocol_version: str
    state: HoldoutState
    trial_id: str
    backtest_result: BacktestResult | None
    status: str
    reason: str | None = None


class HoldoutManager:
    """Manages sealed final holdout evaluation and holdout lifecycle state machine.

    Ensures:
    1. Only candidates passing earlier research gates can evaluate against holdout.
    2. Holdout is evaluated exactly once per candidate/protocol.
    3. Failed candidates cannot retune against the same holdout.
    4. Burned holdouts cannot be evaluated or tuned against.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        ledger: TrialLedger,
        dataset_registry: DatasetRegistry,
        db_path: str | Path | None = None,
    ) -> None:
        self._engine = engine
        self._ledger = ledger
        self._dataset_registry = dataset_registry
        if db_path is not None:
            self._connection = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,
            )
        else:
            # Share the ledger connection if accessible, or create memory db
            self._connection = ledger._connection
        self._connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS holdout_evaluations (
                candidate_strategy_id TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                research_protocol_version TEXT NOT NULL,
                holdout_version TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('UNTOUCHED', 'EVALUATED', 'PASSED', 'FAILED', 'BURNED')),
                trial_id TEXT,
                timestamp_evaluated TEXT NOT NULL,
                result_json TEXT,
                PRIMARY KEY (candidate_strategy_id, dataset_version, research_protocol_version)
            );

            CREATE TABLE IF NOT EXISTS holdout_burns (
                dataset_version TEXT NOT NULL,
                research_protocol_version TEXT NOT NULL,
                burned_at TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (dataset_version, research_protocol_version)
            );

            CREATE TRIGGER IF NOT EXISTS holdout_eval_no_delete
            BEFORE DELETE ON holdout_evaluations
            BEGIN
                SELECT RAISE(ABORT, 'holdout evaluations are permanent and append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS holdout_eval_no_reset
            BEFORE UPDATE OF state ON holdout_evaluations
            WHEN OLD.state IN ('FAILED', 'PASSED', 'BURNED')
            BEGIN
                SELECT RAISE(ABORT, 'evaluated holdout state is permanent and cannot be reset');
            END;
            """
        )

    def is_holdout_burned(self, dataset_version: str, research_protocol_version: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM holdout_burns WHERE dataset_version = ? AND research_protocol_version = ?",
            (dataset_version, research_protocol_version),
        ).fetchone()
        return row is not None

    def burn_holdout(self, dataset_version: str, research_protocol_version: str, reason: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._connection.execute(
            """
            INSERT OR REPLACE INTO holdout_burns(dataset_version, research_protocol_version, burned_at, reason)
            VALUES (?, ?, ?, ?)
            """,
            (dataset_version, research_protocol_version, now, reason),
        )

    def get_holdout_state(
        self,
        candidate_strategy_id: str,
        dataset_version: str,
        research_protocol_version: str,
    ) -> HoldoutState:
        if self.is_holdout_burned(dataset_version, research_protocol_version):
            return HoldoutState.BURNED
        row = self._connection.execute(
            """
            SELECT state FROM holdout_evaluations
            WHERE candidate_strategy_id = ? AND dataset_version = ? AND research_protocol_version = ?
            """,
            (candidate_strategy_id, dataset_version, research_protocol_version),
        ).fetchone()
        if row is None:
            return HoldoutState.UNTOUCHED
        return HoldoutState(row["state"])

    def has_candidate_failed(
        self,
        candidate_strategy_id: str,
        dataset_version: str,
        research_protocol_version: str,
    ) -> bool:
        state = self.get_holdout_state(candidate_strategy_id, dataset_version, research_protocol_version)
        return state == HoldoutState.FAILED

    def final_evaluate(
        self,
        strategy_spec: StrategySpec,
        dataset_version: str,
        research_protocol_version: str,
        *,
        config: BacktestConfig,
        seed: int = 42,
        pass_criterion: Callable[[BacktestResult], bool] | None = None,
        allowed_kinds: set[DatasetKind] | None = None,
        budget: ResearchBudget | None = None,
    ) -> FinalEvaluationResult:
        """Dedicated, strictly controlled API for evaluating candidates on FINAL_HOLDOUT.

        Enforces:
        - Prior gate passing (must have completed RESEARCH or VALIDATION trial)
        - Exactly one holdout evaluation per candidate/protocol
        - Prevention of re-evaluation or retuning after failure
        - Sealing of holdout data via allow_holdout=True internal token
        """
        # 1. Check if holdout is burned
        if self.is_holdout_burned(dataset_version, research_protocol_version):
            raise HoldoutSecurityError(
                f"Holdout for dataset={dataset_version!r}, protocol={research_protocol_version!r} is BURNED"
            )

        # 2. Normalize strategy spec and derive ID
        normalized_spec = normalize_strategy_spec(strategy_spec)
        strategy_id = derive_strategy_id(normalized_spec)

        # 3. Check current holdout state for candidate
        current_state = self.get_holdout_state(strategy_id, dataset_version, research_protocol_version)
        if current_state != HoldoutState.UNTOUCHED:
            raise HoldoutSecurityError(
                f"Candidate {strategy_id} has holdout state {current_state.value}; "
                "repeat evaluation or tuning against this holdout is strictly forbidden."
            )

        # 4. Gate verification: candidate must have passed earlier gates
        completed_prior_trials = self._ledger.find_trials(
            strategy_id=strategy_id,
            dataset_version=dataset_version,
            research_protocol_version=research_protocol_version,
            status="COMPLETED",
        )
        # Filter for RESEARCH or VALIDATION split zones
        research_val_trials = [
            t for t in completed_prior_trials
            if t["split_zone"] in ("RESEARCH", "VALIDATION")
        ]
        if not research_val_trials:
            raise HoldoutSecurityError(
                f"Candidate strategy {strategy_id} has not passed required pre-holdout gates. "
                "Must have at least one COMPLETED trial in RESEARCH or VALIDATION split zone."
            )

        # 5. Load holdout data using internal privilege token
        kinds = allowed_kinds or {DatasetKind.LICENSED}
        record = self._dataset_registry.get(dataset_version)
        if record.kind not in kinds:
            raise HoldoutSecurityError(
                f"dataset {dataset_version} has kind={record.kind.value}; allowed={sorted(k.value for k in kinds)}"
            )

        data = self._dataset_registry.load_zone(
            dataset_version,
            SplitZone.FINAL_HOLDOUT,
            allowed_kinds=kinds,
            allow_holdout=True,
        )

        signal_fn = compile_strategy_spec(normalized_spec)
        task_id = f"HOLDOUT-EVAL-{strategy_id[:12]}"
        trial_id = "TRIAL-HOLDOUT-" + uuid.uuid4().hex

        mode = "PRODUCTION" if record.kind is DatasetKind.LICENSED else "FIXTURE"
        active_budget = budget or ResearchBudget()

        trial_context = TrialContext(
            trial_id=trial_id,
            experiment_id=derive_experiment_id(task_id),
            strategy_id=strategy_id,
            dataset_version=dataset_version,
            split_zone=SplitZone.FINAL_HOLDOUT.value,
            research_protocol_version=research_protocol_version,
            feature_version=normalized_spec.feature_version,
            parameter_set=dict(normalized_spec.parameters),
            strategy_spec_json=normalized_spec.canonical_json(),
            seed=seed,
            execution_model=config.execution_model,
            cost_model=config.cost_schedule.schedule_id if config.cost_schedule else "NONE",
            slippage_model=f"per_side_{config.slippage_bps_per_side:.6f}bps",
            estimated_runtime_minutes=0.01,
            estimated_llm_cost=0.0,
            mode=mode,
            dataset_kind=record.kind.value,
        )

        self._ledger.reserve(trial_context, active_budget)
        started = time.perf_counter()

        try:
            cuts = config.causality_cut_points or self._engine.default_causality_cut_points(len(data))
            assert_causal_signal(data.reset_index(drop=True), signal_fn, cut_points=list(cuts))
            backtest_result = self._engine._run_internal(data, config=config, signal_fn=signal_fn)
        except NonCausalSignalError as exc:
            elapsed = (time.perf_counter() - started) / 60.0
            self._ledger.complete(
                trial_id,
                result={"error": str(exc)},
                actual_runtime_minutes=elapsed,
                actual_llm_cost=0.0,
                status="REJECTED_NONCAUSAL",
            )
            self._record_holdout_evaluation(
                strategy_id=strategy_id,
                dataset_version=dataset_version,
                protocol=research_protocol_version,
                state=HoldoutState.FAILED,
                trial_id=trial_id,
                result_json=json.dumps({"error": str(exc)}),
            )
            return FinalEvaluationResult(
                candidate_strategy_id=strategy_id,
                dataset_version=dataset_version,
                research_protocol_version=research_protocol_version,
                state=HoldoutState.FAILED,
                trial_id=trial_id,
                backtest_result=None,
                status="REJECTED_NONCAUSAL",
                reason=str(exc),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - started) / 60.0
            self._ledger.complete(
                trial_id,
                result={"error": repr(exc)},
                actual_runtime_minutes=elapsed,
                actual_llm_cost=0.0,
                status="FAILED",
            )
            self._record_holdout_evaluation(
                strategy_id=strategy_id,
                dataset_version=dataset_version,
                protocol=research_protocol_version,
                state=HoldoutState.FAILED,
                trial_id=trial_id,
                result_json=json.dumps({"error": repr(exc)}),
            )
            return FinalEvaluationResult(
                candidate_strategy_id=strategy_id,
                dataset_version=dataset_version,
                research_protocol_version=research_protocol_version,
                state=HoldoutState.FAILED,
                trial_id=trial_id,
                backtest_result=None,
                status="FAILED",
                reason=repr(exc),
            )

        elapsed = (time.perf_counter() - started) / 60.0

        # Evaluate pass criterion
        if pass_criterion is not None:
            passed = pass_criterion(backtest_result)
        else:
            passed = backtest_result.net_pnl > 0 and backtest_result.trade_count >= 1

        final_state = HoldoutState.PASSED if passed else HoldoutState.FAILED
        status = "COMPLETED" if passed else "REJECTED_FINAL_HOLDOUT"

        result_dict = {
            "trade_count": backtest_result.trade_count,
            "gross_pnl": backtest_result.gross_pnl,
            "costs": backtest_result.costs,
            "net_pnl": backtest_result.net_pnl,
            "mean_gross_return_bps": backtest_result.mean_gross_return_bps,
            "mean_net_return_bps": backtest_result.mean_net_return_bps,
            "passed": passed,
        }

        self._ledger.complete(
            trial_id,
            result=result_dict,
            actual_runtime_minutes=elapsed,
            actual_llm_cost=0.0,
            status=status,
        )

        manifest_version = record.split_manifest.manifest_version if record.split_manifest else "legacy"
        self._record_holdout_evaluation(
            strategy_id=strategy_id,
            dataset_version=dataset_version,
            protocol=research_protocol_version,
            holdout_version=manifest_version,
            state=final_state,
            trial_id=trial_id,
            result_json=json.dumps(result_dict, sort_keys=True),
        )

        return FinalEvaluationResult(
            candidate_strategy_id=strategy_id,
            dataset_version=dataset_version,
            research_protocol_version=research_protocol_version,
            state=final_state,
            trial_id=trial_id,
            backtest_result=backtest_result,
            status=status,
            reason=None if passed else "Failed holdout pass criterion",
        )

    def _record_holdout_evaluation(
        self,
        *,
        strategy_id: str,
        dataset_version: str,
        protocol: str,
        holdout_version: str = "v1",
        state: HoldoutState,
        trial_id: str,
        result_json: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._connection.execute(
            """
            INSERT INTO holdout_evaluations(
                candidate_strategy_id, dataset_version, research_protocol_version,
                holdout_version, state, trial_id, timestamp_evaluated, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id,
                dataset_version,
                protocol,
                holdout_version,
                state.value,
                trial_id,
                now,
                result_json,
            ),
        )
