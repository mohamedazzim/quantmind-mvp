from __future__ import annotations

import time
import uuid

import pandas as pd

from quantmind.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from quantmind.backtest.strategies import NonCausalSignalError, assert_causal_signal
from quantmind.research_integrity.harness import derive_experiment_id, derive_strategy_id
from quantmind.research_integrity.trial_ledger import ResearchBudget, TrialContext, TrialLedger
from quantmind.strategy import StrategySpec
from typing import Callable
import numpy as np

SignalFunction = Callable[[pd.DataFrame], np.ndarray]


class FixtureResearchHarness:
    """Synthetic regression-only harness. Never imported by production agents."""

    def __init__(self, engine: BacktestEngine, ledger: TrialLedger) -> None:
        self._engine = engine
        self._ledger = ledger

    def run_trial(
        self,
        data: pd.DataFrame,
        *,
        config: BacktestConfig,
        research_task_id: str,
        strategy_spec: StrategySpec,
        dataset_version: str,
        research_protocol_version: str,
        seed: int,
        budget: ResearchBudget,
        signal_fn: SignalFunction | None = None,
        estimated_runtime_minutes: float = 0.01,
    ) -> BacktestResult:
        if signal_fn is None and config.signal_column is None:
            raise ValueError("fixture path requires signal_fn or signal_column")
        if signal_fn is None:
            column = config.signal_column
            signal_fn = lambda frame: frame[column].to_numpy(dtype=float)

        trial_context = TrialContext(
            trial_id="TRIAL-FIXTURE-" + uuid.uuid4().hex,
            experiment_id=derive_experiment_id(research_task_id),
            strategy_id=derive_strategy_id(strategy_spec),
            dataset_version=dataset_version,
            research_protocol_version=research_protocol_version,
            feature_version=strategy_spec.feature_version,
            parameter_set=dict(strategy_spec.parameters),
            strategy_spec_json=strategy_spec.canonical_json(),
            seed=seed,
            execution_model=config.execution_model,
            cost_model=config.cost_schedule.schedule_id if config.cost_schedule else "NONE",
            slippage_model=f"per_side_{config.slippage_bps_per_side:.6f}bps",
            estimated_runtime_minutes=estimated_runtime_minutes,
            mode="FIXTURE",
            dataset_kind="SYNTHETIC",
        )
        self._ledger.reserve(trial_context, budget)
        started = time.perf_counter()
        try:
            cuts = config.causality_cut_points or self._engine.default_causality_cut_points(len(data))
            assert_causal_signal(data.reset_index(drop=True), signal_fn, cut_points=list(cuts))
            result = self._engine._run_internal(data, config=config, signal_fn=signal_fn)
        except NonCausalSignalError as exc:
            elapsed = (time.perf_counter() - started) / 60.0
            self._ledger.complete(
                trial_context.trial_id,
                result={"error": str(exc)},
                actual_runtime_minutes=elapsed,
                status="REJECTED_NONCAUSAL",
            )
            raise
        except Exception as exc:
            elapsed = (time.perf_counter() - started) / 60.0
            self._ledger.complete(
                trial_context.trial_id,
                result={"error": repr(exc)},
                actual_runtime_minutes=elapsed,
                status="FAILED",
            )
            raise
        elapsed = (time.perf_counter() - started) / 60.0
        self._ledger.complete(
            trial_context.trial_id,
            result={"trade_count": result.trade_count, "mean_net_return_bps": result.mean_net_return_bps},
            actual_runtime_minutes=elapsed,
            status="COMPLETED",
        )
        return result
