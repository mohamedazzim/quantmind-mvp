from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
import uuid

from quantmind.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from quantmind.backtest.strategies import NonCausalSignalError, assert_causal_signal
from quantmind.data import DatasetKind, DatasetRegistry
from quantmind.strategy import StrategySpec, compile_strategy_spec, normalize_strategy_spec
from .trial_ledger import ResearchBudget, TrialContext, TrialLedger


@dataclass(frozen=True)
class ResearchTask:
    task_id: str


def derive_experiment_id(research_task_id: str) -> str:
    return "EXP-" + hashlib.sha256(research_task_id.encode()).hexdigest()[:24]


def derive_strategy_id(strategy_spec: StrategySpec) -> str:
    """Derive identity from normalized logic only; strategy_version is metadata."""
    normalized = normalize_strategy_spec(strategy_spec)
    return "STRAT-" + hashlib.sha256(normalized.logic_canonical_json().encode()).hexdigest()[:24]


class ResearchHarness:
    """Mandatory production entry point for research trials.

    Production trials resolve data by dataset_version + split_zone through a
    checksum-verified DatasetRegistry and accept only declarative StrategySpec.
    Synthetic/fixture datasets and arbitrary signal callables are excluded from
    this production API.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        ledger: TrialLedger,
        dataset_registry: DatasetRegistry,
    ) -> None:
        self._engine = engine
        self._ledger = ledger
        self._dataset_registry = dataset_registry

    def run_trial(
        self,
        *,
        config: BacktestConfig,
        research_task_id: str,
        strategy_spec: StrategySpec,
        dataset_version: str,
        split_zone: str,
        research_protocol_version: str,
        seed: int,
        budget: ResearchBudget,
        estimated_runtime_minutes: float = 0.01,
        estimated_llm_cost: float = 0.0,
    ) -> BacktestResult:
        if config.signal_column is not None:
            raise ValueError(
                "signal_column is test-fixture-only; production research trials require StrategySpec"
            )
        # Normalize/validate BEFORE reserving a trial budget.
        normalized_spec = normalize_strategy_spec(strategy_spec)
        signal_fn = compile_strategy_spec(normalized_spec)

        record = self._dataset_registry.get(dataset_version)
        if record.kind is not DatasetKind.LICENSED:
            raise ValueError(
                f"production research requires LICENSED dataset; got {record.kind.value} for {dataset_version}"
            )
        data = self._dataset_registry.load_zone(
            dataset_version,
            split_zone,
            allowed_kinds={DatasetKind.LICENSED},
        )

        trial_context = TrialContext(
            trial_id="TRIAL-" + uuid.uuid4().hex,
            experiment_id=derive_experiment_id(research_task_id),
            strategy_id=derive_strategy_id(normalized_spec),
            dataset_version=dataset_version,
            research_protocol_version=research_protocol_version,
            feature_version=normalized_spec.feature_version,
            parameter_set=dict(normalized_spec.parameters),
            strategy_spec_json=normalized_spec.canonical_json(),
            seed=seed,
            execution_model=config.execution_model,
            cost_model=config.cost_schedule.schedule_id if config.cost_schedule else "NONE",
            slippage_model=f"per_side_{config.slippage_bps_per_side:.6f}bps",
            estimated_runtime_minutes=estimated_runtime_minutes,
            estimated_llm_cost=estimated_llm_cost,
            mode="PRODUCTION",
            dataset_kind=record.kind.value,
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
                actual_llm_cost=0.0,
                status="REJECTED_NONCAUSAL",
            )
            raise
        except Exception as exc:
            elapsed = (time.perf_counter() - started) / 60.0
            self._ledger.complete(
                trial_context.trial_id,
                result={"error": repr(exc)},
                actual_runtime_minutes=elapsed,
                actual_llm_cost=0.0,
                status="FAILED",
            )
            raise

        elapsed = (time.perf_counter() - started) / 60.0
        self._ledger.complete(
            trial_context.trial_id,
            result={
                "trade_count": result.trade_count,
                "gross_pnl": result.gross_pnl,
                "costs": result.costs,
                "net_pnl": result.net_pnl,
                "mean_gross_return_bps": result.mean_gross_return_bps,
                "mean_net_return_bps": result.mean_net_return_bps,
            },
            actual_runtime_minutes=elapsed,
            actual_llm_cost=0.0,
            status="COMPLETED",
        )
        return result
