from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
import uuid

from quantmind.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from quantmind.backtest.strategies import NonCausalSignalError, assert_causal_signal
from quantmind.data import DatasetKind, DatasetRegistry
from quantmind.data.splits import SplitZone
from quantmind.strategy import StrategySpec, compile_strategy_spec, normalize_strategy_spec
from .trial_ledger import ResearchBudget, TrialContext, TrialLedger
from .artifacts import ArtifactRegistry, ArtifactType, write_oos_returns_artifact


@dataclass(frozen=True)
class ResearchTask:
    task_id: str


def derive_experiment_id(research_task_id: str) -> str:
    return "EXP-" + hashlib.sha256(research_task_id.encode()).hexdigest()[:24]


def derive_strategy_id(strategy_spec: StrategySpec) -> str:
    """Derive identity from normalized logic only; strategy_version is metadata."""
    normalized = normalize_strategy_spec(strategy_spec)
    return "STRAT-" + hashlib.sha256(normalized.logic_canonical_json().encode()).hexdigest()[:24]


from .holdout import HoldoutManager, HoldoutSecurityError


class ResearchHarness:
    """Mandatory production entry point for research trials.

    Production trials resolve data by dataset_version + split_zone through a
    checksum-verified DatasetRegistry and accept only declarative StrategySpec.
    Synthetic/fixture datasets and arbitrary signal callables are excluded from
    this production API.

    If *artifact_registry* is provided, every completed PRODUCTION trial will
    have its OOS trade-level return stream written to a Parquet artifact and
    registered in the registry.  *artifact_dir* specifies where to store the
    files on disk (defaults to ``./artifacts`` in the current working directory).
    """

    def __init__(
        self,
        engine: BacktestEngine,
        ledger: TrialLedger,
        dataset_registry: DatasetRegistry,
        holdout_manager: HoldoutManager | None = None,
        artifact_registry: ArtifactRegistry | None = None,
        artifact_dir: Path | str | None = None,
    ) -> None:
        self._engine = engine
        self._ledger = ledger
        self._dataset_registry = dataset_registry
        self._holdout_manager = holdout_manager or HoldoutManager(engine, ledger, dataset_registry)
        self._artifact_registry = artifact_registry
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else Path("artifacts")

    def run_trial(
        self,
        *,
        config: BacktestConfig,
        research_task_id: str,
        strategy_spec: StrategySpec,
        dataset_version: str,
        split_zone: SplitZone | str,
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

        zone_str = split_zone.value if isinstance(split_zone, SplitZone) else str(split_zone).upper()
        if zone_str == SplitZone.FINAL_HOLDOUT.value:
            raise ValueError(
                "FINAL_HOLDOUT zone is sealed; production research trials cannot access holdout data. Use final_evaluate()."
            )

        # Normalize/validate BEFORE reserving a trial budget.
        normalized_spec = normalize_strategy_spec(strategy_spec)
        strategy_id = derive_strategy_id(normalized_spec)

        if self._holdout_manager.has_candidate_failed(strategy_id, dataset_version, research_protocol_version):
            raise HoldoutSecurityError(
                f"Candidate {strategy_id} has failed final holdout for dataset={dataset_version} and cannot be retuned against it."
            )
        if self._holdout_manager.is_holdout_burned(dataset_version, research_protocol_version):
            raise HoldoutSecurityError(
                f"Holdout for dataset={dataset_version} is BURNED; further research or tuning is prohibited."
            )

        signal_fn = compile_strategy_spec(normalized_spec)

        record = self._dataset_registry.get(dataset_version)
        if record.kind is not DatasetKind.LICENSED:
            raise ValueError(
                f"production research requires LICENSED dataset; got {record.kind.value} for {dataset_version}"
            )
        data = self._dataset_registry.load_zone(
            dataset_version,
            zone_str,
            allowed_kinds={DatasetKind.LICENSED},
        )

        split_m = record.split_manifest
        split_m_ver = split_m.manifest_version if split_m else ""
        split_m_sha = split_m.manifest_hash() if split_m else ""
        cfg_ident = f"{config.execution_model}:{config.hold_model}:{config.quantity}:{config.lot_size}:{config.slippage_bps_per_side}:{config.cost_schedule.schedule_id if config.cost_schedule else 'NONE'}"
        config_hash = hashlib.sha256(cfg_ident.encode()).hexdigest()[:16]

        trial_context = TrialContext(
            trial_id="TRIAL-" + uuid.uuid4().hex,
            experiment_id=derive_experiment_id(research_task_id),
            strategy_id=strategy_id,
            dataset_version=dataset_version,
            split_zone=zone_str,
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
            dataset_sha256=record.sha256,
            split_manifest_version=split_m_ver,
            split_manifest_sha256=split_m_sha,
            code_version="0.1.0",
            config_hash=config_hash,
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

        # Write OOS return artifact before completing the trial in the ledger.
        artifact_sha256: str | None = None
        if self._artifact_registry is not None:
            try:
                art_path, art_sha256, art_rows, art_start, art_end = write_oos_returns_artifact(
                    result,
                    trial_id=trial_context.trial_id,
                    dataset_version=dataset_version,
                    artifact_dir=self._artifact_dir,
                )
                art_record = self._artifact_registry.record(
                    trial_id=trial_context.trial_id,
                    artifact_type=ArtifactType.OOS_RETURNS,
                    dataset_version=dataset_version,
                    sha256=art_sha256,
                    format="parquet",
                    row_count=art_rows,
                    start_timestamp=art_start,
                    end_timestamp=art_end,
                    artifact_path=art_path,
                )
                artifact_sha256 = art_record.sha256
            except Exception:
                # Artifact write failure is non-fatal for the trial ledger record,
                # but we propagate since this breaks population eligibility.
                raise

        self._ledger.complete(
            trial_context.trial_id,
            result={
                "trade_count": result.trade_count,
                "gross_pnl": result.gross_pnl,
                "costs": result.costs,
                "net_pnl": result.net_pnl,
                "mean_gross_return_bps": result.mean_gross_return_bps,
                "mean_net_return_bps": result.mean_net_return_bps,
                **({"artifact_sha256": artifact_sha256} if artifact_sha256 is not None else {}),
            },
            actual_runtime_minutes=elapsed,
            actual_llm_cost=0.0,
            status="COMPLETED",
        )
        return result

