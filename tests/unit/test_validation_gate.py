"""Unit tests for StrategyValidationGate & StrategyRegistry integration (PRD v3.8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantmind.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, BacktestTrade
from quantmind.data import DatasetKind, DatasetRegistry, PurgeEmbargoSpec, compute_split_manifest
from quantmind.research_integrity.artifacts import ArtifactRegistry, ArtifactType, write_oos_returns_artifact
from quantmind.research_integrity.holdout import HoldoutManager, HoldoutState
from quantmind.research_integrity.population import ProductionPopulationQuery
from quantmind.research_integrity.qualification import (
    PaperReplayEligibility,
    QualificationLedger,
    RobustnessReport,
    RobustnessStatus,
    StrategyQualificationRecord,
    StrategyValidationGate,
    ValidationReport,
    ValidationStatus,
    check_paper_replay_eligibility,
)
from quantmind.research_integrity.trial_ledger import ResearchBudget, TrialContext, TrialLedger
from quantmind.strategy import (
    StrategyLifecycleState,
    StrategyRegistry,
    StrategyRegistryError,
    StrategySpec,
    derive_strategy_id,
)


def _setup_gate_environment(tmp_path: Path):
    # 1. Dataset registry & licensed dataset with split manifest
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_file = data_dir / "nifty_test.csv"
    timestamps = pd.date_range("2023-01-01 09:15", periods=2000, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )
    df.to_csv(csv_file, index=False)

    registry = DatasetRegistry(tmp_path / "registry.db")
    record = registry.register_file(
        version="DS-GATE-V1",
        path=csv_file,
        kind=DatasetKind.LICENSED,
    )

    manifest = compute_split_manifest(
        dataset_version="DS-GATE-V1",
        manifest_version="m1",
        timestamps=timestamps,
        spec=PurgeEmbargoSpec(
            feature_lookback_bars=10,
            prediction_horizon_bars=10,
            holding_period_bars=10,
            embargo_bars=10,
        ),
    )
    registry.register_split_manifest(manifest)

    # 2. Ledgers and managers
    ledger = TrialLedger(tmp_path / "trials.db")
    engine = BacktestEngine()

    holdout_mgr = HoldoutManager(
        engine=engine,
        ledger=ledger,
        dataset_registry=registry,
        db_path=tmp_path / "holdout.db",
    )
    art_reg = ArtifactRegistry(tmp_path / "artifacts.db")
    pop_query = ProductionPopulationQuery(ledger, art_reg)
    qual_ledger = QualificationLedger(tmp_path / "qual.db")

    gate = StrategyValidationGate(
        dataset_registry=registry,
        trial_ledger=ledger,
        holdout_manager=holdout_mgr,
        population_query=pop_query,
        qualification_ledger=qual_ledger,
    )

    return {
        "registry": registry,
        "ledger": ledger,
        "holdout_mgr": holdout_mgr,
        "art_reg": art_reg,
        "pop_query": pop_query,
        "qual_ledger": qual_ledger,
        "gate": gate,
        "engine": engine,
        "tmp_path": tmp_path,
    }


def _record_completed_trial(
    env: dict[str, Any],
    *,
    trial_id: str,
    spec: StrategySpec,
    split_zone: str = "VALIDATION",
    trade_count: int = 120,
    net_pnl: float = 500.0,
    dataset_version: str = "DS-GATE-V1",
    protocol_version: str = "RP-2",
) -> None:
    ledger = env["ledger"]
    art_reg = env["art_reg"]
    tmp_path = env["tmp_path"]

    strat_id = derive_strategy_id(spec)
    ctx = TrialContext(
        trial_id=trial_id,
        experiment_id=f"EXP-{trial_id}",
        strategy_id=strat_id,
        dataset_version=dataset_version,
        split_zone=split_zone,
        research_protocol_version=protocol_version,
        feature_version=spec.feature_version,
        parameter_set=dict(spec.parameters),
        strategy_spec_json=spec.canonical_json(),
        seed=42,
        execution_model="next_bar_open_v1",
        cost_model="NONE",
        slippage_model="0bps",
        estimated_runtime_minutes=0.01,
        mode="PRODUCTION",
        dataset_kind="LICENSED",
    )
    budget = ResearchBudget(max_trials=1000, max_experiments=1000, max_strategy_variants=1000)
    ledger.reserve(ctx, budget)

    # Generate returns and write OOS return artifact
    rng = np.random.default_rng(42)
    step_ret = net_pnl / float(trade_count)
    returns = [step_ret + rng.normal(0, 0.1) for _ in range(trade_count)]

    trades = tuple(
        BacktestTrade(
            signal_timestamp=pd.Timestamp("2023-01-03") + pd.Timedelta(minutes=m),
            entry_timestamp=pd.Timestamp("2023-01-03") + pd.Timedelta(minutes=m + 1),
            exit_timestamp=pd.Timestamp("2023-01-03") + pd.Timedelta(minutes=m + 2),
            side=1,
            quantity=1,
            entry_price=100.0,
            exit_price=100.0 + r / 100.0,
            gross_return_bps=r,
            cost_bps=0.0,
            net_return_bps=r,
            gross_pnl=r,
            costs=0.0,
            net_pnl=r,
        )
        for m, r in enumerate(returns)
    )
    res = BacktestResult(
        execution_model="next_bar_open_v1",
        trades=trades,
        gross_pnl=sum(returns),
        costs=0.0,
        net_pnl=sum(returns),
        mean_gross_return_bps=float(np.mean(returns)),
        mean_net_return_bps=float(np.mean(returns)),
    )
    path, sha, rows, start, end = write_oos_returns_artifact(
        res, trial_id=trial_id, dataset_version=dataset_version, artifact_dir=tmp_path
    )
    art_reg.record(
        trial_id=trial_id,
        artifact_type=ArtifactType.OOS_RETURNS,
        dataset_version=dataset_version,
        sha256=sha,
        format="parquet",
        row_count=rows,
        start_timestamp=start,
        end_timestamp=end,
        artifact_path=path,
    )
    ledger.complete(
        trial_id,
        result={"trade_count": trade_count, "net_pnl": net_pnl, "artifact_sha256": sha},
        actual_runtime_minutes=0.01,
        status="COMPLETED",
    )


class TestStrategyValidationGate:
    def test_candidate_holdout_required_when_untouched(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        _record_completed_trial(env, trial_id="TRIAL-VAL-1", spec=spec, trade_count=120, net_pnl=350.0)

        record, report = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-VAL-1",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )

        assert record.final_status == ValidationStatus.HOLDOUT_REQUIRED
        assert record.holdout_state == "UNTOUCHED"
        assert record.trade_count == 120
        assert record.verify_digest() is True
        assert len(report.checks_passed) >= 2
        # Verify no ranking or "best strategy" claims
        assert "rank" not in report.summary_text.lower()
        assert "best strategy" not in report.summary_text.lower()

    def test_candidate_paper_eligible_when_holdout_passed(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        strat_id = derive_strategy_id(spec)
        _record_completed_trial(env, trial_id="TRIAL-VAL-2", spec=spec, trade_count=150, net_pnl=500.0)

        # Mark holdout as PASSED in holdout manager
        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id=strat_id,
            dataset_version="DS-GATE-V1",
            protocol="RP-2",
            state=HoldoutState.PASSED,
            trial_id="TRIAL-HOLDOUT-2",
            result_json=json.dumps({"passed": True, "trade_count": 60, "net_pnl": 120.0}),
        )

        record, report = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-VAL-2",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )

        assert record.final_status == ValidationStatus.PAPER_ELIGIBLE
        assert record.holdout_state == "PASSED"
        assert record.verify_digest() is True

        eligibility = check_paper_replay_eligibility(record)
        assert eligibility.is_eligible is True

    def test_candidate_underpowered_when_sample_size_insufficient(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        # 30 trades is below minimum_trades_validation (75)
        _record_completed_trial(env, trial_id="TRIAL-VAL-3", spec=spec, trade_count=30, net_pnl=100.0)

        record, report = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-VAL-3",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )

        assert record.final_status == ValidationStatus.UNDERPOWERED
        assert any("underpowered" in r.lower() for r in record.reasons)

    def test_candidate_rejected_when_unprofitable(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        _record_completed_trial(env, trial_id="TRIAL-VAL-4", spec=spec, trade_count=120, net_pnl=-50.0)

        record, report = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-VAL-4",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )

        assert record.final_status == ValidationStatus.REJECTED
        assert any("net profitability" in r.lower() for r in record.reasons)

    def test_candidate_rejected_when_robustness_fails(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        _record_completed_trial(env, trial_id="TRIAL-VAL-5", spec=spec, trade_count=120, net_pnl=400.0)

        failed_rob = RobustnessReport(
            status=RobustnessStatus.FAILED,
            details={"parameter_cliff": True},
            reasons=("Performance collapses under 5% parameter perturbation",),
        )

        record, report = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-VAL-5",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
            robustness_report=failed_rob,
        )

        assert record.final_status == ValidationStatus.REJECTED
        assert record.robustness_status == RobustnessStatus.FAILED
        assert any("robustness failure" in r.lower() for r in record.reasons)

    def test_candidate_rejected_final_holdout(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        strat_id = derive_strategy_id(spec)
        _record_completed_trial(env, trial_id="TRIAL-VAL-6", spec=spec, trade_count=120, net_pnl=400.0)

        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id=strat_id,
            dataset_version="DS-GATE-V1",
            protocol="RP-2",
            state=HoldoutState.FAILED,
            trial_id="TRIAL-HOLDOUT-6",
            result_json=json.dumps({"passed": False, "trade_count": 60, "net_pnl": -20.0}),
        )

        record, report = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-VAL-6",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )

        assert record.final_status == ValidationStatus.REJECTED_FINAL_HOLDOUT
        assert record.holdout_state == "FAILED"


class TestStrategyRegistryIntegration:
    def test_full_lifecycle_to_paper_eligible(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        strat_id = derive_strategy_id(spec)
        _record_completed_trial(env, trial_id="TRIAL-VAL-REG", spec=spec, trade_count=150, net_pnl=600.0)

        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id=strat_id,
            dataset_version="DS-GATE-V1",
            protocol="RP-2",
            state=HoldoutState.PASSED,
            trial_id="TRIAL-HOLDOUT-REG",
            result_json=json.dumps({"passed": True}),
        )

        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-VAL-REG",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.PAPER_ELIGIBLE

        strat_reg = StrategyRegistry(tmp_path / "strategies.db")
        strat_reg.register_strategy(spec, initial_state=StrategyLifecycleState.IDEA)

        # Transition IDEA -> RESEARCH -> VALIDATION -> PAPER_ELIGIBLE
        strat_reg.transition_state(strat_id, StrategyLifecycleState.RESEARCH, reason="started research")
        strat_reg.transition_state(strat_id, StrategyLifecycleState.VALIDATION, reason="started validation")
        final_record = strat_reg.transition_state(
            strat_id,
            StrategyLifecycleState.PAPER_ELIGIBLE,
            reason="passed all gates",
            qualification_record=record,
        )

        assert final_record.state == StrategyLifecycleState.PAPER_ELIGIBLE
        assert final_record.qualification_id == record.qualification_id
        assert final_record.qualification_hash == record.record_hash
        assert len(final_record.history) == 4

    def test_transition_to_paper_eligible_fails_with_underpowered_record(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        strat_id = derive_strategy_id(spec)
        _record_completed_trial(env, trial_id="TRIAL-VAL-UNDER", spec=spec, trade_count=20, net_pnl=50.0)

        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-VAL-UNDER",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.UNDERPOWERED

        strat_reg = StrategyRegistry(tmp_path / "strategies.db")
        strat_reg.register_strategy(spec, initial_state=StrategyLifecycleState.VALIDATION)

        with pytest.raises(StrategyRegistryError, match="must be 'PAPER_ELIGIBLE'"):
            strat_reg.transition_state(
                strat_id,
                StrategyLifecycleState.PAPER_ELIGIBLE,
                qualification_record=record,
            )
