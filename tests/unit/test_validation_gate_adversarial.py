"""Adversarial and Exploit Tests for Strategy Validation Gate & Strategy Registry (PRD v3.8).

Verifies that the validation gate and registry cannot be bypassed, tampered with,
or fed illegitimate manual overrides, fixture data, or conflicting records.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import pytest

from quantmind.research_integrity.holdout import HoldoutState
from quantmind.research_integrity.qualification import (
    InvalidStateTransitionError,
    QualificationLedger,
    StrategyQualificationRecord,
    StrategyValidationGate,
    ValidationGateError,
    ValidationStatus,
    validate_transition,
)
from quantmind.strategy import (
    StrategyLifecycleState,
    StrategyRegistry,
    StrategyRegistryError,
    StrategySpec,
    derive_strategy_id,
)
import numpy as np
import pandas as pd

from quantmind.backtest.engine import BacktestEngine, BacktestResult, BacktestTrade
from quantmind.data import DatasetKind, DatasetRegistry, PurgeEmbargoSpec, compute_split_manifest
from quantmind.research_integrity.artifacts import ArtifactRegistry, ArtifactType, write_oos_returns_artifact
from quantmind.research_integrity.holdout import HoldoutManager
from quantmind.research_integrity.population import ProductionPopulationQuery
from quantmind.research_integrity.qualification import RobustnessStatus
from quantmind.research_integrity.trial_ledger import ResearchBudget, TrialContext, TrialLedger


def _setup_gate_environment(tmp_path: Path):
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
    env: dict,
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




class TestValidationGateAdversarial:
    def test_exploit_manual_sharpe_override_rejected(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        _record_completed_trial(env, trial_id="TRIAL-EXP-1", spec=spec)

        with pytest.raises(ValueError, match="Manual statistical overrides"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-EXP-1",
                dataset_version="DS-GATE-V1",
                research_protocol_version="RP-2",
                manual_sharpe=3.5,
            )

    def test_exploit_manual_dsr_override_rejected(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        _record_completed_trial(env, trial_id="TRIAL-EXP-2", spec=spec)

        with pytest.raises(ValueError, match="Manual statistical overrides"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-EXP-2",
                dataset_version="DS-GATE-V1",
                research_protocol_version="RP-2",
                manual_dsr=0.99,
            )

    def test_exploit_manual_trials_override_rejected(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        _record_completed_trial(env, trial_id="TRIAL-EXP-3", spec=spec)

        with pytest.raises(ValueError, match="Manual statistical overrides"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-EXP-3",
                dataset_version="DS-GATE-V1",
                research_protocol_version="RP-2",
                manual_trials=1,
            )

    def test_exploit_fixture_trial_rejected(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        # Record a trial in FIXTURE mode directly in ledger
        strat_id = derive_strategy_id(spec)
        env["ledger"]._connection.execute(
            """
            INSERT INTO trials (
                trial_id, experiment_id, strategy_id, dataset_version, split_zone,
                research_protocol_version, feature_version, parameter_set_json, strategy_spec_json,
                seed, execution_model, cost_model, slippage_model,
                timestamp_started, estimated_runtime_minutes,
                estimated_llm_cost, mode, dataset_kind, status
            ) VALUES (?, 'EXP-FIX', ?, 'DS-GATE-V1', 'VALIDATION', 'RP-2', 'f1', '{}', '{}', 1, 'm', 'c', 's', 'now', 0.1, 0.0, 'FIXTURE', 'LICENSED', 'COMPLETED')
            """,
            ("TRIAL-FIXTURE", strat_id),
        )

        with pytest.raises(ValueError, match="requires mode='PRODUCTION'"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-FIXTURE",
                dataset_version="DS-GATE-V1",
                research_protocol_version="RP-2",
            )

    def test_exploit_research_zone_trial_rejected(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        _record_completed_trial(env, trial_id="TRIAL-RESEARCH-ZONE", spec=spec, split_zone="RESEARCH")

        with pytest.raises(ValueError, match="must be from split_zone='VALIDATION'"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-RESEARCH-ZONE",
                dataset_version="DS-GATE-V1",
                research_protocol_version="RP-2",
            )

    def test_exploit_strategy_spec_mismatch_rejected(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec1 = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        spec2 = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.2, 0.8]},
        )
        _record_completed_trial(env, trial_id="TRIAL-SPEC-1", spec=spec1)

        # Attempt to validate spec2 using trial from spec1
        with pytest.raises(ValueError, match="does not match derived strategy_id"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec2,
                validation_trial_id="TRIAL-SPEC-1",
                dataset_version="DS-GATE-V1",
                research_protocol_version="RP-2",
            )

    def test_burned_holdout_rejects_candidate(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        _record_completed_trial(env, trial_id="TRIAL-BURN", spec=spec)
        env["holdout_mgr"].burn_holdout("DS-GATE-V1", "RP-2", "Data snooping detected")

        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-BURN",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.REJECTED
        assert any("burned" in r.lower() for r in record.reasons)

    def test_failed_holdout_rejects_candidate(self, tmp_path: Path) -> None:
        env = _setup_gate_environment(tmp_path)
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        strat_id = derive_strategy_id(spec)
        _record_completed_trial(env, trial_id="TRIAL-FAIL-HOLDOUT", spec=spec)
        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id=strat_id,
            dataset_version="DS-GATE-V1",
            protocol="RP-2",
            state=HoldoutState.FAILED,
            trial_id="TRIAL-H-FAIL",
            result_json=json.dumps({"passed": False}),
        )

        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-FAIL-HOLDOUT",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.REJECTED_FINAL_HOLDOUT
        assert record.holdout_state == "FAILED"


class TestStrategyRegistryAdversarial:
    def test_exploit_transition_to_paper_eligible_without_record_fails(self, tmp_path: Path) -> None:
        registry = StrategyRegistry(tmp_path / "strategies.db")
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.VALIDATION)

        with pytest.raises(StrategyRegistryError, match="requires an authoritative StrategyQualificationRecord"):
            registry.transition_state(
                strat_id,
                StrategyLifecycleState.PAPER_ELIGIBLE,
            )

    def test_exploit_tampered_qualification_record_rejected(self, tmp_path: Path) -> None:
        registry = StrategyRegistry(tmp_path / "strategies.db")
        spec = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        strat_id = registry.register_strategy(spec, initial_state=StrategyLifecycleState.VALIDATION)

        env = _setup_gate_environment(tmp_path)
        _record_completed_trial(env, trial_id="TRIAL-OK", spec=spec)
        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id=strat_id,
            dataset_version="DS-GATE-V1",
            protocol="RP-2",
            state=HoldoutState.PASSED,
            trial_id="TRIAL-H-OK",
            result_json=json.dumps({"passed": True}),
        )
        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-OK",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )

        # Tamper with the record (fake higher Sharpe)
        d = record.to_dict()
        d["observed_sharpe"] = 4.5
        tampered_record = StrategyQualificationRecord.from_dict(d)

        with pytest.raises(StrategyRegistryError, match="failed cryptographic digest"):
            registry.transition_state(
                strat_id,
                StrategyLifecycleState.PAPER_ELIGIBLE,
                qualification_record=tampered_record,
            )

    def test_exploit_mismatched_strategy_record_rejected(self, tmp_path: Path) -> None:
        registry = StrategyRegistry(tmp_path / "strategies.db")
        spec1 = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.1, 0.9]},
        )
        spec2 = StrategySpec(
            strategy_version="v1",
            feature_version="f1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 375, "session_window": [0.2, 0.8]},
        )
        strat_id1 = registry.register_strategy(spec1, initial_state=StrategyLifecycleState.VALIDATION)
        strat_id2 = registry.register_strategy(spec2, initial_state=StrategyLifecycleState.VALIDATION)

        env = _setup_gate_environment(tmp_path)
        _record_completed_trial(env, trial_id="TRIAL-OK1", spec=spec1)
        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id=strat_id1,
            dataset_version="DS-GATE-V1",
            protocol="RP-2",
            state=HoldoutState.PASSED,
            trial_id="TRIAL-H-OK1",
            result_json=json.dumps({"passed": True}),
        )
        record1, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec1,
            validation_trial_id="TRIAL-OK1",
            dataset_version="DS-GATE-V1",
            research_protocol_version="RP-2",
        )

        # Attempt to use record1 to promote strategy 2
        with pytest.raises(StrategyRegistryError, match="does not match registered strategy"):
            registry.transition_state(
                strat_id2,
                StrategyLifecycleState.PAPER_ELIGIBLE,
                qualification_record=record1,
            )

    def test_exploit_tamper_qualification_ledger_sqlite_triggers(self, tmp_path: Path) -> None:
        qual_ledger = QualificationLedger(tmp_path / "qual_tamper.db")
        record = StrategyQualificationRecord.create(
            qualification_id="QUAL-IMMUTABLE",
            strategy_id="STRAT-1",
            strategy_spec_hash="hash",
            dataset_version="DS-1",
            dataset_sha256="sha",
            split_manifest_version="v1",
            research_protocol_version="RP-2",
            population_hash="pophash",
            effective_trial_count=10.0,
            observed_sharpe=1.5,
            dsr=0.96,
            trade_count=100,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
        )
        qual_ledger.record_qualification(record)

        # Attempt DELETE
        with pytest.raises(sqlite3.IntegrityError, match="permanent and append-only"):
            qual_ledger._connection.execute("DELETE FROM strategy_qualifications WHERE qualification_id = 'QUAL-IMMUTABLE'")

        # Attempt UPDATE
        with pytest.raises(sqlite3.IntegrityError, match="permanent and immutable"):
            qual_ledger._connection.execute("UPDATE strategy_qualifications SET observed_sharpe = 5.0")
