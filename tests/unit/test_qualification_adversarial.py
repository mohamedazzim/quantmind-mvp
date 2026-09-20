"""Adversarial Qualification Audit (PRD v3.8 - Step 2 & Step 3).

Comprehensive audit testing the exact production boundary:
- A. Authoritative evidence enforcement (rejection of caller overrides)
- B. Provenance consistency and tampering invalidation
- C. Qualification record tampering (digest mismatch, SQLite trigger blocks)
- D. State machine exploits (all illegal transitions blocked, all legal transitions verified)
- E. Holdout consistency (PASSED required, FAILED/BURNED/UNTOUCHED barred, scope alignment)
- F. Fixture isolation (FIXTURE trials and fixture populations rejected)
- G. Sample-size gate (tested below, at, and above minimum with protocol loading)
- H. Qualification reproducibility and canonical JSON ordering
- I. Paper eligibility boundary (pure eligibility; no broker/order/live feed execution)
- Step 3. ValidationReport audit (prohibition of ranking, leaderboards, winner, best strategy)
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantmind.backtest.engine import BacktestEngine, BacktestResult, BacktestTrade
from quantmind.data import DatasetKind, DatasetRegistry, PurgeEmbargoSpec, compute_split_manifest
from quantmind.research_integrity.artifacts import ArtifactRegistry, ArtifactType, write_oos_returns_artifact
from quantmind.research_integrity.holdout import HoldoutManager, HoldoutState
from quantmind.research_integrity.population import ProductionPopulationQuery
from quantmind.research_integrity.qualification import (
    ALLOWED_VALIDATION_TRANSITIONS,
    InvalidStateTransitionError,
    PaperReplayEligibility,
    QualificationLedger,
    QualificationLedgerError,
    RobustnessReport,
    RobustnessStatus,
    StrategyQualificationRecord,
    StrategyValidationGate,
    ValidationGateError,
    ValidationReport,
    ValidationStatus,
    check_paper_replay_eligibility,
    validate_transition,
)
from quantmind.research_integrity.trial_ledger import ResearchBudget, TrialContext, TrialLedger
from quantmind.strategy import (
    StrategyLifecycleState,
    StrategyRegistry,
    StrategyRegistryError,
    StrategySpec,
    derive_strategy_id,
)


def _setup_adversarial_env(tmp_path: Path):
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
    registry.register_file(
        version="DS-AUDIT-V1",
        path=csv_file,
        kind=DatasetKind.LICENSED,
    )
    manifest = compute_split_manifest(
        dataset_version="DS-AUDIT-V1",
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


def _seed_completed_trial(
    env: dict[str, Any],
    *,
    trial_id: str,
    spec: StrategySpec,
    split_zone: str = "VALIDATION",
    trade_count: int = 120,
    net_pnl: float = 500.0,
    dataset_version: str = "DS-AUDIT-V1",
    protocol_version: str = "RP-2",
    mode: str = "PRODUCTION",
    corrupt_artifact_sha: bool = False,
) -> None:
    ledger: TrialLedger = env["ledger"]
    art_reg: ArtifactRegistry = env["art_reg"]
    tmp_path: Path = env["tmp_path"]

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
        mode=mode,
        dataset_kind="LICENSED" if mode == "PRODUCTION" else "SYNTHETIC",
    )
    budget = ResearchBudget(max_trials=1000, max_experiments=1000, max_strategy_variants=1000)
    ledger.reserve(ctx, budget)

    rng = np.random.default_rng(42)
    step_ret = net_pnl / float(max(1, trade_count))
    returns = [step_ret + rng.normal(0, 0.05) for _ in range(trade_count)]

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
        mean_gross_return_bps=float(np.mean(returns)) if returns else 0.0,
        mean_net_return_bps=float(np.mean(returns)) if returns else 0.0,
    )
    path, sha, rows, start, end = write_oos_returns_artifact(
        res, trial_id=trial_id, dataset_version=dataset_version, artifact_dir=tmp_path
    )
    recorded_sha = "corrupted_sha" if corrupt_artifact_sha else sha
    art_reg.record(
        trial_id=trial_id,
        artifact_type=ArtifactType.OOS_RETURNS,
        dataset_version=dataset_version,
        sha256=recorded_sha,
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


# ---------------------------------------------------------------------------
# A. AUTHORITATIVE EVIDENCE AUDIT
# ---------------------------------------------------------------------------


class TestAuthoritativeEvidenceAudit:
    @pytest.mark.parametrize(
        "override_arg",
        [
            {"manual_sharpe": 2.5},
            {"manual_dsr": 0.99},
            {"manual_trials": 1},
            {"manual_effective_trial_count": 1.0},
            {"manual_holdout": "PASSED"},
            {"manual_trade_count": 500},
            {"manual_population_hash": "f" * 64},
            {"manual_dataset_sha256": "e" * 64},
        ],
    )
    def test_gate_rejects_caller_supplied_overrides(self, tmp_path: Path, override_arg: dict[str, Any]) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        _seed_completed_trial(env, trial_id="TRIAL-A", spec=spec)

        with pytest.raises(ValueError, match="statistical overrides"):
            env["gate"].evaluate_candidate(

                strategy_spec=spec,
                validation_trial_id="TRIAL-A",
                dataset_version="DS-AUDIT-V1",
                research_protocol_version="RP-2",
                **override_arg,
            )


# ---------------------------------------------------------------------------
# B. PROVENANCE CONSISTENCY AUDIT
# ---------------------------------------------------------------------------


class TestProvenanceConsistencyAudit:
    def test_dataset_version_mismatch_fails(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        _seed_completed_trial(env, trial_id="TRIAL-B1", spec=spec, dataset_version="DS-AUDIT-V1")

        with pytest.raises(ValueError, match="does not match requested"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-B1",
                dataset_version="DS-DIFFERENT-V2",
                research_protocol_version="RP-2",
            )

    def test_research_protocol_version_mismatch_fails(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        _seed_completed_trial(env, trial_id="TRIAL-B2", spec=spec, protocol_version="RP-2")

        with pytest.raises(ValueError, match="does not match requested"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-B2",
                dataset_version="DS-AUDIT-V1",
                research_protocol_version="RP-3",
            )

    def test_strategy_id_mismatch_fails(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec1 = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        spec2 = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.2, 0.8]})
        _seed_completed_trial(env, trial_id="TRIAL-B3", spec=spec1)

        with pytest.raises(ValueError, match="does not match derived strategy_id"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec2,
                validation_trial_id="TRIAL-B3",
                dataset_version="DS-AUDIT-V1",
                research_protocol_version="RP-2",
            )

    def test_artifact_sha256_mismatch_excludes_trial_from_population(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        _seed_completed_trial(env, trial_id="TRIAL-B4", spec=spec, corrupt_artifact_sha=True)

        with pytest.raises(ValidationGateError, match="not in the eligible production population query"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-B4",
                dataset_version="DS-AUDIT-V1",
                research_protocol_version="RP-2",
            )

    def test_record_hash_detects_mutated_provenance_fields(self, tmp_path: Path) -> None:
        rec = StrategyQualificationRecord.create(
            qualification_id="QUAL-PROV-1",
            strategy_id="STRAT-1",
            strategy_spec_hash="a" * 64,
            dataset_version="DS-1",
            dataset_sha256="b" * 64,
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="c" * 64,
            effective_trial_count=10.0,
            observed_sharpe=1.5,
            dsr=0.96,
            trade_count=120,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
        )
        assert rec.verify_digest() is True

        for field in [
            "dataset_version",
            "dataset_sha256",
            "split_manifest_version",
            "research_protocol_version",
            "strategy_id",
            "strategy_spec_hash",
            "population_hash",
            "effective_trial_count",
        ]:
            d = rec.to_dict()
            if isinstance(d[field], float):
                d[field] += 1.0
            else:
                d[field] = "MUTATED"
            mutated = StrategyQualificationRecord.from_dict(d)
            assert mutated.verify_digest() is False, f"Failed to detect tampering in field '{field}'"


# ---------------------------------------------------------------------------
# C. QUALIFICATION RECORD TAMPERING AUDIT
# ---------------------------------------------------------------------------


class TestQualificationRecordTamperingAudit:
    def test_record_hash_detects_field_mutation(self) -> None:
        rec = StrategyQualificationRecord.create(
            qualification_id="Q-TAMPER",
            strategy_id="S-1",
            strategy_spec_hash="hash",
            dataset_version="DS-1",
            dataset_sha256="sha",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pophash",
            effective_trial_count=5.0,
            observed_sharpe=1.2,
            dsr=0.95,
            trade_count=100,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
        )
        d = rec.to_dict()
        d["observed_sharpe"] = 3.0
        assert StrategyQualificationRecord.from_dict(d).verify_digest() is False

    def test_sqlite_update_is_blocked_by_trigger(self, tmp_path: Path) -> None:
        ledger = QualificationLedger(tmp_path / "ledger_tamper.db")
        rec = StrategyQualificationRecord.create(
            qualification_id="Q-SQL-1",
            strategy_id="S-1",
            strategy_spec_hash="hash",
            dataset_version="DS-1",
            dataset_sha256="sha",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pophash",
            effective_trial_count=5.0,
            observed_sharpe=1.2,
            dsr=0.95,
            trade_count=100,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
        )
        ledger.record_qualification(rec)

        with pytest.raises(sqlite3.IntegrityError, match="permanent and immutable"):
            ledger._connection.execute("UPDATE strategy_qualifications SET dsr = 0.999 WHERE qualification_id = 'Q-SQL-1'")

        with pytest.raises(sqlite3.IntegrityError, match="permanent and immutable"):
            ledger._connection.execute("UPDATE strategy_qualifications SET final_status = 'REJECTED' WHERE qualification_id = 'Q-SQL-1'")

        with pytest.raises(sqlite3.IntegrityError, match="permanent and immutable"):
            ledger._connection.execute("UPDATE strategy_qualifications SET holdout_state = 'FAILED' WHERE qualification_id = 'Q-SQL-1'")

    def test_sqlite_delete_is_blocked_by_trigger(self, tmp_path: Path) -> None:
        ledger = QualificationLedger(tmp_path / "ledger_del.db")
        rec = StrategyQualificationRecord.create(
            qualification_id="Q-SQL-2",
            strategy_id="S-1",
            strategy_spec_hash="hash",
            dataset_version="DS-1",
            dataset_sha256="sha",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pophash",
            effective_trial_count=5.0,
            observed_sharpe=1.2,
            dsr=0.95,
            trade_count=100,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
        )
        ledger.record_qualification(rec)

        with pytest.raises(sqlite3.IntegrityError, match="permanent and append-only"):
            ledger._connection.execute("DELETE FROM strategy_qualifications WHERE qualification_id = 'Q-SQL-2'")


# ---------------------------------------------------------------------------
# D. STATE MACHINE EXPLOITS AUDIT
# ---------------------------------------------------------------------------


class TestStateMachineExploitsAudit:
    @pytest.mark.parametrize(
        "src,dst",
        [
            (ValidationStatus.REJECTED, ValidationStatus.PAPER_ELIGIBLE),
            (ValidationStatus.UNDERPOWERED, ValidationStatus.PAPER_ELIGIBLE),
            (ValidationStatus.REJECTED_FINAL_HOLDOUT, ValidationStatus.PAPER_ELIGIBLE),
            (ValidationStatus.HOLDOUT_REQUIRED, ValidationStatus.PAPER_ELIGIBLE),
            (ValidationStatus.CANDIDATE, ValidationStatus.PAPER_ELIGIBLE),
            (ValidationStatus.PAPER_ELIGIBLE, ValidationStatus.VALIDATION),
            (ValidationStatus.PAPER_ELIGIBLE, ValidationStatus.REJECTED),
        ],
    )
    def test_illegal_state_transitions_fail(self, src: ValidationStatus, dst: ValidationStatus) -> None:
        with pytest.raises(InvalidStateTransitionError, match="Illegal validation status transition"):
            validate_transition(src, dst)

    def test_every_legal_transition_succeeds(self) -> None:
        for src, targets in ALLOWED_VALIDATION_TRANSITIONS.items():
            for dst in targets:
                validate_transition(src, dst)


# ---------------------------------------------------------------------------
# E. HOLDOUT CONSISTENCY AUDIT
# ---------------------------------------------------------------------------


class TestHoldoutConsistencyAudit:
    def test_untouched_holdout_yields_holdout_required(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        _seed_completed_trial(env, trial_id="TRIAL-E1", spec=spec)

        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-E1",
            dataset_version="DS-AUDIT-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.HOLDOUT_REQUIRED
        assert record.holdout_state == "UNTOUCHED"

    def test_failed_holdout_yields_rejected_final_holdout(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        strat_id = derive_strategy_id(spec)
        _seed_completed_trial(env, trial_id="TRIAL-E2", spec=spec)
        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id=strat_id,
            dataset_version="DS-AUDIT-V1",
            protocol="RP-2",
            state=HoldoutState.FAILED,
            trial_id="TRIAL-H-FAIL",
            result_json=json.dumps({"passed": False}),
        )

        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-E2",
            dataset_version="DS-AUDIT-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.REJECTED_FINAL_HOLDOUT
        assert record.holdout_state == "FAILED"

    def test_burned_holdout_yields_rejected(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        _seed_completed_trial(env, trial_id="TRIAL-E3", spec=spec)
        env["holdout_mgr"].burn_holdout("DS-AUDIT-V1", "RP-2", "Data snooping detected")

        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-E3",
            dataset_version="DS-AUDIT-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.REJECTED
        assert any("burned" in r.lower() for r in record.reasons)

    def test_holdout_belongs_to_same_scope(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        strat_id = derive_strategy_id(spec)
        _seed_completed_trial(env, trial_id="TRIAL-E4", spec=spec)

        # Record PASSED for a DIFFERENT dataset version
        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id=strat_id,
            dataset_version="DS-OTHER-V2",
            protocol="RP-2",
            state=HoldoutState.PASSED,
            trial_id="TRIAL-H-PASSED",
            result_json=json.dumps({"passed": True}),
        )

        # Evaluating for DS-AUDIT-V1 must report holdout as UNTOUCHED, not PASSED
        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-E4",
            dataset_version="DS-AUDIT-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.HOLDOUT_REQUIRED
        assert record.holdout_state == "UNTOUCHED"

    def test_holdout_evaluation_is_immutable(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        env["holdout_mgr"]._record_holdout_evaluation(
            strategy_id="STRAT-IMMUT",
            dataset_version="DS-AUDIT-V1",
            protocol="RP-2",
            state=HoldoutState.PASSED,
            trial_id="TRIAL-H-IMMUT",
            result_json=json.dumps({"passed": True}),
        )

        with pytest.raises(sqlite3.IntegrityError, match="holdout evaluations are permanent"):
            env["holdout_mgr"]._connection.execute("DELETE FROM holdout_evaluations WHERE candidate_strategy_id = 'STRAT-IMMUT'")

        with pytest.raises(sqlite3.IntegrityError, match="evaluated holdout state is permanent"):
            env["holdout_mgr"]._connection.execute("UPDATE holdout_evaluations SET state = 'FAILED' WHERE candidate_strategy_id = 'STRAT-IMMUT'")


# ---------------------------------------------------------------------------
# F. FIXTURE ISOLATION AUDIT
# ---------------------------------------------------------------------------


class TestFixtureIsolationAudit:
    def test_fixture_trial_rejected(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        strat_id = derive_strategy_id(spec)
        env["ledger"]._connection.execute(
            """
            INSERT INTO trials (
                trial_id, experiment_id, strategy_id, dataset_version, split_zone,
                research_protocol_version, feature_version, parameter_set_json, strategy_spec_json,
                seed, execution_model, cost_model, slippage_model,
                timestamp_started, estimated_runtime_minutes,
                estimated_llm_cost, mode, dataset_kind, status
            ) VALUES ('TRIAL-FIX-AUDIT', 'EXP-F', ?, 'DS-AUDIT-V1', 'VALIDATION', 'RP-2', 'f1', '{}', '{}', 1, 'm', 'c', 's', 'now', 0.1, 0.0, 'FIXTURE', 'LICENSED', 'COMPLETED')
            """,
            (strat_id,),
        )

        with pytest.raises(ValueError, match="requires mode='PRODUCTION'"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-FIX-AUDIT",
                dataset_version="DS-AUDIT-V1",
                research_protocol_version="RP-2",
            )

    def test_fixture_population_cannot_qualify(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        # Seed in FIXTURE mode: will not enter ProductionPopulationQuery
        _seed_completed_trial(env, trial_id="TRIAL-FIX-POP", spec=spec, mode="FIXTURE")

        with pytest.raises(ValueError, match="requires mode='PRODUCTION'"):
            env["gate"].evaluate_candidate(
                strategy_spec=spec,
                validation_trial_id="TRIAL-FIX-POP",
                dataset_version="DS-AUDIT-V1",
                research_protocol_version="RP-2",
            )

    def test_fixture_record_never_becomes_paper_eligible(self) -> None:
        rec = StrategyQualificationRecord.create(
            qualification_id="QUAL-FIX",
            strategy_id="STRAT-FIX",
            strategy_spec_hash="hash",
            dataset_version="DS-SYNTHETIC",
            dataset_sha256="sha",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pophash",
            effective_trial_count=1.0,
            observed_sharpe=1.0,
            dsr=0.5,
            trade_count=100,
            holdout_state="UNTOUCHED",
            robustness_status=RobustnessStatus.NOT_TESTED,
            final_status=ValidationStatus.REJECTED,
        )
        eligibility = check_paper_replay_eligibility(rec)
        assert eligibility.is_eligible is False


# ---------------------------------------------------------------------------
# G. SAMPLE-SIZE GATE AUDIT
# ---------------------------------------------------------------------------


class TestSampleSizeGateAudit:
    def test_sample_size_exactly_below_minimum(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        # Protocol requirement is 75 trades; 74 trades must be UNDERPOWERED
        _seed_completed_trial(env, trial_id="TRIAL-74", spec=spec, trade_count=74, net_pnl=200.0)

        record, _ = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-74",
            dataset_version="DS-AUDIT-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.UNDERPOWERED
        assert any("sample size underpowered" in r.lower() for r in record.reasons)

    def test_sample_size_exactly_at_minimum(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        # Exactly 75 trades: passes validation sample check (effective obs requires >= 100 for overall power)
        # Let's test with custom protocol config to test exact threshold dynamic loading
        custom_gate = StrategyValidationGate(
            dataset_registry=env["registry"],
            trial_ledger=env["ledger"],
            holdout_manager=env["holdout_mgr"],
            population_query=env["pop_query"],
            protocol_config={
                "sample_requirements": {
                    "minimum_trades_validation": 50,
                    "minimum_effective_outcome_observations": 50,
                }
            },
        )
        _seed_completed_trial(env, trial_id="TRIAL-50", spec=spec, trade_count=50, net_pnl=200.0)

        record, report = custom_gate.evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-50",
            dataset_version="DS-AUDIT-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.HOLDOUT_REQUIRED
        assert any("trade count (50) >= protocol minimum" in p.lower() for p in report.checks_passed)

    def test_sample_size_above_minimum(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        _seed_completed_trial(env, trial_id="TRIAL-120", spec=spec, trade_count=120, net_pnl=300.0)

        record, report = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-120",
            dataset_version="DS-AUDIT-V1",
            research_protocol_version="RP-2",
        )
        assert record.final_status == ValidationStatus.HOLDOUT_REQUIRED
        assert any("trade count (120) >= protocol minimum" in p.lower() for p in report.checks_passed)


# ---------------------------------------------------------------------------
# H. QUALIFICATION REPRODUCIBILITY AUDIT
# ---------------------------------------------------------------------------


class TestQualificationReproducibilityAudit:
    def test_same_evidence_yields_same_record_hash(self) -> None:
        rec1 = StrategyQualificationRecord.create(
            qualification_id="Q-DET-1",
            strategy_id="S-1",
            strategy_spec_hash="hash1",
            dataset_version="DS-1",
            dataset_sha256="sha1",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pop1",
            effective_trial_count=10.0,
            observed_sharpe=1.5,
            dsr=0.96,
            trade_count=120,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
            created_at="2026-09-20T12:00:00+00:00",
            reasons=["reason1"],
        )
        rec2 = StrategyQualificationRecord.create(
            qualification_id="Q-DET-1",
            strategy_id="S-1",
            strategy_spec_hash="hash1",
            dataset_version="DS-1",
            dataset_sha256="sha1",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pop1",
            effective_trial_count=10.0,
            observed_sharpe=1.5,
            dsr=0.96,
            trade_count=120,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
            created_at="2026-09-20T12:00:00+00:00",
            reasons=["reason1"],
        )
        assert rec1.record_hash == rec2.record_hash

    def test_changed_evidence_yields_different_record_hash(self) -> None:
        rec1 = StrategyQualificationRecord.create(
            qualification_id="Q-DET-1",
            strategy_id="S-1",
            strategy_spec_hash="hash1",
            dataset_version="DS-1",
            dataset_sha256="sha1",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pop1",
            effective_trial_count=10.0,
            observed_sharpe=1.5,
            dsr=0.96,
            trade_count=120,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
            created_at="2026-09-20T12:00:00+00:00",
        )
        rec2 = StrategyQualificationRecord.create(
            qualification_id="Q-DET-1",
            strategy_id="S-1",
            strategy_spec_hash="hash1",
            dataset_version="DS-1",
            dataset_sha256="sha1",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pop1",
            effective_trial_count=10.0,
            observed_sharpe=1.55,  # altered Sharpe
            dsr=0.96,
            trade_count=120,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
            created_at="2026-09-20T12:00:00+00:00",
        )
        assert rec1.record_hash != rec2.record_hash

    def test_canonical_json_ordering_is_deterministic(self) -> None:
        rec = StrategyQualificationRecord.create(
            qualification_id="Q-JSON",
            strategy_id="S-1",
            strategy_spec_hash="hash",
            dataset_version="DS-1",
            dataset_sha256="sha",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pop",
            effective_trial_count=5.0,
            observed_sharpe=1.2,
            dsr=0.95,
            trade_count=100,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
            created_at="2026-09-20T12:00:00+00:00",
        )
        canon = rec.canonical_json()
        parsed = json.loads(canon)
        # Verify keys are sorted in canonical string
        keys = list(parsed.keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# I. PAPER ELIGIBILITY AUDIT
# ---------------------------------------------------------------------------


class TestPaperEligibilityAudit:
    def test_paper_eligible_is_pure_qualification_only(self, tmp_path: Path) -> None:
        rec = StrategyQualificationRecord.create(
            qualification_id="Q-PE",
            strategy_id="S-1",
            strategy_spec_hash="hash",
            dataset_version="DS-1",
            dataset_sha256="sha",
            split_manifest_version="m1",
            research_protocol_version="RP-2",
            population_hash="pop",
            effective_trial_count=5.0,
            observed_sharpe=1.2,
            dsr=0.95,
            trade_count=100,
            holdout_state="PASSED",
            robustness_status=RobustnessStatus.PASSED,
            final_status=ValidationStatus.PAPER_ELIGIBLE,
        )
        eligibility = check_paper_replay_eligibility(rec)
        assert eligibility.is_eligible is True
        # Verify it has NO side effects: no broker, no orders, no live feed
        assert hasattr(eligibility, "is_eligible")
        assert not hasattr(eligibility, "place_order")
        assert not hasattr(eligibility, "broker_client")
        assert not hasattr(eligibility, "live_feed")


# ---------------------------------------------------------------------------
# STEP 3 — QUALIFICATION REPORT AUDIT
# ---------------------------------------------------------------------------


class TestQualificationReportAudit:
    def test_validation_report_prohibits_ranking_and_claims(self, tmp_path: Path) -> None:
        env = _setup_adversarial_env(tmp_path)
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 375, "session_window": [0.1, 0.9]})
        _seed_completed_trial(env, trial_id="TRIAL-RPT", spec=spec, trade_count=120, net_pnl=300.0)

        record, report = env["gate"].evaluate_candidate(
            strategy_spec=spec,
            validation_trial_id="TRIAL-RPT",
            dataset_version="DS-AUDIT-V1",
            research_protocol_version="RP-2",
        )

        combined_text = (
            report.summary_text.lower()
            + " "
            + " ".join(p.lower() for p in report.checks_passed)
            + " "
            + " ".join(f.lower() for f in report.checks_failed)
            + " "
            + " ".join(w.lower() for w in report.checks_warned)
        )

        prohibited_phrases = [
            "best strategy",
            "rank",
            "leaderboard",
            "winner",
            "expected profit",
            "probability of future profit",
        ]

        for phrase in prohibited_phrases:
            assert phrase not in combined_text, f"Prohibited phrase '{phrase}' found in validation report"
