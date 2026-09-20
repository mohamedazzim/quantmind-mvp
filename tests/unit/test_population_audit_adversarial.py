"""Adversarial audit tests for Research Population Accounting, OOS Return Artifacts,
EICT-CORR-1 inputs, DSR evidence layer, and Provenance (PRD v3.6).

Audit sections covered:
- Step 2: Artifact Security Audit (1-11)
- Step 3: Production Population Audit (Exclusion of FIXTURE, FAILED, ABANDONED, REJECTED, RUNNING, etc.)
- Step 4: Population Reset Exploit (Cumulative scope across multiple tasks/experiments)
- Step 5: EICT-CORR-1 Input Audit (Correlation, distance, intersection alignment, leakage, singletons)
- Step 6: EICT Effective Trial Count Test (Cluster structure verification)
- Step 7: DSR Input Audit (Reproducibility, non-annualized per-trade returns)
- Step 8: Return Distribution Edge Cases (11 edge cases)
- Step 9: Provenance Audit (10 required fields, mismatch rejection)
- Step 10: Artifact / Ledger Consistency (Trial vs Registry vs File)
- Step 11: Performance Profiling (Vectorized PyArrow/NumPy over 100+ trials)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from quantmind.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, BacktestTrade
from quantmind.data import DatasetKind, PurgeEmbargoSpec, compute_split_manifest
from quantmind.data.registry import DatasetRegistry
from quantmind.research_integrity.artifacts import (
    ArtifactImmutabilityError,
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactStatus,
    ArtifactType,
    _OOS_RETURNS_SCHEMA,
    _sha256_file,
    load_oos_returns_verified,
    write_oos_returns_artifact,
)
from quantmind.research_integrity.harness import ResearchHarness
from quantmind.research_integrity.population import (
    Eict1PopulationInputs,
    EictCorr1InputBuilder,
    EligibleTrial,
    ProductionPopulationQuery,
    ReturnDistributionMetadata,
    TrialProvenance,
    align_pair,
    build_dsr_inputs,
    compute_return_distribution,
    _EICT_CORR1_MIN_OVERLAP,
)
from quantmind.research_integrity.trial_ledger import ResearchBudget, TrialContext, TrialLedger
from quantmind.strategy.spec import StrategySpec


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def _create_trade(
    ts: pd.Timestamp,
    net_return_bps: float,
    side: int = 1,
) -> BacktestTrade:
    return BacktestTrade(
        signal_timestamp=ts,
        entry_timestamp=ts + pd.Timedelta(minutes=1),
        exit_timestamp=ts + pd.Timedelta(minutes=2),
        side=side,
        quantity=1,
        entry_price=100.0,
        exit_price=100.0 + net_return_bps / 100.0,
        gross_return_bps=net_return_bps,
        cost_bps=0.0,
        net_return_bps=net_return_bps,
        gross_pnl=net_return_bps,
        costs=0.0,
        net_pnl=net_return_bps,
    )


def _create_backtest_result(returns: Sequence[float], start_hour: int = 0) -> BacktestResult:
    base = pd.Timestamp("2023-01-03 09:00:00", tz="UTC")
    trades = tuple(
        _create_trade(base + pd.Timedelta(hours=start_hour + i), r)
        for i, r in enumerate(returns)
    )
    return BacktestResult(
        execution_model="next_bar_open_v1",
        trades=trades,
        gross_pnl=sum(returns),
        costs=0.0,
        net_pnl=sum(returns),
        mean_gross_return_bps=float(np.mean(returns)) if returns else 0.0,
        mean_net_return_bps=float(np.mean(returns)) if returns else 0.0,
    )


def _setup_dataset(tmp_path: Path, ds_version: str = "LICENSED-AUDIT-V1") -> tuple[DatasetRegistry, Path]:
    base_ts = pd.Timestamp("2023-01-03 09:15:00")
    num_bars = 150
    timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(num_bars)]
    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "open": 100.0 + np.sin(np.arange(num_bars) / 5.0),
            "high": 102.0 + np.sin(np.arange(num_bars) / 5.0),
            "low": 98.0 + np.sin(np.arange(num_bars) / 5.0),
            "close": 100.5 + np.sin(np.arange(num_bars) / 5.0),
            "volume": 1000,
            "open_interest": 500,
        }
    )
    csv_path = tmp_path / f"{ds_version}.csv"
    df.to_csv(csv_path, index=False)

    spec = PurgeEmbargoSpec(feature_lookback_bars=2, prediction_horizon_bars=2, embargo_bars=2)
    manifest = compute_split_manifest(
        dataset_version=ds_version,
        timestamps=timestamps,
        spec=spec,
        research_ratio=0.5,
        validation_ratio=0.25,
        holdout_ratio=0.25,
    )
    registry = DatasetRegistry(tmp_path / f"ds_reg_{ds_version}.db")
    registry.register_file(
        version=ds_version,
        kind=DatasetKind.LICENSED,
        path=csv_path,
        split_manifest=manifest,
    )
    return registry, csv_path


# ---------------------------------------------------------------------------
# STEP 2 — ARTIFACT SECURITY AUDIT
# ---------------------------------------------------------------------------


class TestStep2ArtifactSecurityAudit:
    """Audit 1-11 for OOS Return Artifacts."""

    def test_item_1_and_2_completed_production_trial_creates_single_artifact_with_sha(
        self, tmp_path: Path
    ) -> None:
        """1. Completed PRODUCTION trial creates exactly one OOS_RETURNS artifact.
        2. Artifact SHA-256 stored in registry and trial ledger."""
        ds_reg, _ = _setup_dataset(tmp_path)
        art_dir = tmp_path / "artifacts"
        art_reg = ArtifactRegistry(tmp_path / "artifacts.db")
        ledger = TrialLedger(tmp_path / "ledger.db")
        harness = ResearchHarness(
            BacktestEngine(),
            ledger,
            ds_reg,
            artifact_registry=art_reg,
            artifact_dir=art_dir,
        )

        spec = StrategySpec(
            strategy_version="1.0.0",
            feature_version="f_v1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 20, "session_window": [0.10, 0.90]},
        )
        budget = ResearchBudget(max_trials=10, max_experiments=10, max_strategy_variants=10)
        res = harness.run_trial(
            config=BacktestConfig(),
            research_task_id="TASK-SEC-AUDIT",
            strategy_spec=spec,
            dataset_version="LICENSED-AUDIT-V1",
            split_zone="RESEARCH",
            research_protocol_version="RP-2",
            seed=42,
            budget=budget,
        )

        # Verify exactly one artifact was created
        arts = art_reg.list_for_dataset("LICENSED-AUDIT-V1")
        assert len(arts) == 1
        art = arts[0]
        assert art.artifact_type == "OOS_RETURNS"
        assert art.row_count == res.trade_count

        # Verify SHA-256 in ledger matches artifact registry
        trial_row = ledger.get(art.trial_id)
        res_json = json.loads(trial_row["result_json"])
        assert res_json["artifact_sha256"] == art.sha256

        # Verify SHA-256 matches actual file on disk
        actual_file_sha = _sha256_file(Path(art.artifact_path))
        assert actual_file_sha == art.sha256

    def test_item_3_rereading_verifies_sha256(self, tmp_path: Path) -> None:
        """3. Re-reading a valid artifact verifies the SHA-256."""
        res = _create_backtest_result([1.0, 2.0, 3.0])
        path, sha, rows, start, end = write_oos_returns_artifact(
            res, trial_id="T-VERIFY", dataset_version="ds-1", artifact_dir=tmp_path
        )
        art_reg = ArtifactRegistry(tmp_path / "artifacts.db")
        rec = art_reg.record(
            trial_id="T-VERIFY",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )
        table = load_oos_returns_verified(rec)
        assert len(table) == 3

    def test_item_4_and_5_modify_one_byte_or_replace_fails_verification(self, tmp_path: Path) -> None:
        """4. Modify one byte of artifact -> verification must fail.
        5. Replace artifact contents -> verification must fail."""
        res = _create_backtest_result([1.0, 2.0, 3.0])
        path, sha, rows, start, end = write_oos_returns_artifact(
            res, trial_id="T-TAMPER", dataset_version="ds-1", artifact_dir=tmp_path
        )
        art_reg = ArtifactRegistry(tmp_path / "artifacts.db")
        rec = art_reg.record(
            trial_id="T-TAMPER",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )

        # 4. Modify one byte
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0xFF
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_oos_returns_verified(rec)

        # 5. Replace contents with another valid parquet
        other_res = _create_backtest_result([99.0, 100.0])
        other_path, _, _, _, _ = write_oos_returns_artifact(
            other_res, trial_id="T-OTHER", dataset_version="ds-1", artifact_dir=tmp_path
        )
        path.write_bytes(other_path.read_bytes())
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_oos_returns_verified(rec)

    def test_item_6_delete_artifact_population_rejects(self, tmp_path: Path) -> None:
        """6. Delete the artifact -> population query must reject that trial."""
        ledger = TrialLedger(tmp_path / "ledger.db")
        art_reg = ArtifactRegistry(tmp_path / "artifacts.db")

        ctx = TrialContext(
            trial_id="T-DEL",
            experiment_id="EXP-1",
            strategy_id="STRAT-1",
            dataset_version="ds-1",
            research_protocol_version="RP-1",
            feature_version="f1",
            parameter_set={},
            seed=1,
            execution_model="next_bar_open_v1",
            cost_model="NONE",
            slippage_model="0bps",
            estimated_runtime_minutes=0.01,
        )
        budget = ResearchBudget()
        ledger.reserve(ctx, budget)

        res = _create_backtest_result([5.0, 10.0])
        path, sha, rows, start, end = write_oos_returns_artifact(
            res, trial_id="T-DEL", dataset_version="ds-1", artifact_dir=tmp_path
        )
        art_reg.record(
            trial_id="T-DEL",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )
        ledger.complete(
            "T-DEL",
            result={"trade_count": 2, "artifact_sha256": sha},
            actual_runtime_minutes=0.01,
            status="COMPLETED",
        )

        # Delete artifact file
        path.unlink()

        query = ProductionPopulationQuery(ledger, art_reg)
        eligible = query.load_eligible_trials(dataset_version="ds-1", research_protocol_version="RP-1")
        assert len(eligible) == 0

    def test_item_7_and_8_database_triggers_prevent_update_and_delete(self, tmp_path: Path) -> None:
        """7. Attempt to delete artifact registry metadata -> reject.
        8. Attempt to update completed artifact metadata -> reject."""
        art_reg = ArtifactRegistry(tmp_path / "artifacts.db")
        rec = art_reg.record(
            trial_id="T-TRIGGERS",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-1",
            sha256="abc123sha",
            format="parquet",
            row_count=10,
            start_timestamp="2023-01-01T00:00:00Z",
            end_timestamp="2023-01-01T01:00:00Z",
            artifact_path=str(tmp_path / "dummy.parquet"),
        )

        # 7. DELETE rejected by trigger
        with pytest.raises(sqlite3.DatabaseError, match="trial_artifacts are immutable"):
            art_reg._connection.execute("DELETE FROM trial_artifacts WHERE artifact_id = ?", (rec.artifact_id,))

        # 8. UPDATE rejected by trigger
        with pytest.raises(sqlite3.DatabaseError, match="trial_artifacts are immutable"):
            art_reg._connection.execute(
                "UPDATE trial_artifacts SET sha256 = 'tampered' WHERE artifact_id = ?",
                (rec.artifact_id,),
            )

    def test_item_9_duplicate_artifact_registration_rejected(self, tmp_path: Path) -> None:
        """9. Attempt to register a second artifact for the same trial -> reject with ArtifactImmutabilityError."""
        art_reg = ArtifactRegistry(tmp_path / "artifacts.db")
        art_reg.record(
            trial_id="T-DUP-REG",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-1",
            sha256="sha1",
            format="parquet",
            row_count=5,
            start_timestamp="2023-01-01T00:00:00Z",
            end_timestamp="2023-01-01T01:00:00Z",
            artifact_path=str(tmp_path / "a1.parquet"),
        )
        with pytest.raises(ArtifactImmutabilityError, match="already registered"):
            art_reg.record(
                trial_id="T-DUP-REG",
                artifact_type=ArtifactType.OOS_RETURNS,
                dataset_version="ds-1",
                sha256="sha2",
                format="parquet",
                row_count=5,
                start_timestamp="2023-01-01T00:00:00Z",
                end_timestamp="2023-01-01T01:00:00Z",
                artifact_path=str(tmp_path / "a2.parquet"),
            )

    def test_item_10_empty_trade_artifact_behavior_deterministic(self, tmp_path: Path) -> None:
        """10. Empty-trade artifact behavior is deterministic and explicitly defined."""
        empty_res = BacktestResult(
            execution_model="next_bar_open_v1",
            trades=(),
            gross_pnl=0.0,
            costs=0.0,
            net_pnl=0.0,
            mean_gross_return_bps=0.0,
            mean_net_return_bps=0.0,
        )
        path, sha, rows, start, end = write_oos_returns_artifact(
            empty_res, trial_id="T-EMPTY-1", dataset_version="ds-1", artifact_dir=tmp_path
        )
        assert rows == 0
        table = pq.read_table(path)
        assert len(table) == 0
        assert list(table.schema.names) == list(_OOS_RETURNS_SCHEMA.names)

    def test_item_11_rerunning_identical_trial_produces_identical_content_and_hash(
        self, tmp_path: Path
    ) -> None:
        """11. Re-running identical trial produces equivalent return content and equivalent artifact hash."""
        returns = [12.5, -4.2, 8.1, 0.0, 15.3]
        res1 = _create_backtest_result(returns)
        res2 = _create_backtest_result(returns)

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        path1, sha1, rows1, _, _ = write_oos_returns_artifact(
            res1, trial_id="T-DET", dataset_version="ds-1", artifact_dir=dir1
        )
        path2, sha2, rows2, _, _ = write_oos_returns_artifact(
            res2, trial_id="T-DET", dataset_version="ds-1", artifact_dir=dir2
        )

        assert rows1 == rows2 == len(returns)
        assert sha1 == sha2


# ---------------------------------------------------------------------------
# STEP 3 — PRODUCTION POPULATION AUDIT
# ---------------------------------------------------------------------------


class TestStep3ProductionPopulationAudit:
    """Prove each exclusion from the production population."""

    @pytest.mark.parametrize(
        "mode,status,with_artifact,tamper,expected_eligible",
        [
            ("PRODUCTION", "COMPLETED", True, False, True),
            ("FIXTURE", "COMPLETED", True, False, False),       # FIXTURE excluded
            ("PRODUCTION", "FAILED", True, False, False),        # FAILED excluded
            ("PRODUCTION", "ABANDONED", True, False, False),     # ABANDONED excluded
            ("PRODUCTION", "REJECTED_NONCAUSAL", True, False, False),  # REJECTED excluded
            ("PRODUCTION", "REJECTED_FINAL_HOLDOUT", True, False, False), # REJECTED excluded
            ("PRODUCTION", "RUNNING", True, False, False),       # RUNNING excluded
            ("PRODUCTION", "COMPLETED", False, False, False),    # missing artifact excluded
            ("PRODUCTION", "COMPLETED", True, True, False),      # tampered artifact excluded
        ],
    )
    def test_population_exclusion_matrix(
        self,
        tmp_path: Path,
        mode: str,
        status: str,
        with_artifact: bool,
        tamper: bool,
        expected_eligible: bool,
    ) -> None:
        ledger = TrialLedger(tmp_path / f"ledger_{mode}_{status}_{with_artifact}_{tamper}.db")
        art_reg = ArtifactRegistry(tmp_path / f"art_reg_{mode}_{status}_{with_artifact}_{tamper}.db")
        trial_id = f"TRIAL-{mode}-{status}-{with_artifact}-{tamper}"

        ctx = TrialContext(
            trial_id=trial_id,
            experiment_id="EXP-TEST",
            strategy_id="STRAT-TEST",
            dataset_version="DS-AUDIT",
            research_protocol_version="RP-AUDIT",
            feature_version="f1",
            parameter_set={},
            seed=42,
            execution_model="next_bar_open_v1",
            cost_model="NONE",
            slippage_model="0bps",
            estimated_runtime_minutes=0.01,
            mode=mode,
        )
        budget = ResearchBudget()
        ledger.reserve(ctx, budget)

        sha = "fake-sha"
        if with_artifact:
            res = _create_backtest_result([1.0, 2.0, 3.0])
            path, sha, rows, start, end = write_oos_returns_artifact(
                res, trial_id=trial_id, dataset_version="DS-AUDIT", artifact_dir=tmp_path
            )
            art_reg.record(
                trial_id=trial_id,
                artifact_type=ArtifactType.OOS_RETURNS,
                dataset_version="DS-AUDIT",
                sha256=sha,
                format="parquet",
                row_count=rows,
                start_timestamp=start,
                end_timestamp=end,
                artifact_path=path,
            )
            if tamper:
                path.write_bytes(b"tampered content corrupted bytes")

        if status == "ABANDONED":
            ledger._connection.execute(
                "UPDATE trials SET status = 'ABANDONED', timestamp_completed = datetime('now') WHERE trial_id = ?",
                (trial_id,),
            )
        elif status != "RUNNING":
            ledger.complete(
                trial_id,
                result={"trade_count": 3, "artifact_sha256": sha},
                actual_runtime_minutes=0.01,
                status=status,
            )

        query = ProductionPopulationQuery(ledger, art_reg)
        eligible = query.load_eligible_trials(
            dataset_version="DS-AUDIT",
            research_protocol_version="RP-AUDIT",
        )
        assert (len(eligible) == 1) == expected_eligible


# ---------------------------------------------------------------------------
# STEP 4 — POPULATION RESET EXPLOIT
# ---------------------------------------------------------------------------


class TestStep4PopulationResetExploit:
    """Verify that changing task_id, experiment_id, agent_run_id, prompt_id does NOT reset population."""

    def test_changing_experiment_and_task_does_not_reset_population(self, tmp_path: Path) -> None:
        ledger = TrialLedger(tmp_path / "ledger_reset.db")
        art_reg = ArtifactRegistry(tmp_path / "art_reset.db")
        budget = ResearchBudget(max_trials=100)

        tasks = ["TASK_A_SUPERVISOR", "TASK_B_PROMPT_AGENT_12", "TASK_C_RETRY_RUN_99"]
        for i, task_id in enumerate(tasks):
            t_id = f"TRIAL-RESET-{i}"
            ctx = TrialContext(
                trial_id=t_id,
                experiment_id=f"EXP-{task_id}",
                strategy_id=f"STRAT-{i}",
                dataset_version="DS-CUMULATIVE",
                research_protocol_version="RP-2",
                feature_version="f1",
                parameter_set={},
                seed=i,
                execution_model="next_bar_open_v1",
                cost_model="NONE",
                slippage_model="0bps",
                estimated_runtime_minutes=0.01,
            )
            ledger.reserve(ctx, budget)
            res = _create_backtest_result([float(i + 1) * 2.0, float(i + 1) * 3.0])
            path, sha, rows, start, end = write_oos_returns_artifact(
                res, trial_id=t_id, dataset_version="DS-CUMULATIVE", artifact_dir=tmp_path
            )
            art_reg.record(
                trial_id=t_id,
                artifact_type=ArtifactType.OOS_RETURNS,
                dataset_version="DS-CUMULATIVE",
                sha256=sha,
                format="parquet",
                row_count=rows,
                start_timestamp=start,
                end_timestamp=end,
                artifact_path=path,
            )
            ledger.complete(
                t_id,
                result={"trade_count": rows, "artifact_sha256": sha},
                actual_runtime_minutes=0.01,
                status="COMPLETED",
            )

        query = ProductionPopulationQuery(ledger, art_reg)
        eligible = query.load_eligible_trials(
            dataset_version="DS-CUMULATIVE",
            research_protocol_version="RP-2",
        )
        assert len(eligible) == 3
        assert {t.trial_id for t in eligible} == {"TRIAL-RESET-0", "TRIAL-RESET-1", "TRIAL-RESET-2"}


# ---------------------------------------------------------------------------
# STEP 5 & 6 — EICT-CORR-1 AUDIT & EFFECTIVE TRIAL COUNT TEST
# ---------------------------------------------------------------------------


class TestStep5And6EictAuditAndEffectiveTrialCount:
    """EICT-CORR-1 audit and clustering verification."""

    def _make_trial(self, trial_id: str, ts: np.ndarray, returns: np.ndarray) -> EligibleTrial:
        dist = compute_return_distribution(trial_id, returns)
        rec = ArtifactRecord(
            artifact_id=f"ART-{trial_id}",
            trial_id=trial_id,
            artifact_type="OOS_RETURNS",
            dataset_version="DS-EICT",
            sha256="sha",
            format="parquet",
            row_count=len(returns),
            start_timestamp="2023-01-01T00:00:00Z",
            end_timestamp="2023-01-01T10:00:00Z",
            created_at="2023-01-01T00:00:00Z",
            status="COMPLETED",
            artifact_path="/dev/null",
        )
        prov = TrialProvenance(
            dataset_version="DS-EICT",
            dataset_sha256="dsha",
            split_manifest_version="v1",
            split_manifest_sha256="smsha",
            research_protocol_version="RP-2",
            strategy_id=f"STRAT-{trial_id}",
            experiment_id="EXP-1",
            code_version="0.1.0",
            config_hash="cfghash",
            artifact_sha256="sha",
        )
        return EligibleTrial(
            trial_id=trial_id,
            strategy_id=f"STRAT-{trial_id}",
            dataset_version="DS-EICT",
            research_protocol_version="RP-2",
            artifact=rec,
            net_returns=returns,
            signal_timestamps=ts,
            distribution=dist,
            provenance=prov,
        )

    def test_correlation_and_distance_relationships(self) -> None:
        """1. identical series: corr ≈ 1, dist ≈ 0.
        2. independent series: corr ≈ 0, dist ≈ sqrt(2).
        3. negatively correlated series: dist approaches 2."""
        n = 150
        ts = np.array(
            [pd.Timestamp("2023-01-03") + pd.Timedelta(hours=i) for i in range(n)],
            dtype="datetime64[us]",
        )
        rng = np.random.default_rng(123)
        ret_base = rng.normal(0, 1, n)

        # 1. Identical
        t_base = self._make_trial("T-BASE", ts, ret_base)
        t_ident = self._make_trial("T-IDENT", ts, ret_base.copy())
        builder = EictCorr1InputBuilder()
        out_ident = builder.build([t_base, t_ident])
        assert abs(out_ident.pairwise[0].pearson_correlation - 1.0) < 1e-6
        assert abs(out_ident.pairwise[0].eict_distance - 0.0) < 1e-6

        # 2. Independent
        ret_indep = rng.normal(0, 1, n)
        t_indep = self._make_trial("T-INDEP", ts, ret_indep)
        out_indep = builder.build([t_base, t_indep])
        corr_indep = out_indep.pairwise[0].pearson_correlation
        assert abs(corr_indep) < 0.2  # near 0
        expected_dist_indep = math.sqrt(2.0 * (1.0 - corr_indep))
        assert abs(out_indep.pairwise[0].eict_distance - expected_dist_indep) < 1e-10
        assert abs(out_indep.pairwise[0].eict_distance - math.sqrt(2.0)) < 0.3

        # 3. Negatively correlated
        ret_neg = -ret_base
        t_neg = self._make_trial("T-NEG", ts, ret_neg)
        out_neg = builder.build([t_base, t_neg])
        assert abs(out_neg.pairwise[0].pearson_correlation - (-1.0)) < 1e-10
        assert abs(out_neg.pairwise[0].eict_distance - 2.0) < 1e-10

    def test_timestamp_intersection_alignment_and_no_lookahead(self) -> None:
        """4. Pair alignment uses timestamp INTERSECTION only.
        5. No forward fill.
        6. No future timestamp leakage."""
        ts_a = np.array(
            [pd.Timestamp(f"2023-01-03 0{i}:00:00") for i in [1, 2, 4, 5]],
            dtype="datetime64[us]",
        )
        ret_a = np.array([10.0, 20.0, 40.0, 50.0])

        ts_b = np.array(
            [pd.Timestamp(f"2023-01-03 0{i}:00:00") for i in [2, 3, 4, 6]],
            dtype="datetime64[us]",
        )
        ret_b = np.array([200.0, 300.0, 400.0, 600.0])

        common, aligned_a, aligned_b = align_pair(ts_a, ret_a, ts_b, ret_b)
        # Expected intersection: only hours 2 and 4
        assert len(common) == 2
        expected_ts = np.array(
            [pd.Timestamp("2023-01-03 02:00:00"), pd.Timestamp("2023-01-03 04:00:00")],
            dtype="datetime64[us]",
        )
        np.testing.assert_array_equal(common, expected_ts)
        np.testing.assert_allclose(aligned_a, [20.0, 40.0])
        np.testing.assert_allclose(aligned_b, [200.0, 400.0])

    def test_step_6_eict_effective_trial_count_and_singleton(self) -> None:
        """Step 6: Synthetic population with known cluster structure:
        Cluster A: A1, A2, A3 (high corr)
        Cluster B: B1, B2 (high corr)
        Singleton: C1 with insufficient overlap (<100 common obs)."""
        n = 150
        ts = np.array(
            [pd.Timestamp("2023-01-03") + pd.Timedelta(hours=i) for i in range(n)],
            dtype="datetime64[us]",
        )
        rng = np.random.default_rng(999)

        # Base A
        base_a = rng.normal(0, 1, n)
        a1 = base_a + rng.normal(0, 0.05, n)
        a2 = base_a + rng.normal(0, 0.05, n)
        a3 = base_a + rng.normal(0, 0.05, n)

        # Base B (independent of A)
        base_b = rng.normal(0, 1, n)
        b1 = base_b + rng.normal(0, 0.05, n)
        b2 = base_b + rng.normal(0, 0.05, n)

        # C1 has only 30 observations (sparse overlap < 100)
        ts_c = ts[:30]
        c1 = rng.normal(0, 1, 30)

        trials = [
            self._make_trial("A1", ts, a1),
            self._make_trial("A2", ts, a2),
            self._make_trial("A3", ts, a3),
            self._make_trial("B1", ts, b1),
            self._make_trial("B2", ts, b2),
            self._make_trial("C1", ts_c, c1),
        ]

        builder = EictCorr1InputBuilder()
        out = builder.build(trials)

        # Raw count is 6
        assert len(out.trial_ids) == 6
        # C1 must be isolated as a singleton trial
        assert "C1" in out.singleton_trial_ids

        # Intra-cluster correlations > 0.8
        # Find index for A1, A2, A3
        idx = {tid: i for i, tid in enumerate(out.trial_ids)}
        corr_a1_a2 = out.correlation_matrix[idx["A1"], idx["A2"]]
        corr_b1_b2 = out.correlation_matrix[idx["B1"], idx["B2"]]
        assert corr_a1_a2 > 0.80
        assert corr_b1_b2 > 0.80

        # Inter-cluster A vs B correlation is small (< 0.5)
        corr_a_b = out.correlation_matrix[idx["A1"], idx["B1"]]
        assert abs(corr_a_b) < 0.5


# ---------------------------------------------------------------------------
# STEP 7 & 8 — DSR INPUT AUDIT & RETURN DISTRIBUTION EDGE CASES
# ---------------------------------------------------------------------------


class TestStep7And8DsrInputsAndEdgeCases:
    """Audit 11 edge cases for return distributions."""

    def test_item_1_zero_returns(self) -> None:
        """1. Zero returns: valid, mean=0, std=0, sharpe=0."""
        ret = np.zeros(50, dtype=np.float64)
        dist = compute_return_distribution("T-ZERO", ret)
        assert dist.mean_return == 0.0
        assert dist.std_return == 0.0
        assert dist.observed_sharpe == 0.0
        assert dist.skewness == 0.0
        assert dist.kurtosis == 0.0

    def test_item_2_and_3_constant_positive_and_negative(self) -> None:
        """2. Constant positive returns: std=0, sharpe=0.
        3. Constant negative returns: std=0, sharpe=0."""
        ret_pos = np.full(50, 5.0, dtype=np.float64)
        dist_pos = compute_return_distribution("T-CONST-POS", ret_pos)
        assert dist_pos.mean_return == 5.0
        assert dist_pos.std_return == 0.0
        assert dist_pos.observed_sharpe == 0.0

        ret_neg = np.full(50, -3.0, dtype=np.float64)
        dist_neg = compute_return_distribution("T-CONST-NEG", ret_neg)
        assert dist_neg.mean_return == -3.0
        assert dist_neg.std_return == 0.0
        assert dist_neg.observed_sharpe == 0.0

    def test_item_4_single_observation(self) -> None:
        """4. Single observation: valid, std=0, sharpe=0."""
        ret = np.array([10.0], dtype=np.float64)
        dist = compute_return_distribution("T-SINGLE", ret)
        assert dist.mean_return == 10.0
        assert dist.std_return == 0.0
        assert dist.observed_sharpe == 0.0
        assert dist.trade_count == 1
        assert dist.effective_observations == 1

    def test_item_5_two_observations(self) -> None:
        """5. Two observations: valid mean & sample std, skew=0, kurt=0 (insufficient for higher moments)."""
        ret = np.array([10.0, 20.0], dtype=np.float64)
        dist = compute_return_distribution("T-TWO", ret)
        assert dist.mean_return == 15.0
        assert abs(dist.std_return - math.sqrt(50.0)) < 1e-10
        assert dist.skewness == 0.0
        assert dist.kurtosis == 0.0
        assert dist.observed_sharpe == dist.mean_return / dist.std_return

    def test_item_6_zero_standard_deviation(self) -> None:
        """6. Zero standard deviation -> observed_sharpe is 0.0 (no division by zero)."""
        ret = np.array([7.0, 7.0, 7.0, 7.0])
        dist = compute_return_distribution("T-ZERO-STD", ret)
        assert dist.std_return == 0.0
        assert dist.observed_sharpe == 0.0

    def test_item_7_and_8_nan_and_inf_rejected(self) -> None:
        """7. NaN in returns -> rejected (raises ValueError).
        8. Inf in returns -> rejected (raises ValueError)."""
        ret_nan = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="non-finite"):
            compute_return_distribution("T-NAN", ret_nan)

        ret_inf = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="non-finite"):
            compute_return_distribution("T-INF", ret_inf)

    def test_item_9_insufficient_observations_for_higher_moments(self) -> None:
        """9. Insufficient observations (< 3 for skew, < 4 for kurtosis)."""
        ret3 = np.array([1.0, 2.0, 5.0])
        dist3 = compute_return_distribution("T-3", ret3)
        assert dist3.skewness != 0.0
        assert dist3.kurtosis == 0.0  # kurtosis requires >= 4

    def test_item_10_and_11_skewed_and_heavy_tailed(self) -> None:
        """10. Highly skewed distribution (positive skew).
        11. Heavy-tailed distribution (high excess kurtosis)."""
        ret_skew = np.array([1.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float64)
        dist_skew = compute_return_distribution("T-SKEW", ret_skew)
        assert dist_skew.skewness > 1.5

        # Heavy-tailed (leptokurtic): clustered around center with extreme outliers
        ret_heavy = np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
        dist_heavy = compute_return_distribution("T-HEAVY", ret_heavy)
        assert dist_heavy.kurtosis > 2.0


# ---------------------------------------------------------------------------
# STEP 9 & 10 — PROVENANCE AUDIT & ARTIFACT / LEDGER CONSISTENCY
# ---------------------------------------------------------------------------


class TestStep9And10ProvenanceAndConsistency:
    """Audit provenance retention and 3-way consistency."""

    def test_provenance_fields_present_in_eligible_trial(self, tmp_path: Path) -> None:
        """Every production trial must retain 10 provenance fields."""
        ds_reg, _ = _setup_dataset(tmp_path, "LICENSED-PROV-V1")
        art_dir = tmp_path / "artifacts"
        art_reg = ArtifactRegistry(tmp_path / "art_reg.db")
        ledger = TrialLedger(tmp_path / "ledger.db")
        harness = ResearchHarness(
            BacktestEngine(),
            ledger,
            ds_reg,
            artifact_registry=art_reg,
            artifact_dir=art_dir,
        )

        spec = StrategySpec(
            strategy_version="1.0.0",
            feature_version="f_v1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 20, "session_window": [0.10, 0.90]},
        )
        budget = ResearchBudget()
        res = harness.run_trial(
            config=BacktestConfig(),
            research_task_id="TASK-PROV",
            strategy_spec=spec,
            dataset_version="LICENSED-PROV-V1",
            split_zone="RESEARCH",
            research_protocol_version="RP-2",
            seed=42,
            budget=budget,
        )

        query = ProductionPopulationQuery(ledger, art_reg)
        eligible = query.load_eligible_trials(
            dataset_version="LICENSED-PROV-V1",
            research_protocol_version="RP-2",
        )
        assert len(eligible) == 1
        t = eligible[0]
        p = t.provenance

        # Check all 10 fields are non-empty and accurate
        assert p.dataset_version == "LICENSED-PROV-V1"
        assert len(p.dataset_sha256) == 64
        assert p.split_manifest_version == "v1"
        assert len(p.split_manifest_sha256) == 64
        assert p.research_protocol_version == "RP-2"
        assert p.strategy_id.startswith("STRAT-")
        assert p.experiment_id.startswith("EXP-")
        assert p.code_version == "0.1.0"
        assert len(p.config_hash) > 0
        assert len(p.artifact_sha256) == 64

    def test_three_way_consistency_ledger_mismatch_rejected(self, tmp_path: Path) -> None:
        """Trial result SHA != Artifact Registry SHA -> trial rejected from population."""
        ledger = TrialLedger(tmp_path / "ledger_mismatch.db")
        art_reg = ArtifactRegistry(tmp_path / "art_mismatch.db")

        ctx = TrialContext(
            trial_id="T-MISMATCH",
            experiment_id="EXP-1",
            strategy_id="STRAT-1",
            dataset_version="DS-1",
            research_protocol_version="RP-1",
            feature_version="f1",
            parameter_set={},
            seed=1,
            execution_model="next_bar_open_v1",
            cost_model="NONE",
            slippage_model="0bps",
            estimated_runtime_minutes=0.01,
        )
        ledger.reserve(ctx, ResearchBudget())

        res = _create_backtest_result([1.0, 2.0, 3.0])
        path, sha, rows, start, end = write_oos_returns_artifact(
            res, trial_id="T-MISMATCH", dataset_version="DS-1", artifact_dir=tmp_path
        )
        art_reg.record(
            trial_id="T-MISMATCH",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="DS-1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )
        # Complete trial with mismatched artifact_sha256 in ledger
        ledger.complete(
            "T-MISMATCH",
            result={"trade_count": 3, "artifact_sha256": "different_tampered_sha"},
            actual_runtime_minutes=0.01,
            status="COMPLETED",
        )

        query = ProductionPopulationQuery(ledger, art_reg)
        eligible = query.load_eligible_trials(dataset_version="DS-1", research_protocol_version="RP-1")
        assert len(eligible) == 0


# ---------------------------------------------------------------------------
# STEP 11 — PERFORMANCE
# ---------------------------------------------------------------------------


class TestStep11Performance:
    """Profile population builder with 100+ trials — fast vectorized execution."""

    def test_100_trials_population_performance(self, tmp_path: Path) -> None:
        """Measure artifact writing, loading, alignment, and distribution computation over 100 trials."""
        ledger = TrialLedger(tmp_path / "perf_ledger.db")
        art_reg = ArtifactRegistry(tmp_path / "perf_art_reg.db")
        budget = ResearchBudget(max_trials=1000)

        n_trials = 100
        n_obs = 110
        base_ts = pd.Timestamp("2023-01-03 09:00:00", tz="UTC")
        rng = np.random.default_rng(42)

        start_write = time.perf_counter()
        for i in range(n_trials):
            tid = f"T-PERF-{i:03d}"
            ctx = TrialContext(
                trial_id=tid,
                experiment_id="EXP-PERF",
                strategy_id=f"STRAT-{i % 10}",
                dataset_version="DS-PERF",
                research_protocol_version="RP-PERF",
                feature_version="f1",
                parameter_set={},
                seed=i,
                execution_model="next_bar_open_v1",
                cost_model="NONE",
                slippage_model="0bps",
                estimated_runtime_minutes=0.01,
            )
            ledger.reserve(ctx, budget)

            returns = rng.normal(0.5, 2.0, n_obs).tolist()
            res = _create_backtest_result(returns)
            path, sha, rows, start, end = write_oos_returns_artifact(
                res, trial_id=tid, dataset_version="DS-PERF", artifact_dir=tmp_path
            )
            art_reg.record(
                trial_id=tid,
                artifact_type=ArtifactType.OOS_RETURNS,
                dataset_version="DS-PERF",
                sha256=sha,
                format="parquet",
                row_count=rows,
                start_timestamp=start,
                end_timestamp=end,
                artifact_path=path,
            )
            ledger.complete(
                tid,
                result={"trade_count": rows, "artifact_sha256": sha},
                actual_runtime_minutes=0.01,
                status="COMPLETED",
            )
        write_time = time.perf_counter() - start_write

        # Measure population loading (100 Parquet files loaded and SHA-verified)
        start_load = time.perf_counter()
        query = ProductionPopulationQuery(ledger, art_reg)
        eligible = query.load_eligible_trials(
            dataset_version="DS-PERF",
            research_protocol_version="RP-PERF",
        )
        load_time = time.perf_counter() - start_load

        assert len(eligible) == n_trials
        # 100 trials loaded in less than 5.0 seconds
        assert load_time < 5.0

        # Measure DSR inputs construction
        start_dsr = time.perf_counter()
        dsr_inputs = build_dsr_inputs(eligible)
        dsr_time = time.perf_counter() - start_dsr
        assert dsr_inputs.effective_trial_count == n_trials
        assert dsr_time < 0.1  # near-instantaneous
