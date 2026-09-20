"""Tests for OOS return artifacts, artifact registry, population query, EICT-CORR-1
inputs, and DSR input preparation.

Test coverage map (A–O from spec):
  A. write_oos_returns_artifact writes a valid Parquet file
  B. SHA-256 recorded in registry matches file on disk
  C. ArtifactRegistry.record raises ArtifactImmutabilityError on duplicate
  D. Parquet immutability: rewriting file raises ArtifactImmutabilityError
  E. load_oos_returns_verified raises ValueError on tampered file
  F. load_oos_returns_verified raises FileNotFoundError on missing file
  G. Empty trade list produces zero-row artifact that round-trips cleanly
  H. ProductionPopulationQuery excludes FIXTURE trials
  I. ProductionPopulationQuery excludes FAILED/ABANDONED/REJECTED/RUNNING trials
  J. ProductionPopulationQuery excludes trials with no valid OOS artifact
  K. ProductionPopulationQuery returns only COMPLETED PRODUCTION trials with artifacts
  L. align_pair returns correct intersection timestamps and aligned returns
  M. EictCorr1InputBuilder: sufficient overlap → correct Pearson + distance
  N. EictCorr1InputBuilder: insufficient overlap → NaN, singleton_trial_ids populated
  O. compute_return_distribution / build_dsr_inputs produce correct statistics
  P. Protocol constants cannot be overridden via EictCorr1InputBuilder constructor
  Q. ReturnDistributionMetadata for empty trial is all zeros
  R. ResearchHarness.run_trial writes artifact when artifact_registry is provided
  S. Artifact Parquet schema is exact (_OOS_RETURNS_SCHEMA columns + types)
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import math
from pathlib import Path
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantmind.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    CostSchedule,
    CostSchedulePeriod,
)
from quantmind.data import DatasetKind, DatasetRegistry
from quantmind.research_integrity.artifacts import (
    ArtifactImmutabilityError,
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactType,
    _OOS_RETURNS_SCHEMA,
    _sha256_file,
    _trades_to_arrow,
    load_net_returns_array,
    load_oos_returns_verified,
    load_signal_timestamps_array,
    write_oos_returns_artifact,
)
from quantmind.research_integrity.population import (
    EligibleTrial,
    EictCorr1InputBuilder,
    ProductionPopulationQuery,
    ReturnDistributionMetadata,
    TrialProvenance,
    align_pair,
    build_dsr_inputs,
    compute_return_distribution,
    _EICT_CORR1_MIN_OVERLAP,
    _EICT_CORR1_CORRELATION_THRESHOLD,
    _EICT_CORR1_DISTANCE_THRESHOLD,
    _EICT_CORR1_LINKAGE,
    _EICT_CORR1_INSUFFICIENT_OVERLAP_POLICY,
)
from quantmind.research_integrity.trial_ledger import ResearchBudget, TrialContext, TrialLedger
from quantmind.research_integrity.harness import ResearchHarness
from quantmind.strategy.spec import StrategySpec


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_trade(
    signal_ts: pd.Timestamp | None = None,
    net_return_bps: float = 10.0,
    side: int = 1,
) -> BacktestTrade:
    ts = signal_ts or pd.Timestamp("2023-01-03 09:00:00", tz="UTC")
    return BacktestTrade(
        signal_timestamp=ts,
        entry_timestamp=ts + pd.Timedelta(minutes=1),
        exit_timestamp=ts + pd.Timedelta(minutes=2),
        side=side,
        quantity=1,
        entry_price=100.0,
        exit_price=100.0 + net_return_bps / 100,
        gross_return_bps=net_return_bps,
        cost_bps=0.0,
        net_return_bps=net_return_bps,
        gross_pnl=net_return_bps,
        costs=0.0,
        net_pnl=net_return_bps,
    )


def _make_result(net_returns: list[float] | None = None) -> BacktestResult:
    if net_returns is None:
        net_returns = [10.0, -5.0, 8.0, 3.0, -2.0]
    trades = tuple(
        _make_trade(
            signal_ts=pd.Timestamp("2023-01-03") + pd.Timedelta(hours=i),
            net_return_bps=r,
        )
        for i, r in enumerate(net_returns)
    )
    return BacktestResult(
        execution_model="next_bar_open_v1",
        trades=trades,
        gross_pnl=sum(net_returns),
        costs=0.0,
        net_pnl=sum(net_returns),
        mean_gross_return_bps=float(np.mean(net_returns)),
        mean_net_return_bps=float(np.mean(net_returns)),
    )


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


@pytest.fixture
def registry(tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "artifacts.db")


@pytest.fixture
def ledger(tmp_path: Path) -> TrialLedger:
    return TrialLedger(tmp_path / "ledger.db")


# ---------------------------------------------------------------------------
# A. write_oos_returns_artifact writes a valid Parquet file
# ---------------------------------------------------------------------------


class TestWriteOosReturnsArtifact:
    def test_writes_parquet_file(self, tmp_artifact_dir: Path) -> None:
        result = _make_result()
        path, sha256, row_count, start_ts, end_ts = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-abc",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        assert path.exists()
        assert path.suffix == ".parquet"
        assert row_count == 5
        assert sha256  # non-empty
        assert start_ts
        assert end_ts

    def test_parquet_has_expected_columns(self, tmp_artifact_dir: Path) -> None:
        result = _make_result()
        path, *_ = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-col",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        table = pq.read_table(path)
        assert set(table.schema.names) == {
            "signal_timestamp", "entry_timestamp", "exit_timestamp",
            "side", "gross_return_bps", "cost_bps", "net_return_bps",
        }

    def test_net_return_values_preserved(self, tmp_artifact_dir: Path) -> None:
        expected = [10.0, -5.0, 8.0]
        result = _make_result(expected)
        path, *_ = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-val",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        table = pq.read_table(path)
        actual = np.asarray(table.column("net_return_bps"), dtype=np.float64)
        np.testing.assert_allclose(actual, np.array(expected))


# ---------------------------------------------------------------------------
# B. SHA-256 matches
# ---------------------------------------------------------------------------


class TestSha256:
    def test_sha256_matches_file(self, tmp_artifact_dir: Path) -> None:
        result = _make_result()
        path, recorded_sha, *_ = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-sha",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        actual = _sha256_file(path)
        assert actual == recorded_sha


# ---------------------------------------------------------------------------
# C. Duplicate registration raises ArtifactImmutabilityError
# ---------------------------------------------------------------------------


class TestArtifactRegistryImmutability:
    def test_duplicate_record_raises(self, tmp_artifact_dir: Path, registry: ArtifactRegistry) -> None:
        result = _make_result()
        path, sha, rows, start, end = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-dup",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        registry.record(
            trial_id="TRIAL-dup",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-v1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )
        with pytest.raises(ArtifactImmutabilityError):
            registry.record(
                trial_id="TRIAL-dup",
                artifact_type=ArtifactType.OOS_RETURNS,
                dataset_version="ds-v1",
                sha256=sha,
                format="parquet",
                row_count=rows,
                start_timestamp=start,
                end_timestamp=end,
                artifact_path=path,
            )

    def test_sqlite_update_trigger_raises(self, tmp_artifact_dir: Path, registry: ArtifactRegistry) -> None:
        """SQLite trigger artifacts_no_update prevents direct UPDATE."""
        result = _make_result()
        path, sha, rows, start, end = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-upd",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        rec = registry.record(
            trial_id="TRIAL-upd",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-v1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )
        import sqlite3
        with pytest.raises(sqlite3.DatabaseError):
            registry._connection.execute(
                "UPDATE trial_artifacts SET sha256='tampered' WHERE artifact_id=?",
                (rec.artifact_id,),
            )

    def test_sqlite_delete_trigger_raises(self, tmp_artifact_dir: Path, registry: ArtifactRegistry) -> None:
        """SQLite trigger artifacts_no_delete prevents DELETE."""
        result = _make_result()
        path, sha, rows, start, end = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-del",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        rec = registry.record(
            trial_id="TRIAL-del",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-v1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )
        import sqlite3
        with pytest.raises(sqlite3.DatabaseError):
            registry._connection.execute(
                "DELETE FROM trial_artifacts WHERE artifact_id=?",
                (rec.artifact_id,),
            )


# ---------------------------------------------------------------------------
# D. Parquet file overwrite raises ArtifactImmutabilityError
# ---------------------------------------------------------------------------


class TestParquetFileImmutability:
    def test_rewrite_raises(self, tmp_artifact_dir: Path) -> None:
        result = _make_result()
        write_oos_returns_artifact(
            result,
            trial_id="TRIAL-rw",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        with pytest.raises(ArtifactImmutabilityError, match="already exists"):
            write_oos_returns_artifact(
                result,
                trial_id="TRIAL-rw",
                dataset_version="ds-v1",
                artifact_dir=tmp_artifact_dir,
            )


# ---------------------------------------------------------------------------
# E. SHA-256 mismatch raises ValueError
# ---------------------------------------------------------------------------


class TestSha256Verification:
    def test_tampered_file_raises(self, tmp_artifact_dir: Path, registry: ArtifactRegistry) -> None:
        result = _make_result()
        path, sha, rows, start, end = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-tamp",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        rec = registry.record(
            trial_id="TRIAL-tamp",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-v1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )
        # Tamper the file
        path.write_bytes(path.read_bytes() + b"\x00\x00")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_oos_returns_verified(rec)

    # F. Missing file raises FileNotFoundError
    def test_missing_file_raises(self, tmp_artifact_dir: Path, registry: ArtifactRegistry) -> None:
        result = _make_result()
        path, sha, rows, start, end = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-miss",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        rec = registry.record(
            trial_id="TRIAL-miss",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-v1",
            sha256=sha,
            format="parquet",
            row_count=rows,
            start_timestamp=start,
            end_timestamp=end,
            artifact_path=path,
        )
        path.unlink()
        with pytest.raises(FileNotFoundError):
            load_oos_returns_verified(rec)


# ---------------------------------------------------------------------------
# G. Empty trade list → zero-row artifact
# ---------------------------------------------------------------------------


class TestEmptyTrades:
    def test_empty_result_zero_rows(self, tmp_artifact_dir: Path, registry: ArtifactRegistry) -> None:
        result = BacktestResult(
            execution_model="next_bar_open_v1",
            trades=(),
            gross_pnl=0.0, costs=0.0, net_pnl=0.0,
            mean_gross_return_bps=0.0, mean_net_return_bps=0.0,
        )
        path, sha, rows, start, end = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-empty",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        assert rows == 0
        rec = registry.record(
            trial_id="TRIAL-empty",
            artifact_type=ArtifactType.OOS_RETURNS,
            dataset_version="ds-v1",
            sha256=sha, format="parquet", row_count=rows,
            start_timestamp=start, end_timestamp=end, artifact_path=path,
        )
        table = load_oos_returns_verified(rec)
        assert len(table) == 0
        arr = load_net_returns_array(rec)
        assert arr.shape == (0,)


# ---------------------------------------------------------------------------
# Helpers: build ledger + registry with completed PRODUCTION trials
# ---------------------------------------------------------------------------


def _insert_production_trial(
    ledger: TrialLedger,
    trial_id: str,
    mode: str = "PRODUCTION",
    status: str = "COMPLETED",
    dataset_version: str = "ds-v1",
    research_protocol_version: str = "rp-v2",
) -> None:
    """Insert a trial row directly, bypassing harness (for test isolation)."""
    ctx = TrialContext(
        trial_id=trial_id,
        experiment_id="EXP-test",
        strategy_id=f"STRAT-{trial_id}",
        dataset_version=dataset_version,
        research_protocol_version=research_protocol_version,
        feature_version="fv-1",
        parameter_set={},
        seed=42,
        execution_model="next_bar_open_v1",
        cost_model="NONE",
        slippage_model="per_side_0.000000bps",
        estimated_runtime_minutes=0.01,
        mode=mode,
    )
    budget = ResearchBudget(max_trials=10_000, max_experiments=100, max_strategy_variants=1_000,
                            max_runtime_minutes=1_000.0, max_llm_cost=100.0)
    ledger.reserve(ctx, budget)
    if status == "ABANDONED":
        ledger._connection.execute(
            """
            UPDATE trials
            SET timestamp_completed = datetime('now'),
                actual_runtime_minutes = estimated_runtime_minutes,
                actual_llm_cost = estimated_llm_cost,
                result_json = '{"reason":"abandoned"}',
                status = 'ABANDONED'
            WHERE trial_id = ?
            """,
            (trial_id,),
        )
    elif status != "RUNNING":
        ledger.complete(
            trial_id,
            result={"trade_count": 5},
            actual_runtime_minutes=0.01,
            status=status,
        )


def _register_artifact(
    registry: ArtifactRegistry,
    trial_id: str,
    artifact_dir: Path,
    dataset_version: str = "ds-v1",
) -> ArtifactRecord:
    result = _make_result()
    path, sha, rows, start, end = write_oos_returns_artifact(
        result,
        trial_id=trial_id,
        dataset_version=dataset_version,
        artifact_dir=artifact_dir,
    )
    return registry.record(
        trial_id=trial_id,
        artifact_type=ArtifactType.OOS_RETURNS,
        dataset_version=dataset_version,
        sha256=sha, format="parquet", row_count=rows,
        start_timestamp=start, end_timestamp=end, artifact_path=path,
    )


# ---------------------------------------------------------------------------
# H. FIXTURE trials excluded from population
# ---------------------------------------------------------------------------


class TestProductionPopulationExclusions:
    def test_fixture_excluded(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-fix", mode="FIXTURE")
        # FIXTURE mode not eligible, no artifact needed
        query = ProductionPopulationQuery(ledger, registry)
        result = query.load_eligible_trials(
            dataset_version="ds-v1", research_protocol_version="rp-v2"
        )
        assert len(result) == 0

    # I. Non-COMPLETED terminal statuses excluded
    def test_failed_excluded(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-fail", status="FAILED")
        query = ProductionPopulationQuery(ledger, registry)
        assert query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2") == []

    def test_abandoned_excluded(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-aband", status="ABANDONED")
        query = ProductionPopulationQuery(ledger, registry)
        assert query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2") == []

    def test_rejected_noncausal_excluded(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-nc", status="REJECTED_NONCAUSAL")
        query = ProductionPopulationQuery(ledger, registry)
        assert query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2") == []

    def test_rejected_final_holdout_excluded(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-rfh", status="REJECTED_FINAL_HOLDOUT")
        query = ProductionPopulationQuery(ledger, registry)
        assert query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2") == []

    # J. No artifact → excluded
    def test_no_artifact_excluded(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-noart", status="COMPLETED")
        query = ProductionPopulationQuery(ledger, registry)
        assert query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2") == []

    # J2. Tampered artifact → excluded (SHA-256 mismatch)
    def test_tampered_artifact_excluded(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-tamp2", status="COMPLETED")
        rec = _register_artifact(registry, "TRIAL-tamp2", tmp_artifact_dir)
        # Tamper the file so SHA check fails
        Path(rec.artifact_path).write_bytes(b"\xff" * 100)
        query = ProductionPopulationQuery(ledger, registry)
        eligible = query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2")
        trial_ids = [t.trial_id for t in eligible]
        assert "TRIAL-tamp2" not in trial_ids


# ---------------------------------------------------------------------------
# K. Happy path: COMPLETED PRODUCTION with valid artifact → included
# ---------------------------------------------------------------------------


class TestProductionPopulationEligible:
    def test_eligible_trial_included(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-good", status="COMPLETED")
        _register_artifact(registry, "TRIAL-good", tmp_artifact_dir)
        query = ProductionPopulationQuery(ledger, registry)
        eligible = query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2")
        assert len(eligible) == 1
        assert eligible[0].trial_id == "TRIAL-good"
        assert eligible[0].net_returns.dtype == np.float64
        assert len(eligible[0].net_returns) == 5

    def test_multiple_eligible_trials(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        for i in range(3):
            _insert_production_trial(ledger, f"TRIAL-m{i}", status="COMPLETED")
            _register_artifact(registry, f"TRIAL-m{i}", tmp_artifact_dir)
        query = ProductionPopulationQuery(ledger, registry)
        eligible = query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2")
        assert len(eligible) == 3

    def test_distribution_attached(
        self, tmp_artifact_dir: Path, registry: ArtifactRegistry, ledger: TrialLedger
    ) -> None:
        _insert_production_trial(ledger, "TRIAL-dist", status="COMPLETED")
        _register_artifact(registry, "TRIAL-dist", tmp_artifact_dir)
        query = ProductionPopulationQuery(ledger, registry)
        eligible = query.load_eligible_trials(dataset_version="ds-v1", research_protocol_version="rp-v2")
        dist = eligible[0].distribution
        assert isinstance(dist, ReturnDistributionMetadata)
        assert dist.trade_count == 5
        assert dist.effective_observations == 5


# ---------------------------------------------------------------------------
# L. align_pair: timestamp intersection
# ---------------------------------------------------------------------------


class TestAlignPair:
    def _make_ts(self, hours: list[int]) -> np.ndarray:
        return np.array(
            [pd.Timestamp("2023-01-03") + pd.Timedelta(hours=h) for h in hours],
            dtype="datetime64[us]",
        )

    def test_full_overlap(self) -> None:
        ts = self._make_ts([0, 1, 2])
        ret = np.array([1.0, 2.0, 3.0])
        common, a, b = align_pair(ts, ret, ts.copy(), ret.copy())
        assert len(common) == 3
        np.testing.assert_array_equal(a, b)

    def test_partial_overlap(self) -> None:
        ts_a = self._make_ts([0, 1, 2, 3])
        ret_a = np.array([1.0, 2.0, 3.0, 4.0])
        ts_b = self._make_ts([1, 2, 4])
        ret_b = np.array([10.0, 20.0, 30.0])
        common, a, b = align_pair(ts_a, ret_a, ts_b, ret_b)
        assert len(common) == 2  # hours 1, 2
        np.testing.assert_allclose(a, [2.0, 3.0])
        np.testing.assert_allclose(b, [10.0, 20.0])

    def test_no_overlap(self) -> None:
        ts_a = self._make_ts([0, 1])
        ts_b = self._make_ts([5, 6])
        common, a, b = align_pair(ts_a, np.ones(2), ts_b, np.ones(2))
        assert len(common) == 0
        assert len(a) == 0
        assert len(b) == 0


# ---------------------------------------------------------------------------
# M. EICT-CORR-1: sufficient overlap → correct values
# ---------------------------------------------------------------------------


def _make_eligible_trial(
    trial_id: str,
    signal_hours: list[int],
    returns: list[float],
) -> EligibleTrial:
    ts = np.array(
        [pd.Timestamp("2023-01-03") + pd.Timedelta(hours=h) for h in signal_hours],
        dtype="datetime64[us]",
    )
    net = np.array(returns, dtype=np.float64)
    dist = compute_return_distribution(trial_id, net)
    # Construct a minimal ArtifactRecord (won't be file-loaded in these tests)
    rec = ArtifactRecord(
        artifact_id=f"ART-{trial_id}",
        trial_id=trial_id,
        artifact_type="OOS_RETURNS",
        dataset_version="ds-v1",
        sha256="fake",
        format="parquet",
        row_count=len(returns),
        start_timestamp="2023-01-03T00:00:00+00:00",
        end_timestamp="2023-01-03T23:00:00+00:00",
        created_at="2023-01-03T00:00:00+00:00",
        status="COMPLETED",
        artifact_path="/nonexistent/fake.parquet",
    )
    prov = TrialProvenance(
        dataset_version="ds-v1",
        dataset_sha256="fake-sha",
        split_manifest_version="sm-v1",
        split_manifest_sha256="fake-sm-sha",
        research_protocol_version="rp-v2",
        strategy_id=f"STRAT-{trial_id}",
        experiment_id="EXP-test",
        code_version="0.1.0",
        config_hash="fake-cfg",
        artifact_sha256="fake",
    )
    return EligibleTrial(
        trial_id=trial_id,
        strategy_id=f"STRAT-{trial_id}",
        dataset_version="ds-v1",
        research_protocol_version="rp-v2",
        artifact=rec,
        net_returns=net,
        signal_timestamps=ts,
        distribution=dist,
        provenance=prov,
    )


class TestEictCorr1InputBuilder:
    def _builder(self) -> EictCorr1InputBuilder:
        return EictCorr1InputBuilder()

    def test_sufficient_overlap_correct_correlation(self) -> None:
        n = _EICT_CORR1_MIN_OVERLAP + 10
        rng = np.random.default_rng(42)
        returns_a = rng.normal(0, 1, n)
        returns_b = returns_a * 0.9 + rng.normal(0, 0.1, n)  # highly correlated
        hours = list(range(n))

        ta = _make_eligible_trial("T-A", hours, returns_a.tolist())
        tb = _make_eligible_trial("T-B", hours, returns_b.tolist())

        result = self._builder().build([ta, tb])
        assert len(result.pairwise) == 1
        p = result.pairwise[0]
        assert p.sufficient_overlap
        assert p.n_common == n
        assert not math.isnan(p.pearson_correlation)
        # Correlation should be very high for this data
        assert p.pearson_correlation > 0.8

        # distance = sqrt(2*(1-r))
        expected_dist = np.sqrt(2.0 * (1.0 - p.pearson_correlation))
        assert abs(p.eict_distance - expected_dist) < 1e-10

        # Correlation matrix symmetry
        assert result.correlation_matrix.shape == (2, 2)
        assert result.correlation_matrix[0, 0] == 1.0
        assert result.correlation_matrix[1, 1] == 1.0
        assert result.correlation_matrix[0, 1] == result.correlation_matrix[1, 0]

    def test_distance_matrix_diagonal_zero(self) -> None:
        n = _EICT_CORR1_MIN_OVERLAP + 5
        hours = list(range(n))
        ta = _make_eligible_trial("T-C", hours, [1.0] * n)
        result = self._builder().build([ta])
        assert result.distance_matrix.shape == (1, 1)
        assert result.distance_matrix[0, 0] == 0.0

    # N. Insufficient overlap → NaN, singleton_trial_ids
    def test_insufficient_overlap_singleton(self) -> None:
        # Only 50 common observations — below min of 100
        n = _EICT_CORR1_MIN_OVERLAP - 50
        ta = _make_eligible_trial("T-S1", list(range(n)), [1.0] * n)
        tb = _make_eligible_trial("T-S2", list(range(n)), [2.0] * n)

        result = self._builder().build([ta, tb])
        assert len(result.pairwise) == 1
        p = result.pairwise[0]
        assert not p.sufficient_overlap
        assert math.isnan(p.pearson_correlation)
        assert math.isnan(p.eict_distance)

        # Both are singletons since no pair has sufficient overlap
        assert set(result.singleton_trial_ids) == {"T-S1", "T-S2"}

    def test_no_overlap_both_singletons(self) -> None:
        ta = _make_eligible_trial("T-X", list(range(50)), [1.0] * 50)
        tb = _make_eligible_trial("T-Y", list(range(500, 600)), [2.0] * 100)
        result = self._builder().build([ta, tb])
        assert "T-X" in result.singleton_trial_ids
        assert "T-Y" in result.singleton_trial_ids

    def test_protocol_constants_in_output(self) -> None:
        result = self._builder().build([])
        assert result.correlation_threshold == _EICT_CORR1_CORRELATION_THRESHOLD
        assert result.distance_threshold == _EICT_CORR1_DISTANCE_THRESHOLD
        assert result.linkage == _EICT_CORR1_LINKAGE
        assert result.minimum_overlap_observations == _EICT_CORR1_MIN_OVERLAP
        assert result.insufficient_overlap_policy == _EICT_CORR1_INSUFFICIENT_OVERLAP_POLICY

    # P. Protocol constants cannot be overridden
    def test_wrong_min_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="minimum_overlap_observations"):
            EictCorr1InputBuilder(minimum_overlap_observations=50)

    def test_wrong_correlation_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="correlation_threshold"):
            EictCorr1InputBuilder(correlation_threshold=0.95)

    def test_wrong_distance_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="distance_threshold"):
            EictCorr1InputBuilder(distance_threshold=0.5)

    def test_wrong_linkage_raises(self) -> None:
        with pytest.raises(ValueError, match="linkage"):
            EictCorr1InputBuilder(linkage="single")

    def test_wrong_overlap_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="insufficient_overlap_policy"):
            EictCorr1InputBuilder(insufficient_overlap_policy="drop")


# ---------------------------------------------------------------------------
# O. compute_return_distribution / build_dsr_inputs
# ---------------------------------------------------------------------------


class TestReturnDistribution:
    def test_basic_statistics(self) -> None:
        returns = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float64)
        dist = compute_return_distribution("T-stat", returns)
        assert abs(dist.mean_return - np.mean(returns)) < 1e-10
        assert abs(dist.std_return - np.std(returns, ddof=1)) < 1e-10
        assert dist.trade_count == 8
        assert dist.effective_observations == 8
        assert dist.observed_sharpe == dist.mean_return / dist.std_return

    # Q. Empty trial → all zeros
    def test_empty_returns_zeros(self) -> None:
        dist = compute_return_distribution("T-empty", np.array([], dtype=np.float64))
        assert dist.mean_return == 0.0
        assert dist.std_return == 0.0
        assert dist.skewness == 0.0
        assert dist.kurtosis == 0.0
        assert dist.observed_sharpe == 0.0
        assert dist.trade_count == 0
        assert dist.effective_observations == 0

    def test_zero_std_observed_sharpe_is_zero(self) -> None:
        dist = compute_return_distribution("T-const", np.array([5.0, 5.0, 5.0], dtype=np.float64))
        assert dist.std_return == 0.0
        assert dist.observed_sharpe == 0.0


class TestBuildDsrInputs:
    def test_dsr_inputs_count(self) -> None:
        trials = [
            _make_eligible_trial(f"T-d{i}", list(range(10)), [float(i)] * 10)
            for i in range(4)
        ]
        dsr = build_dsr_inputs(trials)
        assert dsr.effective_trial_count == 4
        assert len(dsr.trial_distributions) == 4

    def test_empty_population(self) -> None:
        dsr = build_dsr_inputs([])
        assert dsr.effective_trial_count == 0
        assert dsr.trial_distributions == []


# ---------------------------------------------------------------------------
# R. ResearchHarness writes artifact when artifact_registry provided
# ---------------------------------------------------------------------------


class TestResearchHarnessArtifactIntegration:
    """Integration: harness writes OOS artifact to disk + registry on success."""

    def _make_harness(
        self, ledger: TrialLedger, artifact_registry: ArtifactRegistry, artifact_dir: Path
    ) -> tuple[ResearchHarness, str]:
        from datetime import timedelta
        from quantmind.data import DatasetKind, PurgeEmbargoSpec, compute_split_manifest
        from quantmind.data.registry import DatasetRegistry

        base_ts = pd.Timestamp("2023-01-03 09:15:00")
        num_bars = 120
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
        csv_path = artifact_dir / "licensed_data.csv"
        df.to_csv(csv_path, index=False)

        spec = PurgeEmbargoSpec(feature_lookback_bars=2, prediction_horizon_bars=2, embargo_bars=2)
        manifest = compute_split_manifest(
            dataset_version="ds-harness-v1",
            timestamps=timestamps,
            spec=spec,
            research_ratio=0.5,
            validation_ratio=0.25,
            holdout_ratio=0.25,
        )

        registry = DatasetRegistry()
        ds_version = "ds-harness-v1"
        registry.register_file(
            version=ds_version,
            kind=DatasetKind.LICENSED,
            path=csv_path,
            split_manifest=manifest,
        )

        engine = BacktestEngine()
        harness = ResearchHarness(
            engine,
            ledger,
            registry,
            artifact_registry=artifact_registry,
            artifact_dir=artifact_dir,
        )
        return harness, ds_version

    def test_artifact_written_after_trial(
        self,
        tmp_artifact_dir: Path,
        registry: ArtifactRegistry,
        ledger: TrialLedger,
    ) -> None:
        harness, ds_version = self._make_harness(ledger, registry, tmp_artifact_dir)
        spec = StrategySpec(
            strategy_version="1.0.0",
            feature_version="f_v1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 20, "session_window": [0.10, 0.90]},
        )
        config = BacktestConfig()
        budget = ResearchBudget(max_trials=10, max_experiments=10,
                                max_strategy_variants=10, max_runtime_minutes=60.0, max_llm_cost=10.0)
        result = harness.run_trial(
            config=config,
            research_task_id="task-harness-test",
            strategy_spec=spec,
            dataset_version=ds_version,
            split_zone="RESEARCH",
            research_protocol_version="rp-v2",
            seed=42,
            budget=budget,
        )

        # Check artifact was registered
        arts = registry.list_for_dataset(ds_version)
        assert len(arts) == 1
        art = arts[0]
        assert art.format == "parquet"
        assert art.artifact_type == "OOS_RETURNS"
        assert Path(art.artifact_path).exists()

        # Verify SHA-256 integrity
        actual_sha = _sha256_file(Path(art.artifact_path))
        assert actual_sha == art.sha256

        # Verify row count matches trade count
        assert art.row_count == result.trade_count


# ---------------------------------------------------------------------------
# S. Parquet schema is exactly _OOS_RETURNS_SCHEMA
# ---------------------------------------------------------------------------


class TestParquetSchema:
    def test_schema_exact(self, tmp_artifact_dir: Path) -> None:
        result = _make_result()
        path, *_ = write_oos_returns_artifact(
            result,
            trial_id="TRIAL-schema",
            dataset_version="ds-v1",
            artifact_dir=tmp_artifact_dir,
        )
        table = pq.read_table(path)
        # Column names match
        assert list(table.schema.names) == list(_OOS_RETURNS_SCHEMA.names)
        # Types match
        for i, field in enumerate(_OOS_RETURNS_SCHEMA):
            assert table.schema.field(field.name).type == field.type, (
                f"Column {field.name}: expected {field.type}, got {table.schema.field(field.name).type}"
            )
