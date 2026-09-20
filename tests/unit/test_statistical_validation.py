"""Unit tests for Statistical Validation: EICT-CORR-1 Clustering & Deflated Sharpe Ratio (PRD v3.7).

Coverage:
- LAYER A: EICT-CORR-1 hierarchical clustering, singletons, reproducible population hash
- LAYER B: Deflated Sharpe Ratio calculations against hand-calculated references
- Multiple-testing guards and negative tests
- End-to-end StatisticalValidationPipeline integration
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from quantmind.backtest.engine import BacktestConfig, BacktestResult, BacktestTrade
from quantmind.data import DatasetKind, PurgeEmbargoSpec, compute_split_manifest
from quantmind.data.registry import DatasetRegistry
from quantmind.research_integrity.artifacts import (
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactType,
    write_oos_returns_artifact,
)
from quantmind.research_integrity.population import (
    Eict1PopulationInputs,
    EictCorr1InputBuilder,
    EligibleTrial,
    ProductionPopulationQuery,
    ReturnDistributionMetadata,
    TrialProvenance,
    compute_return_distribution,
)
from quantmind.research_integrity.statistical_validation import (
    DeflatedSharpeCalculator,
    DeflatedSharpeResult,
    EictClusterResult,
    EictCorr1Calculator,
    StatisticalValidationPipeline,
    compute_population_hash,
)
from quantmind.research_integrity.trial_ledger import ResearchBudget, TrialContext, TrialLedger


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_mock_trial(
    trial_id: str,
    returns: Sequence[float],
    start_hour: int = 0,
    dataset_version: str = "DS-TEST",
    protocol_version: str = "RP-2",
) -> EligibleTrial:
    ret_arr = np.asarray(returns, dtype=np.float64)
    n = len(ret_arr)
    ts = np.array(
        [pd.Timestamp("2023-01-03 09:00:00") + pd.Timedelta(hours=start_hour + i) for i in range(n)],
        dtype="datetime64[us]",
    )
    dist = compute_return_distribution(trial_id, ret_arr)
    art = ArtifactRecord(
        artifact_id=f"ART-{trial_id}",
        trial_id=trial_id,
        artifact_type="OOS_RETURNS",
        dataset_version=dataset_version,
        sha256=f"sha_{trial_id}",
        format="parquet",
        row_count=n,
        start_timestamp="2023-01-03T09:00:00Z",
        end_timestamp="2023-01-03T18:00:00Z",
        created_at="2023-01-03T09:00:00Z",
        status="COMPLETED",
        artifact_path="/dummy/path.parquet",
    )
    prov = TrialProvenance(
        dataset_version=dataset_version,
        dataset_sha256="dsha",
        split_manifest_version="v1",
        split_manifest_sha256="smsha",
        research_protocol_version=protocol_version,
        strategy_id=f"STRAT-{trial_id}",
        experiment_id="EXP-1",
        code_version="0.1.0",
        config_hash="cfghash",
        artifact_sha256=f"sha_{trial_id}",
    )
    return EligibleTrial(
        trial_id=trial_id,
        strategy_id=f"STRAT-{trial_id}",
        dataset_version=dataset_version,
        research_protocol_version=protocol_version,
        artifact=art,
        net_returns=ret_arr,
        signal_timestamps=ts,
        distribution=dist,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# LAYER A: EICT-CORR-1 Tests
# ---------------------------------------------------------------------------


class TestEictCorr1Calculator:
    def test_empty_population(self) -> None:
        calc = EictCorr1Calculator()
        res = calc.calculate([])
        assert res.raw_trial_count == 0
        assert res.effective_trial_count == 0
        assert res.clusters == {}
        assert res.singleton_clusters == ()

    def test_single_trial_population(self) -> None:
        t1 = _make_mock_trial("T1", [1.0, 2.0, 3.0] * 40)
        calc = EictCorr1Calculator()
        res = calc.calculate([t1])
        assert res.raw_trial_count == 1
        assert res.effective_trial_count == 1
        assert res.clusters == {1: ("T1",)}
        assert res.cluster_memberships == {"T1": 1}

    def test_two_highly_correlated_trials_merge(self) -> None:
        n = 120
        rng = np.random.default_rng(42)
        r_base = rng.normal(1.0, 2.0, n)
        r1 = r_base + rng.normal(0, 0.05, n)
        r2 = r_base + rng.normal(0, 0.05, n)

        t1 = _make_mock_trial("T1", r1)
        t2 = _make_mock_trial("T2", r2)

        calc = EictCorr1Calculator()
        res = calc.calculate([t1, t2])
        assert res.raw_trial_count == 2
        assert res.effective_trial_count == 1
        assert len(res.clusters) == 1
        assert set(res.clusters[1]) == {"T1", "T2"}
        assert res.cluster_memberships["T1"] == res.cluster_memberships["T2"]

    def test_two_independent_trials_separate(self) -> None:
        n = 120
        rng = np.random.default_rng(42)
        r1 = rng.normal(1.0, 2.0, n)
        r2 = rng.normal(-0.5, 1.5, n)

        t1 = _make_mock_trial("T1", r1)
        t2 = _make_mock_trial("T2", r2)

        calc = EictCorr1Calculator()
        res = calc.calculate([t1, t2])
        assert res.raw_trial_count == 2
        assert res.effective_trial_count == 2
        assert len(res.clusters) == 2
        assert res.cluster_memberships["T1"] != res.cluster_memberships["T2"]

    def test_known_cluster_structure_with_singletons(self) -> None:
        """Synthetic structure:
        Cluster 1: A1, A2, A3 (high corr)
        Cluster 2: B1, B2 (high corr)
        Singleton 1: S1 (< 100 observations)
        Singleton 2: S2 (< 100 observations)
        Total raw = 7, total effective = 2 + 2 = 4."""
        n = 120
        rng = np.random.default_rng(101)

        base_a = rng.normal(0, 1, n)
        a1 = base_a + rng.normal(0, 0.02, n)
        a2 = base_a + rng.normal(0, 0.02, n)
        a3 = base_a + rng.normal(0, 0.02, n)

        base_b = rng.normal(0, 1, n)
        b1 = base_b + rng.normal(0, 0.02, n)
        b2 = base_b + rng.normal(0, 0.02, n)

        # Singletons with only 40 observations
        s1 = rng.normal(0, 1, 40)
        s2 = rng.normal(0, 1, 40)

        trials = [
            _make_mock_trial("A1", a1),
            _make_mock_trial("A2", a2),
            _make_mock_trial("A3", a3),
            _make_mock_trial("B1", b1),
            _make_mock_trial("B2", b2),
            _make_mock_trial("S1", s1),
            _make_mock_trial("S2", s2),
        ]

        calc = EictCorr1Calculator()
        res = calc.calculate(trials)

        assert res.raw_trial_count == 7
        assert res.effective_trial_count == 4
        assert len(res.clusters) == 4
        assert set(res.singleton_clusters) == {"S1", "S2"}

        # Check A1, A2, A3 share same cluster
        cid_a = res.cluster_memberships["A1"]
        assert res.cluster_memberships["A2"] == cid_a
        assert res.cluster_memberships["A3"] == cid_a

        # Check B1, B2 share same cluster
        cid_b = res.cluster_memberships["B1"]
        assert res.cluster_memberships["B2"] == cid_b
        assert cid_a != cid_b

        # S1 and S2 have distinct singleton clusters
        assert res.cluster_memberships["S1"] != res.cluster_memberships["S2"]
        assert res.cluster_memberships["S1"] not in (cid_a, cid_b)

    def test_population_hash_reproducibility(self) -> None:
        trials = [
            _make_mock_trial(f"T{i}", [float(i)] * 110)
            for i in range(5)
        ]
        h1 = compute_population_hash(trials)
        h2 = compute_population_hash(list(reversed(trials)))
        assert h1 == h2
        assert len(h1) == 64


# ---------------------------------------------------------------------------
# LAYER B: Deflated Sharpe Ratio Tests
# ---------------------------------------------------------------------------


class TestDeflatedSharpeCalculator:
    def test_reference_values_n_equals_1_psr_equivalence(self) -> None:
        """When N = 1 (effective_trial_count = 1), E[max] = 0.
        DSR simplifies exactly to Probabilistic Sharpe Ratio (PSR).

        Reference hand-calculation:
        T = 100, observed_sharpe = 0.5, normal returns (skew=0, kurt_excess=0).
        SE = sqrt((1 + 0.5 * 0.5^2) / 99) = sqrt(1.125 / 99) = 0.106600358
        z = 0.5 / 0.106600358 = 4.69041576
        DSR = norm.cdf(4.69041576) = 0.9999986
        """
        # Create normal returns with sample size 100
        n = 100
        rng = np.random.default_rng(42)
        raw_ret = rng.normal(0.5, 1.0, n)
        t1 = _make_mock_trial("T1", raw_ret)

        calc_eict = EictCorr1Calculator()
        eict_res = calc_eict.calculate([t1])

        calc_dsr = DeflatedSharpeCalculator([t1], eict_res)
        res = calc_dsr.calculate("T1")

        assert res.effective_trial_count == 1
        assert res.expected_max_sharpe == 0.0
        assert res.sharpe_std_population == 0.0

        # Theoretical SE using trial's actual observed moments
        sr = res.observed_sharpe
        var_sr = (1.0 - res.skewness * sr + ((res.raw_kurtosis - 1.0) / 4.0) * (sr ** 2)) / (n - 1)
        expected_se = math.sqrt(var_sr)
        assert abs(res.sharpe_standard_error - expected_se) < 1e-10

        expected_z = sr / expected_se
        assert abs(res.z_stat - expected_z) < 1e-10
        assert abs(res.dsr - stats.norm.cdf(expected_z)) < 1e-10

    def test_deflation_with_multiple_trials_n_gt_1(self) -> None:
        """When effective_trial_count > 1 and population has dispersion,
        E[max] > 0 and DSR is strictly less than single-trial PSR."""
        rng = np.random.default_rng(99)
        trials = []
        for i in range(10):
            # Vary mean returns to create variance in observed Sharpe
            mu = (i - 5) * 0.2
            r = rng.normal(mu, 1.0, 120)
            trials.append(_make_mock_trial(f"T{i}", r))

        calc_eict = EictCorr1Calculator()
        eict_res = calc_eict.calculate(trials)

        calc_dsr = DeflatedSharpeCalculator(trials, eict_res)

        # Pick trial with highest observed Sharpe
        best_trial = max(trials, key=lambda t: t.distribution.observed_sharpe)
        res_best = calc_dsr.calculate(best_trial.trial_id)

        assert res_best.effective_trial_count > 1
        assert res_best.expected_max_sharpe > 0.0
        assert res_best.sharpe_std_population > 0.0

        # Without multiplicity correction (PSR), z would be observed_sharpe / SE
        psr_z = res_best.observed_sharpe / res_best.sharpe_standard_error
        psr = stats.norm.cdf(psr_z)

        # DSR must be strictly deflated relative to PSR
        assert res_best.z_stat < psr_z
        assert res_best.dsr < psr

    def test_fat_tails_increase_standard_error(self) -> None:
        """High kurtosis (heavy tails) increases standard error, reducing z."""
        n = 200
        rng = np.random.default_rng(12)
        # Normal returns
        r_normal = rng.normal(0.5, 1.0, n)
        t_normal = _make_mock_trial("T_NORM", r_normal)

        # Leptokurtic returns with same mean and std but extreme outliers
        r_fat = r_normal.copy()
        r_fat[0] += 15.0
        r_fat[1] -= 15.0
        t_fat = _make_mock_trial("T_FAT", r_fat)

        calc_eict = EictCorr1Calculator()
        eict_norm = calc_eict.calculate([t_normal])
        eict_fat = calc_eict.calculate([t_fat])

        res_norm = DeflatedSharpeCalculator([t_normal], eict_norm).calculate("T_NORM")
        res_fat = DeflatedSharpeCalculator([t_fat], eict_fat).calculate("T_FAT")

        assert res_fat.kurtosis_excess > res_norm.kurtosis_excess
        # Standard error should be larger for the heavy-tailed distribution
        assert res_fat.sharpe_standard_error > res_norm.sharpe_standard_error


# ---------------------------------------------------------------------------
# MULTIPLE-TESTING GUARDS & NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestMultipleTestingGuardsAndNegativeTests:
    def test_manual_override_of_trial_count_raises(self) -> None:
        t1 = _make_mock_trial("T1", [1.0] * 110)
        eict_res = EictCorr1Calculator().calculate([t1])
        calc = DeflatedSharpeCalculator([t1], eict_res)

        with pytest.raises(ValueError, match="manual overrides"):
            calc.calculate("T1", manual_trials=1)

    def test_manual_override_of_observed_sharpe_raises(self) -> None:
        t1 = _make_mock_trial("T1", [1.0] * 110)
        eict_res = EictCorr1Calculator().calculate([t1])
        calc = DeflatedSharpeCalculator([t1], eict_res)

        with pytest.raises(ValueError, match="manual overrides"):
            calc.calculate("T1", manual_sharpe=3.5)

    def test_trial_not_in_population_raises(self) -> None:
        t1 = _make_mock_trial("T1", [1.0] * 110)
        eict_res = EictCorr1Calculator().calculate([t1])
        calc = DeflatedSharpeCalculator([t1], eict_res)

        with pytest.raises(KeyError, match="not an eligible trial in population"):
            calc.calculate("T_UNKNOWN")

    def test_mismatched_eict_method_id_raises(self) -> None:
        with pytest.raises(ValueError, match="method_id must be 'EICT-CORR-1'"):
            EictCorr1Calculator(method_id="ARBITRARY_METHOD")

    def test_tampered_distance_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="distance_threshold"):
            EictCorr1Calculator(distance_threshold=0.50)

    def test_tampered_linkage_raises(self) -> None:
        with pytest.raises(ValueError, match="linkage"):
            EictCorr1Calculator(linkage="complete")

    def test_empty_population_calculator_raises(self) -> None:
        eict_empty = EictCorr1Calculator().calculate([])
        with pytest.raises(ValueError, match="requires a non-empty population"):
            DeflatedSharpeCalculator([], eict_empty)


# ---------------------------------------------------------------------------
# Integrated Pipeline End-to-End Test
# ---------------------------------------------------------------------------


class TestStatisticalValidationPipeline:
    def test_end_to_end_pipeline_evaluation(self, tmp_path: Path) -> None:
        """Integration: build 5 production trials, run pipeline evaluation."""
        ledger = TrialLedger(tmp_path / "ledger.db")
        art_reg = ArtifactRegistry(tmp_path / "artifacts.db")
        budget = ResearchBudget(max_trials=100)

        rng = np.random.default_rng(77)
        for i in range(5):
            tid = f"T-PIPE-{i}"
            ctx = TrialContext(
                trial_id=tid,
                experiment_id="EXP-PIPE",
                strategy_id=f"STRAT-{i}",
                dataset_version="DS-PIPE-V1",
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

            returns = rng.normal(float(i) * 0.5, 2.0, 120).tolist()
            res = BacktestResult(
                execution_model="next_bar_open_v1",
                trades=tuple(
                    BacktestTrade(
                        signal_timestamp=pd.Timestamp("2023-01-03") + pd.Timedelta(hours=h),
                        entry_timestamp=pd.Timestamp("2023-01-03") + pd.Timedelta(hours=h, minutes=1),
                        exit_timestamp=pd.Timestamp("2023-01-03") + pd.Timedelta(hours=h, minutes=2),
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
                    for h, r in enumerate(returns)
                ),
                gross_pnl=sum(returns),
                costs=0.0,
                net_pnl=sum(returns),
                mean_gross_return_bps=float(np.mean(returns)),
                mean_net_return_bps=float(np.mean(returns)),
            )
            path, sha, rows, start, end = write_oos_returns_artifact(
                res, trial_id=tid, dataset_version="DS-PIPE-V1", artifact_dir=tmp_path
            )
            art_reg.record(
                trial_id=tid,
                artifact_type=ArtifactType.OOS_RETURNS,
                dataset_version="DS-PIPE-V1",
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

        query = ProductionPopulationQuery(ledger, art_reg)
        pipeline = StatisticalValidationPipeline(query)

        eict_res, dsr_res = pipeline.evaluate_trial(
            "T-PIPE-4",
            dataset_version="DS-PIPE-V1",
            research_protocol_version="RP-2",
        )

        assert eict_res.raw_trial_count == 5
        assert eict_res.effective_trial_count >= 1
        assert eict_res.method_id == "EICT-CORR-1"
        assert dsr_res.trial_id == "T-PIPE-4"
        assert dsr_res.effective_trial_count == eict_res.effective_trial_count
        assert 0.0 <= dsr_res.dsr <= 1.0
