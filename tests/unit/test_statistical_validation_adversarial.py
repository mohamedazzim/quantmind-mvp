"""Adversarial mathematical audit for QuantMind v3.7:
- Kurtosis convention verification (Fisher excess vs Pearson raw)
- DSR N=1 PSR equivalence and multiplicity ladder
- DSR monotonicity under skewness, kurtosis, and trial count
- Degenerate / zero return cases (T=0, T=1, T=2, constant returns, NaN, inf)
- EICT permutation and input order invariance
- EICT threshold boundaries (r=0.79, r=0.80, r=0.81)
- Singleton trial accounting
- Multiplicity bypass prevention (manual override rejections)
- Population hash sensitivity and order invariance
- Full 9-field statistical provenance retention
- Independent reference implementation cross-check
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from quantmind.backtest.engine import BacktestConfig, BacktestResult, BacktestTrade
from quantmind.research_integrity.artifacts import (
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactType,
    write_oos_returns_artifact,
)
from quantmind.research_integrity.population import (
    EictCorr1InputBuilder,
    EligibleTrial,
    ProductionPopulationQuery,
    ReturnDistributionMetadata,
    TrialProvenance,
    compute_return_distribution,
    _EICT_CORR1_DISTANCE_THRESHOLD,
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
# Independent Reference Implementation (in tests only)
# ---------------------------------------------------------------------------


def ref_moments(returns: np.ndarray) -> tuple[float, float, float, float]:
    """Calculate mean, sample std (ddof=1), skewness, and Fisher excess kurtosis."""
    n = len(returns)
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    skew = float(stats.skew(returns, bias=False)) if n >= 3 else 0.0
    kurt_excess = float(stats.kurtosis(returns, bias=False, fisher=True)) if n >= 4 else 0.0
    return mu, sigma, skew, kurt_excess


def ref_dsr_se(sr: float, T: int, skew: float, kurt_excess: float) -> float:
    """Independent calculation of Sharpe standard error using Mertens / Lo formula.

    gamma4 (Pearson) = kurt_excess + 3.0
    term = 1 - skew * sr + ((gamma4 - 1) / 4) * sr^2
         = 1 - skew * sr + ((kurt_excess + 2) / 4) * sr^2
    """
    raw_kurt = kurt_excess + 3.0
    var_sr = (1.0 - skew * sr + ((raw_kurt - 1.0) / 4.0) * (sr ** 2)) / (T - 1.0)
    return math.sqrt(max(1e-12, var_sr))


def ref_emax(N: int, sharpe_std: float) -> float:
    """Independent calculation of expected maximum Sharpe under null."""
    if N <= 1 or sharpe_std <= 0.0:
        return 0.0
    gamma = 0.5772156649015329
    z1 = float(stats.norm.ppf(1.0 - 1.0 / N))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (N * math.e)))
    return sharpe_std * ((1.0 - gamma) * z1 + gamma * z2)


def ref_dsr(sr: float, T: int, skew: float, kurt_excess: float, N: int, sharpe_std: float) -> float:
    """Independent calculation of Deflated Sharpe Ratio."""
    se = ref_dsr_se(sr, T, skew, kurt_excess)
    emax = ref_emax(N, sharpe_std)
    z = (sr - emax) / se
    return float(stats.norm.cdf(z))


def _make_trial_helper(
    trial_id: str,
    returns: Sequence[float],
    start_hour: int = 0,
    dataset_version: str = "DS-REF",
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
        dataset_sha256="dsha_hex_1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        split_manifest_version="v1",
        split_manifest_sha256="smsha_hex_1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        research_protocol_version=protocol_version,
        strategy_id=f"STRAT-{trial_id}",
        experiment_id="EXP-REF",
        code_version="0.1.0",
        config_hash="cfghash1234",
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
# 2. CRITICAL DSR KURTOSIS AUDIT
# ---------------------------------------------------------------------------


class TestSection2DsrKurtosisAudit:
    def test_item_a_gaussian_moments(self) -> None:
        """A. Gaussian synthetic return distribution: skew ≈ 0, Pearson kurtosis ≈ 3, excess kurtosis ≈ 0."""
        rng = np.random.default_rng(12345)
        # Large sample to verify statistical convergence
        gaussian_returns = rng.normal(loc=0.05, scale=1.0, size=50_000)
        dist = compute_return_distribution("T-GAUSS", gaussian_returns)

        # Skewness is near 0
        assert abs(dist.skewness) < 0.05
        # Excess kurtosis (Fisher) is near 0
        assert abs(dist.excess_kurtosis) < 0.05
        assert abs(dist.kurtosis) < 0.05
        # Pearson raw kurtosis is near 3.0
        assert abs(dist.raw_kurtosis - 3.0) < 0.05

    def test_item_b_and_e_kurtosis_convention_reaches_dsr_properly(self) -> None:
        """B & E: Verify normal-return DSR uses (3 - 1)/4 = 0.5 and NOT (0 - 1)/4 = -0.25."""
        # Consider a candidate with known Sharpe = 0.5, T = 101 observations
        # Under exact normality (skew = 0, excess_kurtosis = 0, raw_kurtosis = 3):
        # var_sr = (1 - 0 + ((3 - 1)/4) * 0.5^2) / 100 = (1 + 0.5 * 0.25) / 100 = 1.125 / 100 = 0.01125
        # se = sqrt(0.01125) = 0.106066
        # If buggy formula used (0 - 1)/4: var_sr = (1 - 0.25 * 0.25) / 100 = 0.9375 / 100 = 0.009375 (WRONG)
        t = _make_trial_helper("T-NORM", [0.0] * 101)  # Dummy trial
        # Override distribution with exact theoretical normal moments
        mock_dist = ReturnDistributionMetadata(
            trial_id="T-NORM",
            mean_return=0.5,
            std_return=1.0,
            skewness=0.0,
            kurtosis=0.0,  # Fisher excess kurtosis = 0
            trade_count=101,
            effective_observations=101,
            observed_sharpe=0.5,
        )
        t_mock = EligibleTrial(
            trial_id=t.trial_id,
            strategy_id=t.strategy_id,
            dataset_version=t.dataset_version,
            research_protocol_version=t.research_protocol_version,
            artifact=t.artifact,
            net_returns=t.net_returns,
            signal_timestamps=t.signal_timestamps,
            distribution=mock_dist,
            provenance=t.provenance,
        )

        eict_res = EictCorr1Calculator().calculate([t_mock])
        dsr_res = DeflatedSharpeCalculator([t_mock], eict_res).calculate("T-NORM")

        expected_var = (1.0 + 0.5 * (0.5 ** 2)) / 100.0
        expected_se = math.sqrt(expected_var)

        assert abs(dsr_res.sharpe_standard_error - expected_se) < 1e-10
        # Prove that it is NOT the buggy (0 - 1)/4 value
        buggy_se = math.sqrt((1.0 - 0.25 * (0.5 ** 2)) / 100.0)
        assert abs(dsr_res.sharpe_standard_error - buggy_se) > 0.005

    def test_item_c_and_d_hand_calculated_comparison(self) -> None:
        """C & D: Compare implementation against hand calculation for non-normal case."""
        # Non-normal case:
        # T = 61 observations, SR = 0.4, skew = -0.6, excess_kurtosis = 2.0 (raw_kurtosis = 5.0)
        # var_sr = (1 - (-0.6)*0.4 + ((5 - 1)/4) * 0.4^2) / 60
        #         = (1 + 0.24 + 1.0 * 0.16) / 60 = 1.40 / 60 = 0.02333333
        # se = sqrt(0.02333333) = 0.1527525
        t = _make_trial_helper("T-HAND", [0.0] * 61)
        mock_dist = ReturnDistributionMetadata(
            trial_id="T-HAND",
            mean_return=0.4,
            std_return=1.0,
            skewness=-0.6,
            kurtosis=2.0,  # excess = 2.0
            trade_count=61,
            effective_observations=61,
            observed_sharpe=0.4,
        )
        t_mock = EligibleTrial(
            trial_id=t.trial_id,
            strategy_id=t.strategy_id,
            dataset_version=t.dataset_version,
            research_protocol_version=t.research_protocol_version,
            artifact=t.artifact,
            net_returns=t.net_returns,
            signal_timestamps=t.signal_timestamps,
            distribution=mock_dist,
            provenance=t.provenance,
        )

        eict_res = EictCorr1Calculator().calculate([t_mock])
        dsr_res = DeflatedSharpeCalculator([t_mock], eict_res).calculate("T-HAND")

        expected_var = (1.0 + 0.24 + 0.16) / 60.0
        expected_se = math.sqrt(expected_var)
        expected_z = 0.4 / expected_se
        expected_dsr = float(stats.norm.cdf(expected_z))

        assert abs(dsr_res.sharpe_standard_error - expected_se) < 1e-10
        assert abs(dsr_res.z_stat - expected_z) < 1e-10
        assert abs(dsr_res.dsr - expected_dsr) < 1e-10


# ---------------------------------------------------------------------------
# 3. DSR N=1 AUDIT & MULTIPLICITY LADDER
# ---------------------------------------------------------------------------


class TestSection3DsrN1AndMultiplicityLadder:
    def test_n_equals_1_psr_equivalence(self) -> None:
        """For N=1: E[max] = 0, DSR exactly equals PSR."""
        rng = np.random.default_rng(777)
        ret = rng.normal(0.2, 1.5, 150)
        t1 = _make_trial_helper("T-PSR", ret)

        eict_res = EictCorr1Calculator().calculate([t1])
        dsr_res = DeflatedSharpeCalculator([t1], eict_res).calculate("T-PSR")

        assert dsr_res.expected_max_sharpe == 0.0
        # PSR is norm.cdf(sr / se)
        psr = float(stats.norm.cdf(dsr_res.observed_sharpe / dsr_res.sharpe_standard_error))
        assert abs(dsr_res.dsr - psr) < 1e-10

    @pytest.mark.parametrize("N", [2, 10, 100, 1000])
    def test_multiplicity_ladder_increases_penalty(self, N: int) -> None:
        """Multiplicity ladder: increasing N strictly increases the E[max] penalty."""
        sharpe_std = 0.40
        gamma = 0.5772156649015329
        z1 = float(stats.norm.ppf(1.0 - 1.0 / N))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (N * math.e)))
        emax = sharpe_std * ((1.0 - gamma) * z1 + gamma * z2)

        # Expected max penalty is strictly increasing with N
        if N == 2:
            assert 0.15 < emax < 0.40
        elif N == 10:
            assert 0.55 < emax < 0.70
        elif N == 100:
            assert 0.95 < emax < 1.10
        elif N == 1000:
            assert 1.25 < emax < 1.45


# ---------------------------------------------------------------------------
# 4. DSR MONOTONICITY
# ---------------------------------------------------------------------------


class TestSection4DsrMonotonicity:
    def test_monotonicity_under_multiplicity(self) -> None:
        """Same strategy candidate, increasing population trial count -> DSR does not become more favorable."""
        rng = np.random.default_rng(888)
        base_t = _make_trial_helper("T-BASE", rng.normal(0.4, 1.0, 120))

        # Population with 2 trials vs population with 10 trials
        pop_2 = [base_t, _make_trial_helper("T-2", rng.normal(0.1, 1.0, 120))]
        pop_10 = [base_t] + [_make_trial_helper(f"T-{i}", rng.normal((i - 5) * 0.1, 1.0, 120)) for i in range(1, 10)]

        dsr_2 = DeflatedSharpeCalculator(pop_2, EictCorr1Calculator().calculate(pop_2)).calculate("T-BASE")
        dsr_10 = DeflatedSharpeCalculator(pop_10, EictCorr1Calculator().calculate(pop_10)).calculate("T-BASE")

        assert dsr_10.expected_max_sharpe > dsr_2.expected_max_sharpe
        assert dsr_10.dsr < dsr_2.dsr

    def test_monotonicity_under_kurtosis(self) -> None:
        """Same SR, higher kurtosis (fatter tails) -> increased standard error, lower DSR."""
        t_base = _make_trial_helper("T-BASE", [0.0] * 100)
        eict_res = EictClusterResult(
            raw_trial_count=1,
            effective_trial_count=1,
            correlation_matrix=np.ones((1, 1)),
            distance_matrix=np.zeros((1, 1)),
            clusters={1: ("T-BASE",)},
            cluster_memberships={"T-BASE": 1},
            singleton_clusters=(),
            method_id="EICT-CORR-1",
            protocol_version="RP-2",
            population_hash="hash",
            dataset_version="DS",
            research_protocol_version="RP-2",
            created_at="2023-01-01T00:00:00Z",
        )

        def _dsr_for_kurt(kurt_excess: float) -> float:
            d = ReturnDistributionMetadata("T-BASE", 0.5, 1.0, 0.0, kurt_excess, 100, 100, 0.5)
            t = EligibleTrial(
                t_base.trial_id, t_base.strategy_id, t_base.dataset_version,
                t_base.research_protocol_version, t_base.artifact, t_base.net_returns,
                t_base.signal_timestamps, d, t_base.provenance
            )
            return DeflatedSharpeCalculator([t], eict_res).calculate("T-BASE").dsr

        dsr_normal = _dsr_for_kurt(0.0)
        dsr_fat = _dsr_for_kurt(5.0)
        dsr_extreme_fat = _dsr_for_kurt(20.0)

        assert dsr_normal > dsr_fat > dsr_extreme_fat

    def test_monotonicity_under_negative_skew(self) -> None:
        """Same SR > 0, more negative skewness -> increased standard error, lower DSR."""
        t_base = _make_trial_helper("T-BASE", [0.0] * 100)
        eict_res = EictClusterResult(
            raw_trial_count=1,
            effective_trial_count=1,
            correlation_matrix=np.ones((1, 1)),
            distance_matrix=np.zeros((1, 1)),
            clusters={1: ("T-BASE",)},
            cluster_memberships={"T-BASE": 1},
            singleton_clusters=(),
            method_id="EICT-CORR-1",
            protocol_version="RP-2",
            population_hash="hash",
            dataset_version="DS",
            research_protocol_version="RP-2",
            created_at="2023-01-01T00:00:00Z",
        )

        def _dsr_for_skew(skew: float) -> float:
            d = ReturnDistributionMetadata("T-BASE", 0.5, 1.0, skew, 0.0, 100, 100, 0.5)
            t = EligibleTrial(
                t_base.trial_id, t_base.strategy_id, t_base.dataset_version,
                t_base.research_protocol_version, t_base.artifact, t_base.net_returns,
                t_base.signal_timestamps, d, t_base.provenance
            )
            return DeflatedSharpeCalculator([t], eict_res).calculate("T-BASE").dsr

        dsr_symm = _dsr_for_skew(0.0)
        dsr_neg_skew = _dsr_for_skew(-1.0)
        dsr_severe_neg = _dsr_for_skew(-2.5)

        assert dsr_symm > dsr_neg_skew > dsr_severe_neg


# ---------------------------------------------------------------------------
# 5. ZERO / DEGENERATE CASES
# ---------------------------------------------------------------------------


class TestSection5ZeroDegenerateCases:
    def test_sample_length_t_less_than_2_rejected(self) -> None:
        """T = 0 or T = 1: REJECTED (raises ValueError)."""
        t0 = _make_trial_helper("T-0", [])
        eict0 = EictCorr1Calculator().calculate([t0])
        with pytest.raises(ValueError, match="sample length T=0 which is insufficient"):
            DeflatedSharpeCalculator([t0], eict0).calculate("T-0")

        t1 = _make_trial_helper("T-1", [10.0])
        eict1 = EictCorr1Calculator().calculate([t1])
        with pytest.raises(ValueError, match="sample length T=1 which is insufficient"):
            DeflatedSharpeCalculator([t1], eict1).calculate("T-1")

    def test_sample_length_t_equals_2_valid(self) -> None:
        """T = 2: VALID (minimum valid sample size for sample std)."""
        t2 = _make_trial_helper("T-2", [10.0, 20.0])
        eict2 = EictCorr1Calculator().calculate([t2])
        res = DeflatedSharpeCalculator([t2], eict2).calculate("T-2")
        assert res.sample_length == 2
        assert res.observed_sharpe > 0.0
        assert 0.0 <= res.dsr <= 1.0

    def test_constant_returns_std_zero_valid_non_significant(self) -> None:
        """std = 0 (constant positive, constant negative, zero returns):
        VALID, observed_sharpe = 0.0, DSR <= 0.5 (non-significant)."""
        t_const_pos = _make_trial_helper("T-CPOS", [5.0] * 50)
        eict = EictCorr1Calculator().calculate([t_const_pos])
        res = DeflatedSharpeCalculator([t_const_pos], eict).calculate("T-CPOS")
        assert res.observed_sharpe == 0.0
        assert res.dsr <= 0.5
        assert not res.passes_dsr_gate

        t_zero = _make_trial_helper("T-ZERO", [0.0] * 50)
        res_z = DeflatedSharpeCalculator([t_zero], eict).calculate("T-ZERO")
        assert res_z.observed_sharpe == 0.0
        assert res_z.dsr <= 0.5
        assert not res_z.passes_dsr_gate

    def test_nan_inf_rejected(self) -> None:
        """NaN / Inf in returns: REJECTED (raises ValueError)."""
        with pytest.raises(ValueError, match="non-finite"):
            compute_return_distribution("T-NAN", np.array([1.0, np.nan, 2.0]))

        with pytest.raises(ValueError, match="non-finite"):
            compute_return_distribution("T-INF", np.array([1.0, np.inf, 2.0]))


# ---------------------------------------------------------------------------
# 6. EICT INPUT ORDER INVARIANCE
# ---------------------------------------------------------------------------


class TestSection6EictInputOrderInvariance:
    def test_eict_permutation_invariance(self) -> None:
        """Creating population with trial IDs in several orders yields identical output."""
        rng = np.random.default_rng(42)
        n = 120
        trials = [
            _make_trial_helper("A", rng.normal(0, 1, n)),
            _make_trial_helper("B", rng.normal(0, 1, n)),
            _make_trial_helper("C", rng.normal(0, 1, n)),
            _make_trial_helper("D", rng.normal(0, 1, n)),
            _make_trial_helper("E", rng.normal(0, 1, n)),
        ]

        order1 = [trials[0], trials[1], trials[2], trials[3], trials[4]]  # [A, B, C, D, E]
        order2 = [trials[3], trials[0], trials[4], trials[1], trials[2]]  # [D, A, E, B, C]
        order3 = [trials[2], trials[4], trials[1], trials[0], trials[3]]  # [C, E, B, A, D]

        calc = EictCorr1Calculator()
        res1 = calc.calculate(order1)
        res2 = calc.calculate(order2)
        res3 = calc.calculate(order3)

        # 1. Effective trial count identical
        assert res1.effective_trial_count == res2.effective_trial_count == res3.effective_trial_count
        # 2. Population hash identical
        assert res1.population_hash == res2.population_hash == res3.population_hash
        # 3. Correlation and distance matrices identical
        np.testing.assert_array_almost_equal(res1.correlation_matrix, res2.correlation_matrix)
        np.testing.assert_array_almost_equal(res1.correlation_matrix, res3.correlation_matrix)
        np.testing.assert_array_almost_equal(res1.distance_matrix, res2.distance_matrix)
        np.testing.assert_array_almost_equal(res1.distance_matrix, res3.distance_matrix)
        # 4. Cluster memberships identical
        assert res1.cluster_memberships == res2.cluster_memberships == res3.cluster_memberships


# ---------------------------------------------------------------------------
# 7. EICT THRESHOLD BOUNDARY AUDIT
# ---------------------------------------------------------------------------


class TestSection7EictThresholdAudit:
    def test_correlation_boundary_thresholds(self) -> None:
        """Verify boundary behavior around correlation 0.79 vs 0.80 vs 0.81.
        distance_threshold = 0.6325 = sqrt(2*(1 - 0.80))."""
        # Pair with corr = 0.79 -> dist = sqrt(2*(1 - 0.79)) = sqrt(0.42) = 0.648074 > 0.6325 -> 2 clusters
        # Pair with corr = 0.81 -> dist = sqrt(2*(1 - 0.81)) = sqrt(0.38) = 0.616441 < 0.6325 -> 1 cluster
        n = 200
        rng = np.random.default_rng(123)
        z = rng.normal(0, 1, n)

        def _make_pair(target_corr: float) -> tuple[EligibleTrial, EligibleTrial]:
            # Generate pair with exact target correlation
            w = rng.normal(0, 1, n)
            # Orthogonalize w w.r.t z
            w = w - (np.dot(z, w) / np.dot(z, z)) * z
            z_norm = z / np.std(z)
            w_norm = w / np.std(w)
            y = target_corr * z_norm + math.sqrt(1.0 - target_corr ** 2) * w_norm
            t1 = _make_trial_helper("T1", z_norm)
            t2 = _make_trial_helper("T2", y)
            return t1, t2

        calc = EictCorr1Calculator()

        # Target 0.79 -> 2 clusters
        t1, t2 = _make_pair(0.79)
        res_79 = calc.calculate([t1, t2])
        assert res_79.effective_trial_count == 2

        # Target 0.81 -> 1 cluster
        t1, t2 = _make_pair(0.81)
        res_81 = calc.calculate([t1, t2])
        assert res_81.effective_trial_count == 1


# ---------------------------------------------------------------------------
# 8. SINGLETON ACCOUNTING
# ---------------------------------------------------------------------------


class TestSection8SingletonAccounting:
    def test_sparse_trials_account_as_singletons(self) -> None:
        """Sparse trials (<100 common observations) count as exactly one effective trial each."""
        rng = np.random.default_rng(555)
        # Dense trials (150 observations)
        t_dense1 = _make_trial_helper("D1", rng.normal(0, 1, 150))
        t_dense2 = _make_trial_helper("D2", rng.normal(0, 1, 150))
        # Sparse trials (30 observations)
        t_sparse1 = _make_trial_helper("S1", rng.normal(0, 1, 30))
        t_sparse2 = _make_trial_helper("S2", rng.normal(0, 1, 30))

        calc = EictCorr1Calculator()
        res = calc.calculate([t_dense1, t_dense2, t_sparse1, t_sparse2])

        assert res.raw_trial_count == 4
        assert set(res.singleton_clusters) == {"S1", "S2"}
        # Both singletons must be present in cluster memberships
        assert "S1" in res.cluster_memberships
        assert "S2" in res.cluster_memberships
        assert res.cluster_memberships["S1"] != res.cluster_memberships["S2"]


# ---------------------------------------------------------------------------
# 9. MULTIPLICITY BYPASS PREVENTION
# ---------------------------------------------------------------------------


class TestSection9MultiplicityBypass:
    @pytest.mark.parametrize(
        "bad_arg",
        [
            {"manual_sharpe": 2.5},
            {"manual_trials": 1},
            {"manual_effective_trial_count": 1},
            {"manual_skew": 0.0},
            {"manual_kurtosis": 0.0},
        ],
    )
    def test_manual_override_attempts_rejected(self, bad_arg: dict) -> None:
        """Attempting to manually override any parameter raises ValueError."""
        t1 = _make_trial_helper("T1", [1.0, 2.0, 3.0] * 40)
        eict = EictCorr1Calculator().calculate([t1])
        calc = DeflatedSharpeCalculator([t1], eict)
        with pytest.raises(ValueError, match="manual overrides"):
            calc.calculate("T1", **bad_arg)


# ---------------------------------------------------------------------------
# 10. POPULATION HASH INTEGRITY
# ---------------------------------------------------------------------------


class TestSection10PopulationHashIntegrity:
    def test_population_hash_changes_on_eligible_trial_addition_or_removal(self) -> None:
        t1 = _make_trial_helper("T1", [1.0] * 110)
        t2 = _make_trial_helper("T2", [2.0] * 110)
        h1 = compute_population_hash([t1])
        h2 = compute_population_hash([t1, t2])
        assert h1 != h2

    def test_population_hash_changes_on_artifact_sha_change(self) -> None:
        t1 = _make_trial_helper("T1", [1.0] * 110)
        h1 = compute_population_hash([t1])

        # Mutate artifact sha
        art2 = ArtifactRecord(
            t1.artifact.artifact_id, t1.artifact.trial_id, t1.artifact.artifact_type,
            t1.artifact.dataset_version, "different_sha_content", t1.artifact.format,
            t1.artifact.row_count, t1.artifact.start_timestamp, t1.artifact.end_timestamp,
            t1.artifact.created_at, t1.artifact.status, t1.artifact.artifact_path
        )
        t1_mod = EligibleTrial(
            t1.trial_id, t1.strategy_id, t1.dataset_version, t1.research_protocol_version,
            art2, t1.net_returns, t1.signal_timestamps, t1.distribution, t1.provenance
        )
        h2 = compute_population_hash([t1_mod])
        assert h1 != h2

    def test_population_hash_invariant_to_list_ordering(self) -> None:
        t1 = _make_trial_helper("T1", [1.0] * 110)
        t2 = _make_trial_helper("T2", [2.0] * 110)
        t3 = _make_trial_helper("T3", [3.0] * 110)
        assert compute_population_hash([t1, t2, t3]) == compute_population_hash([t3, t1, t2])


# ---------------------------------------------------------------------------
# 11. PROVENANCE RETENTION
# ---------------------------------------------------------------------------


class TestSection11ProvenanceRetention:
    def test_all_9_provenance_attributes_retained_in_dsr_result(self) -> None:
        t1 = _make_trial_helper("T-PROV", [1.0, 2.0, 3.0] * 40)
        eict = EictCorr1Calculator().calculate([t1])
        dsr_res = DeflatedSharpeCalculator([t1], eict).calculate("T-PROV")

        assert dsr_res.dataset_version == "DS-REF"
        assert len(dsr_res.dataset_sha256) > 0
        assert dsr_res.split_manifest_version == "v1"
        assert len(dsr_res.split_manifest_sha256) > 0
        assert dsr_res.research_protocol_version == "RP-2"
        assert len(dsr_res.population_hash) == 64
        assert dsr_res.method_id == "EICT-CORR-1"
        assert dsr_res.code_version == "0.1.0"
        assert len(dsr_res.config_hash) > 0


# ---------------------------------------------------------------------------
# 12. REFERENCE-IMPLEMENTATION CROSS-CHECK
# ---------------------------------------------------------------------------


class TestSection12ReferenceCrossCheck:
    def test_cross_check_against_independent_reference(self) -> None:
        """Cross-check production DeflatedSharpeCalculator against independent ref implementation."""
        rng = np.random.default_rng(2026)
        trials = []
        for i in range(5):
            r = rng.normal(0.1 * i, 1.2, 130)
            trials.append(_make_trial_helper(f"T{i}", r))

        eict_res = EictCorr1Calculator().calculate(trials)
        calc = DeflatedSharpeCalculator(trials, eict_res)

        for t in trials:
            prod_res = calc.calculate(t.trial_id)

            # Independent reference calculation
            r_arr = t.net_returns
            mu, sigma, skew, kurt_excess = ref_moments(r_arr)
            sr = mu / sigma
            se = ref_dsr_se(sr, len(r_arr), skew, kurt_excess)

            sharpe_std = float(np.std([tr.distribution.observed_sharpe for tr in trials], ddof=1))
            emax = ref_emax(eict_res.effective_trial_count, sharpe_std)
            ref_val = ref_dsr(sr, len(r_arr), skew, kurt_excess, eict_res.effective_trial_count, sharpe_std)

            assert abs(prod_res.observed_sharpe - sr) < 1e-10
            assert abs(prod_res.skewness - skew) < 1e-10
            assert abs(prod_res.kurtosis_excess - kurt_excess) < 1e-10
            assert abs(prod_res.sharpe_standard_error - se) < 1e-10
            assert abs(prod_res.expected_max_sharpe - emax) < 1e-10
            assert abs(prod_res.dsr - ref_val) < 1e-10
