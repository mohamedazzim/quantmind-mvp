"""Statistical Validation Layer: EICT-CORR-1 Clustering & Deflated Sharpe Ratio.

This module implements two decoupled statistical verification layers:
- LAYER A: EICT-CORR-1 (Effective Independent Trial Count via hierarchical clustering)
- LAYER B: Deflated Sharpe Ratio (Bailey & López de Prado 2014 multiple-testing adjustment)

Multiple-Testing Protection Guarantees:
- Calculations must be anchored to an authoritative ProductionPopulationQuery.
- Manual overrides of effective trial count or candidate Sharpe ratio are strictly prohibited.
- New research tasks cannot reset or bypass the cumulative population scope.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import scipy.stats as stats

from quantmind.research_integrity.population import (
    Eict1PopulationInputs,
    EictCorr1InputBuilder,
    EligibleTrial,
    ProductionPopulationQuery,
    _EICT_CORR1_CORRELATION_THRESHOLD,
    _EICT_CORR1_DISTANCE_THRESHOLD,
    _EICT_CORR1_LINKAGE,
    _EICT_CORR1_MIN_OVERLAP,
    _EICT_CORR1_INSUFFICIENT_OVERLAP_POLICY,
)


# Euler-Mascheroni constant
_EULER_MASCHERONI: float = 0.5772156649015329


# ---------------------------------------------------------------------------
# LAYER A: EICT-CORR-1 Result & Calculator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EictClusterResult:
    """Immutable output of the EICT-CORR-1 clustering algorithm."""

    raw_trial_count: int
    effective_trial_count: int
    correlation_matrix: np.ndarray
    distance_matrix: np.ndarray
    clusters: Mapping[int, tuple[str, ...]]        # cluster_id -> tuple of trial_ids
    cluster_memberships: Mapping[str, int]         # trial_id -> cluster_id
    singleton_clusters: tuple[str, ...]            # trial_ids forming singletons
    method_id: str                                 # "EICT-CORR-1"
    protocol_version: str
    population_hash: str
    dataset_version: str
    research_protocol_version: str
    created_at: str


def compute_population_hash(trials: Sequence[EligibleTrial]) -> str:
    """Compute deterministic SHA-256 digest over canonical trial metadata."""
    sorted_trials = sorted(trials, key=lambda t: t.trial_id)
    canonical = [
        {
            "trial_id": t.trial_id,
            "strategy_id": t.strategy_id,
            "artifact_sha256": t.artifact.sha256,
        }
        for t in sorted_trials
    ]
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EictCorr1Calculator:
    """Computes effective independent trial count using the locked EICT-CORR-1 protocol.

    Hierarchical clustering with average linkage at distance cut 0.6325.
    Trials with insufficient overlap (<100 common observations) are preserved
    as singleton clusters and each contributes 1 to the effective trial count.
    """

    METHOD_ID: str = "EICT-CORR-1"

    def __init__(
        self,
        *,
        distance_threshold: float = _EICT_CORR1_DISTANCE_THRESHOLD,
        linkage: str = _EICT_CORR1_LINKAGE,
        method_id: str = "EICT-CORR-1",
    ) -> None:
        if method_id != self.METHOD_ID:
            raise ValueError(
                f"method_id must be '{self.METHOD_ID}' under protocol RP-2; got '{method_id}'"
            )
        if abs(distance_threshold - _EICT_CORR1_DISTANCE_THRESHOLD) > 1e-10:
            raise ValueError(
                f"distance_threshold must be {_EICT_CORR1_DISTANCE_THRESHOLD} (locked protocol value)"
            )
        if linkage != _EICT_CORR1_LINKAGE:
            raise ValueError(f"linkage must be '{_EICT_CORR1_LINKAGE}' (locked protocol value)")

        self._distance_threshold = distance_threshold
        self._linkage = linkage
        self._builder = EictCorr1InputBuilder()

    def calculate(
        self,
        trials: Sequence[EligibleTrial],
        inputs: Eict1PopulationInputs | None = None,
    ) -> EictClusterResult:
        """Execute EICT-CORR-1 clustering over eligible trials."""
        if not trials:
            now = datetime.now(timezone.utc).isoformat()
            return EictClusterResult(
                raw_trial_count=0,
                effective_trial_count=0,
                correlation_matrix=np.empty((0, 0)),
                distance_matrix=np.empty((0, 0)),
                clusters={},
                cluster_memberships={},
                singleton_clusters=(),
                method_id=self.METHOD_ID,
                protocol_version="",
                population_hash="",
                dataset_version="",
                research_protocol_version="",
                created_at=now,
            )

        pop_inputs = inputs or self._builder.build(trials)
        trial_ids = pop_inputs.trial_ids
        raw_count = len(trial_ids)
        dist_matrix = pop_inputs.distance_matrix
        singleton_ids = set(pop_inputs.singleton_trial_ids)

        if raw_count == 1:
            tid = trial_ids[0]
            now = datetime.now(timezone.utc).isoformat()
            return EictClusterResult(
                raw_trial_count=1,
                effective_trial_count=1,
                correlation_matrix=pop_inputs.correlation_matrix,
                distance_matrix=pop_inputs.distance_matrix,
                clusters={1: (tid,)},
                cluster_memberships={tid: 1},
                singleton_clusters=(tid,) if tid in singleton_ids else (),
                method_id=self.METHOD_ID,
                protocol_version=trials[0].research_protocol_version,
                population_hash=compute_population_hash(trials),
                dataset_version=trials[0].dataset_version,
                research_protocol_version=trials[0].research_protocol_version,
                created_at=now,
            )

        # Separate into connected trials and singletons
        connected_tids = [tid for tid in trial_ids if tid not in singleton_ids]
        singletons_list = [tid for tid in trial_ids if tid in singleton_ids]

        clusters: dict[int, list[str]] = {}
        memberships: dict[str, int] = {}
        next_cid = 1

        if len(connected_tids) == 1:
            tid = connected_tids[0]
            clusters[next_cid] = [tid]
            memberships[tid] = next_cid
            next_cid += 1
        elif len(connected_tids) > 1:
            idx_map = [trial_ids.index(tid) for tid in connected_tids]
            sub_D = dist_matrix[np.ix_(idx_map, idx_map)].copy()
            # Replace any residual NaNs with maximum distance 2.0 (no overlap)
            np.nan_to_num(sub_D, copy=False, nan=2.0)
            np.fill_diagonal(sub_D, 0.0)
            sub_D = 0.5 * (sub_D + sub_D.T)
            # Clip between 0 and 2
            sub_D = np.clip(sub_D, 0.0, 2.0)

            condensed = squareform(sub_D)
            Z = sch.linkage(condensed, method=self._linkage)
            labels = sch.fcluster(Z, t=self._distance_threshold, criterion="distance")

            # Group deterministically by sorted label
            unique_labels = sorted(set(labels))
            for lbl in unique_labels:
                indices = np.where(labels == lbl)[0]
                members = sorted(connected_tids[i] for i in indices)
                clusters[next_cid] = members
                for m in members:
                    memberships[m] = next_cid
                next_cid += 1

        # Each singleton trial becomes its own cluster
        for stid in sorted(singletons_list):
            clusters[next_cid] = [stid]
            memberships[stid] = next_cid
            next_cid += 1

        effective_count = len(clusters)
        pop_hash = compute_population_hash(trials)
        now = datetime.now(timezone.utc).isoformat()

        return EictClusterResult(
            raw_trial_count=raw_count,
            effective_trial_count=effective_count,
            correlation_matrix=pop_inputs.correlation_matrix,
            distance_matrix=pop_inputs.distance_matrix,
            clusters={k: tuple(v) for k, v in clusters.items()},
            cluster_memberships=memberships,
            singleton_clusters=tuple(sorted(singletons_list)),
            method_id=self.METHOD_ID,
            protocol_version=trials[0].research_protocol_version,
            population_hash=pop_hash,
            dataset_version=trials[0].dataset_version,
            research_protocol_version=trials[0].research_protocol_version,
            created_at=now,
        )


# ---------------------------------------------------------------------------
# LAYER B: Deflated Sharpe Ratio Result & Calculator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio validation result for a specific strategy trial."""

    trial_id: str
    strategy_id: str
    observed_sharpe: float
    sample_length: int
    skewness: float
    kurtosis_excess: float
    raw_kurtosis: float
    effective_trial_count: int
    sharpe_std_population: float
    expected_max_sharpe: float
    sharpe_standard_error: float
    z_stat: float
    dsr: float
    passes_dsr_gate: bool
    population_hash: str
    dataset_version: str
    dataset_sha256: str
    split_manifest_version: str
    split_manifest_sha256: str
    research_protocol_version: str
    method_id: str
    code_version: str
    config_hash: str


class DeflatedSharpeCalculator:
    """Computes the Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Inputs are drawn exclusively from the authoritative production population
    and EICT cluster results. Manual overrides of candidate Sharpe, trial count,
    skewness, or kurtosis are strictly rejected.
    """

    def __init__(
        self,
        population: Sequence[EligibleTrial],
        eict_result: EictClusterResult,
        *,
        dsr_significance_level: float = 0.95,
    ) -> None:
        if not population:
            raise ValueError("DeflatedSharpeCalculator requires a non-empty population")
        if eict_result.raw_trial_count != len(population):
            raise ValueError(
                f"population length ({len(population)}) does not match EICT raw trial count ({eict_result.raw_trial_count})"
            )

        self._population = list(population)
        self._eict_result = eict_result
        self._significance_level = dsr_significance_level
        self._trial_map = {t.trial_id: t for t in self._population}

        # Distribution of Sharpes across the full population
        sharpes = [t.distribution.observed_sharpe for t in self._population]
        if len(sharpes) > 1:
            self._sharpe_std = float(np.std(sharpes, ddof=1))
        else:
            self._sharpe_std = 0.0

    @property
    def population_hash(self) -> str:
        return self._eict_result.population_hash

    @property
    def effective_trial_count(self) -> int:
        return self._eict_result.effective_trial_count

    def expected_max_sharpe(self) -> float:
        """Compute E[max_{N} {SR}] under null where true Sharpe = 0."""
        N = float(self._eict_result.effective_trial_count)
        if N <= 1.0 or self._sharpe_std == 0.0:
            return 0.0

        z1 = float(stats.norm.ppf(1.0 - 1.0 / N))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (N * math.e)))
        emax_factor = (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
        return float(self._sharpe_std * emax_factor)

    def calculate(
        self,
        trial_id: str,
        *,
        manual_sharpe: float | None = None,
        manual_trials: int | None = None,
        manual_effective_trial_count: int | None = None,
        manual_skew: float | None = None,
        manual_kurtosis: float | None = None,
    ) -> DeflatedSharpeResult:
        """Calculate DSR for a specific trial within the population.

        Manual overrides are prohibited to prevent multiplicity bypassing.
        """
        if (
            manual_sharpe is not None
            or manual_trials is not None
            or manual_effective_trial_count is not None
            or manual_skew is not None
            or manual_kurtosis is not None
        ):
            raise ValueError(
                "manual overrides of observed_sharpe, effective_trial_count, skewness, or kurtosis "
                "are prohibited; DSR must derive strictly from the authoritative production population."
            )

        if trial_id not in self._trial_map:
            raise KeyError(
                f"trial_id '{trial_id}' is not an eligible trial in population "
                f"scope dataset={self._eict_result.dataset_version}, protocol={self._eict_result.research_protocol_version}"
            )

        trial = self._trial_map[trial_id]
        dist = trial.distribution

        sr = dist.observed_sharpe
        T = dist.effective_observations
        if T < 2:
            raise ValueError(
                f"trial '{trial_id}' has sample length T={T} which is insufficient for "
                f"DSR calculation (minimum T >= 2 required for sample standard error)"
            )

        skew = dist.skewness
        kurt_excess = dist.excess_kurtosis
        raw_kurt = dist.raw_kurtosis
        eff_count = self._eict_result.effective_trial_count
        emax = self.expected_max_sharpe()

        # Mertens / Lo / Bailey-López de Prado standard error
        # Formula: sqrt( (1 - gamma3 * SR + ((gamma4 - 1)/4) * SR^2) / (T - 1) )
        # where gamma3 = skewness, gamma4 = raw (Pearson) kurtosis = kurt_excess + 3
        var_term = (
            1.0
            - skew * sr
            + ((raw_kurt - 1.0) / 4.0) * (sr ** 2)
        ) / (T - 1.0)
        se = math.sqrt(max(1e-12, var_term))
        z = (sr - emax) / se
        dsr = float(stats.norm.cdf(z))

        passes_gate = dsr >= self._significance_level

        prov = trial.provenance
        return DeflatedSharpeResult(
            trial_id=trial.trial_id,
            strategy_id=trial.strategy_id,
            observed_sharpe=sr,
            sample_length=T,
            skewness=skew,
            kurtosis_excess=kurt_excess,
            raw_kurtosis=raw_kurt,
            effective_trial_count=eff_count,
            sharpe_std_population=self._sharpe_std,
            expected_max_sharpe=emax,
            sharpe_standard_error=se,
            z_stat=z,
            dsr=dsr,
            passes_dsr_gate=passes_gate,
            population_hash=self._eict_result.population_hash,
            dataset_version=trial.dataset_version,
            dataset_sha256=prov.dataset_sha256,
            split_manifest_version=prov.split_manifest_version,
            split_manifest_sha256=prov.split_manifest_sha256,
            research_protocol_version=trial.research_protocol_version,
            method_id=self._eict_result.method_id,
            code_version=prov.code_version,
            config_hash=prov.config_hash,
        )


# ---------------------------------------------------------------------------
# Integrated Statistical Validation Pipeline
# ---------------------------------------------------------------------------


class StatisticalValidationPipeline:
    """Authoritative gateway executing EICT-CORR-1 and DSR over production trials."""

    def __init__(
        self,
        population_query: ProductionPopulationQuery,
        eict_calculator: EictCorr1Calculator | None = None,
        significance_level: float = 0.95,
    ) -> None:
        self._query = population_query
        self._eict_calc = eict_calculator or EictCorr1Calculator()
        self._significance_level = significance_level

    def evaluate_trial(
        self,
        trial_id: str,
        *,
        dataset_version: str,
        research_protocol_version: str,
    ) -> tuple[EictClusterResult, DeflatedSharpeResult]:
        """Load population, compute EICT-CORR-1 clustering, and evaluate DSR for trial_id."""
        trials = self._query.load_eligible_trials(
            dataset_version=dataset_version,
            research_protocol_version=research_protocol_version,
        )
        trial_ids = [t.trial_id for t in trials]
        if trial_id not in trial_ids:
            raise KeyError(
                f"trial '{trial_id}' not found in eligible production population for "
                f"dataset={dataset_version}, protocol={research_protocol_version}"
            )

        eict_res = self._eict_calc.calculate(trials)
        dsr_calc = DeflatedSharpeCalculator(
            trials, eict_res, dsr_significance_level=self._significance_level
        )
        dsr_res = dsr_calc.calculate(trial_id)
        return eict_res, dsr_res
