"""Production population accounting, EICT-CORR-1 input preparation, and DSR inputs.

This module provides the **evidence layer** required by the EICT-CORR-1 and DSR
statistical methods.  It does NOT implement the statistical decisions themselves.

Population scope
----------------
* ``mode = PRODUCTION``
* Key dimensions: ``dataset_version`` + ``research_protocol_version``
* Excluded statuses: FAILED, ABANDONED, REJECTED_NONCAUSAL, REJECTED_FINAL_HOLDOUT,
  RUNNING (any non-terminal status)
* FIXTURE trials are **never** eligible — excluded by mode filter
* Must have a valid ``ArtifactType.OOS_RETURNS`` artifact with SHA-256 verified

EICT-CORR-1 constants (read from configs/research_protocol_v2.yaml)
--------------------------------------------------------------------
DO NOT change these protocol values in code; they are loaded from config.
  correlation_threshold:       0.80
  distance_threshold:          0.6325
  linkage:                     average
  minimum_overlap_observations: 100
  insufficient_overlap_policy: singleton_cluster

DSR inputs (evidence layer only — no DSR formula implemented here)
-------------------------------------------------------------------
* observed_sharpe = mean_net_return_bps / std_net_return_bps
* sample_length   = effective_observations per trial
* skewness        = scipy.stats.skew(net_return_bps)
* kurtosis        = scipy.stats.kurtosis(net_return_bps, fisher=True)
* effective_trial_count = len(eligible trials) after EICT-CORR-1 filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import sqlite3

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.stats as stats

from quantmind.research_integrity.artifacts import (
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactType,
    load_oos_returns_verified,
    load_net_returns_array,
    load_signal_timestamps_array,
)
from quantmind.research_integrity.trial_ledger import TrialLedger

# ---------------------------------------------------------------------------
# Protocol constants — loaded once from YAML; defined here as typed defaults.
# These must NOT be altered without a research_protocol_version bump.
# ---------------------------------------------------------------------------

_EICT_CORR1_CORRELATION_THRESHOLD: float = 0.80
_EICT_CORR1_DISTANCE_THRESHOLD: float = 0.6325
_EICT_CORR1_LINKAGE: str = "average"
_EICT_CORR1_MIN_OVERLAP: int = 100
_EICT_CORR1_INSUFFICIENT_OVERLAP_POLICY: str = "singleton_cluster"


# ---------------------------------------------------------------------------
# Return distribution metadata (DSR inputs per trial)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReturnDistributionMetadata:
    """Statistical summary of a trial's OOS return stream — DSR inputs."""

    trial_id: str
    mean_return: float           # mean net_return_bps
    std_return: float            # std dev net_return_bps
    skewness: float              # scipy skew
    kurtosis: float              # excess kurtosis (Fisher)
    trade_count: int             # total trade count (rows in artifact)
    effective_observations: int  # same as trade_count (MVP: no autocorr adjustment)
    observed_sharpe: float       # mean / std if std > 0, else 0.0

    @property
    def excess_kurtosis(self) -> float:
        """Fisher's excess kurtosis (normal distribution = 0.0)."""
        return self.kurtosis

    @property
    def raw_kurtosis(self) -> float:
        """Pearson's raw kurtosis gamma4 = excess_kurtosis + 3.0 (normal distribution = 3.0)."""
        return self.kurtosis + 3.0


def compute_return_distribution(trial_id: str, net_returns: np.ndarray) -> ReturnDistributionMetadata:
    """Compute DSR input statistics from a net_return_bps array."""
    if len(net_returns) > 0 and not np.all(np.isfinite(net_returns)):
        raise ValueError(f"net_returns for trial {trial_id} contains non-finite values (NaN or Inf)")

    n = len(net_returns)
    if n == 0:
        return ReturnDistributionMetadata(
            trial_id=trial_id,
            mean_return=0.0,
            std_return=0.0,
            skewness=0.0,
            kurtosis=0.0,
            trade_count=0,
            effective_observations=0,
            observed_sharpe=0.0,
        )
    mu = float(np.mean(net_returns))
    sigma = float(np.std(net_returns, ddof=1)) if n > 1 else 0.0
    if sigma == 0.0 or not np.isfinite(sigma):
        skew = 0.0
        kurt = 0.0
        sharpe = 0.0
    else:
        skew = float(stats.skew(net_returns, bias=False)) if n >= 3 else 0.0
        kurt = float(stats.kurtosis(net_returns, bias=False, fisher=True)) if n >= 4 else 0.0
        sharpe = mu / sigma
    return ReturnDistributionMetadata(
        trial_id=trial_id,
        mean_return=mu,
        std_return=sigma,
        skewness=skew,
        kurtosis=kurt,
        trade_count=n,
        effective_observations=n,
        observed_sharpe=sharpe,
    )


# ---------------------------------------------------------------------------
# Eligible trial record for population
# ---------------------------------------------------------------------------

# Terminal statuses that *pass* into the production population.
_ELIGIBLE_TERMINAL = frozenset({"COMPLETED"})

# Non-eligible statuses — any terminal status that is not COMPLETED.
_EXCLUDED_TERMINAL = frozenset({
    "FAILED",
    "ABANDONED",
    "REJECTED_NONCAUSAL",
    "REJECTED_FINAL_HOLDOUT",
})


@dataclass(frozen=True)
class TrialProvenance:
    """Immutable audit trail ensuring reproducible population provenance."""

    dataset_version: str
    dataset_sha256: str
    split_manifest_version: str
    split_manifest_sha256: str
    research_protocol_version: str
    strategy_id: str
    experiment_id: str
    code_version: str
    config_hash: str
    artifact_sha256: str


@dataclass(frozen=True)
class EligibleTrial:
    """A production trial that passed all population eligibility filters."""

    trial_id: str
    strategy_id: str
    dataset_version: str
    research_protocol_version: str
    artifact: ArtifactRecord
    net_returns: np.ndarray       # float64 array — shape (N,)
    signal_timestamps: np.ndarray # datetime64[us] — shape (N,)
    distribution: ReturnDistributionMetadata
    provenance: TrialProvenance


# ---------------------------------------------------------------------------
# Production population query
# ---------------------------------------------------------------------------


class ProductionPopulationQuery:
    """Query the TrialLedger + ArtifactRegistry for eligible production trials.

    An eligible trial must:
    1. Have ``mode = PRODUCTION``
    2. Have ``status = COMPLETED`` (terminal passing)
    3. Belong to the requested (dataset_version, research_protocol_version) scope
    4. Have a registered ``OOS_RETURNS`` artifact with a SHA-256-verified file
    5. Have consistent artifact_sha256 between ledger result and artifact registry
    """

    def __init__(self, ledger: TrialLedger, artifact_registry: ArtifactRegistry) -> None:
        self._ledger = ledger
        self._registry = artifact_registry

    def load_eligible_trials(
        self,
        *,
        dataset_version: str,
        research_protocol_version: str,
    ) -> list[EligibleTrial]:
        """Return all eligible production trials for the given population scope.

        Trials are sorted by their artifact ``created_at`` timestamp ascending.
        SHA-256 verification is performed on every artifact; trials with
        missing or tampered artifacts are silently excluded.
        """
        import json

        rows = self._ledger.find_trials(
            dataset_version=dataset_version,
            research_protocol_version=research_protocol_version,
            mode="PRODUCTION",
            status="COMPLETED",
        )

        eligible: list[EligibleTrial] = []
        for row in rows:
            trial_id = row["trial_id"]
            artifact = self._registry.get_for_trial(trial_id, ArtifactType.OOS_RETURNS)
            if artifact is None:
                # No artifact registered — trial excluded
                continue

            # Consistency check: ledger result_json vs artifact registry
            result_json = row["result_json"]
            if result_json:
                try:
                    res_dict = json.loads(result_json)
                    ledger_sha = res_dict.get("artifact_sha256")
                    if ledger_sha is not None and ledger_sha != artifact.sha256:
                        # Ledger and registry disagree on artifact SHA-256
                        continue
                except Exception:
                    continue

            # Load and verify artifact
            try:
                table = load_oos_returns_verified(artifact)
            except (FileNotFoundError, ValueError):
                # Missing file or SHA-256 mismatch — trial excluded
                continue

            net_col = table.column("net_return_bps")
            net_returns = np.asarray(net_col, dtype=np.float64)

            sig_col = table.column("signal_timestamp")
            signal_timestamps = sig_col.to_pandas().values.astype("datetime64[us]")

            try:
                dist = compute_return_distribution(trial_id, net_returns)
            except ValueError:
                # Non-finite or invalid distribution — trial excluded
                continue

            col_keys = row.keys() if hasattr(row, "keys") else ()
            provenance = TrialProvenance(
                dataset_version=dataset_version,
                dataset_sha256=row["dataset_sha256"] if "dataset_sha256" in col_keys else "",
                split_manifest_version=row["split_manifest_version"] if "split_manifest_version" in col_keys else "",
                split_manifest_sha256=row["split_manifest_sha256"] if "split_manifest_sha256" in col_keys else "",
                research_protocol_version=research_protocol_version,
                strategy_id=row["strategy_id"],
                experiment_id=row["experiment_id"],
                code_version=row["code_version"] if "code_version" in col_keys else "0.1.0",
                config_hash=row["config_hash"] if "config_hash" in col_keys else "",
                artifact_sha256=artifact.sha256,
            )

            eligible.append(
                EligibleTrial(
                    trial_id=trial_id,
                    strategy_id=row["strategy_id"],
                    dataset_version=dataset_version,
                    research_protocol_version=research_protocol_version,
                    artifact=artifact,
                    net_returns=net_returns,
                    signal_timestamps=signal_timestamps,
                    distribution=dist,
                    provenance=provenance,
                )
            )

        return eligible


# ---------------------------------------------------------------------------
# Common timestamp alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedPair:
    """A pair of trials aligned to their common timestamp intersection."""

    trial_a: str
    trial_b: str
    returns_a: np.ndarray  # float64, aligned
    returns_b: np.ndarray  # float64, aligned
    n_common: int


def align_pair(
    ts_a: np.ndarray,
    ret_a: np.ndarray,
    ts_b: np.ndarray,
    ret_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute intersection of timestamps and align return arrays.

    Parameters
    ----------
    ts_a, ts_b : datetime64[us] arrays of signal timestamps
    ret_a, ret_b : float64 net_return_bps arrays (same length as corresponding ts)

    Returns
    -------
    common_ts, aligned_a, aligned_b
    """
    # Use pandas Index for fast set-intersection; keep sorted order.
    idx_a = pd.Index(ts_a)
    idx_b = pd.Index(ts_b)
    common = idx_a.intersection(idx_b).sort_values()

    if len(common) == 0:
        empty = np.empty(0, dtype=np.float64)
        return common.to_numpy().astype("datetime64[us]"), empty, empty

    mask_a = np.isin(ts_a, common.to_numpy())
    mask_b = np.isin(ts_b, common.to_numpy())

    # Align by common timestamp order
    df_a = pd.DataFrame({"ts": ts_a, "ret": ret_a}).set_index("ts").loc[common]
    df_b = pd.DataFrame({"ts": ts_b, "ret": ret_b}).set_index("ts").loc[common]

    aligned_a = df_a["ret"].to_numpy(dtype=np.float64)
    aligned_b = df_b["ret"].to_numpy(dtype=np.float64)
    common_ts = common.to_numpy().astype("datetime64[us]")
    return common_ts, aligned_a, aligned_b


# ---------------------------------------------------------------------------
# EICT-CORR-1 input preparation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Eict1PairwiseInput:
    """Pairwise input record for EICT-CORR-1 clustering."""

    trial_a: str
    trial_b: str
    n_common: int
    pearson_correlation: float   # NaN if insufficient overlap
    eict_distance: float         # sqrt(2*(1-corr)), NaN if insufficient overlap
    sufficient_overlap: bool     # n_common >= minimum_overlap_observations


@dataclass
class Eict1PopulationInputs:
    """Full EICT-CORR-1 input bundle for a population scope."""

    dataset_version: str
    research_protocol_version: str
    trial_ids: list[str]               # ordered list
    pairwise: list[Eict1PairwiseInput]
    correlation_matrix: np.ndarray     # shape (N, N), NaN for insufficient pairs
    distance_matrix: np.ndarray        # shape (N, N), NaN for insufficient pairs
    singleton_trial_ids: list[str]     # trials with no sufficient-overlap partner
    # Protocol constants (read-only, do NOT mutate)
    correlation_threshold: float = _EICT_CORR1_CORRELATION_THRESHOLD
    distance_threshold: float = _EICT_CORR1_DISTANCE_THRESHOLD
    linkage: str = _EICT_CORR1_LINKAGE
    minimum_overlap_observations: int = _EICT_CORR1_MIN_OVERLAP
    insufficient_overlap_policy: str = _EICT_CORR1_INSUFFICIENT_OVERLAP_POLICY


@dataclass(frozen=True)
class DsrPopulationInputs:
    """DSR input bundle for a population scope (evidence layer — no formula)."""

    dataset_version: str
    research_protocol_version: str
    trial_distributions: list[ReturnDistributionMetadata]
    effective_trial_count: int   # len(trial_distributions) after EICT filtering


class EictCorr1InputBuilder:
    """Build EICT-CORR-1 pairwise distance inputs from eligible trials.

    This class prepares the clustering inputs as specified by the research
    protocol.  It does NOT perform clustering or make any accept/reject
    decisions — those are reserved for the statistical layer.

    Protocol parameters are fixed and must not be overridden here.
    """

    def __init__(
        self,
        *,
        minimum_overlap_observations: int = _EICT_CORR1_MIN_OVERLAP,
        insufficient_overlap_policy: str = _EICT_CORR1_INSUFFICIENT_OVERLAP_POLICY,
        correlation_threshold: float = _EICT_CORR1_CORRELATION_THRESHOLD,
        distance_threshold: float = _EICT_CORR1_DISTANCE_THRESHOLD,
        linkage: str = _EICT_CORR1_LINKAGE,
    ) -> None:
        # Guard: protocol constants should not be changed via constructor
        if minimum_overlap_observations != _EICT_CORR1_MIN_OVERLAP:
            raise ValueError(
                f"minimum_overlap_observations must be {_EICT_CORR1_MIN_OVERLAP} (protocol constant)"
            )
        if insufficient_overlap_policy != _EICT_CORR1_INSUFFICIENT_OVERLAP_POLICY:
            raise ValueError(
                f"insufficient_overlap_policy must be '{_EICT_CORR1_INSUFFICIENT_OVERLAP_POLICY}' (protocol constant)"
            )
        if abs(correlation_threshold - _EICT_CORR1_CORRELATION_THRESHOLD) > 1e-10:
            raise ValueError(
                f"correlation_threshold must be {_EICT_CORR1_CORRELATION_THRESHOLD} (protocol constant)"
            )
        if abs(distance_threshold - _EICT_CORR1_DISTANCE_THRESHOLD) > 1e-10:
            raise ValueError(
                f"distance_threshold must be {_EICT_CORR1_DISTANCE_THRESHOLD} (protocol constant)"
            )
        if linkage != _EICT_CORR1_LINKAGE:
            raise ValueError(
                f"linkage must be '{_EICT_CORR1_LINKAGE}' (protocol constant)"
            )

    def build(self, trials: Sequence[EligibleTrial]) -> Eict1PopulationInputs:
        """Compute pairwise EICT-CORR-1 inputs from a list of eligible trials.

        Steps
        -----
        1. Intersect signal timestamps for each pair.
        2. Check minimum_overlap_observations.
        3. If sufficient: compute Pearson correlation and distance sqrt(2*(1-r)).
        4. If insufficient: distance = NaN, mark as singleton candidate.
        5. Identify trials that have no sufficient-overlap partner → singletons.
        """
        sorted_trials = sorted(trials, key=lambda t: t.trial_id)
        trial_ids = [t.trial_id for t in sorted_trials]
        n = len(sorted_trials)

        pairwise: list[Eict1PairwiseInput] = []
        corr_matrix = np.full((n, n), np.nan)
        dist_matrix = np.full((n, n), np.nan)
        np.fill_diagonal(corr_matrix, 1.0)
        np.fill_diagonal(dist_matrix, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                ta, tb = sorted_trials[i], sorted_trials[j]
                _, aligned_a, aligned_b = align_pair(
                    ta.signal_timestamps, ta.net_returns,
                    tb.signal_timestamps, tb.net_returns,
                )
                n_common = len(aligned_a)
                sufficient = n_common >= _EICT_CORR1_MIN_OVERLAP

                if sufficient:
                    # Pearson correlation (ddof=1 via np.corrcoef)
                    r = float(np.corrcoef(aligned_a, aligned_b)[0, 1])
                    # Clip to [-1, 1] for numerical safety
                    r = max(-1.0, min(1.0, r))
                    dist = float(np.sqrt(2.0 * (1.0 - r)))
                    corr_matrix[i, j] = corr_matrix[j, i] = r
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
                else:
                    r = float("nan")
                    dist = float("nan")

                pairwise.append(Eict1PairwiseInput(
                    trial_a=ta.trial_id,
                    trial_b=tb.trial_id,
                    n_common=n_common,
                    pearson_correlation=r,
                    eict_distance=dist,
                    sufficient_overlap=sufficient,
                ))

        # Identify singleton trials: any trial that has no sufficient-overlap
        # partner is treated as a singleton cluster per protocol.
        has_partner = set()
        for p in pairwise:
            if p.sufficient_overlap:
                has_partner.add(p.trial_a)
                has_partner.add(p.trial_b)
        singleton_ids = [t.trial_id for t in trials if t.trial_id not in has_partner]

        return Eict1PopulationInputs(
            dataset_version=trials[0].dataset_version if trials else "",
            research_protocol_version=trials[0].research_protocol_version if trials else "",
            trial_ids=trial_ids,
            pairwise=pairwise,
            correlation_matrix=corr_matrix,
            distance_matrix=dist_matrix,
            singleton_trial_ids=singleton_ids,
        )


# ---------------------------------------------------------------------------
# DSR input preparation
# ---------------------------------------------------------------------------


def build_dsr_inputs(trials: Sequence[EligibleTrial]) -> DsrPopulationInputs:
    """Collect DSR inputs from eligible trials.

    This is the evidence layer only.  The DSR formula is NOT implemented here.
    ``effective_trial_count`` is simply the count of eligible trials passed in;
    the caller is responsible for applying any EICT-based reduction first.
    """
    distributions = [t.distribution for t in trials]
    return DsrPopulationInputs(
        dataset_version=trials[0].dataset_version if trials else "",
        research_protocol_version=trials[0].research_protocol_version if trials else "",
        trial_distributions=list(distributions),
        effective_trial_count=len(distributions),
    )
