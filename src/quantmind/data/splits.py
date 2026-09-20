from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

import pandas as pd


class SplitZone(str, Enum):
    RESEARCH = "RESEARCH"
    VALIDATION = "VALIDATION"
    FINAL_HOLDOUT = "FINAL_HOLDOUT"
    FORWARD_PAPER = "FORWARD_PAPER"


class SplitManifestError(ValueError):
    """Raised when a SplitManifest is malformed or invalid."""


@dataclass(frozen=True)
class PurgeEmbargoSpec:
    """Temporal dependency specification defining purge and embargo windows."""

    feature_lookback_bars: int = 0
    prediction_horizon_bars: int = 0
    holding_period_bars: int = 0
    forward_dependency_bars: int = 0
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        for name in (
            "feature_lookback_bars",
            "prediction_horizon_bars",
            "holding_period_bars",
            "forward_dependency_bars",
            "embargo_bars",
        ):
            val = getattr(self, name)
            if not isinstance(val, int) or val < 0:
                raise SplitManifestError(f"{name} must be a non-negative integer, got {val!r}")

    @property
    def purge_bars(self) -> int:
        return (
            self.feature_lookback_bars
            + self.prediction_horizon_bars
            + self.holding_period_bars
            + self.forward_dependency_bars
        )

    @property
    def total_boundary_bars(self) -> int:
        return self.purge_bars + self.embargo_bars


@dataclass(frozen=True)
class RemovedRange:
    name: str
    start: str
    end: str
    bar_count: int


@dataclass(frozen=True)
class SplitManifest:
    """Versioned and immutable manifest defining split zones and purge/embargo boundaries."""

    manifest_version: str
    dataset_version: str
    research_start: str
    research_end: str
    research_validation_purge_start: str
    research_validation_purge_end: str
    research_validation_embargo_start: str
    research_validation_embargo_end: str
    validation_start: str
    validation_end: str
    validation_holdout_purge_start: str
    validation_holdout_purge_end: str
    validation_holdout_embargo_start: str
    validation_holdout_embargo_end: str
    holdout_start: str
    holdout_end: str
    paper_forward_start: str
    paper_forward_end: str | None = None
    spec: PurgeEmbargoSpec | None = None
    removed_ranges: tuple[RemovedRange, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.manifest_version.strip():
            raise SplitManifestError("manifest_version must not be empty")
        if not self.dataset_version.strip():
            raise SplitManifestError("dataset_version must not be empty")

        def _to_ts(name: str, val: str | None) -> pd.Timestamp | None:
            if val is None:
                return None
            try:
                return pd.Timestamp(val)
            except Exception as exc:
                raise SplitManifestError(f"invalid timestamp for {name}: {val!r}") from exc

        r_s = _to_ts("research_start", self.research_start)
        r_e = _to_ts("research_end", self.research_end)
        rv_ps = _to_ts("research_validation_purge_start", self.research_validation_purge_start)
        rv_pe = _to_ts("research_validation_purge_end", self.research_validation_purge_end)
        rv_es = _to_ts("research_validation_embargo_start", self.research_validation_embargo_start)
        rv_ee = _to_ts("research_validation_embargo_end", self.research_validation_embargo_end)
        v_s = _to_ts("validation_start", self.validation_start)
        v_e = _to_ts("validation_end", self.validation_end)
        vh_ps = _to_ts("validation_holdout_purge_start", self.validation_holdout_purge_start)
        vh_pe = _to_ts("validation_holdout_purge_end", self.validation_holdout_purge_end)
        vh_es = _to_ts("validation_holdout_embargo_start", self.validation_holdout_embargo_start)
        vh_ee = _to_ts("validation_holdout_embargo_end", self.validation_holdout_embargo_end)
        h_s = _to_ts("holdout_start", self.holdout_start)
        h_e = _to_ts("holdout_end", self.holdout_end)
        p_s = _to_ts("paper_forward_start", self.paper_forward_start)
        p_e = _to_ts("paper_forward_end", self.paper_forward_end)

        # Basic ordering checks: zone interiors must be non-empty (start < end)
        if not (r_s < r_e):
            raise SplitManifestError(f"research_start ({r_s}) must be strictly earlier than research_end ({r_e})")
        if not (v_s < v_e):
            raise SplitManifestError(f"validation_start ({v_s}) must be strictly earlier than validation_end ({v_e})")
        if not (h_s < h_e):
            raise SplitManifestError(f"holdout_start ({h_s}) must be strictly earlier than holdout_end ({h_e})")
        if p_e is not None and not (p_s < p_e):
            raise SplitManifestError(f"paper_forward_start ({p_s}) must be strictly earlier than paper_forward_end ({p_e})")

        # Boundary ordering:
        # research_end <= rv_purge_start <= rv_purge_end <= rv_embargo_start <= rv_embargo_end <= validation_start
        if not (r_e <= rv_ps <= rv_pe <= rv_es <= rv_ee <= v_s):
            raise SplitManifestError(
                "research -> validation boundary ordering violated: "
                f"expected research_end ({r_e}) <= rv_ps ({rv_ps}) <= rv_pe ({rv_pe}) <= "
                f"rv_es ({rv_es}) <= rv_ee ({rv_ee}) <= validation_start ({v_s})"
            )

        # validation_end <= vh_purge_start <= vh_purge_end <= vh_embargo_start <= vh_embargo_end <= holdout_start
        if not (v_e <= vh_ps <= vh_pe <= vh_es <= vh_ee <= h_s):
            raise SplitManifestError(
                "validation -> holdout boundary ordering violated: "
                f"expected validation_end ({v_e}) <= vh_ps ({vh_ps}) <= vh_pe ({rv_pe}) <= "
                f"vh_es ({vh_es}) <= vh_ee ({vh_ee}) <= holdout_start ({h_s})"
            )

        if not (h_e <= p_s):
            raise SplitManifestError(
                f"holdout_end ({h_e}) must be <= paper_forward_start ({p_s})"
            )

        # If purge or embargo spec is provided with > 0 bars, verify interval is non-zero
        if self.spec is not None:
            if self.spec.purge_bars > 0:
                if rv_ps == rv_pe:
                    raise SplitManifestError("spec declared purge_bars > 0 but research_validation purge interval is zero-width")
                if vh_ps == vh_pe:
                    raise SplitManifestError("spec declared purge_bars > 0 but validation_holdout purge interval is zero-width")
            if self.spec.embargo_bars > 0:
                if rv_es == rv_ee:
                    raise SplitManifestError("spec declared embargo_bars > 0 but research_validation embargo interval is zero-width")
                if vh_es == vh_ee:
                    raise SplitManifestError("spec declared embargo_bars > 0 but validation_holdout embargo interval is zero-width")

    def get_zone_bounds(self, zone: SplitZone | str) -> tuple[str, str | None]:
        zone_str = zone.value if isinstance(zone, SplitZone) else str(zone).upper()
        if zone_str == SplitZone.RESEARCH.value:
            return (self.research_start, self.research_end)
        elif zone_str == SplitZone.VALIDATION.value:
            return (self.validation_start, self.validation_end)
        elif zone_str == SplitZone.FINAL_HOLDOUT.value:
            return (self.holdout_start, self.holdout_end)
        elif zone_str == SplitZone.FORWARD_PAPER.value:
            return (self.paper_forward_start, self.paper_forward_end)
        else:
            raise SplitManifestError(f"unknown split zone: {zone!r}")

    def canonical_json(self) -> str:
        data = {
            "manifest_version": self.manifest_version,
            "dataset_version": self.dataset_version,
            "research_start": self.research_start,
            "research_end": self.research_end,
            "research_validation_purge_start": self.research_validation_purge_start,
            "research_validation_purge_end": self.research_validation_purge_end,
            "research_validation_embargo_start": self.research_validation_embargo_start,
            "research_validation_embargo_end": self.research_validation_embargo_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "validation_holdout_purge_start": self.validation_holdout_purge_start,
            "validation_holdout_purge_end": self.validation_holdout_purge_end,
            "validation_holdout_embargo_start": self.validation_holdout_embargo_start,
            "validation_holdout_embargo_end": self.validation_holdout_embargo_end,
            "holdout_start": self.holdout_start,
            "holdout_end": self.holdout_end,
            "paper_forward_start": self.paper_forward_start,
            "paper_forward_end": self.paper_forward_end,
            "spec": asdict(self.spec) if self.spec is not None else None,
            "removed_ranges": [asdict(r) for r in self.removed_ranges],
            "metadata": dict(sorted(self.metadata.items())),
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def manifest_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SplitManifest:
        spec_data = data.get("spec")
        spec = PurgeEmbargoSpec(**spec_data) if spec_data else None
        removed_raw = data.get("removed_ranges", ())
        removed = tuple(
            RemovedRange(
                name=r["name"],
                start=r["start"],
                end=r["end"],
                bar_count=r["bar_count"],
            )
            for r in removed_raw
        )
        return cls(
            manifest_version=str(data["manifest_version"]),
            dataset_version=str(data["dataset_version"]),
            research_start=str(data["research_start"]),
            research_end=str(data["research_end"]),
            research_validation_purge_start=str(data["research_validation_purge_start"]),
            research_validation_purge_end=str(data["research_validation_purge_end"]),
            research_validation_embargo_start=str(data["research_validation_embargo_start"]),
            research_validation_embargo_end=str(data["research_validation_embargo_end"]),
            validation_start=str(data["validation_start"]),
            validation_end=str(data["validation_end"]),
            validation_holdout_purge_start=str(data["validation_holdout_purge_start"]),
            validation_holdout_purge_end=str(data["validation_holdout_purge_end"]),
            validation_holdout_embargo_start=str(data["validation_holdout_embargo_start"]),
            validation_holdout_embargo_end=str(data["validation_holdout_embargo_end"]),
            holdout_start=str(data["holdout_start"]),
            holdout_end=str(data["holdout_end"]),
            paper_forward_start=str(data["paper_forward_start"]),
            paper_forward_end=data.get("paper_forward_end"),
            spec=spec,
            removed_ranges=removed,
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> SplitManifest:
        return cls.from_dict(json.loads(json_str))


def compute_split_manifest(
    *,
    dataset_version: str,
    manifest_version: str = "v1",
    timestamps: Sequence[Any] | pd.Series,
    spec: PurgeEmbargoSpec,
    research_ratio: float = 0.5,
    validation_ratio: float = 0.2,
    holdout_ratio: float = 0.2,
    metadata: Mapping[str, Any] | None = None,
) -> SplitManifest:
    """Derive deterministic split boundaries with exact purge and embargo windows.

    Excludes purge and embargo rows from both Research, Validation, and Holdout zones.
    """
    if isinstance(timestamps, pd.Series):
        ts_series = pd.to_datetime(timestamps).sort_values().drop_duplicates().reset_index(drop=True)
    else:
        ts_series = pd.Series(sorted(set(pd.to_datetime(t) for t in timestamps)))

    total_bars = len(ts_series)
    if total_bars < 20:
        raise SplitManifestError(f"insufficient bars to compute split manifest: {total_bars}")

    purge_bars = spec.purge_bars
    embargo_bars = spec.embargo_bars
    boundary_buffer_bars = (purge_bars + embargo_bars) * 2

    if total_bars <= boundary_buffer_bars + 10:
        raise SplitManifestError(
            f"total bars ({total_bars}) insufficient for purge ({purge_bars}) and embargo ({embargo_bars}) buffers"
        )

    available_bars = total_bars - boundary_buffer_bars
    total_ratio = research_ratio + validation_ratio + holdout_ratio
    if total_ratio <= 0 or total_ratio > 1.0:
        raise SplitManifestError(f"invalid split ratios sum: {total_ratio}")

    n_res = max(1, int(available_bars * (research_ratio / total_ratio)))
    n_val = max(1, int(available_bars * (validation_ratio / total_ratio)))
    # Holdout gets remainder of available_bars up to holdout_ratio
    n_hold = max(1, int(available_bars * (holdout_ratio / total_ratio)))

    # Adjust indices along the sorted timestamps
    # 1. Research
    idx_r_start = 0
    idx_r_end = idx_r_start + n_res

    # 2. Research-Validation Purge
    idx_rv_p_start = idx_r_end
    idx_rv_p_end = idx_rv_p_start + purge_bars

    # 3. Research-Validation Embargo
    idx_rv_e_start = idx_rv_p_end
    idx_rv_e_end = idx_rv_e_start + embargo_bars

    # 4. Validation
    idx_v_start = idx_rv_e_end
    idx_v_end = idx_v_start + n_val

    # 5. Validation-Holdout Purge
    idx_vh_p_start = idx_v_end
    idx_vh_p_end = idx_vh_p_start + purge_bars

    # 6. Validation-Holdout Embargo
    idx_vh_e_start = idx_vh_p_end
    idx_vh_e_end = idx_vh_e_start + embargo_bars

    # 7. Final Holdout
    idx_h_start = idx_vh_e_end
    idx_h_end = min(total_bars, idx_h_start + n_hold)

    # 8. Forward Paper
    idx_p_start = idx_h_end

    def _fmt(idx: int) -> str:
        if idx >= total_bars:
            # past end: use last timestamp
            return ts_series.iloc[-1].isoformat()
        return ts_series.iloc[idx].isoformat()

    removed: list[RemovedRange] = []
    if purge_bars > 0:
        removed.append(
            RemovedRange(
                name="research_validation_purge",
                start=_fmt(idx_rv_p_start),
                end=_fmt(idx_rv_p_end),
                bar_count=purge_bars,
            )
        )
    if embargo_bars > 0:
        removed.append(
            RemovedRange(
                name="research_validation_embargo",
                start=_fmt(idx_rv_e_start),
                end=_fmt(idx_rv_e_end),
                bar_count=embargo_bars,
            )
        )
    if purge_bars > 0:
        removed.append(
            RemovedRange(
                name="validation_holdout_purge",
                start=_fmt(idx_vh_p_start),
                end=_fmt(idx_vh_p_end),
                bar_count=purge_bars,
            )
        )
    if embargo_bars > 0:
        removed.append(
            RemovedRange(
                name="validation_holdout_embargo",
                start=_fmt(idx_vh_e_start),
                end=_fmt(idx_vh_e_end),
                bar_count=embargo_bars,
            )
        )

    paper_end = _fmt(total_bars - 1) if idx_p_start < total_bars - 1 else None

    return SplitManifest(
        manifest_version=manifest_version,
        dataset_version=dataset_version,
        research_start=_fmt(idx_r_start),
        research_end=_fmt(idx_rv_p_start),
        research_validation_purge_start=_fmt(idx_rv_p_start),
        research_validation_purge_end=_fmt(idx_rv_p_end),
        research_validation_embargo_start=_fmt(idx_rv_e_start),
        research_validation_embargo_end=_fmt(idx_rv_e_end),
        validation_start=_fmt(idx_v_start),
        validation_end=_fmt(idx_vh_p_start),
        validation_holdout_purge_start=_fmt(idx_vh_p_start),
        validation_holdout_purge_end=_fmt(idx_vh_p_end),
        validation_holdout_embargo_start=_fmt(idx_vh_e_start),
        validation_holdout_embargo_end=_fmt(idx_vh_e_end),
        holdout_start=_fmt(idx_h_start),
        holdout_end=_fmt(idx_h_end),
        paper_forward_start=_fmt(idx_p_start),
        paper_forward_end=paper_end,
        spec=spec,
        removed_ranges=tuple(removed),
        metadata=dict(metadata or {}),
    )
