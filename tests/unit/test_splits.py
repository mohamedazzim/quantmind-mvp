from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pytest
import pandas as pd
import numpy as np

from quantmind.data import (
    DatasetKind,
    DatasetRecord,
    DatasetRegistry,
    DatasetRegistryError,
    DatasetZone,
    PurgeEmbargoSpec,
    RemovedRange,
    SplitManifest,
    SplitManifestError,
    SplitZone,
    compute_split_manifest,
)


def _generate_test_csv(path: Path, num_bars: int = 100) -> Path:
    base_ts = pd.Timestamp("2026-01-01 09:15:00")
    timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(num_bars)]
    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "open": 100.0 + np.arange(num_bars),
            "high": 101.0 + np.arange(num_bars),
            "low": 99.0 + np.arange(num_bars),
            "close": 100.5 + np.arange(num_bars),
            "volume": 1000,
            "open_interest": 500,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_purge_embargo_spec_validation_and_properties():
    spec = PurgeEmbargoSpec(
        feature_lookback_bars=5,
        prediction_horizon_bars=2,
        holding_period_bars=3,
        forward_dependency_bars=1,
        embargo_bars=4,
    )
    assert spec.purge_bars == 11
    assert spec.total_boundary_bars == 15

    with pytest.raises(SplitManifestError, match="must be a non-negative integer"):
        PurgeEmbargoSpec(feature_lookback_bars=-1)


def test_split_manifest_validation_ordering():
    # Valid manifest
    manifest = SplitManifest(
        manifest_version="v1",
        dataset_version="DATA-1",
        research_start="2026-01-01 09:15:00",
        research_end="2026-01-01 12:00:00",
        research_validation_purge_start="2026-01-01 12:00:00",
        research_validation_purge_end="2026-01-01 12:30:00",
        research_validation_embargo_start="2026-01-01 12:30:00",
        research_validation_embargo_end="2026-01-01 13:00:00",
        validation_start="2026-01-01 13:00:00",
        validation_end="2026-01-01 14:00:00",
        validation_holdout_purge_start="2026-01-01 14:00:00",
        validation_holdout_purge_end="2026-01-01 14:30:00",
        validation_holdout_embargo_start="2026-01-01 14:30:00",
        validation_holdout_embargo_end="2026-01-01 15:00:00",
        holdout_start="2026-01-01 15:00:00",
        holdout_end="2026-01-01 15:30:00",
        paper_forward_start="2026-01-01 15:30:00",
    )
    assert manifest.manifest_hash() is not None
    assert manifest.get_zone_bounds(SplitZone.RESEARCH) == ("2026-01-01 09:15:00", "2026-01-01 12:00:00")
    assert manifest.get_zone_bounds(SplitZone.VALIDATION) == ("2026-01-01 13:00:00", "2026-01-01 14:00:00")
    assert manifest.get_zone_bounds(SplitZone.FINAL_HOLDOUT) == ("2026-01-01 15:00:00", "2026-01-01 15:30:00")

    # Inverted research interval
    with pytest.raises(SplitManifestError, match="earlier than research_end"):
        SplitManifest(
            manifest_version="v1",
            dataset_version="DATA-1",
            research_start="2026-01-01 12:00:00",
            research_end="2026-01-01 09:15:00",
            research_validation_purge_start="2026-01-01 12:00:00",
            research_validation_purge_end="2026-01-01 12:30:00",
            research_validation_embargo_start="2026-01-01 12:30:00",
            research_validation_embargo_end="2026-01-01 13:00:00",
            validation_start="2026-01-01 13:00:00",
            validation_end="2026-01-01 14:00:00",
            validation_holdout_purge_start="2026-01-01 14:00:00",
            validation_holdout_purge_end="2026-01-01 14:30:00",
            validation_holdout_embargo_start="2026-01-01 14:30:00",
            validation_holdout_embargo_end="2026-01-01 15:00:00",
            holdout_start="2026-01-01 15:00:00",
            holdout_end="2026-01-01 15:30:00",
            paper_forward_start="2026-01-01 15:30:00",
        )

    # Inverted boundary ordering (purge end > embargo start is not violated, but purge end > validation_start)
    with pytest.raises(SplitManifestError, match="boundary ordering violated"):
        SplitManifest(
            manifest_version="v1",
            dataset_version="DATA-1",
            research_start="2026-01-01 09:15:00",
            research_end="2026-01-01 12:00:00",
            research_validation_purge_start="2026-01-01 12:00:00",
            research_validation_purge_end="2026-01-01 13:30:00",  # crosses validation start
            research_validation_embargo_start="2026-01-01 12:30:00",
            research_validation_embargo_end="2026-01-01 13:00:00",
            validation_start="2026-01-01 13:00:00",
            validation_end="2026-01-01 14:00:00",
            validation_holdout_purge_start="2026-01-01 14:00:00",
            validation_holdout_purge_end="2026-01-01 14:30:00",
            validation_holdout_embargo_start="2026-01-01 14:30:00",
            validation_holdout_embargo_end="2026-01-01 15:00:00",
            holdout_start="2026-01-01 15:00:00",
            holdout_end="2026-01-01 15:30:00",
            paper_forward_start="2026-01-01 15:30:00",
        )


def test_compute_split_manifest_derives_exact_ranges():
    base_ts = pd.Timestamp("2026-01-01 09:15:00")
    timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(100)]
    spec = PurgeEmbargoSpec(
        feature_lookback_bars=2,
        prediction_horizon_bars=2,
        holding_period_bars=2,
        forward_dependency_bars=0,
        embargo_bars=3,
    )
    manifest = compute_split_manifest(
        dataset_version="SYNTH-100",
        timestamps=timestamps,
        spec=spec,
        research_ratio=0.5,
        validation_ratio=0.25,
        holdout_ratio=0.25,
    )

    assert spec.purge_bars == 6
    assert spec.embargo_bars == 3
    assert len(manifest.removed_ranges) == 4
    names = [r.name for r in manifest.removed_ranges]
    assert names == [
        "research_validation_purge",
        "research_validation_embargo",
        "validation_holdout_purge",
        "validation_holdout_embargo",
    ]
    counts = [r.bar_count for r in manifest.removed_ranges]
    assert counts == [6, 3, 6, 3]

    # Serialization and deserialization roundtrip
    serialized = manifest.canonical_json()
    deserialized = SplitManifest.from_json(serialized)
    assert deserialized.manifest_hash() == manifest.manifest_hash()
    assert deserialized.dataset_version == manifest.dataset_version
    assert deserialized.removed_ranges == manifest.removed_ranges


def test_dataset_immutability_same_version_same_checksum_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "data.csv"
        _generate_test_csv(csv_path)

        registry = DatasetRegistry()
        rec1 = registry.register_file(
            version="DATA-V1",
            kind=DatasetKind.LICENSED,
            path=csv_path,
        )
        rec2 = registry.register_file(
            version="DATA-V1",
            kind=DatasetKind.LICENSED,
            path=csv_path,
        )
        assert rec1.version == rec2.version
        assert rec1.sha256 == rec2.sha256


def test_dataset_immutability_same_version_different_checksum_rejected():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path1 = Path(tmp_dir) / "data1.csv"
        csv_path2 = Path(tmp_dir) / "data2.csv"
        _generate_test_csv(csv_path1, num_bars=50)
        _generate_test_csv(csv_path2, num_bars=60)

        registry = DatasetRegistry()
        registry.register_file(
            version="DATA-V1",
            kind=DatasetKind.LICENSED,
            path=csv_path1,
        )

        with pytest.raises(DatasetRegistryError, match="immutable"):
            registry.register_file(
                version="DATA-V1",
                kind=DatasetKind.LICENSED,
                path=csv_path2,
            )


def test_dataset_immutability_same_version_different_split_manifest_rejected():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "data.csv"
        _generate_test_csv(csv_path, num_bars=100)

        timestamps = [pd.Timestamp("2026-01-01 09:15:00") + timedelta(minutes=5 * i) for i in range(100)]
        spec1 = PurgeEmbargoSpec(feature_lookback_bars=2, prediction_horizon_bars=2)
        manifest1 = compute_split_manifest(
            dataset_version="DATA-V1",
            manifest_version="v1",
            timestamps=timestamps,
            spec=spec1,
        )

        spec2 = PurgeEmbargoSpec(feature_lookback_bars=10, prediction_horizon_bars=5)
        manifest2 = compute_split_manifest(
            dataset_version="DATA-V1",
            manifest_version="v2",
            timestamps=timestamps,
            spec=spec2,
        )

        registry = DatasetRegistry()
        registry.register_file(
            version="DATA-V1",
            kind=DatasetKind.LICENSED,
            path=csv_path,
            split_manifest=manifest1,
        )

        # Idempotent re-registration with same manifest
        registry.register_file(
            version="DATA-V1",
            kind=DatasetKind.LICENSED,
            path=csv_path,
            split_manifest=manifest1,
        )

        # Re-registration with modified split manifest rejected
        with pytest.raises(DatasetRegistryError, match="immutable"):
            registry.register_file(
                version="DATA-V1",
                kind=DatasetKind.LICENSED,
                path=csv_path,
                split_manifest=manifest2,
            )

        # Direct manifest modification rejected
        with pytest.raises(DatasetRegistryError, match="immutable"):
            registry.register_split_manifest(manifest2)
