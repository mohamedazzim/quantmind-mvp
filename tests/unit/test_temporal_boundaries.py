from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pytest
import pandas as pd
import numpy as np

from quantmind.backtest.strategies import NonCausalSignalError, assert_causal_signal
from quantmind.data import (
    DatasetKind,
    DatasetRegistry,
    PurgeEmbargoSpec,
    SplitZone,
    compute_split_manifest,
)
from quantmind.strategy import StrategySpec, compile_strategy_spec


def _generate_bar_series(num_bars: int = 200, start_time: str = "2026-01-01 09:15:00") -> pd.DataFrame:
    base_ts = pd.Timestamp(start_time)
    timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(num_bars)]
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.standard_normal(num_bars))
    return pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices + 0.1,
            "volume": 1000,
            "open_interest": 500,
        }
    )


def test_research_validation_holdout_boundaries_strictly_disjoint():
    df = _generate_bar_series(num_bars=200)
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "data.csv"
        df.to_csv(csv_path, index=False)

        spec = PurgeEmbargoSpec(
            feature_lookback_bars=5,
            prediction_horizon_bars=3,
            holding_period_bars=4,
            forward_dependency_bars=1,
            embargo_bars=5,
        )
        manifest = compute_split_manifest(
            dataset_version="DATA-V1",
            timestamps=df["timestamp"],
            spec=spec,
            research_ratio=0.4,
            validation_ratio=0.3,
            holdout_ratio=0.3,
        )

        registry = DatasetRegistry()
        registry.register_file(
            version="DATA-V1",
            kind=DatasetKind.LICENSED,
            path=csv_path,
            split_manifest=manifest,
        )

        research_df = registry.load_zone("DATA-V1", SplitZone.RESEARCH)
        validation_df = registry.load_zone("DATA-V1", SplitZone.VALIDATION)
        holdout_df = registry.load_zone("DATA-V1", SplitZone.FINAL_HOLDOUT, allow_holdout=True)

        r_ts = set(pd.to_datetime(research_df["timestamp"]))
        v_ts = set(pd.to_datetime(validation_df["timestamp"]))
        h_ts = set(pd.to_datetime(holdout_df["timestamp"]))

        # 1. Strictly disjoint sets
        assert len(r_ts.intersection(v_ts)) == 0, "research and validation overlap!"
        assert len(v_ts.intersection(h_ts)) == 0, "validation and holdout overlap!"
        assert len(r_ts.intersection(h_ts)) == 0, "research and holdout overlap!"

        # 2. Strict chronological order
        max_r = max(r_ts)
        min_v = min(v_ts)
        max_v = max(v_ts)
        min_h = min(h_ts)

        assert max_r < min_v
        assert max_v < min_h


def test_purge_and_embargo_ranges_are_correct_and_absent():
    df = _generate_bar_series(num_bars=200)
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "data.csv"
        df.to_csv(csv_path, index=False)

        spec = PurgeEmbargoSpec(
            feature_lookback_bars=4,
            prediction_horizon_bars=2,
            holding_period_bars=3,
            forward_dependency_bars=1,
            embargo_bars=6,
        )
        # purge_bars = 4 + 2 + 3 + 1 = 10; embargo_bars = 6
        assert spec.purge_bars == 10
        assert spec.embargo_bars == 6

        manifest = compute_split_manifest(
            dataset_version="DATA-V1",
            timestamps=df["timestamp"],
            spec=spec,
            research_ratio=0.4,
            validation_ratio=0.3,
            holdout_ratio=0.3,
        )

        registry = DatasetRegistry()
        registry.register_file(
            version="DATA-V1",
            kind=DatasetKind.LICENSED,
            path=csv_path,
            split_manifest=manifest,
        )

        research_df = registry.load_zone("DATA-V1", SplitZone.RESEARCH)
        validation_df = registry.load_zone("DATA-V1", SplitZone.VALIDATION)
        all_loaded = pd.concat([research_df, validation_df], ignore_index=True)
        loaded_ts = set(pd.to_datetime(all_loaded["timestamp"]))

        # Verify each removed range rows are completely absent from research & validation
        for removed in manifest.removed_ranges:
            r_start = pd.Timestamp(removed.start)
            r_end = pd.Timestamp(removed.end)
            for t in loaded_ts:
                assert not (r_start <= t < r_end), f"timestamp {t} leaked into {removed.name}"


def test_feature_lookback_and_forward_horizon_cannot_cross_boundary():
    """Verify that feature lookback and label holding periods cannot cross the split boundary."""
    df = _generate_bar_series(num_bars=200)
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "data.csv"
        df.to_csv(csv_path, index=False)

        lookback = 5
        horizon = 3
        holding = 4
        forward_dep = 1
        embargo = 5
        spec = PurgeEmbargoSpec(
            feature_lookback_bars=lookback,
            prediction_horizon_bars=horizon,
            holding_period_bars=holding,
            forward_dependency_bars=forward_dep,
            embargo_bars=embargo,
        )
        manifest = compute_split_manifest(
            dataset_version="DATA-V1",
            timestamps=df["timestamp"],
            spec=spec,
            research_ratio=0.4,
            validation_ratio=0.3,
            holdout_ratio=0.3,
        )

        all_ts = pd.to_datetime(df["timestamp"]).tolist()
        r_end_idx = all_ts.index(pd.Timestamp(manifest.research_end))
        v_start_idx = all_ts.index(pd.Timestamp(manifest.validation_start))

        # Separation between research end and validation start in bars
        gap_bars = v_start_idx - r_end_idx
        assert gap_bars == spec.purge_bars + spec.embargo_bars

        # 1. Forward trade initiated at last research bar cannot reach validation start
        # Forward reach = horizon + holding + forward_dep
        forward_reach = horizon + holding + forward_dep
        assert forward_reach <= spec.purge_bars
        assert (r_end_idx + forward_reach) <= v_start_idx

        # 2. Backward feature computed at validation start cannot reach research end
        assert (v_start_idx - lookback) >= r_end_idx


def test_mid_session_truncation_does_not_change_completed_causal_signals():
    """Causal preflight: signals computed on truncated prefixes must match full evaluation."""
    df = _generate_bar_series(num_bars=100)
    spec = StrategySpec(
        strategy_version="1.0.0",
        feature_version="f_v1",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 20, "session_window": [0.10, 0.90]},
    )
    signal_fn = compile_strategy_spec(spec)

    # Full session signal
    full_signal = signal_fn(df)

    # Check that at any truncation cut point, the prefix signals are identical
    for cut in [25, 50, 75]:
        prefix_df = df.iloc[:cut].copy().reset_index(drop=True)
        prefix_signal = signal_fn(prefix_df)
        np.testing.assert_array_equal(
            full_signal[:cut],
            prefix_signal,
            err_msg=f"signal changed under prefix truncation at bar {cut}!",
        )

    # Also assert causality preflight passes
    assert_causal_signal(df, signal_fn, cut_points=[25, 50, 75])


def test_non_causal_signal_fails_preflight():
    """Verify that a future-peeking signal is caught and rejected."""
    df = _generate_bar_series(num_bars=100)

    # Signal using future close return
    def future_peeking_signal(frame: pd.DataFrame) -> np.ndarray:
        close = frame["close"].to_numpy(dtype=float)
        # shifts backward by 1: peeks into next bar!
        future = np.roll(close, -1)
        future[-1] = close[-1]
        return np.where(future > close, 1.0, -1.0)

    with pytest.raises(NonCausalSignalError):
        assert_causal_signal(df, future_peeking_signal, cut_points=[25, 50, 75])
