import numpy as np
import pandas as pd
import pytest

from quantmind.synthetic import (
    SyntheticConfig,
    SyntheticFuturesGenerator,
    block_bootstrap_returns,
    directional_intraday_null,
    inject_feature_conditioned_edge,
    inject_gap_lookahead_canary,
)
from quantmind.synthetic.generator import _causal_signal


def _session_return_matrix(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].to_numpy(dtype=float)
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    sessions = pd.to_datetime(df["timestamp"]).dt.normalize()
    matrix = []
    for _, idx in sessions.groupby(sessions, sort=False).groups.items():
        matrix.append(returns[np.asarray(idx, dtype=int)])
    return np.vstack(matrix)


def _squared_return_lag1_autocorr(df: pd.DataFrame) -> float:
    matrix = _session_return_matrix(df)
    x = matrix[:, 1:].reshape(-1) ** 2
    y = matrix[:, :-1].reshape(-1) ** 2
    return float(np.corrcoef(x, y)[0, 1])


def _intraday_abs_return_profile(df: pd.DataFrame) -> np.ndarray:
    matrix = _session_return_matrix(df)
    return np.mean(np.abs(matrix), axis=0)


def _directional_lag1_autocorr(df: pd.DataFrame) -> float:
    matrix = _session_return_matrix(df)
    x = matrix[:, 1:].reshape(-1)
    y = matrix[:, :-1].reshape(-1)
    return float(np.corrcoef(x, y)[0, 1])


def test_generator_is_reproducible_has_real_gaps_and_valid_ohlc():
    cfg = SyntheticConfig(sessions=10, bars_per_session=40, seed=123)
    a = SyntheticFuturesGenerator(cfg).generate()
    b = SyntheticFuturesGenerator(cfg).generate()

    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 400
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert (a["volume"] > 0).all()
    assert (a["open_interest"] > 0).all()

    # The old fixture had open[t] == close[t-1] everywhere; that would hide look-ahead fills.
    prev_close = a["close"].shift(1)
    gap_bps = (np.log(a["open"] / prev_close).abs() * 10000.0).dropna()
    assert gap_bps.max() > 0.1
    session_start = pd.to_datetime(a["timestamp"]).dt.time == pd.Timestamp("09:15:00").time()
    assert gap_bps.loc[session_start].max() > 5.0


def test_intraday_profile_is_smoothly_interpolated_not_tiled():
    cfg = SyntheticConfig(sessions=80, bars_per_session=60, seed=2)
    df = SyntheticFuturesGenerator(cfg).generate()
    profile = _intraday_abs_return_profile(df)
    thirds = [profile[:20].mean(), profile[20:40].mean(), profile[40:].mean()]
    assert thirds[1] > thirds[0]
    assert thirds[1] > thirds[2]


def test_volatility_is_near_configured_annualized_target():
    cfg = SyntheticConfig(sessions=250, bars_per_session=375, seed=8)
    df = SyntheticFuturesGenerator(cfg).generate()
    intraday = np.log(df["close"] / df["open"]).to_numpy()
    realized = np.std(intraday, ddof=1) * np.sqrt(252 * cfg.bars_per_session)
    assert 0.14 < realized < 0.22


def test_block_bootstrap_preserves_session_boundaries_profile_and_joint_fields():
    cfg = SyntheticConfig(sessions=100, bars_per_session=30, seed=5)
    original = SyntheticFuturesGenerator(cfg).generate()
    surrogate = block_bootstrap_returns(original, block_size=5, seed=44)

    assert surrogate["timestamp"].equals(original["timestamp"])
    assert surrogate["contract"].equals(original["contract"])
    assert surrogate["timestamp"].dt.normalize().equals(original["timestamp"].dt.normalize())
    assert surrogate["volume"].nunique() > 1
    assert surrogate["open_interest"].nunique() > 1

    original_profile = _intraday_abs_return_profile(original)
    surrogate_profile = _intraday_abs_return_profile(surrogate)
    ratio = surrogate_profile / np.maximum(original_profile, 1e-12)
    assert np.all((ratio > 0.55) & (ratio < 1.45))


def test_block_bootstrap_preserves_squared_return_autocorrelation():
    cfg = SyntheticConfig(sessions=140, bars_per_session=30, seed=15)
    original = SyntheticFuturesGenerator(cfg).generate()
    surrogate = block_bootstrap_returns(original, block_size=5, seed=99)

    original_ac = _squared_return_lag1_autocorr(original)
    surrogate_ac = _squared_return_lag1_autocorr(surrogate)
    assert original_ac > 0.0
    assert abs(surrogate_ac - original_ac) < 0.10


def test_directional_null_destroys_directional_predictability_but_preserves_squared_structure():
    cfg = SyntheticConfig(sessions=120, bars_per_session=40, seed=23)
    base = SyntheticFuturesGenerator(cfg).generate()
    null = directional_intraday_null(base, seed=91)

    null_dir = _directional_lag1_autocorr(null)
    base_sq = _squared_return_lag1_autocorr(base)
    null_sq = _squared_return_lag1_autocorr(null)
    assert abs(null_dir) < 0.03
    assert abs(null_sq - base_sq) < 0.03


def test_feature_conditioned_edge_is_observable_and_symmetric():
    cfg = SyntheticConfig(sessions=40, bars_per_session=40, seed=9)
    base = SyntheticFuturesGenerator(cfg).generate()
    controlled = inject_feature_conditioned_edge(base, net_edge_bps=2.0, round_trip_cost_bps=2.0, seed=1)

    signal = controlled["control_signal"].to_numpy(dtype=int)
    assert set(np.unique(signal)) <= {-1, 0, 1}
    assert np.any(signal == 1)
    assert np.any(signal == -1)
    # The strategy condition is computable from past data, not an unobservable random flag.
    assert "edge_condition" not in controlled.columns


def test_control_signal_recomputable_from_delivered_data():
    cfg = SyntheticConfig(sessions=60, bars_per_session=60, seed=17)
    base = SyntheticFuturesGenerator(cfg).generate()
    delivered = inject_feature_conditioned_edge(base, net_edge_bps=2.0, round_trip_cost_bps=2.0)

    recomputed = _causal_signal(
        delivered.drop(columns=["control_signal"]),
        calendar_session_bars=60,
    )
    assert np.array_equal(recomputed, delivered["control_signal"].to_numpy(dtype=int))


def test_null_and_control_keep_independent_wicks():
    cfg = SyntheticConfig(sessions=60, bars_per_session=60, seed=29)
    base = SyntheticFuturesGenerator(cfg).generate()

    for df in (
        directional_intraday_null(base, seed=91),
        inject_feature_conditioned_edge(base, net_edge_bps=2.0, round_trip_cost_bps=2.0),
    ):
        up = np.log(df["high"] / np.maximum(df["open"], df["close"]))
        body = np.abs(np.log(df["close"] / df["open"]))
        corr = float(np.corrcoef(up.to_numpy(), body.to_numpy())[0, 1])
        assert abs(corr) < 0.1



def test_positive_control_is_sparse_and_does_not_create_dense_momentum():
    cfg = SyntheticConfig(sessions=250, bars_per_session=375, seed=55)
    base = SyntheticFuturesGenerator(cfg).generate()
    controlled = inject_feature_conditioned_edge(base, net_edge_bps=10.0)
    signal = controlled["control_signal"].to_numpy(dtype=int)
    sessions = pd.to_datetime(controlled["timestamp"]).dt.normalize().to_numpy()
    counts = [int(np.count_nonzero(signal[sessions == day])) for day in pd.unique(sessions)]
    bar_returns = np.log(controlled["close"].to_numpy(dtype=float) / controlled["open"].to_numpy(dtype=float))
    autocorr = float(np.corrcoef(bar_returns[:-1], bar_returns[1:])[0, 1])

    assert 1.0 <= float(np.mean(counts)) <= 3.0
    assert float(np.median(counts)) <= 3.0
    assert abs(autocorr) <= 0.05

def test_positive_control_edge_ladder_is_monotonic():
    cfg = SyntheticConfig(sessions=120, bars_per_session=60, seed=12)
    base = SyntheticFuturesGenerator(cfg).generate()
    results = []
    for edge in (0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
        controlled = inject_feature_conditioned_edge(base, net_edge_bps=edge, round_trip_cost_bps=2.0)
        signal = controlled["control_signal"].to_numpy(dtype=int)
        opens = controlled["open"].to_numpy(dtype=float)
        closes = controlled["close"].to_numpy(dtype=float)
        triggered = np.flatnonzero(signal[:-1] != 0) + 1
        signed_returns = signal[triggered - 1] * np.log(closes[triggered] / opens[triggered]) * 10000.0
        net_mean = float(np.mean(signed_returns) - 2.0)
        results.append(net_mean)
    assert np.all(np.diff(results) > 0)
    assert results[-1] > 0.0


def test_lookahead_canary_separates_same_bar_bug_from_next_open_fill():
    cfg = SyntheticConfig(sessions=40, bars_per_session=40, seed=77)
    base = SyntheticFuturesGenerator(cfg).generate()
    canary = inject_gap_lookahead_canary(base, magnitude_bps=30.0)

    signal = canary["canary_signal"].to_numpy(dtype=int)
    base_closes = base["close"].to_numpy(dtype=float)
    base_opens = base["open"].to_numpy(dtype=float)
    closes = canary["close"].to_numpy(dtype=float)
    opens = canary["open"].to_numpy(dtype=float)

    triggered = np.flatnonzero(signal[:-1] != 0)
    assert len(triggered) > 10

    # Correct engine: enter after the gap at open(t+1), then exit at close(t+1).
    # The canary must leave this open-to-close return unchanged relative to base.
    correct_canary = []
    correct_base = []
    forbidden_same_close_delta = []
    for t in triggered:
        direction = signal[t]
        correct_canary.append(direction * np.log(closes[t + 1] / opens[t + 1]) * 10000.0)
        correct_base.append(direction * np.log(base_closes[t + 1] / base_opens[t + 1]) * 10000.0)
        forbidden = direction * np.log(closes[t + 1] / closes[t]) * 10000.0
        base_forbidden = direction * np.log(base_closes[t + 1] / base_closes[t]) * 10000.0
        forbidden_same_close_delta.append(forbidden - base_forbidden)

    assert abs(np.mean(correct_canary) - np.mean(correct_base)) < 0.1
    assert np.mean(forbidden_same_close_delta) > 20.0


def test_invalid_synthetic_parameters_raise():
    with pytest.raises(ValueError):
        SyntheticFuturesGenerator(SyntheticConfig(sessions=0))

    with pytest.raises(ValueError):
        SyntheticFuturesGenerator(SyntheticConfig(annualized_sigma=0.0))

    with pytest.raises(ValueError):
        block_bootstrap_returns(pd.DataFrame({"close": [1.0, 1.1]}), block_size=0)

    with pytest.raises(ValueError):
        block_bootstrap_returns(
            pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=2), "close": [1.0, 1.1]}),
            block_size=3,
        )


def test_positive_control_edge_ladder_is_monotonic_in_mean_net_return():
    cfg = SyntheticConfig(sessions=250, bars_per_session=375, seed=123)
    base = SyntheticFuturesGenerator(cfg).generate()
    ladder = (0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
    measurements = []
    for edge in ladder:
        controlled = inject_feature_conditioned_edge(base, net_edge_bps=edge, round_trip_cost_bps=2.0)
        signal = controlled["control_signal"].to_numpy(dtype=int)
        idx = np.flatnonzero(signal[:-1] != 0) + 1
        signed_net = signal[idx - 1] * np.log(
            controlled["close"].to_numpy()[idx] / controlled["open"].to_numpy()[idx]
        ) * 10000.0 - 2.0
        mean_net = float(np.mean(signed_net))
        t_stat = float(mean_net / (np.std(signed_net, ddof=1) / np.sqrt(len(signed_net))))
        measurements.append((edge, len(idx), mean_net, t_stat))

    assert all(b[2] > a[2] for a, b in zip(measurements, measurements[1:]))
    assert all(np.isfinite(row[3]) for row in measurements)
    assert measurements[-1][2] > 0.0

    # Detectability thresholds belong to the eventual research-integrity pipeline.
    # This fixture only verifies that increasing the injected edge increases the
    # observed net-return signal; it must not bake fixture-size-specific t-statistics
    # into the tests.


def test_causal_control_is_truncation_invariant_with_calendar_session_length():
    cfg = SyntheticConfig(sessions=25, bars_per_session=60, seed=120)
    base = SyntheticFuturesGenerator(cfg).generate()
    delivered = inject_feature_conditioned_edge(base, net_edge_bps=2.0, calendar_session_bars=60)
    full = _causal_signal(delivered.drop(columns=["control_signal"]), calendar_session_bars=60)
    for cut in (71, 143, 311, 599, len(delivered) - 1):
        truncated = delivered.iloc[:cut].drop(columns=["control_signal"])
        prefix = _causal_signal(truncated, calendar_session_bars=60)
        assert np.array_equal(prefix, full[:cut])
