from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    sessions: int = 250
    bars_per_session: int = 375
    start: str = "2022-01-03 09:15:00"
    seed: int = 7
    start_price: float = 18000.0
    annualized_sigma: float = 0.18
    intraday_volatility_profile: tuple[float, ...] = (0.75, 1.0, 1.15, 0.9)
    overnight_gap_sigma_bps: float = 18.0
    intraday_open_jitter_bps: float = 2.5
    wick_scale_bps: float = 7.0


class SyntheticFuturesGenerator:
    """Generate deterministic NIFTY-futures-like 1-minute bars for tests.

    This fixture intentionally models:
    - volatility clustering;
    - a smooth intraday volatility profile;
    - overnight and within-session open gaps;
    - independent wick noise;
    - jointly generated volume and open interest.

    It is a research-integrity fixture, not a market simulator.
    """

    def __init__(self, config: SyntheticConfig):
        if config.sessions <= 0 or config.bars_per_session <= 1:
            raise ValueError("sessions and bars_per_session must be positive")
        if config.annualized_sigma <= 0:
            raise ValueError("annualized_sigma must be > 0")
        if not config.intraday_volatility_profile:
            raise ValueError("intraday_volatility_profile must not be empty")
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _intraday_profile(self) -> np.ndarray:
        controls = np.asarray(self.config.intraday_volatility_profile, dtype=float)
        if np.any(controls <= 0):
            raise ValueError("intraday_volatility_profile values must be > 0")
        x_old = np.linspace(0.0, 1.0, len(controls))
        x_new = np.linspace(0.0, 1.0, self.config.bars_per_session)
        profile = np.interp(x_new, x_old, controls)
        return profile / profile.mean()

    def generate_returns(self) -> np.ndarray:
        """Generate open-to-close log returns with GARCH-like clustering."""
        n = self.config.sessions * self.config.bars_per_session
        z = self.rng.standard_normal(n)

        bars_per_year = 252 * self.config.bars_per_session
        target_bar_var = (self.config.annualized_sigma / np.sqrt(bars_per_year)) ** 2
        omega = target_bar_var * (1.0 - 0.08 - 0.90)
        alpha = 0.08
        beta = 0.90

        sigma = np.empty(n, dtype=float)
        returns = np.empty(n, dtype=float)
        sigma[0] = np.sqrt(target_bar_var)
        returns[0] = sigma[0] * z[0]
        for i in range(1, n):
            variance = omega + alpha * returns[i - 1] ** 2 + beta * sigma[i - 1] ** 2
            sigma[i] = np.sqrt(max(variance, 1e-16))
            returns[i] = sigma[i] * z[i]

        returns *= np.tile(self._intraday_profile(), self.config.sessions)
        return returns

    def _timestamps(self) -> pd.DatetimeIndex:
        cfg = self.config
        start = pd.Timestamp(cfg.start)
        session_days = pd.bdate_range(start=start.normalize(), periods=cfg.sessions, freq="B")
        timestamps: list[pd.Timestamp] = []
        for session_day in session_days:
            session_start = session_day + pd.Timedelta(hours=9, minutes=15)
            timestamps.extend(
                session_start + pd.to_timedelta(np.arange(cfg.bars_per_session), unit="min")
            )
        return pd.DatetimeIndex(timestamps)

    def generate(self, contract: str = "NIFTY_FUT") -> pd.DataFrame:
        cfg = self.config
        returns = self.generate_returns()
        total = len(returns)
        timestamps = self._timestamps()

        session_ids = np.repeat(np.arange(cfg.sessions), cfg.bars_per_session)
        first_in_session = np.zeros(total, dtype=bool)
        first_in_session[:: cfg.bars_per_session] = True

        # Overnight gaps are larger; normal within-session opens also contain small jitter.
        gap_sigmas = np.full(total, cfg.intraday_open_jitter_bps / 10000.0)
        gap_sigmas[first_in_session] = cfg.overnight_gap_sigma_bps / 10000.0
        open_gap = self.rng.normal(0.0, gap_sigmas)

        close = np.empty(total, dtype=float)
        open_price = np.empty(total, dtype=float)
        previous_close = cfg.start_price
        for i in range(total):
            open_price[i] = previous_close * np.exp(open_gap[i])
            close[i] = open_price[i] * np.exp(returns[i])
            previous_close = close[i]

        # Independent wicks: high/low are not deterministic functions of close/prev-close.
        wick_up = np.abs(self.rng.normal(0.0, cfg.wick_scale_bps / 10000.0, total))
        wick_down = np.abs(self.rng.normal(0.0, cfg.wick_scale_bps / 10000.0, total))
        body_high = np.maximum(open_price, close)
        body_low = np.minimum(open_price, close)
        high = body_high * np.exp(wick_up)
        low = body_low * np.exp(-wick_down)

        base_volume = np.exp(self.rng.normal(np.log(1000), 0.35, total)).astype(int)
        oi_changes = self.rng.normal(0, 200, total)
        open_interest = np.maximum(100000, 250000 + np.cumsum(oi_changes)).astype(int)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "contract": contract,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": base_volume,
                "open_interest": open_interest,
            }
        )


def _session_blocks(df: pd.DataFrame) -> list[np.ndarray]:
    """Return contiguous integer index arrays, one for each timestamped session."""
    timestamps = pd.to_datetime(df["timestamp"], errors="raise")
    session_keys = timestamps.dt.normalize()
    groups: list[np.ndarray] = []
    for _, idx in session_keys.groupby(session_keys, sort=False).groups.items():
        groups.append(np.asarray(idx, dtype=int))
    return groups


def _return_components(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bar open gap and open-to-close log-return components."""
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    previous_close = np.empty_like(closes)
    previous_close[0] = closes[0]
    previous_close[1:] = closes[:-1]
    gap = np.log(opens / previous_close)
    intrabar = np.log(closes / opens)
    return gap, intrabar


def _wick_log_ratios(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return log-distance wick ratios relative to each bar's body range."""
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    body_high = np.maximum(opens, closes)
    body_low = np.minimum(opens, closes)
    wick_up = np.log(highs / body_high)
    wick_down = np.log(body_low / lows)
    return wick_up, wick_down


def _rebuild_from_components(
    out: pd.DataFrame,
    gap: np.ndarray,
    intrabar: np.ndarray,
    *,
    start_price: float,
) -> pd.DataFrame:
    previous_close = start_price
    opens = np.empty_like(gap)
    closes = np.empty_like(gap)
    for i in range(len(gap)):
        opens[i] = previous_close * np.exp(gap[i])
        closes[i] = opens[i] * np.exp(intrabar[i])
        previous_close = closes[i]

    out["open"] = opens
    out["close"] = closes

    # Preserve sampled wick ratios if callers have supplied them; otherwise create
    # conservative deterministic wicks from the rebuilt body range.
    if "_wick_up" in out.columns and "_wick_down" in out.columns:
        body_high = np.maximum(opens, closes)
        body_low = np.minimum(opens, closes)
        out["high"] = body_high * np.exp(out.pop("_wick_up").to_numpy(dtype=float))
        out["low"] = body_low * np.exp(-out.pop("_wick_down").to_numpy(dtype=float))
    else:
        spread = np.maximum(np.abs(closes - opens) * 0.5, start_price * 1e-5)
        out["high"] = np.maximum(opens, closes) + spread
        out["low"] = np.minimum(opens, closes) - spread
    return out


def block_bootstrap_returns(
    df: pd.DataFrame,
    *,
    block_size: int = 5,
    seed: int = 11,
) -> pd.DataFrame:
    """Cross-session null: sample complete session blocks jointly.

    This null deliberately preserves within-session directional structure. It is
    intended for cross-session robustness tests, not for intraday directional-null
    hypothesis tests.

    OHLCV and open-interest observations are sampled together at the session level.
    """
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    out = df.copy().reset_index(drop=True)
    required = {"timestamp", "open", "high", "low", "close", "volume", "open_interest"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    sessions = _session_blocks(out)
    if len(sessions) < block_size:
        raise ValueError("block_size cannot exceed the number of sessions")

    source_gap, source_intrabar = _return_components(out)
    wick_up = np.log(
        out["high"].to_numpy(dtype=float)
        / np.maximum(out["open"].to_numpy(dtype=float), out["close"].to_numpy(dtype=float))
    )
    wick_down = np.log(
        np.minimum(out["open"].to_numpy(dtype=float), out["close"].to_numpy(dtype=float))
        / out["low"].to_numpy(dtype=float)
    )

    rng = np.random.default_rng(seed)
    sampled_indices: list[np.ndarray] = []
    while len(sampled_indices) < len(sessions):
        start = int(rng.integers(0, len(sessions) - block_size + 1))
        sampled_indices.extend(sessions[start : start + block_size])
    sampled_indices = sampled_indices[: len(sessions)]

    picked = np.concatenate(sampled_indices)
    sampled_gap = source_gap[picked]
    sampled_intrabar = source_intrabar[picked]

    out = out.iloc[: len(picked)].copy().reset_index(drop=True)
    out["volume"] = df["volume"].to_numpy(dtype=int)[picked]
    out["open_interest"] = df["open_interest"].to_numpy(dtype=int)[picked]
    out["_wick_up"] = wick_up[picked]
    out["_wick_down"] = wick_down[picked]
    return _rebuild_from_components(out, sampled_gap, sampled_intrabar, start_price=float(df["close"].iloc[0]))


def directional_intraday_null(
    df: pd.DataFrame,
    *,
    seed: int = 31,
) -> pd.DataFrame:
    """Destroy directional predictability while preserving volatility structure.

    Signed open gaps and open-to-close returns are independently sign-randomized
    within each session. Absolute return magnitudes remain attached to their original
    timestamps, so the intraday volatility profile and squared-return clustering are
    preserved. Volume and open interest stay attached to the same bars.
    """
    out = df.copy().reset_index(drop=True)
    required = {"timestamp", "open", "high", "low", "close", "volume", "open_interest"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    sessions = _session_blocks(out)
    gap, intrabar = _return_components(out)
    wick_up, wick_down = _wick_log_ratios(out)
    rng = np.random.default_rng(seed)

    null_gap = gap.copy()
    null_intrabar = intrabar.copy()
    for idx in sessions:
        null_gap[idx] *= rng.choice(np.array([-1.0, 1.0]), size=len(idx))
        null_intrabar[idx] *= rng.choice(np.array([-1.0, 1.0]), size=len(idx))

    out["_wick_up"] = wick_up
    out["_wick_down"] = wick_down
    return _rebuild_from_components(
        out,
        null_gap,
        null_intrabar,
        start_price=float(df["close"].iloc[0]),
    )


def _causal_signal(
    df: pd.DataFrame,
    *,
    session_window: tuple[float, float] = (0.10, 0.90),
    condition: str = "prior_volume_tail_oi_direction",
    tail_quantile: float = 0.993,
    lookback_sessions: int = 20,
    calendar_session_bars: int | None = None,
) -> np.ndarray:
    """Return decision-time signals derived from completed bars.

    Signal ``t`` is computed using only data available at the close of bar ``t``
    and is therefore intended for a next-bar-open execution model. When
    ``calendar_session_bars`` is supplied, fractional session windows are anchored
    to the full exchange session length rather than the observed (possibly truncated)
    input. The injected
    positive control applies its edge to bar ``t+1`` using that signal, so the
    signal cannot feed back into the feature used to create itself.
    """
    if not 0 <= session_window[0] < session_window[1] <= 1:
        raise ValueError("session_window must satisfy 0 < start < end <= 1")
    if not 0.5 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must be between 0.5 and 1.0")
    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be > 0")
    if calendar_session_bars is not None and calendar_session_bars <= 1:
        raise ValueError("calendar_session_bars must be > 1")
    if condition != "prior_volume_tail_oi_direction":
        raise ValueError(f"unsupported synthetic control condition: {condition}")

    volumes = df["volume"].to_numpy(dtype=float)
    oi = df["open_interest"].to_numpy(dtype=float)
    signal = np.zeros(len(df), dtype=int)
    sessions = _session_blocks(df)

    for session_idx, idx in enumerate(sessions):
        prior_sessions = sessions[max(0, session_idx - lookback_sessions) : session_idx]
        if not prior_sessions:
            continue
        prior_volume = np.concatenate([volumes[s] for s in prior_sessions])
        threshold = float(np.quantile(prior_volume, tail_quantile))

        n = calendar_session_bars if calendar_session_bars is not None else len(idx)
        start_slot = int(np.floor(session_window[0] * n))
        end_slot = int(np.ceil(session_window[1] * n))
        for j in range(max(1, start_slot), min(end_slot, len(idx))):
            decision_idx = idx[j]
            oi_delta = oi[decision_idx] - oi[decision_idx - 1]
            if volumes[decision_idx] >= threshold and oi_delta != 0:
                signal[decision_idx] = 1 if oi_delta > 0 else -1
    return signal


def inject_feature_conditioned_edge(
    df: pd.DataFrame,
    *,
    net_edge_bps: float,
    round_trip_cost_bps: float = 2.0,
    seed: int = 41,
    condition_column: str = "control_signal",
    session_window: tuple[float, float] = (0.10, 0.90),
    condition: str = "prior_volume_tail_oi_direction",
    tail_quantile: float = 0.993,
    lookback_sessions: int = 20,
    calendar_session_bars: int | None = None,
) -> pd.DataFrame:
    """Inject a sparse, observable, symmetric next-bar open-to-close edge.

    The decision signal is computed from the completed bar and written to that
    bar. Its edge is injected into the following bar's open-to-close return, which
    matches the deterministic next-bar-open execution model.
    """
    del seed
    if net_edge_bps < 0 or round_trip_cost_bps < 0:
        raise ValueError("edge and cost must be >= 0")

    out = df.copy().reset_index(drop=True)
    gap, intrabar = _return_components(out)
    wick_up, wick_down = _wick_log_ratios(out)
    signal = _causal_signal(
        out,
        session_window=session_window,
        condition=condition,
        tail_quantile=tail_quantile,
        lookback_sessions=lookback_sessions,
        calendar_session_bars=calendar_session_bars,
    )
    injected_intrabar = intrabar.copy()
    gross_edge_bps = net_edge_bps + round_trip_cost_bps

    sessions = _session_blocks(out)
    for idx in sessions:
        for j in range(len(idx) - 1):
            decision_idx = idx[j]
            target_idx = idx[j + 1]
            if signal[decision_idx] != 0:
                injected_intrabar[target_idx] += signal[decision_idx] * gross_edge_bps / 10000.0

    out[condition_column] = signal
    out["_wick_up"] = wick_up
    out["_wick_down"] = wick_down
    return _rebuild_from_components(
        out,
        gap,
        injected_intrabar,
        start_price=float(df["close"].iloc[0]),
    )

def inject_gap_lookahead_canary(
    df: pd.DataFrame,
    *,
    magnitude_bps: float = 25.0,
    seed: int = 51,
    condition_column: str = "canary_signal",
) -> pd.DataFrame:
    """Inject an edge entirely inside close(t)->open(t+1).

    A correct next-bar-open engine cannot capture this move because it enters only
    after the gap. A forbidden same-bar-close fill can capture it. The signal itself
    is based only on prior-bar information.
    """
    if magnitude_bps < 0:
        raise ValueError("magnitude_bps must be >= 0")

    out = df.copy().reset_index(drop=True)
    signal = _causal_signal(out)
    out[condition_column] = signal

    base_open = out["open"].to_numpy(dtype=float)
    base_close = out["close"].to_numpy(dtype=float)
    wick_up = np.abs(np.log(out["high"].to_numpy(dtype=float) / np.maximum(base_open, base_close)))
    wick_down = np.abs(np.log(np.minimum(base_open, base_close) / out["low"].to_numpy(dtype=float)))

    # Apply a persistent level shift from t+1 onward. This creates a single
    # close(t)->open(t+1) jump while preserving every affected bar's open-to-close
    # return. A correct next-open fill therefore cannot capture the jump.
    log_shift = np.zeros(len(out), dtype=float)
    cumulative = 0.0
    for i in range(len(out)):
        if i > 0 and signal[i - 1] != 0:
            cumulative += signal[i - 1] * magnitude_bps / 10000.0
        log_shift[i] = cumulative

    open_price = base_open * np.exp(log_shift)
    close_price = base_close * np.exp(log_shift)
    out["open"] = open_price
    out["close"] = close_price
    out["high"] = np.maximum(open_price, close_price) * np.exp(wick_up)
    out["low"] = np.minimum(open_price, close_price) * np.exp(-wick_down)
    return out


# Backward-compatible alias retained for early scaffold users.
def inject_next_bar_edge(
    df: pd.DataFrame,
    *,
    probability: float = 0.10,
    magnitude_bps: float = 12.0,
    seed: int = 21,
    condition_column: str = "edge_condition",
) -> pd.DataFrame:
    """Deprecated alias for the observable positive-control injector.

    ``probability`` is retained only for API compatibility; the new protocol uses
    the causal feature condition rather than an unobservable random flag.
    """
    del probability
    return inject_feature_conditioned_edge(
        df,
        net_edge_bps=max(0.0, magnitude_bps - 2.0),
        round_trip_cost_bps=2.0,
        seed=seed,
        condition_column=condition_column,
    )
