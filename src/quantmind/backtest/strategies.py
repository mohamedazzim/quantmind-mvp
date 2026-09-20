from __future__ import annotations

import numpy as np
import pandas as pd


class NonCausalSignalError(RuntimeError):
    """Raised when a signal changes after future rows are removed."""


def _session_blocks(df: pd.DataFrame) -> list[np.ndarray]:
    keys = pd.to_datetime(df["timestamp"]).dt.normalize()
    return [np.asarray(idx, dtype=int) for _, idx in keys.groupby(keys, sort=False).groups.items()]


def prior_bar_momentum_signal(
    df: pd.DataFrame,
    *,
    session_window: tuple[float, float] = (0.10, 0.90),
    calendar_session_bars: int,
) -> np.ndarray:
    """Return a causal current-bar direction signal for next-bar execution.

    Despite the historical function name, the signal at bar ``t`` uses the
    completed open-to-close return of bar ``t`` itself. The backtester then
    executes it at bar ``t+1``. ``calendar_session_bars`` must be supplied by
    a session calendar when truncation-invariant fractional session windows are
    required; otherwise the observed session length is used.
    """
    if not 0 <= session_window[0] < session_window[1] <= 1:
        raise ValueError("session_window must satisfy 0 < start < end <= 1")
    if calendar_session_bars <= 1:
        raise ValueError("calendar_session_bars must be > 1")

    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    intrabar = np.log(closes / opens)
    signal = np.zeros(len(df), dtype=int)
    for idx in _session_blocks(df):
        n = calendar_session_bars
        start = int(np.floor(session_window[0] * n))
        end = int(np.ceil(session_window[1] * n))
        for j in range(max(0, start), min(end, len(idx))):
            current_return = intrabar[idx[j]]
            signal[idx[j]] = 1 if current_return > 0 else -1 if current_return < 0 else 0
    return signal


def assert_causal_signal(
    df: pd.DataFrame,
    signal_fn,
    *,
    cut_points: list[int] | np.ndarray,
) -> None:
    """Raise if a signal changes any completed prefix when future rows are removed."""
    full = np.asarray(signal_fn(df), dtype=int)
    if full.shape != (len(df),):
        raise ValueError("signal_fn must return one signal per row")

    for cut in sorted({int(c) for c in cut_points}):
        if cut <= 1 or cut > len(df):
            raise ValueError(f"cut point out of range: {cut}")
        truncated = df.iloc[:cut].copy()
        prefix = np.asarray(signal_fn(truncated), dtype=int)
        if prefix.shape != (cut,):
            raise ValueError("signal_fn returned invalid truncated shape")
        if not np.array_equal(prefix, full[:cut]):
            raise NonCausalSignalError(f"non-causal signal detected at cut={cut}")
