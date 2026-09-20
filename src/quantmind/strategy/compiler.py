from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from quantmind.backtest.strategies import prior_bar_momentum_signal
from .spec import StrategySpec, StrategySpecError

SignalFunction = Callable[[pd.DataFrame], np.ndarray]


SUPPORTED_SIGNALS = frozenset({
    "current_bar_momentum",
    "current_bar_mean_reversion",
})

_PARAMETER_KEYS = {
    "current_bar_momentum": {"calendar_session_bars", "session_window"},
    "current_bar_mean_reversion": {"calendar_session_bars", "session_window"},
}


def _require_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise StrategySpecError(f"{name} must be an int, not {type(value).__name__}")
    return value


def _require_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategySpecError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise StrategySpecError(f"{name} must be finite")
    return value


def normalize_strategy_spec(spec: StrategySpec) -> StrategySpec:
    """Validate and normalize the typed parameter schema before reservation."""
    if spec.signal_name not in SUPPORTED_SIGNALS:
        raise StrategySpecError(
            f"unsupported signal_name={spec.signal_name!r}; allowed={sorted(SUPPORTED_SIGNALS)}"
        )
    params = dict(spec.parameters)
    expected = _PARAMETER_KEYS[spec.signal_name]
    unknown = set(params) - expected
    missing = expected - set(params)
    if unknown:
        raise StrategySpecError(
            f"unsupported parameters for {spec.signal_name}: {sorted(unknown)}"
        )
    if missing:
        raise StrategySpecError(
            f"missing parameters for {spec.signal_name}: {sorted(missing)}"
        )

    calendar_session_bars = _require_int(
        params["calendar_session_bars"], name="calendar_session_bars"
    )
    if calendar_session_bars <= 1:
        raise StrategySpecError("calendar_session_bars must be > 1")

    window_raw = params["session_window"]
    if not isinstance(window_raw, (list, tuple)) or len(window_raw) != 2:
        raise StrategySpecError("session_window must contain exactly two numeric values")
    start = _require_real(window_raw[0], name="session_window[0]")
    end = _require_real(window_raw[1], name="session_window[1]")
    if not 0.0 <= start < end <= 1.0:
        raise StrategySpecError("session_window must satisfy 0 <= start < end <= 1")

    normalized_parameters = {
        "calendar_session_bars": calendar_session_bars,
        "session_window": [start, end],
    }
    return StrategySpec(
        strategy_version=spec.strategy_version,
        feature_version=spec.feature_version,
        signal_name=spec.signal_name,
        parameters=normalized_parameters,
    )


def _parse_parameters(spec: StrategySpec) -> tuple[int, tuple[float, float]]:
    normalized = normalize_strategy_spec(spec)
    return (
        int(normalized.parameters["calendar_session_bars"]),
        tuple(normalized.parameters["session_window"]),
    )


class StrategyCompiler:
    """Compile only validated, whitelisted declarative strategy specifications."""

    @staticmethod
    def compile(spec: StrategySpec) -> SignalFunction:
        normalized = normalize_strategy_spec(spec)
        calendar_session_bars, session_window = _parse_parameters(normalized)

        def current_bar_momentum(df: pd.DataFrame) -> np.ndarray:
            return prior_bar_momentum_signal(
                df,
                session_window=session_window,
                calendar_session_bars=calendar_session_bars,
            )

        if normalized.signal_name == "current_bar_momentum":
            return current_bar_momentum

        def current_bar_mean_reversion(df: pd.DataFrame) -> np.ndarray:
            return -current_bar_momentum(df)

        return current_bar_mean_reversion


def compile_strategy_spec(spec: StrategySpec) -> SignalFunction:
    return StrategyCompiler.compile(spec)
