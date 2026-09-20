from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

from .strategies import assert_causal_signal

import numpy as np
import pandas as pd

SignalFunction = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class CostSchedulePeriod:
    effective_from: date
    effective_to: date | None
    round_trip_bps: float

    def contains(self, on_date: date) -> bool:
        return self.effective_from <= on_date and (
            self.effective_to is None or on_date <= self.effective_to
        )


@dataclass(frozen=True)
class CostSchedule:
    schedule_id: str
    periods: tuple[CostSchedulePeriod, ...]

    def round_trip_bps_at(self, timestamp: datetime | pd.Timestamp) -> float:
        on_date = timestamp.date() if hasattr(timestamp, "date") else timestamp
        matches = [period for period in self.periods if period.contains(on_date)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one cost schedule period for {on_date}")
        return matches[0].round_trip_bps


@dataclass(frozen=True)
class BacktestConfig:
    execution_model: str = "next_bar_open_v1"
    hold_model: str = "one_bar_v1"
    quantity: int = 1
    lot_size: int = 1
    slippage_bps_per_side: float = 0.0
    cost_schedule: CostSchedule | None = None
    signal_column: str | None = None
    enforce_session_boundaries: bool = True
    causality_preflight: bool = False
    causality_cut_points: tuple[int, ...] = ()


@dataclass(frozen=True)
class BacktestTrade:
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    side: int
    quantity: int
    entry_price: float
    exit_price: float
    gross_return_bps: float
    cost_bps: float
    net_return_bps: float
    gross_pnl: float
    costs: float
    net_pnl: float


@dataclass(frozen=True)
class BacktestResult:
    execution_model: str
    trades: tuple[BacktestTrade, ...]
    gross_pnl: float
    costs: float
    net_pnl: float
    mean_gross_return_bps: float
    mean_net_return_bps: float

    @property
    def trade_count(self) -> int:
        return len(self.trades)


class BacktestEngine:
    """Small deterministic one-bar futures backtest engine.

    The MVP intentionally keeps the interface narrow. A signal observed at the
    close of bar t is filled at bar t+1 open under ``next_bar_open_v1`` and exits
    at bar t+1 close. ``same_bar_close_v0`` exists only as a deliberate canary
    for catching forbidden same-close look-ahead behaviour.
    """

    REQUIRED_COLUMNS = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    }

    def run(self, *args, **kwargs):
        raise RuntimeError("BacktestEngine.run is internal; use ResearchHarness.run_trial")

    def _run_internal(
        self,
        data: pd.DataFrame,
        *,
        config: BacktestConfig | None = None,
        signal_fn: SignalFunction | None = None,
    ) -> BacktestResult:
        """Internal deterministic engine entry point; ResearchHarness owns research execution."""
        cfg = config or BacktestConfig()
        self._validate(data, cfg, signal_fn)
        frame = data.reset_index(drop=True).copy()
        signals = self._resolve_signals(frame, cfg, signal_fn)
        timestamps = pd.to_datetime(frame["timestamp"])
        opens = frame["open"].to_numpy(dtype=float)
        closes = frame["close"].to_numpy(dtype=float)

        n = len(frame)
        if n < 2:
            return BacktestResult(
                execution_model=cfg.execution_model,
                trades=(), gross_pnl=0.0, costs=0.0, net_pnl=0.0,
                mean_gross_return_bps=0.0, mean_net_return_bps=0.0,
            )

        session_keys = timestamps.dt.normalize().to_numpy()
        eligible = signals[:-1] != 0
        if cfg.enforce_session_boundaries:
            eligible &= session_keys[:-1] == session_keys[1:]
        indices = np.flatnonzero(eligible)
        if len(indices) == 0:
            return BacktestResult(
                execution_model=cfg.execution_model,
                trades=(), gross_pnl=0.0, costs=0.0, net_pnl=0.0,
                mean_gross_return_bps=0.0, mean_net_return_bps=0.0,
            )

        sides = signals[indices].astype(int)
        entry_indices = indices + 1
        exit_indices = indices + 1
        if cfg.execution_model == "next_bar_open_v1":
            raw_entry = opens[entry_indices]
            raw_exit = closes[exit_indices]
        elif cfg.execution_model == "same_bar_close_v0":
            raw_entry = closes[indices]
            raw_exit = opens[entry_indices]
            exit_indices = entry_indices
        else:
            raise ValueError(f"unsupported execution model: {cfg.execution_model}")

        slip = cfg.slippage_bps_per_side / 10000.0
        entry_prices = raw_entry * (1.0 + sides * slip)
        exit_prices = raw_exit * (1.0 - sides * slip)
        gross_returns = sides * (exit_prices / entry_prices - 1.0) * 10000.0

        if cfg.cost_schedule is None:
            costs_bps = np.zeros(len(indices), dtype=float)
        else:
            entry_dates = pd.to_datetime(timestamps.iloc[entry_indices]).dt.date.to_numpy()
            costs_bps = np.asarray([cfg.cost_schedule.round_trip_bps_at(d) for d in entry_dates], dtype=float)

        net_returns = gross_returns - costs_bps
        multiplier = cfg.quantity * cfg.lot_size
        gross_pnls = sides * (exit_prices - entry_prices) * multiplier
        costs = entry_prices * multiplier * (costs_bps / 10000.0)
        net_pnls = gross_pnls - costs

        trades = tuple(
            BacktestTrade(
                signal_timestamp=pd.Timestamp(timestamps.iloc[i]),
                entry_timestamp=pd.Timestamp(timestamps.iloc[e]),
                exit_timestamp=pd.Timestamp(timestamps.iloc[x]),
                side=int(s), quantity=cfg.quantity,
                entry_price=float(ep), exit_price=float(xp),
                gross_return_bps=float(gr), cost_bps=float(cb),
                net_return_bps=float(nr), gross_pnl=float(gp),
                costs=float(c), net_pnl=float(npnl),
            )
            for i, e, x, s, ep, xp, gr, cb, nr, gp, c, npnl in zip(
                indices, entry_indices, exit_indices, sides, entry_prices, exit_prices,
                gross_returns, costs_bps, net_returns, gross_pnls, costs, net_pnls
            )
        )
        return BacktestResult(
            execution_model=cfg.execution_model,
            trades=trades,
            gross_pnl=float(gross_pnls.sum()),
            costs=float(costs.sum()),
            net_pnl=float(net_pnls.sum()),
            mean_gross_return_bps=float(gross_returns.mean()),
            mean_net_return_bps=float(net_returns.mean()),
        )

    @staticmethod
    def default_causality_cut_points(length: int) -> tuple[int, ...]:
        if length <= 10:
            return tuple(range(2, length + 1))
        raw = np.linspace(2, length - 1, num=min(8, length - 2), dtype=int)
        return tuple(sorted(set(int(v) for v in raw)))

    @staticmethod
    def _validate(data: pd.DataFrame, config: BacktestConfig, signal_fn: SignalFunction | None) -> None:
        missing = BacktestEngine.REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"missing backtest columns: {sorted(missing)}")
        if len(data) < 2:
            raise ValueError("at least two bars are required")
        if config.quantity <= 0 or config.lot_size <= 0:
            raise ValueError("quantity and lot_size must be positive")
        if config.slippage_bps_per_side < 0:
            raise ValueError("slippage_bps_per_side must be >= 0")
        if config.signal_column is None and signal_fn is None:
            raise ValueError("signal_column or signal_fn is required")
        if config.signal_column is not None and config.signal_column not in data.columns:
            raise ValueError(f"signal column not found: {config.signal_column}")

    @staticmethod
    def _resolve_signals(
        frame: pd.DataFrame,
        config: BacktestConfig,
        signal_fn: SignalFunction | None,
    ) -> np.ndarray:
        raw = signal_fn(frame) if signal_fn is not None else frame[config.signal_column].to_numpy(dtype=float)
        signals = np.asarray(raw, dtype=float)
        if signals.shape != (len(frame),):
            raise ValueError("signal function must return one value per bar")
        if np.any(~np.isfinite(signals)):
            raise ValueError("signals must be finite")
        return np.sign(signals).astype(int)

    @staticmethod
    def _slippage_adjust(price: float, *, side: int, bps: float) -> float:
        # Positive side is a buy; negative side is a sell.
        return float(price * (1.0 + side * bps / 10000.0))
