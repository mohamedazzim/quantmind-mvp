"""Market Data Replay Feed (PRD v3.9).

Provides deterministic streaming of recorded market data through a normalized
market-data interface, compatible with future live market feeds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from quantmind.data.registry import DatasetKind, DatasetRegistry
from quantmind.data.splits import SplitZone


class MarketFeedSecurityError(RuntimeError):
    """Raised when an illegal or unauthorized dataset partition is requested for replay."""


@dataclass(frozen=True)
class ReplayBar:
    """A normalized market-data bar streamed during replay."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float
    session_id: str
    bar_index: int


class MarketDataFeed(ABC):
    """Normalized interface for market data feeds (replay and future live feeds)."""

    @property
    @abstractmethod
    def symbol(self) -> str:
        """Trading symbol or instrument identifier."""

    @property
    @abstractmethod
    def lot_size(self) -> int:
        """Contract lot size multiplier."""

    @property
    @abstractmethod
    def tick_size(self) -> float:
        """Minimum price movement."""

    @abstractmethod
    def stream_bars(self) -> Iterator[ReplayBar]:
        """Stream normalized market bars chronologically."""


class ReplayFeed(MarketDataFeed):
    """Deterministic recorded-data replay feed.

    Guarantees:
    - Chronological ordering of bars
    - Fast streaming via pre-extracted NumPy arrays (no per-bar .iloc calls)
    - Session boundary tracking
    - Pause / resume capability at the API level
    - Strict prohibition of FINAL_HOLDOUT partition access for research replay
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        symbol: str = "NIFTY_FUT",
        lot_size: int = 50,
        tick_size: float = 0.05,
        expiry_date: date | str | None = None,
        dataset_version: str = "",
        split_zone: SplitZone | str = SplitZone.FORWARD_PAPER,
    ) -> None:
        zone_val = split_zone.value if isinstance(split_zone, SplitZone) else str(split_zone)
        if zone_val == SplitZone.FINAL_HOLDOUT.value:
            raise MarketFeedSecurityError(
                "ReplayFeed is prohibited from accessing the sealed FINAL_HOLDOUT zone directly"
            )

        if len(data) == 0:
            raise ValueError("ReplayFeed requires non-empty market data")

        required_cols = {"timestamp", "open", "high", "low", "close"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"ReplayFeed data missing required columns: {sorted(missing_cols)}")

        # Check for NaNs
        for col in required_cols:
            if data[col].isna().any():
                raise ValueError(f"ReplayFeed data contains NaN values in column '{col}'")

        # Check for non-finite / infs in numeric columns
        for col in ("open", "high", "low", "close"):
            numeric_vals = pd.to_numeric(data[col], errors="coerce")
            if not np.isfinite(numeric_vals).all():
                raise ValueError(f"ReplayFeed data contains non-finite or infinite values in column '{col}'")

        # Check timestamp chronological ordering and duplicates
        ts_series = pd.to_datetime(data["timestamp"])
        if ts_series.duplicated().any():
            raise ValueError("ReplayFeed data contains duplicate timestamps")
        if not ts_series.is_monotonic_increasing:
            raise ValueError("ReplayFeed data must be sorted in strictly increasing chronological order")

        # Check OHLC price sanity
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)

        if (opens <= 0).any() or (highs <= 0).any() or (lows <= 0).any() or (closes <= 0).any():
            raise ValueError("ReplayFeed prices must be strictly positive")

        if (highs < lows).any() or (highs < opens).any() or (highs < closes).any() or (lows > opens).any() or (lows > closes).any():
            raise ValueError("ReplayFeed OHLC invariant violated: high must be >= open, low, close and low must be <= open, high, close")

        self._symbol = symbol
        self._lot_size = lot_size
        self._tick_size = tick_size
        self._expiry_date = expiry_date
        self._dataset_version = dataset_version
        self._split_zone = zone_val
        self._is_paused = False

        # Pre-extract columnar NumPy arrays for fast zero-overhead iteration
        frame = data.reset_index(drop=True)
        self._timestamps = ts_series.to_numpy()
        self._opens = frame["open"].to_numpy(dtype=float)
        self._highs = frame["high"].to_numpy(dtype=float)
        self._lows = frame["low"].to_numpy(dtype=float)
        self._closes = frame["close"].to_numpy(dtype=float)
        self._volumes = frame["volume"].to_numpy(dtype=float) if "volume" in frame.columns else np.zeros(len(frame), dtype=float)
        self._open_interests = (
            frame["open_interest"].to_numpy(dtype=float) if "open_interest" in frame.columns else np.zeros(len(frame), dtype=float)
        )

        ts_series = pd.Series(self._timestamps)
        self._session_ids = ts_series.dt.normalize().dt.strftime("%Y-%m-%d").to_numpy()
        self._n_bars = len(frame)
        self._cursor = 0

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def lot_size(self) -> int:
        return self._lot_size

    @property
    def tick_size(self) -> float:
        return self._tick_size

    @property
    def expiry_date(self) -> date | str | None:
        return self._expiry_date

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    @property
    def split_zone(self) -> str:
        return self._split_zone

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def total_bars(self) -> int:
        return self._n_bars

    def pause(self) -> None:
        """Pause playback stream."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume playback stream."""
        self._is_paused = False

    def reset(self) -> None:
        """Reset replay playback to the beginning."""
        self._cursor = 0
        self._is_paused = False

    def stream_bars(self) -> Iterator[ReplayBar]:
        """Iterate chronologically over replay bars."""
        while self._cursor < self._n_bars:
            if self._is_paused:
                return

            i = self._cursor
            bar = ReplayBar(
                timestamp=pd.Timestamp(self._timestamps[i]),
                open=float(self._opens[i]),
                high=float(self._highs[i]),
                low=float(self._lows[i]),
                close=float(self._closes[i]),
                volume=float(self._volumes[i]),
                open_interest=float(self._open_interests[i]),
                session_id=str(self._session_ids[i]),
                bar_index=i,
            )
            self._cursor += 1
            yield bar

    def __iter__(self) -> Iterator[ReplayBar]:
        return self.stream_bars()

    @classmethod
    def from_dataset_registry(
        cls,
        registry: DatasetRegistry,
        dataset_version: str,
        *,
        split_zone: SplitZone | str = SplitZone.FORWARD_PAPER,
        symbol: str = "NIFTY_FUT",
        lot_size: int = 50,
        tick_size: float = 0.05,
        expiry_date: date | str | None = None,
        require_licensed: bool = True,
    ) -> ReplayFeed:
        """Create a ReplayFeed directly from an authoritative DatasetRegistry entry."""
        record = registry.get(dataset_version)
        if record is None:
            raise MarketFeedSecurityError(f"Dataset '{dataset_version}' not found in registry")

        # Verify disk checksum against registry record
        record = registry.verify(dataset_version)

        if require_licensed and record.kind is not DatasetKind.LICENSED:
            raise MarketFeedSecurityError(
                f"Production replay requires LICENSED dataset, but '{dataset_version}' has kind '{record.kind.value}'"
            )

        zone_val = split_zone.value if isinstance(split_zone, SplitZone) else str(split_zone)
        if zone_val == SplitZone.FINAL_HOLDOUT.value:
            raise MarketFeedSecurityError(
                "ReplayFeed is prohibited from accessing the sealed FINAL_HOLDOUT zone directly"
            )

        if record.zones and any(z.name == zone_val for z in record.zones):
            df = registry.load_zone(dataset_version, zone_val)
        else:
            df = registry._read_file(record.path, record.format, record.sha256)

        return cls(
            df,
            symbol=symbol,
            lot_size=lot_size,
            tick_size=tick_size,
            expiry_date=expiry_date,
            dataset_version=dataset_version,
            split_zone=split_zone,
        )
