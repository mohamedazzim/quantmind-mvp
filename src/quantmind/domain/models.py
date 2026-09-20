from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class ContractSpecPeriod:
    effective_from: date
    effective_to: date | None
    lot_size: int
    tick_size: float

    def contains(self, on_date: date) -> bool:
        return self.effective_from <= on_date and (
            self.effective_to is None or on_date <= self.effective_to
        )


@dataclass(frozen=True)
class FuturesContract:
    symbol: str
    expiry: date
    specs: tuple[ContractSpecPeriod, ...]

    def spec_at(self, timestamp: date | datetime) -> ContractSpecPeriod:
        on_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
        matches = [spec for spec in self.specs if spec.contains(on_date)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one contract specification for {on_date}")
        return matches[0]


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    contract: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int


@dataclass(frozen=True)
class Order:
    order_id: str
    timestamp: datetime
    contract: str
    side: Side
    quantity: int
    order_type: str = "MARKET"


@dataclass(frozen=True)
class Fill:
    order_id: str
    timestamp: datetime
    contract: str
    side: Side
    quantity: int
    price: float
    slippage: float
    fees: float


@dataclass(frozen=True)
class Trial:
    trial_id: str
    experiment_id: str
    strategy_id: str
    dataset_version: str
    feature_version: str
    parameter_set: Mapping[str, Any]
    seed: int
    execution_model: str
    cost_model: str
    slippage_model: str
    timestamp_started: datetime
    timestamp_completed: datetime
    result: Mapping[str, Any]
    status: str
    split_zone: str = "RESEARCH"
