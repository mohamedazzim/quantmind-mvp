"""Deterministic Paper Replay Engine (PRD v3.9).

Executes historical or forward recorded-data replay of qualified strategies
through a normalized market-data interface with pre-trade risk controls.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Callable, Sequence
import uuid

import numpy as np
import pandas as pd

from quantmind.backtest.engine import CostSchedule
from quantmind.data.registry import DatasetKind, DatasetRegistry
from quantmind.data.splits import SplitZone
from quantmind.paper.feed import MarketFeedSecurityError, ReplayBar, ReplayFeed
from quantmind.paper.ledger import PaperLedger
from quantmind.paper.models import (
    PaperFill,
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    ReplayReport,
    ReplaySessionSummary,
)
from quantmind.paper.risk import PaperRiskConfig, PaperRiskEngine
from quantmind.research_integrity.holdout import HoldoutState
from quantmind.research_integrity.qualification import (
    StrategyQualificationRecord,
    ValidationStatus,
)
from quantmind.strategy.compiler import compile_strategy_spec, derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.spec import StrategySpec


class PaperReplaySecurityError(RuntimeError):
    """Raised when paper replay security constraints or qualification rules are violated."""


class PaperReplayEngine:
    """Deterministic paper execution engine.

    Guarantees:
    - Accepts ONLY valid StrategyQualificationRecords with PAPER_ELIGIBLE and holdout PASSED.
    - Raw StrategySpecs cannot bypass the qualification boundary.
    - Validates dataset and protocol consistency with the qualification record.
    - Executes locked next_bar_open semantics (signal at close(t) -> fill at open(t+1)).
    - Enforces deterministic pre-trade risk controls via PaperRiskEngine.
    - Strictly simulated; no broker credentials or live external routing endpoints exist.
    """

    def __init__(
        self,
        *,
        ledger: PaperLedger | None = None,
        risk_config: PaperRiskConfig | None = None,
        cost_schedule: CostSchedule | None = None,
        slippage_bps_per_side: float = 0.0,
        enforce_session_boundaries: bool = True,
        dataset_registry: DatasetRegistry | None = None,
        expected_protocol_version: str | None = None,
    ) -> None:
        self.ledger = ledger or PaperLedger()
        self.risk_engine = PaperRiskEngine(risk_config)
        self.cost_schedule = cost_schedule
        self.slippage_bps_per_side = slippage_bps_per_side
        self.enforce_session_boundaries = enforce_session_boundaries
        self.dataset_registry = dataset_registry
        self.expected_protocol_version = expected_protocol_version

    def _round_to_tick(self, price: float, tick_size: float) -> float:
        """Round price to nearest valid tick increment."""
        if tick_size <= 0:
            return price
        return round(round(price / tick_size) * tick_size, 4)

    def run_replay(
        self,
        qualification_record: StrategyQualificationRecord,
        strategy_spec: StrategySpec,
        feed: ReplayFeed,
        *,
        initial_capital: float = 100_000.0,
        quantity: int = 1,
        hold_bars: int = 1,
    ) -> ReplayReport:
        # 0. Strict type verification (fail closed against fake/duck-typed objects)
        if not isinstance(qualification_record, StrategyQualificationRecord):
            raise PaperReplaySecurityError(
                f"qualification_record must be an instance of StrategyQualificationRecord, got {type(qualification_record)}"
            )
        if not isinstance(strategy_spec, StrategySpec):
            raise PaperReplaySecurityError(
                f"strategy_spec must be an instance of StrategySpec, got {type(strategy_spec)}"
            )
        if not isinstance(feed, ReplayFeed):
            raise PaperReplaySecurityError(
                f"feed must be an instance of ReplayFeed, got {type(feed)}"
            )

        # 1. Authoritative qualification verification
        if not qualification_record.verify_digest():
            raise PaperReplaySecurityError(
                f"Qualification record '{qualification_record.qualification_id}' failed cryptographic digest verification"
            )

        if qualification_record.final_status != ValidationStatus.PAPER_ELIGIBLE:
            raise PaperReplaySecurityError(
                f"Strategy '{qualification_record.strategy_id}' has status '{qualification_record.final_status.value}'; "
                "requires 'PAPER_ELIGIBLE' to run paper replay"
            )

        if qualification_record.holdout_state != HoldoutState.PASSED.value:
            raise PaperReplaySecurityError(
                f"Strategy '{qualification_record.strategy_id}' has holdout state '{qualification_record.holdout_state}'; "
                "requires 'PASSED' to run paper replay"
            )

        # 2. Protocol version verification
        if (
            self.expected_protocol_version is not None
            and qualification_record.research_protocol_version != self.expected_protocol_version
        ):
            raise PaperReplaySecurityError(
                f"Qualification record research_protocol_version '{qualification_record.research_protocol_version}' "
                f"does not match expected protocol version '{self.expected_protocol_version}'"
            )

        # 3. Strategy specification identity verification
        norm_spec = normalize_strategy_spec(strategy_spec)
        derived_strat_id = derive_strategy_id(norm_spec)
        if qualification_record.strategy_id != derived_strat_id:
            raise PaperReplaySecurityError(
                f"Qualification record strategy_id '{qualification_record.strategy_id}' does not match "
                f"strategy spec derived ID '{derived_strat_id}'"
            )

        expected_spec_hash = hashlib.sha256(norm_spec.canonical_json().encode("utf-8")).hexdigest()
        if qualification_record.strategy_spec_hash != expected_spec_hash:
            raise PaperReplaySecurityError(
                f"Qualification record strategy_spec_hash '{qualification_record.strategy_spec_hash}' does not match "
                f"strategy spec hash '{expected_spec_hash}'"
            )

        # 4. Dataset scope & registry verification
        if not feed.dataset_version or feed.dataset_version != qualification_record.dataset_version:
            raise PaperReplaySecurityError(
                f"ReplayFeed dataset_version '{feed.dataset_version}' does not match qualification record "
                f"dataset_version '{qualification_record.dataset_version}'"
            )

        if self.dataset_registry is not None:
            reg_entry = self.dataset_registry.get(qualification_record.dataset_version)
            if reg_entry is None:
                raise PaperReplaySecurityError(
                    f"Qualification record dataset_version '{qualification_record.dataset_version}' is not registered"
                )
            if reg_entry.sha256 != qualification_record.dataset_sha256:
                raise PaperReplaySecurityError(
                    f"Qualification record dataset_sha256 '{qualification_record.dataset_sha256}' does not match "
                    f"registry dataset_sha256 '{reg_entry.sha256}'"
                )
            if reg_entry.kind is not DatasetKind.LICENSED:
                raise PaperReplaySecurityError(
                    f"Dataset '{qualification_record.dataset_version}' is kind '{reg_entry.kind.value}'; "
                    "synthetic/fixture datasets are barred from production paper replay"
                )

        # 5. Prohibit sealed holdout access
        if feed.split_zone == SplitZone.FINAL_HOLDOUT.value:
            raise MarketFeedSecurityError("Paper replay cannot execute against the sealed FINAL_HOLDOUT partition")

        # 6. Compile strategy signal function
        signal_fn = compile_strategy_spec(norm_spec)

        # Extract all bars for strategy signal generation
        feed.reset()
        bars: list[ReplayBar] = list(feed.stream_bars())
        if len(bars) < 2:
            return ReplayReport.create(
                strategy_id=derived_strat_id,
                qualification_id=qualification_record.qualification_id,
                dataset_version=qualification_record.dataset_version,
                trade_count=0,
                gross_pnl=0.0,
                net_pnl=0.0,
                costs=0.0,
                slippage=0.0,
                max_drawdown_bps=0.0,
                exposure=0.0,
                win_rate=0.0,
                expectancy=0.0,
                sharpe_ratio=None,
                session_breakdown=[],
                strategy_spec_hash=qualification_record.strategy_spec_hash,
                qualification_hash=qualification_record.record_hash,
                dataset_sha256=qualification_record.dataset_sha256,
                research_protocol_version=qualification_record.research_protocol_version,
                symbol=feed.symbol,
                lot_size=feed.lot_size,
                tick_size=feed.tick_size,
                slippage_bps_per_side=self.slippage_bps_per_side,
                cost_schedule_id=self.cost_schedule.schedule_id if self.cost_schedule else "",
                execution_policy="next_bar_open_v1",
            )

        df = pd.DataFrame(
            {
                "timestamp": [b.timestamp for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
                "open_interest": [b.open_interest for b in bars],
            }
        )
        signals = signal_fn(df)

        # 6. Sequential bar-by-bar execution loop
        lot_size = feed.lot_size
        tick_size = feed.tick_size
        slip_rate = self.slippage_bps_per_side / 10000.0

        current_position = PaperPosition(
            symbol=feed.symbol,
            quantity=0,
            entry_price=0.0,
            current_price=bars[0].open,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees_costs=0.0,
        )

        total_gross_pnl = 0.0
        total_costs = 0.0
        total_slippage = 0.0
        trade_pnls: list[float] = []

        pending_order: PaperOrder | None = None
        open_trade: dict[str, Any] | None = None
        session_summaries: dict[str, dict[str, Any]] = {}

        n_bars = len(bars)
        for i in range(n_bars):
            bar = bars[i]
            session_id = bar.session_id

            if session_id not in session_summaries:
                session_summaries[session_id] = {
                    "start_ts": bar.timestamp.isoformat(),
                    "end_ts": bar.timestamp.isoformat(),
                    "trades": 0,
                    "gross_pnl": 0.0,
                    "net_pnl": 0.0,
                }
            session_summaries[session_id]["end_ts"] = bar.timestamp.isoformat()

            # STEP A: Execute pending order at bar OPEN (next-bar-open execution)
            if pending_order is not None:
                order = pending_order
                pending_order = None

                # Pre-trade risk check
                risk_event = self.risk_engine.evaluate_order(
                    order=order,
                    current_position=current_position,
                    current_price=bar.open,
                    session_id=session_id,
                    lot_size=lot_size,
                )

                if risk_event is not None:
                    # Rejected by risk controls
                    rejected_order = PaperOrder(
                        order_id=order.order_id,
                        strategy_id=order.strategy_id,
                        qualification_id=order.qualification_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        order_type=order.order_type,
                        signal_timestamp=order.signal_timestamp,
                        submit_timestamp=order.submit_timestamp,
                        requested_price=order.requested_price,
                        status=PaperOrderStatus.REJECTED,
                        rejection_reason=risk_event.reason,
                    )
                    self.ledger.record_order(rejected_order)
                    self.ledger.record_risk_event(risk_event)
                else:
                    # Approved by risk controls -> generate execution fill
                    raw_open = bar.open
                    slip_amount = raw_open * slip_rate
                    fill_price = self._round_to_tick(
                        raw_open * (1.0 + order.side * slip_rate), tick_size
                    )
                    slippage_val = abs(fill_price - raw_open) * order.quantity * lot_size

                    # Calculate date-effective transaction costs
                    if self.cost_schedule is not None:
                        round_trip_bps = self.cost_schedule.round_trip_bps_at(bar.timestamp)
                        cost_bps = round_trip_bps / 2.0  # Per-side cost
                    else:
                        cost_bps = 0.0
                    fill_cost = (fill_price * order.quantity * lot_size) * (cost_bps / 10000.0)

                    fill = PaperFill(
                        fill_id=f"FILL-{order.order_id}",
                        order_id=order.order_id,
                        strategy_id=order.strategy_id,
                        symbol=order.symbol,
                        fill_timestamp=bar.timestamp.isoformat(),
                        fill_price=fill_price,
                        quantity=order.quantity,
                        side=order.side,
                        cost=fill_cost,
                        slippage=slippage_val,
                    )
                    filled_order = PaperOrder(
                        order_id=order.order_id,
                        strategy_id=order.strategy_id,
                        qualification_id=order.qualification_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        order_type=order.order_type,
                        signal_timestamp=order.signal_timestamp,
                        submit_timestamp=order.submit_timestamp,
                        requested_price=order.requested_price,
                        status=PaperOrderStatus.FILLED,
                    )
                    self.ledger.record_order(filled_order)
                    self.ledger.record_fill(fill)

                    total_costs += fill_cost
                    total_slippage += slippage_val

                    # Update position and trade accounting
                    if open_trade is None:
                        # Opening new trade
                        open_trade = {
                            "entry_bar": i,
                            "entry_price": fill_price,
                            "side": order.side,
                            "quantity": order.quantity,
                            "entry_cost": fill_cost,
                            "session_id": session_id,
                        }
                        current_position = PaperPosition(
                            symbol=feed.symbol,
                            quantity=order.side * order.quantity,
                            entry_price=fill_price,
                            current_price=bar.open,
                            realized_pnl=current_position.realized_pnl,
                            unrealized_pnl=0.0,
                            fees_costs=current_position.fees_costs + fill_cost,
                        )
                    else:
                        # Closing existing trade
                        trade_gross = (
                            open_trade["side"]
                            * (fill_price - open_trade["entry_price"])
                            * order.quantity
                            * lot_size
                        )
                        trade_net = trade_gross - (open_trade["entry_cost"] + fill_cost)
                        total_gross_pnl += trade_gross
                        trade_pnls.append(trade_net)

                        session_summaries[open_trade["session_id"]]["trades"] += 1
                        session_summaries[open_trade["session_id"]]["gross_pnl"] += trade_gross
                        session_summaries[open_trade["session_id"]]["net_pnl"] += trade_net

                        self.risk_engine.update_portfolio_state(
                            current_equity=initial_capital + sum(trade_pnls),
                            realized_pnl_delta=trade_net,
                        )

                        open_trade = None
                        current_position = PaperPosition(
                            symbol=feed.symbol,
                            quantity=0,
                            entry_price=0.0,
                            current_price=bar.open,
                            realized_pnl=current_position.realized_pnl + trade_net,
                            unrealized_pnl=0.0,
                            fees_costs=current_position.fees_costs + fill_cost,
                        )

            # STEP B: Evaluate current holding and signal at bar CLOSE
            current_signal = signals[i]
            next_is_different_session = (
                (i + 1 < n_bars) and (bars[i + 1].session_id != session_id)
            )

            # Exit on session boundary if enforced
            if open_trade is not None:
                bars_held = i - open_trade["entry_bar"] + 1
                should_exit = False
                if self.enforce_session_boundaries and next_is_different_session:
                    should_exit = True
                elif bars_held >= hold_bars:
                    should_exit = True

                if should_exit and pending_order is None and (i + 1 < n_bars):
                    exit_side = -open_trade["side"]
                    pending_order = PaperOrder(
                        order_id=f"ORD-EXIT-{i}",
                        strategy_id=derived_strat_id,
                        qualification_id=qualification_record.qualification_id,
                        symbol=feed.symbol,
                        side=exit_side,
                        quantity=open_trade["quantity"],
                        order_type="NEXT_BAR_OPEN",
                        signal_timestamp=bar.timestamp.isoformat(),
                        submit_timestamp=bar.timestamp.isoformat(),
                        requested_price=bar.close,
                    )
            elif current_signal != 0 and pending_order is None and (i + 1 < n_bars):
                # Don't enter on session boundary if enforce_session_boundaries is True
                if not (self.enforce_session_boundaries and next_is_different_session):
                    pending_order = PaperOrder(
                        order_id=f"ORD-ENTRY-{i}",
                        strategy_id=derived_strat_id,
                        qualification_id=qualification_record.qualification_id,
                        symbol=feed.symbol,
                        side=int(current_signal),
                        quantity=quantity,
                        order_type="NEXT_BAR_OPEN",
                        signal_timestamp=bar.timestamp.isoformat(),
                        submit_timestamp=bar.timestamp.isoformat(),
                        requested_price=bar.close,
                    )

            # Snapshot position at end of bar
            self.ledger.record_position_snapshot(
                timestamp=bar.timestamp.isoformat(),
                strategy_id=derived_strat_id,
                pos=current_position,
            )

        # 7. Aggregate metrics and build ReplayReport
        trade_count = len(trade_pnls)
        net_pnl = sum(trade_pnls)
        wins = [p for p in trade_pnls if p > 0]
        win_rate = len(wins) / float(trade_count) if trade_count > 0 else 0.0
        expectancy = (net_pnl / float(trade_count)) if trade_count > 0 else 0.0

        # Maximum Drawdown in bps
        if trade_pnls:
            equity_curve = np.cumsum([0.0] + trade_pnls)
            peak = np.maximum.accumulate(equity_curve)
            dd = peak - equity_curve
            max_dd = float(np.max(dd))
            max_dd_bps = (max_dd / initial_capital) * 10000.0
            sharpe = float(np.mean(trade_pnls) / (np.std(trade_pnls, ddof=1) or 1.0)) * math.sqrt(250) if len(trade_pnls) > 1 else None
        else:
            max_dd_bps = 0.0
            sharpe = None

        exposure = float(trade_count * quantity * lot_size)

        breakdown = [
            ReplaySessionSummary(
                session_id=s_id,
                start_ts=s_data["start_ts"],
                end_ts=s_data["end_ts"],
                trades=s_data["trades"],
                gross_pnl=s_data["gross_pnl"],
                net_pnl=s_data["net_pnl"],
            )
            for s_id, s_data in sorted(session_summaries.items())
        ]

        report = ReplayReport.create(
            strategy_id=derived_strat_id,
            qualification_id=qualification_record.qualification_id,
            dataset_version=qualification_record.dataset_version,
            trade_count=trade_count,
            gross_pnl=total_gross_pnl,
            net_pnl=net_pnl,
            costs=total_costs,
            slippage=total_slippage,
            max_drawdown_bps=max_dd_bps,
            exposure=exposure,
            win_rate=win_rate,
            expectancy=expectancy,
            sharpe_ratio=sharpe,
            session_breakdown=breakdown,
            strategy_spec_hash=qualification_record.strategy_spec_hash,
            qualification_hash=qualification_record.record_hash,
            dataset_sha256=qualification_record.dataset_sha256,
            research_protocol_version=qualification_record.research_protocol_version,
            symbol=feed.symbol,
            lot_size=feed.lot_size,
            tick_size=feed.tick_size,
            slippage_bps_per_side=self.slippage_bps_per_side,
            cost_schedule_id=self.cost_schedule.schedule_id if self.cost_schedule else "",
            execution_policy="next_bar_open_v1",
        )

        self.ledger.record_report(report)
        return report
