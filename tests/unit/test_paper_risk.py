"""Unit tests for Paper Risk Engine (PRD v3.9)."""

from __future__ import annotations

import pytest

from quantmind.paper.models import PaperOrder, PaperPosition
from quantmind.paper.risk import PaperRiskConfig, PaperRiskEngine


def _make_order(quantity: int = 1, side: int = 1) -> PaperOrder:
    return PaperOrder(
        order_id="ORD-TEST",
        strategy_id="STRAT-1",
        qualification_id="QUAL-1",
        symbol="NIFTY_FUT",
        side=side,
        quantity=quantity,
        signal_timestamp="2023-01-01T09:15:00",
        submit_timestamp="2023-01-01T09:15:00",
        requested_price=100.0,
    )


def _make_pos(quantity: int = 0) -> PaperPosition:
    return PaperPosition(
        symbol="NIFTY_FUT",
        quantity=quantity,
        entry_price=100.0,
        current_price=100.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        fees_costs=0.0,
    )


class TestPaperRiskEngine:
    def test_kill_switch_blocks_all_orders(self) -> None:
        engine = PaperRiskEngine(PaperRiskConfig(kill_switch=True))
        order = _make_order()
        event = engine.evaluate_order(order, _make_pos(), 100.0, "2023-01-01")
        assert event is not None
        assert event.rule_name == "kill_switch"
        assert len(engine.risk_events) == 1

    def test_max_order_quantity_breached(self) -> None:
        engine = PaperRiskEngine(PaperRiskConfig(max_order_quantity=2))
        order = _make_order(quantity=3)
        event = engine.evaluate_order(order, _make_pos(), 100.0, "2023-01-01")
        assert event is not None
        assert event.rule_name == "max_order_quantity"

    def test_max_position_breached(self) -> None:
        engine = PaperRiskEngine(PaperRiskConfig(max_position=5))
        order = _make_order(quantity=3, side=1)
        # Existing position is +4 long, order would push to +7 > 5
        event = engine.evaluate_order(order, _make_pos(quantity=4), 100.0, "2023-01-01")
        assert event is not None
        assert event.rule_name == "max_position"

    def test_max_trades_per_session_breached(self) -> None:
        engine = PaperRiskEngine(PaperRiskConfig(max_trades_per_session=2))
        order = _make_order()
        # Trade 1
        assert engine.evaluate_order(order, _make_pos(), 100.0, "2023-01-01") is None
        # Trade 2
        assert engine.evaluate_order(order, _make_pos(), 100.0, "2023-01-01") is None
        # Trade 3 breaches
        event = engine.evaluate_order(order, _make_pos(), 100.0, "2023-01-01")
        assert event is not None
        assert event.rule_name == "max_trades_per_session"

    def test_max_daily_loss_breached(self) -> None:
        engine = PaperRiskEngine(PaperRiskConfig(max_daily_loss=1000.0))
        engine.on_session_change("2023-01-01")
        # Update daily loss to 1500.0
        engine.update_portfolio_state(current_equity=98_500.0, realized_pnl_delta=-1500.0)

        order = _make_order()
        event = engine.evaluate_order(order, _make_pos(), 100.0, "2023-01-01")
        assert event is not None
        assert event.rule_name == "max_daily_loss"

    def test_max_drawdown_breached(self) -> None:
        engine = PaperRiskEngine(PaperRiskConfig(max_strategy_drawdown=5000.0))
        # Initial peak at 100,000
        engine.update_portfolio_state(current_equity=100_000.0, realized_pnl_delta=0.0)
        # Drop to 94,000 (drawdown = 6000 > 5000)
        engine.update_portfolio_state(current_equity=94_000.0, realized_pnl_delta=-6000.0)

        order = _make_order()
        event = engine.evaluate_order(order, _make_pos(), 100.0, "2023-01-01")
        assert event is not None
        assert event.rule_name == "max_strategy_drawdown"

    def test_max_exposure_breached(self) -> None:
        engine = PaperRiskEngine(PaperRiskConfig(max_exposure=50_000.0, max_position=10))
        # Order quantity 1 at price 60,000 -> exposure 60,000 > 50,000
        order = _make_order(quantity=1)
        event = engine.evaluate_order(order, _make_pos(), 60_000.0, "2023-01-01")
        assert event is not None
        assert event.rule_name == "max_exposure"
