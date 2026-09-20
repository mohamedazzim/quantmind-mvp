"""Paper Trading Risk Engine (PRD v3.9).

Enforces pre-trade and intra-session deterministic risk constraints,
logging any breaches as auditable PaperRiskEvents.
"""

from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Mapping

from quantmind.paper.models import PaperOrder, PaperPosition, PaperRiskEvent


@dataclass(frozen=True)
class PaperRiskConfig:
    """Configurable limits for deterministic paper replay risk management."""

    max_position: int = 10
    max_order_quantity: int = 5
    max_daily_loss: float = 50_000.0
    max_strategy_drawdown: float = 100_000.0
    max_exposure: float = 1_000_000.0
    max_trades_per_session: int = 100
    kill_switch: bool = False


class PaperRiskEngine:
    """Pre-trade risk gateway for paper replay orders."""

    def __init__(self, config: PaperRiskConfig | None = None) -> None:
        self.config = config or PaperRiskConfig()
        self._current_session_id: str | None = None
        self._session_trades_count: int = 0
        self._daily_realized_loss: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._risk_events: list[PaperRiskEvent] = []

    @property
    def risk_events(self) -> tuple[PaperRiskEvent, ...]:
        return tuple(self._risk_events)

    def on_session_change(self, new_session_id: str) -> None:
        """Reset intra-day tracking counters when transitioning across calendar sessions."""
        if self._current_session_id != new_session_id:
            self._current_session_id = new_session_id
            self._session_trades_count = 0
            self._daily_realized_loss = 0.0

    def update_portfolio_state(self, current_equity: float, realized_pnl_delta: float) -> None:
        """Update intra-day portfolio equity and drawdown tracking."""
        self._current_equity = current_equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if realized_pnl_delta < 0:
            self._daily_realized_loss += abs(realized_pnl_delta)

    def evaluate_order(
        self,
        order: PaperOrder,
        current_position: PaperPosition,
        current_price: float,
        session_id: str,
    ) -> PaperRiskEvent | None:
        """Evaluate an order before execution against deterministic risk limits.

        Returns PaperRiskEvent if rejected; None if approved.
        """
        self.on_session_change(session_id)

        # 1. Kill switch check
        if self.config.kill_switch:
            event = PaperRiskEvent(
                event_id=f"RISK-{uuid.uuid4().hex[:12]}",
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                rule_name="kill_switch",
                limit_value=1.0,
                requested_value=1.0,
                timestamp=order.submit_timestamp,
                reason="Emergency kill switch is ACTIVE; all new orders are prohibited",
            )
            self._risk_events.append(event)
            return event

        # 2. Maximum order quantity check
        if order.quantity > self.config.max_order_quantity:
            event = PaperRiskEvent(
                event_id=f"RISK-{uuid.uuid4().hex[:12]}",
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                rule_name="max_order_quantity",
                limit_value=float(self.config.max_order_quantity),
                requested_value=float(order.quantity),
                timestamp=order.submit_timestamp,
                reason=f"Order quantity ({order.quantity}) exceeds max allowed ({self.config.max_order_quantity})",
            )
            self._risk_events.append(event)
            return event

        # 3. Maximum position check (post-execution position size)
        projected_position = current_position.quantity + (order.side * order.quantity)
        if abs(projected_position) > self.config.max_position:
            event = PaperRiskEvent(
                event_id=f"RISK-{uuid.uuid4().hex[:12]}",
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                rule_name="max_position",
                limit_value=float(self.config.max_position),
                requested_value=float(abs(projected_position)),
                timestamp=order.submit_timestamp,
                reason=f"Projected position ({abs(projected_position)}) exceeds max position ({self.config.max_position})",
            )
            self._risk_events.append(event)
            return event

        # 4. Maximum trades per session check
        if self._session_trades_count >= self.config.max_trades_per_session:
            event = PaperRiskEvent(
                event_id=f"RISK-{uuid.uuid4().hex[:12]}",
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                rule_name="max_trades_per_session",
                limit_value=float(self.config.max_trades_per_session),
                requested_value=float(self._session_trades_count + 1),
                timestamp=order.submit_timestamp,
                reason=f"Session trade limit reached ({self.config.max_trades_per_session} trades in session {session_id})",
            )
            self._risk_events.append(event)
            return event

        # 5. Maximum daily loss check
        if self._daily_realized_loss >= self.config.max_daily_loss:
            event = PaperRiskEvent(
                event_id=f"RISK-{uuid.uuid4().hex[:12]}",
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                rule_name="max_daily_loss",
                limit_value=float(self.config.max_daily_loss),
                requested_value=float(self._daily_realized_loss),
                timestamp=order.submit_timestamp,
                reason=f"Daily realized loss ({self._daily_realized_loss:.2f}) breached limit ({self.config.max_daily_loss:.2f})",
            )
            self._risk_events.append(event)
            return event

        # 6. Maximum strategy drawdown check
        current_drawdown = max(0.0, self._peak_equity - self._current_equity)
        if current_drawdown >= self.config.max_strategy_drawdown:
            event = PaperRiskEvent(
                event_id=f"RISK-{uuid.uuid4().hex[:12]}",
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                rule_name="max_strategy_drawdown",
                limit_value=float(self.config.max_strategy_drawdown),
                requested_value=float(current_drawdown),
                timestamp=order.submit_timestamp,
                reason=f"Strategy drawdown ({current_drawdown:.2f}) breached max limit ({self.config.max_strategy_drawdown:.2f})",
            )
            self._risk_events.append(event)
            return event

        # 7. Maximum exposure check
        projected_exposure = abs(projected_position) * current_price
        if projected_exposure > self.config.max_exposure:
            event = PaperRiskEvent(
                event_id=f"RISK-{uuid.uuid4().hex[:12]}",
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                rule_name="max_exposure",
                limit_value=float(self.config.max_exposure),
                requested_value=float(projected_exposure),
                timestamp=order.submit_timestamp,
                reason=f"Projected exposure ({projected_exposure:.2f}) exceeds limit ({self.config.max_exposure:.2f})",
            )
            self._risk_events.append(event)
            return event

        # Approved
        self._session_trades_count += 1
        return None
