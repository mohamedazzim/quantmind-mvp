"""Pure Functional Monitoring Metric Calculators (PRD v4.0 Milestone 3).

This module contains strictly pure mathematical functions for evaluating forward
paper trading performance, execution quality, exposure, and stability metrics:
- cumulative_net_pnl
- rolling_net_pnl
- compute_session_returns
- compute_rolling_sharpe
- compute_rolling_drawdown_bps
- compute_rolling_win_rate
- compute_rolling_expectancy
- compute_cost_to_turnover_bps
- compute_realized_slippage_bps
- compute_trade_frequency_ratio
- compute_drawdown_expansion_ratio
- risk and execution helper metrics (risk_event_count, rejection_rate, holding_duration)
- exposure metrics (peak_gross_exposure, limit_utilization_ratio)
- session boundary metrics (session_overnight_spill_count)

ARCHITECTURAL PRINCIPLES:
1. Pure Functions Only: Zero database access, zero SQLite queries, zero global
   state, zero wall-clock reads, and zero side effects.
2. Authoritative Execution Truth: P&L is taken directly from authoritative net
   realized values. Costs and slippage are NEVER subtracted a second time.
3. Slippage Semantics: Modeled execution slippage is measured strictly via recorded
   slippage amounts and traded notional turnover. Inter-bar market movement between
   signal close and bar open is not misclassified as slippage.
4. Numerical Determinism: Explicit zero handling, sample size requirements, and
   fixed-point rounding matching QuantMind conventions.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from quantmind.paper.models import PaperFill, PaperOrder, PaperOrderStatus, PaperPosition, PaperRiskEvent


# ---------------------------------------------------------------------------
# 1. Performance Metrics (Net P&L, Returns, Sharpe, Drawdown)
# ---------------------------------------------------------------------------


def compute_cumulative_net_pnl(net_pnls: Sequence[float]) -> float:
    """Calculate the cumulative net realized P&L across a sequence of closed trades.

    CRITICAL SEMANTIC RULE:
    In QuantMind paper execution (PaperPosition, ReplayReport), realized P&L is
    ALREADY net of transaction fees and slippage. This function calculates the pure
    sum of net realized values without duplicate cost deductions.

    Returns:
        Cumulative net P&L rounded to 4 decimal places, or 0.0 if empty.
    """
    if not net_pnls:
        return 0.0
    return round(float(sum(net_pnls)), 4)


def compute_rolling_net_pnl(session_net_pnls: Sequence[float], window_size: int) -> float:
    """Calculate net realized P&L over the last W completed calendar sessions.

    Args:
        session_net_pnls: Chronological sequence of realized net P&L per session.
        window_size: Number of completed sessions W to look back.

    Returns:
        Rolling net P&L rounded to 4 decimal places, or 0.0 if empty or window_size <= 0.
    """
    if not session_net_pnls or window_size <= 0:
        return 0.0
    slice_pnls = session_net_pnls[-window_size:]
    return round(float(sum(slice_pnls)), 4)


def compute_session_returns(
    session_net_pnls: Sequence[float],
    initial_capital: float,
) -> list[float]:
    """Calculate fractional session returns R_s = ΔNetPnL_s / E_(s-1).

    Equity base E_(s-1) is constructed sequentially from initial capital C_0 and
    prior realized session net P&Ls:
        E_0 = initial_capital
        R_s = session_net_pnls[s] / E_(s-1)
        E_s = E_(s-1) + session_net_pnls[s]

    Args:
        session_net_pnls: Chronological net P&L realized in each completed session.
        initial_capital: Starting capital base C_0 > 0.

    Raises:
        ValueError: If initial_capital <= 0 or if insolvency (E <= 0) is encountered.

    Returns:
        List of session return fractions rounded to 8 decimal places.
    """
    if initial_capital <= 0.0:
        raise ValueError(f"initial_capital must be strictly positive, got {initial_capital}")

    returns: list[float] = []
    equity = float(initial_capital)

    for pnl in session_net_pnls:
        if equity <= 0.0:
            raise ValueError(f"Insolvent equity base ({equity:.4f} <= 0) encountered during session return calculation")
        ret = pnl / equity
        returns.append(round(ret, 8))
        equity += pnl

    return returns


def compute_rolling_sharpe(
    session_returns: Sequence[float],
    window_size: int = 30,
    min_sessions: int = 10,
    annualization_factor: float = 15.874507866387544,  # sqrt(252)
) -> float | None:
    """Calculate rolling annualized Sharpe ratio over the last W session returns.

    Formula:
        Sharpe_w = sqrt(252) * mean(R_w) / sample_std(R_w)
    where sample_std uses (W - 1) degrees of freedom.

    Rules:
        - Requires at least min_sessions (default 10) completed sessions.
        - If sample standard deviation is 0.0, returns None.
        - Negative Sharpe ratios are permitted and preserved.

    Returns:
        Annualized Sharpe ratio rounded to 4 decimal places, or None if undefined.
    """
    if window_size <= 0 or len(session_returns) < min_sessions:
        return None

    window = session_returns[-window_size:]
    n = len(window)
    if n < min_sessions:
        return None

    mean_ret = sum(window) / n
    variance = sum((r - mean_ret) ** 2 for r in window) / (n - 1)

    if variance <= 1e-14:
        return None

    std_dev = math.sqrt(variance)
    sharpe = annualization_factor * (mean_ret / std_dev)
    return round(float(sharpe), 4)


def compute_rolling_drawdown_bps(equity_curve: Sequence[float]) -> float:
    """Calculate maximum peak-to-trough equity drawdown in basis points (bps).

    Formula:
        DD_t = (PeakEquity_t - Equity_t) / PeakEquity_t * 10,000

    Rules:
        - Chronological sequence of portfolio equity snapshots E_t.
        - If peak equity <= 0.0 (insolvency), returns 10,000.0 bps (100% loss).
        - If equity is monotonically increasing, returns 0.0 bps.

    Returns:
        Max drawdown in bps rounded to 4 decimal places (non-negative).
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd_bps = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak <= 0.0 or equity <= 0.0:
            return 10000.0
        dd_bps = ((peak - equity) / peak) * 10000.0
        if dd_bps > max_dd_bps:
            max_dd_bps = dd_bps

    return round(float(min(10000.0, max(0.0, max_dd_bps))), 4)


# ---------------------------------------------------------------------------
# 2. Trade Quality Metrics (Win Rate, Expectancy)
# ---------------------------------------------------------------------------


def compute_rolling_win_rate(
    trade_net_pnls: Sequence[float],
    min_trades: int = 5,
) -> float | None:
    """Calculate the ratio of profitable closed trades over the last trades.

    Rules:
        - A winning trade strictly requires net_pnl > 0.0.
        - Breakeven trades (net_pnl == 0.0) are NOT winners.
        - Requires at least min_trades closed round-trip trades.

    Returns:
        Win rate in [0.0, 1.0] rounded to 4 decimal places, or None if insufficient trades.
    """
    if len(trade_net_pnls) < min_trades:
        return None

    wins = sum(1 for pnl in trade_net_pnls if pnl > 0.0)
    return round(float(wins / len(trade_net_pnls)), 4)


def compute_rolling_expectancy(
    trade_net_pnls: Sequence[float],
    min_trades: int = 5,
) -> float | None:
    """Calculate average net P&L realized per closed trade.

    Formula:
        Expectancy = sum(trade_net_pnls) / len(trade_net_pnls)

    Returns:
        Net currency expectancy per trade rounded to 4 decimal places, or None if < min_trades.
    """
    if len(trade_net_pnls) < min_trades:
        return None

    return round(float(sum(trade_net_pnls) / len(trade_net_pnls)), 4)


# ---------------------------------------------------------------------------
# 3. Execution Quality Metrics (Turnover Cost, Realized Slippage)
# ---------------------------------------------------------------------------


def compute_cost_to_turnover_bps(
    total_costs: float,
    total_slippage: float,
    total_turnover: float,
) -> float:
    """Calculate total transaction friction relative to traded notional turnover in bps.

    Formula:
        cost_to_turnover_bps = ((total_costs + total_slippage) / total_turnover) * 10,000

    Args:
        total_costs: Sum of exchange/brokerage transaction fees.
        total_slippage: Sum of execution slippage currency drag.
        total_turnover: Total traded notional volume (Sum of fill_price * quantity * lot_size).

    Returns:
        Turnover drag in basis points rounded to 4 decimal places, or 0.0 if turnover == 0.
    """
    if total_turnover <= 0.0:
        return 0.0
    friction = max(0.0, total_costs) + max(0.0, total_slippage)
    return round(float((friction / total_turnover) * 10000.0), 4)


def compute_cost_to_turnover_bps_from_fills(
    fills: Sequence[PaperFill | Mapping[str, Any]],
    lot_size: int,
) -> float:
    """Calculate turnover cost bps directly from authoritative execution fill records."""
    if not fills or lot_size <= 0:
        return 0.0

    total_costs = 0.0
    total_slippage = 0.0
    total_turnover = 0.0

    for f in fills:
        if isinstance(f, PaperFill):
            cost = f.cost
            slip = f.slippage
            price = f.fill_price
            qty = f.quantity
        else:
            cost = float(f["cost"])
            slip = float(f["slippage"])
            price = float(f["fill_price"])
            qty = int(f["quantity"])

        total_costs += cost
        total_slippage += slip
        total_turnover += price * qty * lot_size

    return compute_cost_to_turnover_bps(total_costs, total_slippage, total_turnover)


def compute_realized_slippage_bps(
    total_slippage: float,
    total_turnover: float,
) -> float:
    """Calculate realized execution slippage relative to traded notional turnover in bps.

    SLIPPAGE SEMANTICS:
    In QuantMind, PaperFill.slippage records the exact execution-friction amount:
        abs(fill_price - bar.open) * quantity * lot_size
    Realized slippage bps measures this friction relative to traded notional:
        realized_slippage_bps = (total_slippage / total_turnover) * 10,000

    Returns:
        Slippage in basis points rounded to 4 decimal places, or 0.0 if turnover == 0.
    """
    if total_turnover <= 0.0:
        return 0.0
    return round(float((max(0.0, total_slippage) / total_turnover) * 10000.0), 4)


def compute_realized_slippage_bps_from_fills(
    fills: Sequence[PaperFill | Mapping[str, Any]],
    lot_size: int,
) -> float:
    """Calculate realized slippage bps directly from authoritative execution fill records."""
    if not fills or lot_size <= 0:
        return 0.0

    total_slippage = 0.0
    total_turnover = 0.0

    for f in fills:
        if isinstance(f, PaperFill):
            slip = f.slippage
            price = f.fill_price
            qty = f.quantity
        else:
            slip = float(f["slippage"])
            price = float(f["fill_price"])
            qty = int(f["quantity"])

        total_slippage += slip
        total_turnover += price * qty * lot_size

    return compute_realized_slippage_bps(total_slippage, total_turnover)


# ---------------------------------------------------------------------------
# 4. Stability & Benchmark Deviation Metrics
# ---------------------------------------------------------------------------


def compute_trade_frequency_ratio(
    observed_trades: int,
    observed_sessions: int,
    baseline_trades: int,
    baseline_sessions: int,
    min_observed_sessions: int = 10,
) -> float | None:
    """Calculate the ratio of observed trading frequency to qualification baseline frequency.

    Formula:
        Ratio = (observed_trades / observed_sessions) / (baseline_trades / baseline_sessions)

    Rules:
        - Requires at least min_observed_sessions (default 10).
        - If baseline trading frequency is 0.0 or baseline_sessions <= 0, returns None.
        - This metric is strictly a dimensionless ratio.

    Returns:
        Frequency ratio rounded to 4 decimal places, or None if undefined/insufficient data.
    """
    if observed_sessions < min_observed_sessions:
        return None
    if baseline_sessions <= 0 or baseline_trades <= 0:
        return None

    obs_rate = observed_trades / observed_sessions
    base_rate = baseline_trades / baseline_sessions

    if base_rate <= 0.0:
        return None

    return round(float(obs_rate / base_rate), 4)


def compute_drawdown_expansion_ratio(
    observed_drawdown_bps: float,
    baseline_max_drawdown_bps: float,
    observed_sessions: int,
    min_sessions: int = 5,
) -> float | None:
    """Calculate the expansion of observed paper drawdown relative to qualification baseline.

    Formula:
        Expansion = observed_drawdown_bps / max(1.0, baseline_max_drawdown_bps)

    Rules:
        - Requires at least min_sessions (default 5).
        - Baseline drawdown is clamped to minimum 1.0 bps to prevent zero division.
        - Strictly observational; does not trigger lifecycle actions directly.

    Returns:
        Expansion ratio rounded to 4 decimal places, or None if observed_sessions < min_sessions.
    """
    if observed_sessions < min_sessions:
        return None

    base_dd = max(1.0, float(baseline_max_drawdown_bps))
    obs_dd = max(0.0, float(observed_drawdown_bps))

    return round(float(obs_dd / base_dd), 4)


# ---------------------------------------------------------------------------
# 5. Risk, Order, and Holding Duration Metrics
# ---------------------------------------------------------------------------


def count_risk_events(
    risk_events: Sequence[PaperRiskEvent | Mapping[str, Any]],
) -> int:
    """Count the total number of pre-trade risk events recorded in the evaluation window."""
    return len(risk_events)


def count_rejected_orders(
    orders: Sequence[PaperOrder | Mapping[str, Any]],
) -> int:
    """Count orders whose authoritative status is REJECTED."""
    count = 0
    for o in orders:
        if isinstance(o, PaperOrder):
            if o.status == PaperOrderStatus.REJECTED:
                count += 1
        else:
            if o.get("status") == PaperOrderStatus.REJECTED.value:
                count += 1
    return count


def compute_rejection_rate(
    rejected_orders: int,
    submitted_orders: int,
) -> float | None:
    """Calculate the ratio of rejected orders to total submitted orders.

    Returns:
        Rejection rate in [0.0, 1.0] rounded to 4 decimal places, or None if submitted_orders == 0.
    """
    if submitted_orders <= 0:
        return None
    return round(float(max(0, rejected_orders) / submitted_orders), 4)


def compute_mean_holding_duration_bars(
    holding_durations: Sequence[int],
) -> float | None:
    """Calculate the mean holding horizon in bars across completed round-trip trades.

    Returns:
        Average bars held rounded to 4 decimal places, or None if no completed trades.
    """
    if not holding_durations:
        return None
    return round(float(sum(holding_durations) / len(holding_durations)), 4)


# ---------------------------------------------------------------------------
# 6. Exposure & Session Boundary Metrics
# ---------------------------------------------------------------------------


def compute_peak_gross_exposure(
    positions: Sequence[PaperPosition | Mapping[str, Any]],
    lot_size: int,
) -> float:
    """Calculate the peak gross notional portfolio exposure observed across position snapshots.

    Formula:
        Peak Exposure = max(|quantity| * current_price * lot_size)

    Returns:
        Max gross exposure in currency units rounded to 4 decimal places, or 0.0 if empty.
    """
    if not positions or lot_size <= 0:
        return 0.0

    peak = 0.0
    for p in positions:
        if isinstance(p, PaperPosition):
            qty = abs(p.quantity)
            price = p.current_price
        else:
            qty = abs(int(p["quantity"]))
            price = float(p["current_price"])
        exp = qty * price * lot_size
        if exp > peak:
            peak = exp

    return round(float(peak), 4)


def compute_limit_utilization_ratio(
    peak_observed: float,
    configured_limit: float,
) -> float | None:
    """Calculate the peak utilization ratio against a configured risk limit.

    Formula:
        Utilization = peak_observed / configured_limit

    Returns:
        Ratio rounded to 4 decimal places, or None if configured_limit <= 0.
    """
    if configured_limit <= 0.0:
        return None
    return round(float(max(0.0, peak_observed) / configured_limit), 4)


def count_session_overnight_spills(
    trades: Sequence[Mapping[str, Any] | tuple[str, str]],
) -> int:
    """Count trades that span across a calendar session boundary.

    A session spill occurs when a position opened in session S_entry remains open
    into a different session S_exit (e.g. forced intraday close or overnight carry).

    Args:
        trades: Sequence of trades providing entry_session and exit_session identifiers.

    Returns:
        Integer count of trades where entry_session != exit_session.
    """
    spills = 0
    for t in trades:
        if isinstance(t, tuple):
            entry_s, exit_s = t
        else:
            entry_s = t.get("entry_session_id") or t.get("entry_session")
            exit_s = t.get("exit_session_id") or t.get("exit_session")

        if entry_s and exit_s and entry_s != exit_s:
            spills += 1

    return spills
