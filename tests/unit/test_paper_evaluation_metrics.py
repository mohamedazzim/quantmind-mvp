"""Unit tests for PRD v4.0 Pure Functional Monitoring Metric Calculators.

Tests:
- cumulative_net_pnl & rolling_net_pnl accounting (no double deduction of costs/slippage).
- compute_session_returns sequencing, denominator E_(s-1), and insolvency safeguards.
- compute_rolling_sharpe degrees of freedom, minimum sessions, zero std dev, and negative values.
- compute_rolling_drawdown_bps peak tracking, recovery, and insolvency.
- compute_rolling_win_rate (breakeven not a win, minimum 5 trades).
- compute_rolling_expectancy currency units and trade count thresholds.
- compute_cost_to_turnover_bps and compute_realized_slippage_bps notional scaling.
- compute_trade_frequency_ratio and compute_drawdown_expansion_ratio.
- risk_event_count, rejected_order_count, rejection_rate, mean_holding_duration.
- peak_gross_exposure lot-size scaling and limit_utilization_ratio.
- count_session_overnight_spills across calendar sessions.
- Invariants: determinism, non-negativity of turnover, bounds on win rate.
"""

import math
import pytest

from quantmind.paper.evaluation.metrics import (
    compute_cost_to_turnover_bps,
    compute_cost_to_turnover_bps_from_fills,
    compute_cumulative_net_pnl,
    compute_drawdown_expansion_ratio,
    compute_limit_utilization_ratio,
    compute_mean_holding_duration_bars,
    compute_peak_gross_exposure,
    compute_realized_slippage_bps,
    compute_realized_slippage_bps_from_fills,
    compute_rejection_rate,
    compute_rolling_drawdown_bps,
    compute_rolling_expectancy,
    compute_rolling_net_pnl,
    compute_rolling_sharpe,
    compute_rolling_win_rate,
    compute_session_returns,
    compute_trade_frequency_ratio,
    count_rejected_orders,
    count_risk_events,
    count_session_overnight_spills,
)
from quantmind.paper.models import (
    PaperFill,
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    PaperRiskEvent,
)


# ---------------------------------------------------------------------------
# 1. P&L and Returns Tests
# ---------------------------------------------------------------------------


class TestPnLAndReturnMetrics:
    def test_cumulative_net_pnl_empty_and_normal(self) -> None:
        assert compute_cumulative_net_pnl([]) == 0.0
        assert compute_cumulative_net_pnl([100.25, -50.10, 200.0]) == 250.15

    def test_cumulative_net_pnl_does_not_subtract_costs_again(self) -> None:
        # In QuantMind, closed trade net_pnl is already net of costs!
        trade_net_pnls = [500.0, -150.0]  # Already net!
        total = compute_cumulative_net_pnl(trade_net_pnls)
        assert total == 350.0

    def test_rolling_net_pnl_windowing(self) -> None:
        sessions = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert compute_rolling_net_pnl(sessions, window_size=3) == 120.0  # 30 + 40 + 50
        assert compute_rolling_net_pnl(sessions, window_size=10) == 150.0
        assert compute_rolling_net_pnl(sessions, window_size=0) == 0.0
        assert compute_rolling_net_pnl([], window_size=5) == 0.0

    def test_session_returns_sequential_denominator(self) -> None:
        # Initial capital: 100,000
        # Session 1: +5,000 -> R1 = 5000 / 100000 = 0.05, End Equity = 105,000
        # Session 2: -2,100 -> R2 = -2100 / 105000 = -0.02, End Equity = 102,900
        pnls = [5000.0, -2100.0]
        returns = compute_session_returns(pnls, initial_capital=100000.0)
        assert len(returns) == 2
        assert returns[0] == 0.05
        assert returns[1] == -0.02

    def test_session_returns_invalid_capital(self) -> None:
        with pytest.raises(ValueError, match="initial_capital must be strictly positive"):
            compute_session_returns([100.0], initial_capital=0.0)
        with pytest.raises(ValueError, match="initial_capital must be strictly positive"):
            compute_session_returns([100.0], initial_capital=-5000.0)

    def test_session_returns_insolvency_safeguard(self) -> None:
        # Capital: 10,000; Session 1: -12,000 -> Equity drops to -2,000
        with pytest.raises(ValueError, match="Insolvent equity base"):
            compute_session_returns([-12000.0, 500.0], initial_capital=10000.0)


# ---------------------------------------------------------------------------
# 2. Sharpe Ratio Tests
# ---------------------------------------------------------------------------


class TestRollingSharpe:
    def test_insufficient_sessions_returns_none(self) -> None:
        returns = [0.01] * 9  # Only 9 sessions (< 10)
        assert compute_rolling_sharpe(returns, window_size=30, min_sessions=10) is None

    def test_zero_variance_returns_none(self) -> None:
        returns = [0.01] * 15  # All identical -> std = 0.0
        assert compute_rolling_sharpe(returns, window_size=15, min_sessions=10) is None

    def test_positive_sharpe_exact_math(self) -> None:
        # 10 returns: alternating 0.02 and 0.00
        # Mean = 0.01
        # Sample Var = sum((r - 0.01)^2) / 9 = 10 * 0.0001 / 9 = 0.001 / 9 = 0.00011111...
        # Sample Std = sqrt(0.00011111...) ≈ 0.0105409
        # Sharpe = sqrt(252) * (0.01 / 0.0105409) ≈ 15.8745 * 0.94868 ≈ 15.0596
        returns = [0.02, 0.00] * 5
        sharpe = compute_rolling_sharpe(returns, window_size=10, min_sessions=10)
        assert sharpe is not None
        assert 14.5 < sharpe < 15.5

    def test_negative_sharpe_permitted(self) -> None:
        # Mean negative return
        returns = [-0.02, -0.01] * 5
        sharpe = compute_rolling_sharpe(returns, window_size=10, min_sessions=10)
        assert sharpe is not None
        assert sharpe < 0.0

    def test_window_slicing_uses_last_w_only(self) -> None:
        # First 20 are huge negatives, last 10 are positives
        returns = [-0.10] * 20 + [0.02, 0.01] * 5
        sharpe = compute_rolling_sharpe(returns, window_size=10, min_sessions=10)
        assert sharpe is not None
        assert sharpe > 0.0  # Last 10 are positive!


# ---------------------------------------------------------------------------
# 3. Drawdown Tests
# ---------------------------------------------------------------------------


class TestRollingDrawdown:
    def test_monotonic_gain_zero_drawdown(self) -> None:
        curve = [100000.0, 102000.0, 105000.0, 110000.0]
        assert compute_rolling_drawdown_bps(curve) == 0.0

    def test_single_peak_decline(self) -> None:
        # Peak 100,000 -> drops to 90,000 (10% drop = 1,000 bps)
        curve = [100000.0, 95000.0, 90000.0, 92000.0]
        assert compute_rolling_drawdown_bps(curve) == 1000.0

    def test_multiple_peaks_largest_drawdown_captured(self) -> None:
        # Peak 100 -> drop to 95 (5% = 500 bps) -> new peak 110 -> drop to 99 (11/110 = 10% = 1000 bps)
        curve = [100000.0, 95000.0, 110000.0, 99000.0]
        assert compute_rolling_drawdown_bps(curve) == 1000.0

    def test_insolvency_returns_10000_bps(self) -> None:
        curve = [100000.0, 0.0, -1000.0]
        assert compute_rolling_drawdown_bps(curve) == 10000.0

    def test_empty_curve(self) -> None:
        assert compute_rolling_drawdown_bps([]) == 0.0


# ---------------------------------------------------------------------------
# 4. Win Rate & Expectancy Tests
# ---------------------------------------------------------------------------


class TestWinRateAndExpectancy:
    def test_win_rate_insufficient_trades(self) -> None:
        assert compute_rolling_win_rate([100.0, 200.0, 50.0, -20.0], min_trades=5) is None

    def test_win_rate_zero_pnl_not_counted_as_winner(self) -> None:
        # 5 trades: 2 positive, 1 zero, 2 negative -> 2 / 5 = 0.40
        trades = [100.0, 50.0, 0.0, -20.0, -80.0]
        assert compute_rolling_win_rate(trades, min_trades=5) == 0.40

    def test_win_rate_all_winners_and_all_losers(self) -> None:
        assert compute_rolling_win_rate([10.0] * 5, min_trades=5) == 1.0
        assert compute_rolling_win_rate([-10.0] * 5, min_trades=5) == 0.0

    def test_expectancy_exact_math(self) -> None:
        # 5 trades: sum = 500 -> expectancy = 100.0
        trades = [100.0, 200.0, -50.0, -50.0, 300.0]
        assert compute_rolling_expectancy(trades, min_trades=5) == 100.0

    def test_expectancy_insufficient_trades(self) -> None:
        assert compute_rolling_expectancy([100.0, 200.0], min_trades=5) is None


# ---------------------------------------------------------------------------
# 5. Cost & Slippage Turnover Tests
# ---------------------------------------------------------------------------


class TestCostAndSlippageTurnover:
    def test_cost_to_turnover_bps_exact_math(self) -> None:
        # Turnover: 1,000,000
        # Costs: 200, Slippage: 100 -> Total friction: 300
        # 300 / 1,000,000 * 10,000 = 3.0 bps
        assert compute_cost_to_turnover_bps(200.0, 100.0, 1000000.0) == 3.0

    def test_cost_to_turnover_bps_zero_turnover(self) -> None:
        assert compute_cost_to_turnover_bps(50.0, 10.0, 0.0) == 0.0

    def test_cost_to_turnover_bps_from_fills(self) -> None:
        fills = [
            PaperFill(
                fill_id="F1",
                order_id="O1",
                strategy_id="S1",
                symbol="NIFTY_FUT",
                fill_timestamp="2023-01-01T09:16:00Z",
                fill_price=18000.0,
                quantity=2,
                side=1,
                cost=18.0,
                slippage=9.0,
            ),
            PaperFill(
                fill_id="F2",
                order_id="O2",
                strategy_id="S1",
                symbol="NIFTY_FUT",
                fill_timestamp="2023-01-01T09:20:00Z",
                fill_price=18050.0,
                quantity=2,
                side=-1,
                cost=18.05,
                slippage=9.025,
            ),
        ]
        # lot_size = 50
        # Notional 1: 18000 * 2 * 50 = 1,800,000
        # Notional 2: 18050 * 2 * 50 = 1,805,000
        # Total turnover: 3,605,000
        # Total friction: (18.0 + 18.05) + (9.0 + 9.025) = 36.05 + 18.025 = 54.075
        # Friction bps: (54.075 / 3,605,000) * 10000 = 0.15 bps
        bps = compute_cost_to_turnover_bps_from_fills(fills, lot_size=50)
        assert bps == 0.15

    def test_realized_slippage_bps_exact_math(self) -> None:
        # Total slippage: 180.0 on turnover 3,600,000 -> (180 / 3,600,000) * 10,000 = 0.50 bps
        assert compute_realized_slippage_bps(180.0, 3600000.0) == 0.50
        assert compute_realized_slippage_bps(180.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 6. Frequency & Drawdown Expansion Tests
# ---------------------------------------------------------------------------


class TestFrequencyAndDrawdownExpansion:
    def test_trade_frequency_ratio_exact(self) -> None:
        # Observed: 30 trades over 15 sessions = 2.0 trades/session
        # Baseline: 200 trades over 200 sessions = 1.0 trades/session
        # Ratio = 2.0 / 1.0 = 2.0
        ratio = compute_trade_frequency_ratio(
            observed_trades=30,
            observed_sessions=15,
            baseline_trades=200,
            baseline_sessions=200,
            min_observed_sessions=10,
        )
        assert ratio == 2.0

    def test_trade_frequency_ratio_insufficient_sessions(self) -> None:
        assert (
            compute_trade_frequency_ratio(
                observed_trades=10,
                observed_sessions=5,  # < 10
                baseline_trades=200,
                baseline_sessions=200,
                min_observed_sessions=10,
            )
            is None
        )

    def test_trade_frequency_ratio_zero_baseline(self) -> None:
        assert (
            compute_trade_frequency_ratio(
                observed_trades=15,
                observed_sessions=15,
                baseline_trades=0,
                baseline_sessions=200,
            )
            is None
        )

    def test_drawdown_expansion_ratio_exact(self) -> None:
        # Observed DD: 1,200 bps
        # Baseline Max DD: 800 bps
        # Expansion = 1200 / 800 = 1.50
        expansion = compute_drawdown_expansion_ratio(
            observed_drawdown_bps=1200.0,
            baseline_max_drawdown_bps=800.0,
            observed_sessions=10,
            min_sessions=5,
        )
        assert expansion == 1.50

    def test_drawdown_expansion_ratio_insufficient_sessions(self) -> None:
        assert (
            compute_drawdown_expansion_ratio(
                observed_drawdown_bps=1200.0,
                baseline_max_drawdown_bps=800.0,
                observed_sessions=3,
                min_sessions=5,
            )
            is None
        )

    def test_drawdown_expansion_ratio_zero_baseline_clamped(self) -> None:
        # Baseline max DD is 0.0 -> clamped to 1.0 bps to prevent zero division
        expansion = compute_drawdown_expansion_ratio(
            observed_drawdown_bps=50.0,
            baseline_max_drawdown_bps=0.0,
            observed_sessions=10,
            min_sessions=5,
        )
        assert expansion == 50.0


# ---------------------------------------------------------------------------
# 7. Risk, Order, and Holding Duration Tests
# ---------------------------------------------------------------------------


class TestRiskOrderAndDurationMetrics:
    def test_count_risk_events(self) -> None:
        events = [
            PaperRiskEvent(
                event_id="E1",
                order_id="O1",
                strategy_id="S1",
                rule_name="max_position",
                limit_value=5.0,
                requested_value=6.0,
                timestamp="2023-01-01T09:15:00Z",
                reason="limit breached",
            )
        ]
        assert count_risk_events(events) == 1
        assert count_risk_events([]) == 0

    def test_count_rejected_orders_and_rejection_rate(self) -> None:
        orders = [
            PaperOrder(
                order_id="O1",
                strategy_id="S1",
                qualification_id="Q1",
                symbol="NIFTY",
                side=1,
                quantity=1,
                status=PaperOrderStatus.FILLED,
            ),
            PaperOrder(
                order_id="O2",
                strategy_id="S1",
                qualification_id="Q1",
                symbol="NIFTY",
                side=1,
                quantity=1,
                status=PaperOrderStatus.REJECTED,
            ),
            PaperOrder(
                order_id="O3",
                strategy_id="S1",
                qualification_id="Q1",
                symbol="NIFTY",
                side=1,
                quantity=1,
                status=PaperOrderStatus.REJECTED,
            ),
            PaperOrder(
                order_id="O4",
                strategy_id="S1",
                qualification_id="Q1",
                symbol="NIFTY",
                side=1,
                quantity=1,
                status=PaperOrderStatus.SUBMITTED,
            ),
        ]
        assert count_rejected_orders(orders) == 2
        # 2 rejected / 4 submitted = 0.50
        assert compute_rejection_rate(rejected_orders=2, submitted_orders=4) == 0.50
        assert compute_rejection_rate(rejected_orders=0, submitted_orders=0) is None

    def test_mean_holding_duration_bars(self) -> None:
        assert compute_mean_holding_duration_bars([1, 2, 3, 4]) == 2.5
        assert compute_mean_holding_duration_bars([]) is None


# ---------------------------------------------------------------------------
# 8. Exposure & Session Spill Tests
# ---------------------------------------------------------------------------


class TestExposureAndSpillMetrics:
    def test_peak_gross_exposure(self) -> None:
        positions = [
            PaperPosition(
                symbol="NIFTY",
                quantity=2,
                entry_price=18000.0,
                current_price=18000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                fees_costs=0.0,
            ),
            PaperPosition(
                symbol="NIFTY",
                quantity=-4,
                entry_price=18100.0,
                current_price=18100.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                fees_costs=0.0,
            ),
        ]
        # lot_size = 50
        # Pos 1: |2| * 18000 * 50 = 1,800,000
        # Pos 2: |-4| * 18100 * 50 = 3,620,000
        # Peak = 3,620,000
        peak = compute_peak_gross_exposure(positions, lot_size=50)
        assert peak == 3620000.0
        assert compute_peak_gross_exposure([], lot_size=50) == 0.0

    def test_limit_utilization_ratio(self) -> None:
        # Peak observed: 3,620,000 on limit 5,000,000 -> 3.62 / 5.0 = 0.724
        assert compute_limit_utilization_ratio(3620000.0, 5000000.0) == 0.724
        assert compute_limit_utilization_ratio(3620000.0, 0.0) is None

    def test_count_session_overnight_spills(self) -> None:
        trades = [
            {"entry_session_id": "2023-01-01", "exit_session_id": "2023-01-01"},  # Intraday
            {"entry_session_id": "2023-01-01", "exit_session_id": "2023-01-02"},  # Spilled across session boundary!
            ("2023-01-02", "2023-01-02"),  # Intraday tuple
            ("2023-01-02", "2023-01-03"),  # Spilled tuple
        ]
        assert count_session_overnight_spills(trades) == 2
        assert count_session_overnight_spills([]) == 0


# ---------------------------------------------------------------------------
# 9. Property-Style Invariant Tests
# ---------------------------------------------------------------------------


class TestPropertyInvariants:
    def test_identical_inputs_produce_identical_outputs(self) -> None:
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002, 0.01, -0.008, 0.012]
        s1 = compute_rolling_sharpe(returns, window_size=10, min_sessions=10)
        s2 = compute_rolling_sharpe(returns, window_size=10, min_sessions=10)
        assert s1 == s2

    def test_win_rate_strictly_bounded_in_zero_one(self) -> None:
        for wins, total in [(0, 10), (5, 10), (10, 10)]:
            trades = [1.0] * wins + [-1.0] * (total - wins)
            wr = compute_rolling_win_rate(trades, min_trades=5)
            assert wr is not None
            assert 0.0 <= wr <= 1.0

    def test_cost_to_turnover_bps_non_negative(self) -> None:
        bps = compute_cost_to_turnover_bps(100.0, 50.0, 100000.0)
        assert bps >= 0.0
