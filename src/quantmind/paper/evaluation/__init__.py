"""QuantMind Paper Evaluation & Monitoring Subsystem (PRD v4.0).

This package provides window-bounded evaluation, rolling performance and execution
quality metrics, degradation detection, and immutable evaluation evidence.
"""

from .ledger import (
    EvaluationLedger,
    EvaluationLedgerError,
    EvaluationLedgerIntegrityError,
)
from .metrics import (
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
from .models import (
    DegradationEvent,
    MonitoringConfig,
    MonitoringConfigError,
    MonitoringProvenanceError,
    MonitoringSnapshot,
    PaperEvaluationTransition,
    ResearchFeedbackRecord,
)

__all__ = [
    "DegradationEvent",
    "EvaluationLedger",
    "EvaluationLedgerError",
    "EvaluationLedgerIntegrityError",
    "MonitoringConfig",
    "MonitoringConfigError",
    "MonitoringProvenanceError",
    "MonitoringSnapshot",
    "PaperEvaluationTransition",
    "ResearchFeedbackRecord",
    "compute_cost_to_turnover_bps",
    "compute_cost_to_turnover_bps_from_fills",
    "compute_cumulative_net_pnl",
    "compute_drawdown_expansion_ratio",
    "compute_limit_utilization_ratio",
    "compute_mean_holding_duration_bars",
    "compute_peak_gross_exposure",
    "compute_realized_slippage_bps",
    "compute_realized_slippage_bps_from_fills",
    "compute_rejection_rate",
    "compute_rolling_drawdown_bps",
    "compute_rolling_expectancy",
    "compute_rolling_net_pnl",
    "compute_rolling_sharpe",
    "compute_rolling_win_rate",
    "compute_session_returns",
    "compute_trade_frequency_ratio",
    "count_rejected_orders",
    "count_risk_events",
    "count_session_overnight_spills",
]
