from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.stats import beta


@dataclass(frozen=True)
class AcquisitionGateConfig:
    min_surrogate_runs: int
    false_pass_rate_limit: float
    false_pass_confidence: float
    max_false_passes: int
    min_positive_control_runs: int
    positive_detection_rate_limit: float
    positive_detection_confidence: float
    min_positive_detections: int
    round_trip_cost_bps: float
    strong_positive_control_net_edge_bps: float
    strong_positive_control_gross_edge_bps: float
    positive_control_edge_ladder_net_bps: tuple[float, ...]
    positive_control_condition: str
    positive_control_signal_window: tuple[float, float]
    positive_control_tail_quantile: float
    positive_control_lookback_sessions: int
    positive_control_calendar_session_bars: int
    positive_control_target_triggers_per_session: tuple[float, float]
    positive_control_max_lag1_autocorrelation: float


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML research protocol; YAML is the source of truth."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_acquisition_gate_config(path: str | Path) -> AcquisitionGateConfig:
    raw = load_protocol_config(path)["synthetic_acquisition_gate"]
    control = raw["strong_positive_control"]
    return AcquisitionGateConfig(
        min_surrogate_runs=int(raw["min_surrogate_runs"]),
        false_pass_rate_limit=float(raw["false_pass_rate_limit"]),
        false_pass_confidence=float(raw["false_pass_confidence"]),
        max_false_passes=int(raw["max_false_passes"]),
        min_positive_control_runs=int(raw["min_positive_control_runs"]),
        positive_detection_rate_limit=float(raw["positive_detection_rate_limit"]),
        positive_detection_confidence=float(raw["positive_detection_confidence"]),
        min_positive_detections=int(raw["min_positive_detections"]),
        round_trip_cost_bps=float(control["round_trip_cost_bps"]),
        strong_positive_control_net_edge_bps=float(control["net_edge_bps"]),
        strong_positive_control_gross_edge_bps=float(control["gross_edge_bps"]),
        positive_control_edge_ladder_net_bps=tuple(float(v) for v in raw["edge_ladder_net_bps"]),
        positive_control_condition=str(control["condition"]),
        positive_control_signal_window=tuple(float(v) for v in control["signal_window"]),
        positive_control_tail_quantile=float(control["tail_quantile"]),
        positive_control_lookback_sessions=int(control["lookback_sessions"]),
        positive_control_calendar_session_bars=int(control["calendar_session_bars"]),
        positive_control_target_triggers_per_session=tuple(float(v) for v in control["target_triggers_per_session"]),
        positive_control_max_lag1_autocorrelation=float(control["max_lag1_autocorrelation"]),
    )


def clopper_pearson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    alpha = 1.0 - confidence
    if successes == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def clopper_pearson_lower(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    alpha = 1.0 - confidence
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def false_pass_gate_passes(false_passes: int, surrogate_runs: int, config: AcquisitionGateConfig) -> bool:
    if surrogate_runs < config.min_surrogate_runs:
        return False
    if false_passes > config.max_false_passes:
        return False
    upper = clopper_pearson_upper(false_passes, surrogate_runs, config.false_pass_confidence)
    return upper <= config.false_pass_rate_limit


def positive_control_gate_passes(
    detections: int,
    control_runs: int,
    config: AcquisitionGateConfig,
) -> bool:
    if control_runs < config.min_positive_control_runs:
        return False
    if detections < config.min_positive_detections:
        return False
    lower = clopper_pearson_lower(detections, control_runs, config.positive_detection_confidence)
    return lower >= config.positive_detection_rate_limit
