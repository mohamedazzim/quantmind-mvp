from pathlib import Path

import numpy as np

from quantmind.research_integrity.acquisition_gate import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    false_pass_gate_passes,
    load_acquisition_gate_config,
    load_protocol_config,
    positive_control_gate_passes,
)
from quantmind.synthetic import SyntheticConfig, SyntheticFuturesGenerator, block_bootstrap_returns, directional_intraday_null, inject_feature_conditioned_edge


PROTOCOL = Path(__file__).parents[2] / "configs" / "research_protocol_v2.yaml"


def test_acquisition_gate_reads_yaml_source_of_truth():
    raw = load_protocol_config(PROTOCOL)
    cfg = load_acquisition_gate_config(PROTOCOL)
    gate = raw["synthetic_acquisition_gate"]
    strong = gate["strong_positive_control"]

    assert cfg.min_surrogate_runs == gate["min_surrogate_runs"]
    assert cfg.max_false_passes == gate["max_false_passes"]
    assert cfg.false_pass_rate_limit == gate["false_pass_rate_limit"]
    assert cfg.false_pass_confidence == gate["false_pass_confidence"]
    assert cfg.min_positive_control_runs == gate["min_positive_control_runs"]
    assert cfg.min_positive_detections == gate["min_positive_detections"]
    assert cfg.positive_detection_rate_limit == gate["positive_detection_rate_limit"]
    assert cfg.positive_detection_confidence == gate["positive_detection_confidence"]
    assert cfg.round_trip_cost_bps == strong["round_trip_cost_bps"]
    assert cfg.strong_positive_control_net_edge_bps == strong["net_edge_bps"]
    assert cfg.strong_positive_control_gross_edge_bps == strong["gross_edge_bps"]
    assert cfg.positive_control_condition == strong["condition"]
    assert cfg.positive_control_signal_window == tuple(float(v) for v in strong["signal_window"])
    assert cfg.positive_control_tail_quantile == strong["tail_quantile"]
    assert cfg.positive_control_lookback_sessions == strong["lookback_sessions"]
    assert cfg.positive_control_target_triggers_per_session == tuple(float(v) for v in strong["target_triggers_per_session"])
    assert cfg.positive_control_max_lag1_autocorrelation == strong["max_lag1_autocorrelation"]


def test_acquisition_gate_false_pass_threshold_is_one_sided_statistically_locked():
    cfg = load_acquisition_gate_config(PROTOCOL)
    assert clopper_pearson_upper(4, 200, cfg.false_pass_confidence) <= 0.05
    assert clopper_pearson_upper(5, 200, cfg.false_pass_confidence) > 0.05
    assert false_pass_gate_passes(4, 200, cfg)
    assert not false_pass_gate_passes(5, 200, cfg)


def test_acquisition_gate_positive_control_threshold_is_one_sided_statistically_locked():
    cfg = load_acquisition_gate_config(PROTOCOL)
    assert clopper_pearson_lower(87, 100, cfg.positive_detection_confidence) >= 0.80
    assert clopper_pearson_lower(86, 100, cfg.positive_detection_confidence) < 0.80
    assert positive_control_gate_passes(87, 100, cfg)
    assert not positive_control_gate_passes(86, 100, cfg)


def test_surrogate_is_cross_session_not_directional_intraday_null():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=60, bars_per_session=30, seed=4)).generate()
    cross_session = block_bootstrap_returns(base, block_size=5, seed=99)
    directional_null = directional_intraday_null(base, seed=100)

    assert cross_session["timestamp"].tolist() == base["timestamp"].tolist()
    assert "control_signal" not in cross_session.columns
    assert directional_null["volume"].equals(base["volume"])
    assert directional_null["open_interest"].equals(base["open_interest"])


def test_full_gate_thresholds_have_expected_integer_boundaries():
    cfg = load_acquisition_gate_config(PROTOCOL)
    assert cfg.min_surrogate_runs == 200
    assert cfg.max_false_passes == 4
    assert cfg.min_positive_control_runs == 100
    assert cfg.min_positive_detections == 87
    assert cfg.round_trip_cost_bps == 2.0
    assert cfg.strong_positive_control_net_edge_bps == 10.0
    assert cfg.strong_positive_control_gross_edge_bps == 12.0
    assert cfg.strong_positive_control_net_edge_bps == (
        cfg.strong_positive_control_gross_edge_bps - cfg.round_trip_cost_bps
    )


def test_null_gate_fixture_can_be_generated_without_future_information():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=20, bars_per_session=30, seed=8)).generate()
    null = directional_intraday_null(base, seed=12)
    close = null["close"].to_numpy(dtype=float)
    prior = np.r_[0.0, np.diff(np.log(close))]
    # No future column/label is emitted by the null generator.
    assert len(prior) == len(null)
    assert "future_return" not in null.columns


def test_positive_control_parameters_are_read_from_protocol_config():
    raw = load_protocol_config(PROTOCOL)
    cfg = load_acquisition_gate_config(PROTOCOL)
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=30, bars_per_session=60, seed=88)).generate()
    protocol_driven = inject_feature_conditioned_edge(
        base,
        net_edge_bps=cfg.strong_positive_control_net_edge_bps,
        round_trip_cost_bps=cfg.round_trip_cost_bps,
        condition=cfg.positive_control_condition,
        session_window=cfg.positive_control_signal_window,
        tail_quantile=cfg.positive_control_tail_quantile,
        lookback_sessions=cfg.positive_control_lookback_sessions,
        calendar_session_bars=cfg.positive_control_calendar_session_bars,
    )
    default_same_protocol = inject_feature_conditioned_edge(
        base,
        net_edge_bps=raw["synthetic_acquisition_gate"]["strong_positive_control"]["net_edge_bps"],
        round_trip_cost_bps=raw["synthetic_acquisition_gate"]["strong_positive_control"]["round_trip_cost_bps"],
        condition=raw["synthetic_acquisition_gate"]["strong_positive_control"]["condition"],
        session_window=tuple(raw["synthetic_acquisition_gate"]["strong_positive_control"]["signal_window"]),
        tail_quantile=raw["synthetic_acquisition_gate"]["strong_positive_control"]["tail_quantile"],
        lookback_sessions=raw["synthetic_acquisition_gate"]["strong_positive_control"]["lookback_sessions"],
        calendar_session_bars=raw["synthetic_acquisition_gate"]["strong_positive_control"]["calendar_session_bars"],
    )
    np.testing.assert_array_equal(protocol_driven["control_signal"].to_numpy(), default_same_protocol["control_signal"].to_numpy())
