import pandas as pd
import pytest

from quantmind.research_integrity.harness import derive_strategy_id
from quantmind.strategy import StrategySpec, StrategySpecError, compile_strategy_spec


def momentum_spec():
    return StrategySpec(
        strategy_version="V1",
        feature_version="F1",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 20, "session_window": [0.10, 0.90]},
    )


def mean_reversion_spec():
    return StrategySpec(
        strategy_version="V1",
        feature_version="F1",
        signal_name="current_bar_mean_reversion",
        parameters={"calendar_session_bars": 20, "session_window": [0.10, 0.90]},
    )


def test_strategy_spec_hash_is_stable_and_logic_sensitive():
    assert derive_strategy_id(momentum_spec()) == derive_strategy_id(momentum_spec())
    assert derive_strategy_id(momentum_spec()) != derive_strategy_id(mean_reversion_spec())


def test_compiler_only_accepts_whitelisted_signal_names():
    with pytest.raises(StrategySpecError, match="unsupported signal_name"):
        compile_strategy_spec(
            StrategySpec("V1", "F1", "arbitrary_python", {"code": "..."})
        )


def test_compiler_rejects_unknown_parameters():
    with pytest.raises(StrategySpecError, match="unsupported parameters"):
        compile_strategy_spec(
            StrategySpec(
                "V1",
                "F1",
                "current_bar_momentum",
                {"calendar_session_bars": 20, "session_window": [0.1, 0.9], "code": "..."},
            )
        )


def test_compiled_strategy_is_deterministic_and_causal():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:15", periods=20, freq="min"),
            "open": [100.0 + i for i in range(20)],
            "close": [101.0 + i for i in range(20)],
        }
    )
    fn = compile_strategy_spec(momentum_spec())
    first = fn(data)
    second = fn(data)
    assert first.tolist() == second.tolist()
    truncated = fn(data.iloc[:10])
    assert truncated.tolist() == first[:10].tolist()
