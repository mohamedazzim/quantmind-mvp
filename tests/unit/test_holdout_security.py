from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pytest
import pandas as pd
import numpy as np

from quantmind.backtest.engine import BacktestConfig, BacktestEngine
from quantmind.data import (
    DatasetKind,
    DatasetRegistry,
    DatasetRegistryError,
    PurgeEmbargoSpec,
    SplitZone,
    compute_split_manifest,
)
from quantmind.research_integrity import (
    HoldoutManager,
    HoldoutSecurityError,
    HoldoutState,
    ResearchBudget,
    ResearchHarness,
    TrialLedger,
)
from quantmind.strategy import StrategySpec


def _generate_dataset(path: Path, num_bars: int = 120) -> tuple[Path, list[pd.Timestamp]]:
    base_ts = pd.Timestamp("2026-01-01 09:15:00")
    timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(num_bars)]
    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "open": 100.0 + np.sin(np.arange(num_bars) / 5.0),
            "high": 102.0 + np.sin(np.arange(num_bars) / 5.0),
            "low": 98.0 + np.sin(np.arange(num_bars) / 5.0),
            "close": 100.5 + np.sin(np.arange(num_bars) / 5.0),
            "volume": 1000,
            "open_interest": 500,
        }
    )
    df.to_csv(path, index=False)
    return path, timestamps


@pytest.fixture
def test_setup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "licensed_data.csv"
        _, timestamps = _generate_dataset(csv_path, num_bars=120)

        spec = PurgeEmbargoSpec(feature_lookback_bars=2, prediction_horizon_bars=2, embargo_bars=2)
        manifest = compute_split_manifest(
            dataset_version="LICENSED-V1",
            timestamps=timestamps,
            spec=spec,
            research_ratio=0.4,
            validation_ratio=0.3,
            holdout_ratio=0.3,
        )

        registry = DatasetRegistry()
        registry.register_file(
            version="LICENSED-V1",
            kind=DatasetKind.LICENSED,
            path=csv_path,
            split_manifest=manifest,
        )

        engine = BacktestEngine()
        ledger = TrialLedger()
        holdout_mgr = HoldoutManager(engine, ledger, registry)
        harness = ResearchHarness(engine, ledger, registry, holdout_manager=holdout_mgr)

        strategy_spec = StrategySpec(
            strategy_version="1.0.0",
            feature_version="f_v1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 20, "session_window": [0.10, 0.90]},
        )

        yield {
            "registry": registry,
            "engine": engine,
            "ledger": ledger,
            "holdout_mgr": holdout_mgr,
            "harness": harness,
            "manifest": manifest,
            "strategy_spec": strategy_spec,
            "dataset_version": "LICENSED-V1",
            "protocol_version": "v1.0",
        }


def test_exploit_a_research_call_requests_final_holdout(test_setup):
    """A. Research call requests FINAL_HOLDOUT -> reject."""
    harness = test_setup["harness"]
    with pytest.raises(ValueError, match="FINAL_HOLDOUT zone is sealed"):
        harness.run_trial(
            config=BacktestConfig(),
            research_task_id="task-exploit-a",
            strategy_spec=test_setup["strategy_spec"],
            dataset_version=test_setup["dataset_version"],
            split_zone="FINAL_HOLDOUT",
            research_protocol_version=test_setup["protocol_version"],
            seed=42,
            budget=ResearchBudget(),
        )


def test_exploit_b_agent_or_tool_directly_requests_final_holdout(test_setup):
    """B. Agent/tool directly requests FINAL_HOLDOUT via load_zone -> reject."""
    registry = test_setup["registry"]
    with pytest.raises(DatasetRegistryError, match="FINAL_HOLDOUT zone is sealed"):
        registry.load_zone(test_setup["dataset_version"], "FINAL_HOLDOUT")

    with pytest.raises(DatasetRegistryError, match="FINAL_HOLDOUT zone is sealed"):
        registry.load_zone(test_setup["dataset_version"], SplitZone.FINAL_HOLDOUT)


def test_exploit_c_optimization_job_requests_final_holdout(test_setup):
    """C. Optimization / tuning job requests FINAL_HOLDOUT across multiple trials -> reject."""
    harness = test_setup["harness"]
    for param_val in [0.05, 0.10, 0.15]:
        spec = StrategySpec(
            strategy_version="1.0.0",
            feature_version="f_v1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 20, "session_window": [param_val, 0.90]},
        )
        with pytest.raises(ValueError, match="FINAL_HOLDOUT zone is sealed"):
            harness.run_trial(
                config=BacktestConfig(),
                research_task_id=f"opt-task-{param_val}",
                strategy_spec=spec,
                dataset_version=test_setup["dataset_version"],
                split_zone="FINAL_HOLDOUT",
                research_protocol_version=test_setup["protocol_version"],
                seed=42,
                budget=ResearchBudget(),
            )


def test_exploit_d_and_g_final_evaluation_path_requests_holdout_allowed_and_e2e(test_setup):
    """D & G. Final evaluation path requests FINAL_HOLDOUT -> allowed through controlled gateway.
    Also tests synthetic/e2e controlled evaluation."""
    harness = test_setup["harness"]
    holdout_mgr = test_setup["holdout_mgr"]
    spec = test_setup["strategy_spec"]
    dataset_version = test_setup["dataset_version"]
    protocol_version = test_setup["protocol_version"]

    # 1. Gate check: candidate cannot evaluate holdout directly without passing earlier research trial
    with pytest.raises(HoldoutSecurityError, match="not passed required pre-holdout gates"):
        holdout_mgr.final_evaluate(
            strategy_spec=spec,
            dataset_version=dataset_version,
            research_protocol_version=protocol_version,
            config=BacktestConfig(),
        )

    # 2. Run valid research trial on RESEARCH zone first
    res = harness.run_trial(
        config=BacktestConfig(),
        research_task_id="legitimate-research-task",
        strategy_spec=spec,
        dataset_version=dataset_version,
        split_zone=SplitZone.RESEARCH,
        research_protocol_version=protocol_version,
        seed=42,
        budget=ResearchBudget(),
    )
    assert res is not None

    # 3. Now final evaluation is permitted
    eval_result = holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version=dataset_version,
        research_protocol_version=protocol_version,
        config=BacktestConfig(),
    )
    assert eval_result is not None
    assert eval_result.state in (HoldoutState.PASSED, HoldoutState.FAILED)
    assert eval_result.trial_id.startswith("TRIAL-HOLDOUT-")

    # Verify trial ledger recorded split_zone="FINAL_HOLDOUT"
    holdout_trial = test_setup["ledger"].get(eval_result.trial_id)
    assert holdout_trial["split_zone"] == "FINAL_HOLDOUT"


def test_exploit_e_candidate_attempts_second_holdout_evaluation(test_setup):
    """E. Candidate attempts second holdout evaluation -> reject."""
    harness = test_setup["harness"]
    holdout_mgr = test_setup["holdout_mgr"]
    spec = test_setup["strategy_spec"]
    dataset_version = test_setup["dataset_version"]
    protocol_version = test_setup["protocol_version"]

    # Pass pre-holdout gate
    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="research-gate",
        strategy_spec=spec,
        dataset_version=dataset_version,
        split_zone="RESEARCH",
        research_protocol_version=protocol_version,
        seed=42,
        budget=ResearchBudget(),
    )

    # First evaluation succeeds
    holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version=dataset_version,
        research_protocol_version=protocol_version,
        config=BacktestConfig(),
    )

    # Second evaluation attempt is rejected
    with pytest.raises(HoldoutSecurityError, match="repeat evaluation or tuning.*strictly forbidden"):
        holdout_mgr.final_evaluate(
            strategy_spec=spec,
            dataset_version=dataset_version,
            research_protocol_version=protocol_version,
            config=BacktestConfig(),
        )


def test_exploit_f_failed_candidate_attempts_to_retune_against_same_holdout(test_setup):
    """F. Failed candidate attempts to retune against same holdout -> reject."""
    harness = test_setup["harness"]
    holdout_mgr = test_setup["holdout_mgr"]
    spec = test_setup["strategy_spec"]
    dataset_version = test_setup["dataset_version"]
    protocol_version = test_setup["protocol_version"]

    # Pass pre-holdout gate
    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="research-gate",
        strategy_spec=spec,
        dataset_version=dataset_version,
        split_zone="RESEARCH",
        research_protocol_version=protocol_version,
        seed=42,
        budget=ResearchBudget(),
    )

    # Force failure with impossible pass criterion
    eval_result = holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version=dataset_version,
        research_protocol_version=protocol_version,
        config=BacktestConfig(),
        pass_criterion=lambda res: False,  # unconditionally fail
    )
    assert eval_result.state == HoldoutState.FAILED
    assert eval_result.status == "REJECTED_FINAL_HOLDOUT"

    # Attempt to re-run / tune in ResearchHarness with the failed candidate -> rejected
    with pytest.raises(HoldoutSecurityError, match="failed final holdout.*cannot be retuned"):
        harness.run_trial(
            config=BacktestConfig(),
            research_task_id="retuning-attempt",
            strategy_spec=spec,
            dataset_version=dataset_version,
            split_zone="RESEARCH",
            research_protocol_version=protocol_version,
            seed=42,
            budget=ResearchBudget(),
        )


def test_burned_holdout_blocks_all_evaluations_and_tuning(test_setup):
    """Burned holdout blocks all evaluations and tuning."""
    harness = test_setup["harness"]
    holdout_mgr = test_setup["holdout_mgr"]
    spec = test_setup["strategy_spec"]
    dataset_version = test_setup["dataset_version"]
    protocol_version = test_setup["protocol_version"]

    holdout_mgr.burn_holdout(dataset_version, protocol_version, reason="suspected data snooping leak")

    assert holdout_mgr.is_holdout_burned(dataset_version, protocol_version)
    assert holdout_mgr.get_holdout_state("ANY_STRAT", dataset_version, protocol_version) == HoldoutState.BURNED

    with pytest.raises(HoldoutSecurityError, match="is BURNED"):
        harness.run_trial(
            config=BacktestConfig(),
            research_task_id="tuning-burned",
            strategy_spec=spec,
            dataset_version=dataset_version,
            split_zone="RESEARCH",
            research_protocol_version=protocol_version,
            seed=42,
            budget=ResearchBudget(),
        )

    with pytest.raises(HoldoutSecurityError, match="is BURNED"):
        holdout_mgr.final_evaluate(
            strategy_spec=spec,
            dataset_version=dataset_version,
            research_protocol_version=protocol_version,
            config=BacktestConfig(),
        )
