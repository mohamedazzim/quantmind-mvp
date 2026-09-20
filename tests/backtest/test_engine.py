from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from quantmind.backtest import (
    BacktestConfig,
    BacktestEngine,
    CostSchedule,
    CostSchedulePeriod,
    NonCausalSignalError,
    assert_causal_signal,
    prior_bar_momentum_signal,
)
from quantmind.data import DatasetKind, DatasetRegistry, DatasetZone
from quantmind.research_integrity import (
    ResearchBudget,
    ResearchHarness,
    TrialBudgetExceeded,
    TrialContext,
    TrialLedger,
)
from quantmind.strategy import StrategySpec
from quantmind.synthetic import (
    SyntheticConfig,
    SyntheticFuturesGenerator,
    directional_intraday_null,
    inject_feature_conditioned_edge,
    inject_gap_lookahead_canary,
)
from quantmind.testing import FixtureResearchHarness


def cost_schedule(bps: float = 2.0) -> CostSchedule:
    return CostSchedule(
        "SYNTH-RT-1",
        (CostSchedulePeriod(pd.Timestamp("2020-01-01").date(), None, bps),),
    )


def register_licensed_dataset(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    version: str,
    registry_path: Path | None = None,
    zones: dict[str, DatasetZone] | None = None,
) -> DatasetRegistry:
    csv_path = tmp_path / f"{version}.csv"
    frame.to_csv(csv_path, index=False)
    registry = DatasetRegistry(registry_path or (tmp_path / f"datasets-{version}.sqlite"))
    registry.register_file(
        version=version,
        kind=DatasetKind.LICENSED,
        path=csv_path,
        zones=zones or {"RESEARCH": DatasetZone("RESEARCH", None, None)},
    )
    return registry


def momentum_spec(*, version="S-1", feature="F-1", session_bars=20, window=(0.10, 0.90)):
    return StrategySpec(
        strategy_version=version,
        feature_version=feature,
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": session_bars, "session_window": list(window)},
    )


def mean_reversion_spec(*, version="S-1", feature="F-1", session_bars=20, window=(0.10, 0.90)):
    return StrategySpec(
        strategy_version=version,
        feature_version=feature,
        signal_name="current_bar_mean_reversion",
        parameters={"calendar_session_bars": session_bars, "session_window": list(window)},
    )


def test_next_bar_open_canary_is_not_capturable_but_same_close_is():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=25, bars_per_session=30, seed=102)).generate()
    canary = inject_gap_lookahead_canary(base, magnitude_bps=30.0)
    engine = BacktestEngine()
    next_cfg = BacktestConfig(execution_model="next_bar_open_v1", signal_column="canary_signal", cost_schedule=cost_schedule(0.0))
    same_cfg = BacktestConfig(execution_model="same_bar_close_v0", signal_column="canary_signal", cost_schedule=cost_schedule(0.0))
    base_next = engine._run_internal(base.assign(canary_signal=canary["canary_signal"]), config=next_cfg)
    canary_next = engine._run_internal(canary, config=next_cfg)
    base_same = engine._run_internal(base.assign(canary_signal=canary["canary_signal"]), config=same_cfg)
    canary_same = engine._run_internal(canary, config=same_cfg)
    assert abs(canary_next.mean_net_return_bps - base_next.mean_net_return_bps) < 0.05
    assert abs((canary_same.mean_net_return_bps - base_same.mean_net_return_bps) - 30.0) < 0.10


def test_backtester_recovers_control_net_edge_each_rung_with_paired_difference():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=250, bars_per_session=375, seed=103)).generate()
    engine = BacktestEngine()
    zero = inject_feature_conditioned_edge(base, net_edge_bps=0.0, round_trip_cost_bps=2.0, calendar_session_bars=375)
    zero_result = engine._run_internal(zero, config=BacktestConfig(execution_model="next_bar_open_v1", signal_column="control_signal", cost_schedule=cost_schedule(2.0)))
    observed = []
    for edge in (0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
        data = inject_feature_conditioned_edge(base, net_edge_bps=edge, round_trip_cost_bps=2.0, calendar_session_bars=375)
        result = engine._run_internal(data, config=BacktestConfig(execution_model="next_bar_open_v1", signal_column="control_signal", cost_schedule=cost_schedule(2.0)))
        paired_delta = result.mean_net_return_bps - zero_result.mean_net_return_bps
        observed.append(result.mean_net_return_bps)
        assert abs(paired_delta - edge) < 0.01
    assert all(b > a for a, b in zip(observed, observed[1:]))


def test_directional_null_momentum_is_cost_negative():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=100, bars_per_session=120, seed=104)).generate()
    null = directional_intraday_null(base, seed=105)
    result = BacktestEngine()._run_internal(null, config=BacktestConfig(execution_model="next_bar_open_v1", cost_schedule=cost_schedule(2.0)), signal_fn=lambda df: prior_bar_momentum_signal(df, calendar_session_bars=120))
    assert result.trade_count > 5000
    assert -2.6 < result.mean_net_return_bps < -1.4


def test_backtester_uses_date_effective_cost_schedule():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=10, bars_per_session=20, seed=106)).generate()
    split_day = pd.to_datetime(base["timestamp"]).dt.normalize().iloc[5 * 20].date()
    schedule = CostSchedule("DATED", (
        CostSchedulePeriod(pd.Timestamp("2020-01-01").date(), split_day - pd.Timedelta(days=1), 2.0),
        CostSchedulePeriod(split_day, None, 5.0),
    ))
    signals = np.zeros(len(base), dtype=int)
    for idx in base.groupby(pd.to_datetime(base["timestamp"]).dt.normalize(), sort=False).groups.values():
        ids = list(idx)
        signals[ids[1]] = 1
    result = BacktestEngine()._run_internal(base.assign(signal=signals), config=BacktestConfig(signal_column="signal", cost_schedule=schedule))
    assert result.trade_count == 10
    assert {trade.cost_bps for trade in result.trades if trade.entry_timestamp.date() < split_day} == {2.0}
    assert {trade.cost_bps for trade in result.trades if trade.entry_timestamp.date() >= split_day} == {5.0}


def _future_direction_signal(df: pd.DataFrame) -> np.ndarray:
    closes = df["close"].to_numpy(dtype=float)
    signal = np.zeros(len(df), dtype=int)
    for i in range(len(df) - 1):
        r = np.log(closes[i + 1] / closes[i])
        signal[i] = 1 if r > 0 else -1 if r < 0 else 0
    return signal


def test_causality_preflight_accepts_current_bar_momentum_with_calendar_length():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=30, bars_per_session=60, seed=117)).generate()
    assert_causal_signal(base, lambda df: prior_bar_momentum_signal(df, calendar_session_bars=60), cut_points=[137, 389, 811, len(base) - 1])


def test_causality_preflight_rejects_future_looking_signal():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=20, bars_per_session=60, seed=118)).generate()
    ledger = TrialLedger()
    harness = FixtureResearchHarness(BacktestEngine(), ledger)
    with pytest.raises(NonCausalSignalError, match="non-causal signal"):
        harness.run_trial(
            base,
            signal_fn=_future_direction_signal,
            config=BacktestConfig(execution_model="next_bar_open_v1", cost_schedule=cost_schedule(0.0), causality_cut_points=(77, 221, 599, 911)),
            research_task_id="TASK-FUTURE",
            strategy_spec=momentum_spec(version="future-v1", session_bars=60),
            dataset_version="D",
            research_protocol_version="RP",
            seed=1,
            budget=ResearchBudget(max_trials=3, max_experiments=3, max_strategy_variants=3, max_runtime_minutes=1),
        )
    assert ledger.usage("D", "RP", mode="FIXTURE")["trials"] == 1
    row = ledger._connection.execute("SELECT mode, dataset_kind, status FROM trials").fetchone()
    assert row["mode"] == "FIXTURE" and row["dataset_kind"] == "SYNTHETIC" and row["status"] == "REJECTED_NONCAUSAL"
    ledger.close()


def test_current_bar_momentum_uses_bar_t():
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=5, bars_per_session=20, seed=119)).generate()
    signal = prior_bar_momentum_signal(base, calendar_session_bars=20, session_window=(0.0, 1.0))
    intrabar = np.log(base["close"].to_numpy(dtype=float) / base["open"].to_numpy(dtype=float))
    session = pd.to_datetime(base["timestamp"]).dt.normalize()
    for _, idx in session.groupby(session, sort=False).groups.items():
        ids = np.asarray(idx, dtype=int)
        assert np.array_equal(signal[ids], np.sign(intrabar[ids]).astype(int))


def test_engine_public_entrypoint_is_blocked_without_harness():
    with pytest.raises(RuntimeError, match="ResearchHarness.run_trial"):
        BacktestEngine().run(pd.DataFrame(), config=BacktestConfig())


def test_harness_owns_trial_count_and_loads_data_from_registry(tmp_path):
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=4, bars_per_session=20, seed=151)).generate()
    ledger = TrialLedger()
    registry = register_licensed_dataset(tmp_path, base, version="D-1")
    harness = ResearchHarness(BacktestEngine(), ledger, registry)
    result = harness.run_trial(config=BacktestConfig(cost_schedule=cost_schedule(2.0)), research_task_id="TASK-1", strategy_spec=momentum_spec(session_bars=20), dataset_version="D-1", split_zone="RESEARCH", research_protocol_version="RP-2", seed=1, budget=ResearchBudget(max_trials=10, max_experiments=10, max_strategy_variants=10, max_runtime_minutes=1))
    assert result.trade_count > 0
    assert ledger.usage("D-1", "RP-2")["trials"] == 1
    ledger.close(); registry.close()


def test_production_harness_rejects_signal_column_before_reserve(tmp_path):
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=2, bars_per_session=20, seed=156)).generate()
    ledger = TrialLedger(); registry = register_licensed_dataset(tmp_path, base, version="D")
    harness = ResearchHarness(BacktestEngine(), ledger, registry)
    with pytest.raises(ValueError, match="signal_column is test-fixture-only"):
        harness.run_trial(config=BacktestConfig(signal_column="signal"), research_task_id="TASK-COL", strategy_spec=momentum_spec(session_bars=20), dataset_version="D", split_zone="RESEARCH", research_protocol_version="RP", seed=1, budget=ResearchBudget(max_trials=3, max_experiments=3, max_strategy_variants=3, max_runtime_minutes=1))
    assert ledger.usage("D", "RP")["trials"] == 0
    ledger.close(); registry.close()


def test_production_harness_rejects_arbitrary_signal_callable():
    ledger = TrialLedger(); registry = DatasetRegistry(); harness = ResearchHarness(BacktestEngine(), ledger, registry)
    with pytest.raises(TypeError, match="unexpected keyword argument 'signal_fn'"):
        harness.run_trial(config=BacktestConfig(), research_task_id="TASK-FN", strategy_spec=momentum_spec(session_bars=20), dataset_version="D", split_zone="RESEARCH", research_protocol_version="RP", seed=1, budget=ResearchBudget(max_trials=3, max_experiments=3, max_strategy_variants=3, max_runtime_minutes=1), signal_fn=lambda df: np.ones(len(df)))
    assert ledger.usage("D", "RP")["trials"] == 0
    ledger.close(); registry.close()


def test_production_harness_refuses_synthetic_dataset(tmp_path):
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=2, bars_per_session=20, seed=158)).generate()
    csv_path = tmp_path / "synthetic.csv"; base.to_csv(csv_path, index=False)
    registry = DatasetRegistry(tmp_path / "datasets.sqlite")
    registry.register_file(version="SYN", kind=DatasetKind.SYNTHETIC, path=csv_path, zones={"RESEARCH": DatasetZone("RESEARCH", None, None)})
    ledger = TrialLedger(); harness = ResearchHarness(BacktestEngine(), ledger, registry)
    with pytest.raises(ValueError, match="production research requires LICENSED dataset"):
        harness.run_trial(config=BacktestConfig(cost_schedule=cost_schedule(0.0)), research_task_id="TASK-SYN", strategy_spec=momentum_spec(session_bars=20), dataset_version="SYN", split_zone="RESEARCH", research_protocol_version="RP", seed=1, budget=ResearchBudget(max_trials=3, max_experiments=3, max_strategy_variants=3, max_runtime_minutes=1))
    assert ledger.usage("SYN", "RP")["trials"] == 0
    ledger.close(); registry.close()


def test_fixture_trials_have_separate_mode_and_do_not_pollute_production(tmp_path):
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=2, bars_per_session=20, seed=159)).generate()
    ledger = TrialLedger(); fixture = FixtureResearchHarness(BacktestEngine(), ledger)
    fixture.run_trial(base, signal_fn=lambda df: np.sign(np.ones(len(df))), config=BacktestConfig(execution_model="next_bar_open_v1", cost_schedule=cost_schedule(0)), research_task_id="TASK-F", strategy_spec=momentum_spec(), dataset_version="D", research_protocol_version="RP", seed=1, budget=ResearchBudget(max_trials=3, max_experiments=3, max_strategy_variants=3, max_runtime_minutes=1))
    assert ledger.usage("D", "RP")["trials"] == 0
    assert ledger.usage("D", "RP", mode="FIXTURE")["trials"] == 1
    row = ledger._connection.execute("SELECT mode, dataset_kind FROM trials").fetchone()
    assert row["mode"] == "FIXTURE" and row["dataset_kind"] == "SYNTHETIC"
    ledger.close()


def test_harness_derives_logic_identity_not_strategy_version(tmp_path):
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=3, bars_per_session=20, seed=152)).generate()
    ledger = TrialLedger(); registry = register_licensed_dataset(tmp_path, base, version="D")
    harness = ResearchHarness(BacktestEngine(), ledger, registry)
    budget = ResearchBudget(max_trials=5, max_experiments=10, max_strategy_variants=2, max_runtime_minutes=1)
    common = dict(config=BacktestConfig(cost_schedule=cost_schedule(0.0)), research_task_id="TASK-A", dataset_version="D", split_zone="RESEARCH", research_protocol_version="RP", seed=1, budget=budget, estimated_runtime_minutes=0.01)
    harness.run_trial(**common, strategy_spec=momentum_spec(version="V1"))
    harness.run_trial(**{**common, "seed": 2}, strategy_spec=momentum_spec(version="V2"))
    assert ledger.usage("D", "RP")["strategy_variants"] == 1
    harness.run_trial(**{**common, "seed": 3}, strategy_spec=mean_reversion_spec(version="V2"))
    assert ledger.usage("D", "RP")["strategy_variants"] == 2
    ledger.close(); registry.close()


def test_budget_reservation_is_atomic_across_connections(tmp_path):
    db = tmp_path / "ledger.sqlite"; registry_db = tmp_path / "datasets.sqlite"
    budget = ResearchBudget(max_trials=10, max_experiments=100, max_strategy_variants=100, max_runtime_minutes=10)
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=2, bars_per_session=20, seed=153)).generate()
    register_licensed_dataset(tmp_path, base, version="D", registry_path=registry_db)

    def worker(i: int):
        ledger = TrialLedger(db); registry = DatasetRegistry(registry_db); harness = ResearchHarness(BacktestEngine(), ledger, registry)
        try:
            harness.run_trial(config=BacktestConfig(cost_schedule=cost_schedule(0.0)), research_task_id=f"T-{i}", strategy_spec=momentum_spec(version="V1", feature="F1", session_bars=20, window=(0.10, 0.90 if i % 2 == 0 else 0.80)), dataset_version="D", split_zone="RESEARCH", research_protocol_version="RP", seed=i, budget=budget, estimated_runtime_minutes=0.01)
            return True
        except TrialBudgetExceeded:
            return False
        finally:
            registry.close(); ledger.close()

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(worker, range(20)))
    final = TrialLedger(db)
    assert sum(results) == 10 and final.usage("D", "RP")["trials"] == 10
    final.close()


def test_invalid_strategy_spec_fails_before_reservation(tmp_path):
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=2, bars_per_session=20, seed=160)).generate()
    ledger = TrialLedger(); registry = register_licensed_dataset(tmp_path, base, version="D")
    harness = ResearchHarness(BacktestEngine(), ledger, registry)
    bad = StrategySpec("V1", "F1", "current_bar_momentum", {"calendar_session_bars": 0, "session_window": [0.9, 0.1]})
    with pytest.raises(ValueError, match="calendar_session_bars"):
        harness.run_trial(config=BacktestConfig(cost_schedule=cost_schedule(0)), research_task_id="TASK-BAD", strategy_spec=bad, dataset_version="D", split_zone="RESEARCH", research_protocol_version="RP", seed=1, budget=ResearchBudget(max_trials=1, max_experiments=1, max_strategy_variants=1, max_runtime_minutes=1))
    assert ledger.usage("D", "RP")["trials"] == 0
    ledger.close(); registry.close()


def test_string_and_float_integer_params_are_rejected(tmp_path):
    from quantmind.strategy import compile_strategy_spec, StrategySpecError
    for value in ("60", 60.0, True):
        with pytest.raises(StrategySpecError, match="calendar_session_bars"):
            compile_strategy_spec(StrategySpec("V1", "F1", "current_bar_momentum", {"calendar_session_bars": value, "session_window": [0.1, 0.9]}))


def test_normalized_equivalent_logic_has_same_strategy_id():
    from quantmind.research_integrity.harness import derive_strategy_id
    a = momentum_spec(version="V1", session_bars=60, window=(0.1, 0.9))
    b = momentum_spec(version="V999", session_bars=60, window=(0.10, 0.90))
    assert derive_strategy_id(a) == derive_strategy_id(b)


def test_dataset_registry_versions_are_immutable(tmp_path):
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=2, bars_per_session=10, seed=162)).generate()
    csv_path = tmp_path / "licensed.csv"; base.to_csv(csv_path, index=False)
    registry = DatasetRegistry(tmp_path / "datasets.sqlite")
    zones = {"RESEARCH": DatasetZone("RESEARCH", None, None)}
    registry.register_file(version="D", kind=DatasetKind.LICENSED, path=csv_path, zones=zones)
    base.iloc[0, base.columns.get_loc("close")] += 1.0
    base.to_csv(csv_path, index=False)
    with pytest.raises(Exception, match="immutable"):
        registry.register_file(version="D", kind=DatasetKind.LICENSED, path=csv_path, zones=zones)
    registry.close()


def test_dataset_registry_zone_and_checksum(tmp_path):
    base = SyntheticFuturesGenerator(SyntheticConfig(sessions=2, bars_per_session=10, seed=161)).generate()
    csv_path = tmp_path / "licensed.csv"; base.to_csv(csv_path, index=False)
    registry = DatasetRegistry(tmp_path / "datasets.sqlite")
    first = pd.Timestamp(base["timestamp"].iloc[0]); last = pd.Timestamp(base["timestamp"].iloc[-1])
    split = pd.Timestamp(base["timestamp"].iloc[10])
    registry.register_file(version="D", kind=DatasetKind.LICENSED, path=csv_path, zones={
        "RESEARCH": DatasetZone("RESEARCH", first.isoformat(), split.isoformat()),
        "VALIDATION": DatasetZone("VALIDATION", split.isoformat(), None),
    })
    research = registry.load_zone("D", "RESEARCH", allowed_kinds={DatasetKind.LICENSED})
    validation = registry.load_zone("D", "VALIDATION", allowed_kinds={DatasetKind.LICENSED})
    assert len(research) == 10 and len(validation) == 10
    with csv_path.open("a", encoding="utf-8") as handle: handle.write("\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        registry.load_zone("D", "RESEARCH", allowed_kinds={DatasetKind.LICENSED})
    registry.close()


def test_inflight_runtime_is_reserved_and_stale_trials_can_be_reaped(tmp_path):
    db = tmp_path / "ledger.sqlite"; ledger = TrialLedger(db)
    budget = ResearchBudget(max_trials=10, max_experiments=10, max_strategy_variants=10, max_runtime_minutes=0.5)
    ctx = TrialContext(trial_id="T-STALE", experiment_id="E", strategy_id="S", dataset_version="D", research_protocol_version="RP", feature_version="F", parameter_set={}, seed=1, execution_model="next_bar_open_v1", cost_model="C", slippage_model="S", estimated_runtime_minutes=0.5)
    ledger.reserve(ctx, budget)
    with pytest.raises(TrialBudgetExceeded):
        ledger.reserve(TrialContext(trial_id="T-2", experiment_id="E2", strategy_id="S2", dataset_version="D", research_protocol_version="RP", feature_version="F", parameter_set={}, seed=2, execution_model="next_bar_open_v1", cost_model="C", slippage_model="S", estimated_runtime_minutes=0.1), budget)
    time.sleep(0.01)
    assert ledger.reap_stale(max_age_minutes=0.00005) == 1
    assert ledger.get("T-STALE")["status"] == "ABANDONED"
    ledger.close()


def test_running_metadata_is_immutable():
    ledger = TrialLedger()
    ctx = TrialContext(trial_id="T", experiment_id="E", strategy_id="S", dataset_version="D", research_protocol_version="RP", feature_version="F", parameter_set={}, seed=1, execution_model="next_bar_open_v1", cost_model="C", slippage_model="S", estimated_runtime_minutes=0.1)
    ledger.reserve(ctx, ResearchBudget(max_trials=2, max_experiments=2, max_strategy_variants=2, max_runtime_minutes=1))
    with pytest.raises(Exception, match="metadata is immutable"):
        ledger._connection.execute("UPDATE trials SET dataset_version='D2' WHERE trial_id='T'")
    ledger.close()
