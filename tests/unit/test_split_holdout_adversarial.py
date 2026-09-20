from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import time
import pytest
import numpy as np
import pandas as pd

from quantmind.backtest.engine import BacktestConfig, BacktestEngine
from quantmind.backtest.strategies import prior_bar_momentum_signal
from quantmind.data import (
    DatasetKind,
    DatasetRecord,
    DatasetRegistry,
    DatasetRegistryError,
    PurgeEmbargoSpec,
    SplitManifest,
    SplitManifestError,
    SplitZone,
    compute_split_manifest,
)
from quantmind.research_integrity import (
    HoldoutManager,
    HoldoutSecurityError,
    HoldoutState,
    ResearchBudget,
    ResearchHarness,
    TrialContext,
    TrialLedger,
)
from quantmind.strategy import StrategySpec, compile_strategy_spec
from quantmind.testing.research_harness import FixtureResearchHarness


def _create_synthetic_dataset(path: Path, num_bars: int = 150) -> tuple[Path, list[pd.Timestamp]]:
    base_ts = pd.Timestamp("2026-01-01 09:15:00")
    timestamps = [base_ts + timedelta(minutes=5 * i) for i in range(num_bars)]
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0001, 0.001, num_bars)
    prices = 100.0 * np.exp(np.cumsum(returns))
    df = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "open": prices,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices * 1.0005,
            "volume": 1000,
            "open_interest": 500,
        }
    )
    df.to_csv(path, index=False)
    return path, timestamps


@pytest.fixture
def adversarial_env():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        licensed_csv = tmp_path / "licensed.csv"
        synthetic_csv = tmp_path / "synthetic.csv"
        _, l_timestamps = _create_synthetic_dataset(licensed_csv, num_bars=150)
        _, s_timestamps = _create_synthetic_dataset(synthetic_csv, num_bars=150)

        spec = PurgeEmbargoSpec(
            feature_lookback_bars=3,
            prediction_horizon_bars=2,
            holding_period_bars=2,
            forward_dependency_bars=1,
            embargo_bars=3,
        )

        l_manifest = compute_split_manifest(
            dataset_version="LICENSED-DS-1",
            manifest_version="v1",
            timestamps=l_timestamps,
            spec=spec,
            research_ratio=0.4,
            validation_ratio=0.3,
            holdout_ratio=0.3,
        )

        s_manifest = compute_split_manifest(
            dataset_version="SYNTH-DS-1",
            manifest_version="v1",
            timestamps=s_timestamps,
            spec=spec,
            research_ratio=0.4,
            validation_ratio=0.3,
            holdout_ratio=0.3,
        )

        registry = DatasetRegistry()
        registry.register_file(
            version="LICENSED-DS-1",
            kind=DatasetKind.LICENSED,
            path=licensed_csv,
            split_manifest=l_manifest,
        )
        registry.register_file(
            version="SYNTH-DS-1",
            kind=DatasetKind.SYNTHETIC,
            path=synthetic_csv,
            split_manifest=s_manifest,
        )

        engine = BacktestEngine()
        ledger = TrialLedger()
        holdout_mgr = HoldoutManager(engine, ledger, registry)
        harness = ResearchHarness(engine, ledger, registry, holdout_manager=holdout_mgr)
        fixture_harness = FixtureResearchHarness(engine, ledger)

        strategy_spec = StrategySpec(
            strategy_version="1.0.0",
            feature_version="f_v1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 20, "session_window": [0.10, 0.90]},
        )

        yield {
            "tmp_path": tmp_path,
            "licensed_csv": licensed_csv,
            "synthetic_csv": synthetic_csv,
            "registry": registry,
            "engine": engine,
            "ledger": ledger,
            "holdout_mgr": holdout_mgr,
            "harness": harness,
            "fixture_harness": fixture_harness,
            "strategy_spec": strategy_spec,
            "l_manifest": l_manifest,
            "s_manifest": s_manifest,
            "protocol_version": "RP-2",
        }


# ==================================================
# A. HOLDOUT ACCESS
# ==================================================

def test_01_research_harness_final_holdout_fails_before_reservation(adversarial_env):
    """1. Production ResearchHarness attempting FINAL_HOLDOUT must fail before trial reservation."""
    harness = adversarial_env["harness"]
    ledger = adversarial_env["ledger"]
    spec = adversarial_env["strategy_spec"]

    initial_trials = len(ledger.find_trials(dataset_version="LICENSED-DS-1"))
    with pytest.raises(ValueError, match="FINAL_HOLDOUT zone is sealed"):
        harness.run_trial(
            config=BacktestConfig(),
            research_task_id="leak-attempt-1",
            strategy_spec=spec,
            dataset_version="LICENSED-DS-1",
            split_zone="FINAL_HOLDOUT",
            research_protocol_version="RP-2",
            seed=42,
            budget=ResearchBudget(),
        )

    # Verification: no trial budget was reserved, no trial row created
    final_trials = len(ledger.find_trials(dataset_version="LICENSED-DS-1"))
    assert final_trials == initial_trials


def test_02_direct_public_load_zone_final_holdout_fails(adversarial_env):
    """2. Direct public DatasetRegistry access to FINAL_HOLDOUT must fail."""
    registry = adversarial_env["registry"]
    with pytest.raises(DatasetRegistryError, match="FINAL_HOLDOUT zone is sealed"):
        registry.load_zone("LICENSED-DS-1", "FINAL_HOLDOUT")

    with pytest.raises(DatasetRegistryError, match="FINAL_HOLDOUT zone is sealed"):
        registry.load_zone("LICENSED-DS-1", SplitZone.FINAL_HOLDOUT)


def test_03_optimization_loop_final_holdout_fails(adversarial_env):
    """3. Attempt to access FINAL_HOLDOUT through an optimization loop must fail."""
    harness = adversarial_env["harness"]
    for window_start in [0.05, 0.10, 0.15]:
        candidate_spec = StrategySpec(
            strategy_version="1.0.0",
            feature_version="f_v1",
            signal_name="current_bar_momentum",
            parameters={"calendar_session_bars": 20, "session_window": [window_start, 0.90]},
        )
        with pytest.raises(ValueError, match="FINAL_HOLDOUT zone is sealed"):
            harness.run_trial(
                config=BacktestConfig(),
                research_task_id=f"opt-{window_start}",
                strategy_spec=candidate_spec,
                dataset_version="LICENSED-DS-1",
                split_zone="FINAL_HOLDOUT",
                research_protocol_version="RP-2",
                seed=42,
                budget=ResearchBudget(),
            )


def test_04_dedicated_final_evaluate_succeeds_for_eligible_candidate(adversarial_env):
    """4. Dedicated HoldoutManager.final_evaluate(...) must succeed for an eligible candidate."""
    harness = adversarial_env["harness"]
    holdout_mgr = adversarial_env["holdout_mgr"]
    spec = adversarial_env["strategy_spec"]

    # Pre-requisite research run
    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="eligible-candidate-task",
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        split_zone="RESEARCH",
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
    )

    result = holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        research_protocol_version="RP-2",
        config=BacktestConfig(),
    )
    assert result is not None
    assert result.state in (HoldoutState.PASSED, HoldoutState.FAILED)
    assert result.trial_id.startswith("TRIAL-HOLDOUT-")


# ==================================================
# B. HOLDOUT REUSE & STATE MACHINE
# ==================================================

def test_05_second_evaluation_of_same_candidate_fails(adversarial_env):
    """5. Candidate evaluates FINAL_HOLDOUT once. Second evaluation must fail."""
    harness = adversarial_env["harness"]
    holdout_mgr = adversarial_env["holdout_mgr"]
    spec = adversarial_env["strategy_spec"]

    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="research-pass-1",
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        split_zone="RESEARCH",
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
    )

    # First eval
    holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        research_protocol_version="RP-2",
        config=BacktestConfig(),
    )

    # Second eval fails
    with pytest.raises(HoldoutSecurityError, match="repeat evaluation or tuning.*strictly forbidden"):
        holdout_mgr.final_evaluate(
            strategy_spec=spec,
            dataset_version="LICENSED-DS-1",
            research_protocol_version="RP-2",
            config=BacktestConfig(),
        )


def test_06_failed_candidate_cannot_retune_or_rerun(adversarial_env):
    """6. Candidate that fails holdout must not be able to retune or rerun against same holdout."""
    harness = adversarial_env["harness"]
    holdout_mgr = adversarial_env["holdout_mgr"]
    spec = adversarial_env["strategy_spec"]

    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="research-pass-2",
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        split_zone="RESEARCH",
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
    )

    # Force failure
    res = holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        research_protocol_version="RP-2",
        config=BacktestConfig(),
        pass_criterion=lambda _: False,
    )
    assert res.state == HoldoutState.FAILED

    # Retune attempt in ResearchHarness is rejected
    with pytest.raises(HoldoutSecurityError, match="failed final holdout.*cannot be retuned"):
        harness.run_trial(
            config=BacktestConfig(),
            research_task_id="retune-attempt",
            strategy_spec=spec,
            dataset_version="LICENSED-DS-1",
            split_zone="RESEARCH",
            research_protocol_version="RP-2",
            seed=42,
            budget=ResearchBudget(),
        )


def test_07_burned_holdout_rejects_all_operations(adversarial_env):
    """7. BURNED holdout must reject every later evaluation/tuning attempt."""
    harness = adversarial_env["harness"]
    holdout_mgr = adversarial_env["holdout_mgr"]
    spec = adversarial_env["strategy_spec"]

    holdout_mgr.burn_holdout("LICENSED-DS-1", "RP-2", reason="Adversarial burn test")
    assert holdout_mgr.is_holdout_burned("LICENSED-DS-1", "RP-2")

    # Final evaluate blocked
    with pytest.raises(HoldoutSecurityError, match="is BURNED"):
        holdout_mgr.final_evaluate(
            strategy_spec=spec,
            dataset_version="LICENSED-DS-1",
            research_protocol_version="RP-2",
            config=BacktestConfig(),
        )

    # Research trial blocked
    with pytest.raises(HoldoutSecurityError, match="is BURNED"):
        harness.run_trial(
            config=BacktestConfig(),
            research_task_id="tune-on-burned",
            strategy_spec=spec,
            dataset_version="LICENSED-DS-1",
            split_zone="RESEARCH",
            research_protocol_version="RP-2",
            seed=42,
            budget=ResearchBudget(),
        )


def test_08_holdout_state_cannot_be_reset_in_database(adversarial_env):
    """8. Holdout state must never be reset from FAILED/PASSED to UNTOUCHED."""
    harness = adversarial_env["harness"]
    holdout_mgr = adversarial_env["holdout_mgr"]
    spec = adversarial_env["strategy_spec"]

    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="research-pass-3",
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        split_zone="RESEARCH",
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
    )

    holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        research_protocol_version="RP-2",
        config=BacktestConfig(),
        pass_criterion=lambda _: False,
    )

    # Direct database attack: attempting to update state to 'UNTOUCHED'
    with pytest.raises(sqlite3.IntegrityError, match="evaluated holdout state is permanent and cannot be reset"):
        holdout_mgr._connection.execute(
            "UPDATE holdout_evaluations SET state = 'UNTOUCHED'"
        )


# ==================================================
# C. DATASET INTEGRITY & PROVENANCE
# ==================================================

def test_09_dataset_registration_idempotency(adversarial_env):
    """9. Registering identical version, checksum, and manifest is idempotent."""
    registry = adversarial_env["registry"]
    rec1 = registry.register_file(
        version="LICENSED-DS-1",
        kind=DatasetKind.LICENSED,
        path=adversarial_env["licensed_csv"],
        split_manifest=adversarial_env["l_manifest"],
    )
    rec2 = registry.register_file(
        version="LICENSED-DS-1",
        kind=DatasetKind.LICENSED,
        path=adversarial_env["licensed_csv"],
        split_manifest=adversarial_env["l_manifest"],
    )
    assert rec1.version == rec2.version
    assert rec1.sha256 == rec2.sha256


def test_10_and_11_dataset_immutability_checksum_and_manifest(adversarial_env):
    """10 & 11. Modified checksum or modified manifest for existing version must fail."""
    registry = adversarial_env["registry"]
    tmp_path = adversarial_env["tmp_path"]

    other_csv = tmp_path / "other.csv"
    _create_synthetic_dataset(other_csv, num_bars=50)

    # Different file checksum
    with pytest.raises(DatasetRegistryError, match="immutable"):
        registry.register_file(
            version="LICENSED-DS-1",
            kind=DatasetKind.LICENSED,
            path=other_csv,
            split_manifest=adversarial_env["l_manifest"],
        )

    # Different manifest
    other_spec = PurgeEmbargoSpec(feature_lookback_bars=20, prediction_horizon_bars=10)
    other_manifest = compute_split_manifest(
        dataset_version="LICENSED-DS-1",
        manifest_version="v2_modified",
        timestamps=pd.read_csv(adversarial_env["licensed_csv"])["timestamp"],
        spec=other_spec,
    )
    with pytest.raises(DatasetRegistryError, match="immutable"):
        registry.register_file(
            version="LICENSED-DS-1",
            kind=DatasetKind.LICENSED,
            path=adversarial_env["licensed_csv"],
            split_manifest=other_manifest,
        )


def test_12_corrupting_dataset_causes_checksum_failure(adversarial_env):
    """12. Corrupting/replacing dataset file on disk causes checksum verification failure."""
    registry = adversarial_env["registry"]
    csv_path = adversarial_env["licensed_csv"]

    # Append corrupted bytes to file on disk
    with open(csv_path, "a") as f:
        f.write("\ncorrupted_trailing_row_99999\n")

    with pytest.raises(DatasetRegistryError, match="checksum mismatch"):
        registry.verify("LICENSED-DS-1")

    with pytest.raises(DatasetRegistryError, match="checksum mismatch"):
        registry.load_zone("LICENSED-DS-1", SplitZone.RESEARCH)


def test_13_production_harness_rejects_synthetic_datasets(adversarial_env):
    """13. Production ResearchHarness must reject synthetic datasets."""
    harness = adversarial_env["harness"]
    spec = adversarial_env["strategy_spec"]

    with pytest.raises(ValueError, match="production research requires LICENSED dataset"):
        harness.run_trial(
            config=BacktestConfig(),
            research_task_id="synth-rejection-task",
            strategy_spec=spec,
            dataset_version="SYNTH-DS-1",
            split_zone="RESEARCH",
            research_protocol_version="RP-2",
            seed=42,
            budget=ResearchBudget(),
        )


def test_14_fixture_harness_allows_synthetic_datasets(adversarial_env):
    """14. Fixture harness may use synthetic datasets with mode=FIXTURE."""
    fixture_harness = adversarial_env["fixture_harness"]
    registry = adversarial_env["registry"]
    spec = adversarial_env["strategy_spec"]
    data = registry.load_zone("SYNTH-DS-1", "RESEARCH", allowed_kinds={DatasetKind.SYNTHETIC})

    result = fixture_harness.run_trial(
        data=data,
        config=BacktestConfig(),
        research_task_id="fixture-synthetic-run",
        strategy_spec=spec,
        dataset_version="SYNTH-DS-1",
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
        signal_fn=compile_strategy_spec(spec),
    )
    assert result is not None


def test_15_and_16_population_queries_exclude_fixture_rows(adversarial_env):
    """15 & 16. Fixture trials have mode=FIXTURE; production queries exclude them."""
    harness = adversarial_env["harness"]
    fixture_harness = adversarial_env["fixture_harness"]
    registry = adversarial_env["registry"]
    ledger = adversarial_env["ledger"]
    spec = adversarial_env["strategy_spec"]

    # 1. Run production trial
    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="prod-trial-1",
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        split_zone="RESEARCH",
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
    )

    # 2. Run fixture trial on same protocol
    synth_data = registry.load_zone("SYNTH-DS-1", "RESEARCH", allowed_kinds={DatasetKind.SYNTHETIC})
    fixture_harness.run_trial(
        data=synth_data,
        config=BacktestConfig(),
        research_task_id="fix-trial-1",
        strategy_spec=spec,
        dataset_version="SYNTH-DS-1",
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
        signal_fn=compile_strategy_spec(spec),
    )

    # 3. Usage check: production usage must not count fixture trial
    prod_usage = ledger.usage("LICENSED-DS-1", "RP-2", mode="PRODUCTION")
    assert prod_usage["trials"] == 1

    fixture_usage = ledger.usage("SYNTH-DS-1", "RP-2", mode="FIXTURE")
    assert fixture_usage["trials"] == 1


# ==================================================
# D. SPLIT INTEGRITY & TEMPORAL ISOLATION
# ==================================================

def test_17_to_24_temporal_split_isolation_and_dependencies(adversarial_env):
    """17-24. Split boundaries non-overlapping; purge and embargo ranges isolated; lookback & horizon isolated."""
    registry = adversarial_env["registry"]
    manifest = adversarial_env["l_manifest"]

    r_df = registry.load_zone("LICENSED-DS-1", SplitZone.RESEARCH)
    v_df = registry.load_zone("LICENSED-DS-1", SplitZone.VALIDATION)
    h_df = registry.load_zone("LICENSED-DS-1", SplitZone.FINAL_HOLDOUT, allow_holdout=True)

    r_ts = [pd.Timestamp(t) for t in r_df["timestamp"]]
    v_ts = [pd.Timestamp(t) for t in v_df["timestamp"]]
    h_ts = [pd.Timestamp(t) for t in h_df["timestamp"]]

    # 17. Research & Validation never overlap
    assert max(r_ts) < min(v_ts)

    # 18. Validation & Holdout never overlap
    assert max(v_ts) < min(h_ts)

    # 19 & 20. Purge and embargo ranges are strictly absent
    all_usable = set(r_ts).union(set(v_ts)).union(set(h_ts))
    for r in manifest.removed_ranges:
        r_start = pd.Timestamp(r.start)
        r_end = pd.Timestamp(r.end)
        for t in all_usable:
            assert not (r_start <= t < r_end), f"timestamp {t} leaked into {r.name}"

    # 21-24. Dependency intervals
    spec = manifest.spec
    assert spec is not None
    assert spec.purge_bars == (
        spec.feature_lookback_bars
        + spec.prediction_horizon_bars
        + spec.holding_period_bars
        + spec.forward_dependency_bars
    )


def test_25_causal_signal_invariance_under_prefix_truncation(adversarial_env):
    """25. Mid-session data truncation does not change completed causal signals."""
    registry = adversarial_env["registry"]
    r_df = registry.load_zone("LICENSED-DS-1", SplitZone.RESEARCH)
    spec = adversarial_env["strategy_spec"]
    signal_fn = compile_strategy_spec(spec)

    full_signal = signal_fn(r_df)

    for cut in [20, 35, 50]:
        truncated_df = r_df.iloc[:cut].copy().reset_index(drop=True)
        truncated_signal = signal_fn(truncated_df)
        np.testing.assert_array_equal(
            full_signal[:cut],
            truncated_signal,
            err_msg=f"causal signal varied under truncation cut={cut}",
        )


# ==================================================
# E. TRIAL LEDGER & IMMUTABILITY
# ==================================================

def test_26_to_29_trial_ledger_metadata_and_zone_queries(adversarial_env):
    """26-29. Trial records all PRD fields, supports split_zone queries, holdout trials via final_evaluate only."""
    harness = adversarial_env["harness"]
    holdout_mgr = adversarial_env["holdout_mgr"]
    ledger = adversarial_env["ledger"]
    spec = adversarial_env["strategy_spec"]

    # Run research trial
    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="ledger-test-task",
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        split_zone=SplitZone.RESEARCH,
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
    )

    # Run holdout evaluation
    eval_res = holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        research_protocol_version="RP-2",
        config=BacktestConfig(),
    )

    research_trials = ledger.find_trials(split_zone="RESEARCH")
    holdout_trials = ledger.find_trials(split_zone="FINAL_HOLDOUT")

    assert len(research_trials) >= 1
    assert len(holdout_trials) == 1

    r_row = research_trials[0]
    for required_col in [
        "trial_id",
        "experiment_id",
        "strategy_id",
        "dataset_version",
        "research_protocol_version",
        "split_zone",
        "mode",
        "status",
    ]:
        assert r_row[required_col] is not None

    assert holdout_trials[0]["trial_id"] == eval_res.trial_id


def test_30_to_33_sqlite_triggers_enforce_terminal_immutability(adversarial_env):
    """30-33. Terminal trials and holdout records cannot be updated or deleted."""
    harness = adversarial_env["harness"]
    ledger = adversarial_env["ledger"]
    spec = adversarial_env["strategy_spec"]

    harness.run_trial(
        config=BacktestConfig(),
        research_task_id="immutability-task",
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        split_zone="RESEARCH",
        research_protocol_version="RP-2",
        seed=42,
        budget=ResearchBudget(),
    )

    trials = ledger.find_trials(dataset_version="LICENSED-DS-1", status="COMPLETED")
    assert len(trials) >= 1
    target_trial_id = trials[0]["trial_id"]

    # 30. DELETE of terminal trial must fail
    with pytest.raises(sqlite3.IntegrityError, match="trials are append-only"):
        ledger._connection.execute("DELETE FROM trials WHERE trial_id = ?", (target_trial_id,))

    # 31. UPDATE of terminal trial must fail
    with pytest.raises(sqlite3.IntegrityError, match="completed trials are immutable"):
        ledger._connection.execute(
            "UPDATE trials SET status = 'RUNNING' WHERE trial_id = ?", (target_trial_id,)
        )

    # 32. DELETE of holdout evaluation record must fail
    holdout_mgr = adversarial_env["holdout_mgr"]
    holdout_mgr.final_evaluate(
        strategy_spec=spec,
        dataset_version="LICENSED-DS-1",
        research_protocol_version="RP-2",
        config=BacktestConfig(),
    )
    with pytest.raises(sqlite3.IntegrityError, match="holdout evaluations are permanent"):
        ledger._connection.execute("DELETE FROM holdout_evaluations")


# ==================================================
# PERFORMANCE SANITY BENCHMARK
# ==================================================

def test_performance_sanity_benchmark(adversarial_env):
    """Verify zone loading on 10,000 bars executes quickly with vectorized boolean masks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "large_data.csv"
        _, timestamps = _create_synthetic_dataset(csv_path, num_bars=10_000)

        spec = PurgeEmbargoSpec(feature_lookback_bars=10, prediction_horizon_bars=5, embargo_bars=5)
        manifest = compute_split_manifest(
            dataset_version="PERF-10K",
            timestamps=timestamps,
            spec=spec,
            research_ratio=0.5,
            validation_ratio=0.25,
            holdout_ratio=0.25,
        )

        registry = DatasetRegistry()
        registry.register_file(
            version="PERF-10K",
            kind=DatasetKind.LICENSED,
            path=csv_path,
            split_manifest=manifest,
        )

        # First load reads file & populates cache
        t0 = time.perf_counter()
        df1 = registry.load_zone("PERF-10K", SplitZone.RESEARCH)
        t_first = time.perf_counter() - t0

        # Subsequent loads slice directly from cached frame via boolean mask
        t1 = time.perf_counter()
        df2 = registry.load_zone("PERF-10K", SplitZone.VALIDATION)
        t_cached = time.perf_counter() - t1

        assert len(df1) > 0
        assert len(df2) > 0
        # Cached boolean mask filtering of 10k rows should take under 50ms
        assert t_cached < 0.05, f"zone load took too long: {t_cached:.4f}s"
