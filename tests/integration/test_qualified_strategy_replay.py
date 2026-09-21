"""Integration tests for Qualified Strategy Paper Replay (PRD v3.9).

End-to-end verification of:
1. StrategySpec -> Registration -> Lifecycle Transition -> StrategyQualificationRecord
2. Promotion to PAPER_ELIGIBLE in StrategyRegistry
3. Deterministic PaperReplayEngine execution against ReplayFeed (FORWARD_PAPER)
4. Audit ledger persistence (orders, fills, position snapshots, replay report)
5. Strict determinism: identical inputs produce bitwise identical report_hash
6. Pre-trade risk rule intervention in replay execution
"""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from quantmind.backtest.engine import CostSchedule, CostSchedulePeriod
from quantmind.data.registry import DatasetKind, DatasetRegistry, DatasetZone
from quantmind.data.splits import SplitZone
from quantmind.paper.engine import PaperReplayEngine
from quantmind.paper.feed import ReplayFeed
from quantmind.paper.ledger import PaperLedger
from quantmind.paper.models import PaperOrderStatus
from quantmind.paper.risk import PaperRiskConfig
from quantmind.research_integrity.holdout import HoldoutState
from quantmind.research_integrity.qualification import (
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationStatus,
    check_paper_replay_eligibility,
)
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.registry import StrategyLifecycleState, StrategyRegistry
from quantmind.strategy.spec import StrategySpec


@pytest.fixture
def replay_environment(tmp_path: Path):
    """Fixture providing integrated registry, dataset, and strategy setup."""
    db_path = tmp_path / "quantmind_system.db"
    dataset_registry = DatasetRegistry(db_path)
    strategy_registry = StrategyRegistry(db_path)

    # 1. Create realistic multi-session market data (100 bars across 2 days)
    day1 = pd.date_range("2023-01-02 09:15", periods=50, freq="1min")
    day2 = pd.date_range("2023-01-03 09:15", periods=50, freq="1min")
    timestamps = day1.append(day2)

    # Alternate up and down bars to trigger momentum entries and exits
    closes = []
    opens = []
    price = 1000.0
    for i in range(len(timestamps)):
        o = price
        c = price + (1.5 if i % 2 == 0 else -1.5)
        opens.append(o)
        closes.append(c)
        price = c

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": [max(o, c) + 0.5 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 0.5 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1000.0] * len(timestamps),
            "open_interest": [5000.0] * len(timestamps),
        }
    )

    csv_path = tmp_path / "licensed_market_data.csv"
    df.to_csv(csv_path, index=False)

    zones = {
        "DISCOVERY": DatasetZone("DISCOVERY", "2023-01-02 09:15", "2023-01-02 09:30"),
        "VALIDATION": DatasetZone("VALIDATION", "2023-01-02 09:31", "2023-01-02 09:50"),
        "FINAL_HOLDOUT": DatasetZone("FINAL_HOLDOUT", "2023-01-02 09:51", "2023-01-02 10:04"),
        "FORWARD_PAPER": DatasetZone("FORWARD_PAPER", "2023-01-03 09:15", "2023-01-03 10:04"),
    }

    dataset_record = dataset_registry.register_file(
        version="DS-NIFTY-2023-V1",
        kind=DatasetKind.LICENSED,
        path=csv_path,
        timestamp_column="timestamp",
        zones=zones,
        metadata={"source": "NSE", "asset": "NIFTY_FUT"},
    )

    # 2. Define valid strategy spec
    spec = StrategySpec(
        strategy_version="v1.0",
        feature_version="f1",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 50, "session_window": [0.0, 1.0]},
    )
    norm_spec = normalize_strategy_spec(spec)
    strat_id = derive_strategy_id(norm_spec)
    spec_hash = hashlib.sha256(norm_spec.canonical_json().encode("utf-8")).hexdigest()

    # 3. Create authoritative qualification record
    qualification_record = StrategyQualificationRecord.create(
        qualification_id="QUAL-E2E-2023-001",
        strategy_id=strat_id,
        strategy_spec_hash=spec_hash,
        dataset_version=dataset_record.version,
        dataset_sha256=dataset_record.sha256,
        split_manifest_version="split-manifest-v1",
        research_protocol_version="RP-2",
        population_hash="pophash-prod-2023",
        effective_trial_count=18.5,
        observed_sharpe=1.92,
        dsr=0.975,
        trade_count=180,
        holdout_state=HoldoutState.PASSED.value,
        robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE,
        created_at="2026-09-20T14:00:00+00:00",
        reasons=["All DSR and EICT criteria satisfied", "Sealed holdout passed"],
    )

    return {
        "db_path": db_path,
        "dataset_registry": dataset_registry,
        "strategy_registry": strategy_registry,
        "dataset_record": dataset_record,
        "raw_df": df,
        "spec": spec,
        "strat_id": strat_id,
        "qualification_record": qualification_record,
    }


def test_end_to_end_qualified_strategy_replay(replay_environment) -> None:
    env = replay_environment
    strat_reg: StrategyRegistry = env["strategy_registry"]
    strat_id: str = env["strat_id"]
    spec: StrategySpec = env["spec"]
    qual_rec: StrategyQualificationRecord = env["qualification_record"]
    data_reg: DatasetRegistry = env["dataset_registry"]

    # 1. Register candidate strategy in strategy registry
    strat_reg.register_strategy(spec)

    # 2. Advance lifecycle: IDEA -> RESEARCH -> VALIDATION
    strat_reg.transition_state(strat_id, StrategyLifecycleState.RESEARCH, reason="Starting research")
    strat_reg.transition_state(strat_id, StrategyLifecycleState.VALIDATION, reason="Entering statistical validation")

    # 3. Promote to PAPER_ELIGIBLE via authoritative qualification record
    eligibility = check_paper_replay_eligibility(qual_rec)
    assert eligibility.is_eligible is True

    record = strat_reg.transition_state(
        strat_id,
        StrategyLifecycleState.PAPER_ELIGIBLE,
        reason="Certified by StrategyValidationGate",
        qualification_record=qual_rec,
    )
    assert record.state == StrategyLifecycleState.PAPER_ELIGIBLE
    assert record.qualification_id == qual_rec.qualification_id
    assert record.qualification_hash == qual_rec.record_hash

    # 4. Construct ReplayFeed from DatasetRegistry for FORWARD_PAPER
    feed = ReplayFeed.from_dataset_registry(
        data_reg,
        env["dataset_record"].version,
        split_zone=SplitZone.FORWARD_PAPER,
        lot_size=50,
        tick_size=0.05,
    )

    # 5. Set up engine with date-effective cost schedule and slippage
    cost_sched = CostSchedule(
        schedule_id="SCHED-2023",
        periods=(
            CostSchedulePeriod(effective_from=date(2023, 1, 1), effective_to=date(2023, 12, 31), round_trip_bps=2.0),
        ),
    )
    ledger = PaperLedger()
    engine = PaperReplayEngine(
        ledger=ledger,
        cost_schedule=cost_sched,
        slippage_bps_per_side=2.5,
        dataset_registry=data_reg,
        expected_protocol_version="RP-2",
    )

    # 6. Execute deterministic replay
    report = engine.run_replay(
        qualification_record=qual_rec,
        strategy_spec=spec,
        feed=feed,
        initial_capital=200_000.0,
        quantity=2,
        hold_bars=2,
    )

    # Verify report integrity and metrics
    assert report.strategy_id == strat_id
    assert report.qualification_id == qual_rec.qualification_id
    assert report.dataset_version == env["dataset_record"].version
    assert report.trade_count > 0
    assert report.total_costs > 0.0
    assert report.total_slippage > 0.0
    assert report.report_hash == report.compute_report_hash()

    # 7. Verify ledger persistence
    orders = ledger.get_orders(strategy_id=strat_id)
    fills = ledger.get_fills(strategy_id=strat_id)
    assert len(orders) >= report.trade_count * 2
    assert len(fills) >= report.trade_count * 2

    # Check order lifecycle consistency
    for order in orders:
        assert order["status"] == PaperOrderStatus.FILLED.value
        assert order["strategy_id"] == strat_id
        assert order["qualification_id"] == qual_rec.qualification_id

    # Check fill prices rounded to tick size (0.05)
    for fill in fills:
        assert round(fill["fill_price"] % 0.05, 4) in (0.0, 0.05)
        assert fill["cost"] > 0.0


def test_replay_strict_determinism(replay_environment) -> None:
    """Verifies that two runs with identical inputs produce bitwise identical report_hash."""
    env = replay_environment
    qual_rec: StrategyQualificationRecord = env["qualification_record"]
    spec: StrategySpec = env["spec"]
    data_reg: DatasetRegistry = env["dataset_registry"]

    cost_sched = CostSchedule(
        schedule_id="SCHED-DET",
        periods=(
            CostSchedulePeriod(effective_from=date(2023, 1, 1), effective_to=date(2023, 12, 31), round_trip_bps=1.5),
        ),
    )

    # Run 1
    feed1 = ReplayFeed.from_dataset_registry(data_reg, env["dataset_record"].version)
    engine1 = PaperReplayEngine(cost_schedule=cost_sched, slippage_bps_per_side=1.0, dataset_registry=data_reg)
    report1 = engine1.run_replay(qual_rec, spec, feed1, initial_capital=100_000.0, hold_bars=2)

    # Run 2
    feed2 = ReplayFeed.from_dataset_registry(data_reg, env["dataset_record"].version)
    engine2 = PaperReplayEngine(cost_schedule=cost_sched, slippage_bps_per_side=1.0, dataset_registry=data_reg)
    report2 = engine2.run_replay(qual_rec, spec, feed2, initial_capital=100_000.0, hold_bars=2)

    # Assert bitwise deterministic match across all metrics and hash
    assert report1.report_hash == report2.report_hash
    assert report1.trade_count == report2.trade_count
    assert report1.net_pnl == report2.net_pnl
    assert report1.gross_pnl == report2.gross_pnl
    assert report1.total_costs == report2.total_costs
    assert report1.total_slippage == report2.total_slippage
    assert report1.max_drawdown_bps == report2.max_drawdown_bps
    assert report1.win_rate == report2.win_rate
    assert report1.canonical_json() == report2.canonical_json()


def test_replay_risk_intervention_recording(replay_environment) -> None:
    """Verifies that pre-trade risk blocks illegal orders and writes audit risk events to ledger."""
    env = replay_environment
    qual_rec: StrategyQualificationRecord = env["qualification_record"]
    spec: StrategySpec = env["spec"]
    data_reg: DatasetRegistry = env["dataset_registry"]

    feed = ReplayFeed.from_dataset_registry(data_reg, env["dataset_record"].version)
    ledger = PaperLedger()

    # Configure risk engine with kill_switch active
    risk_cfg = PaperRiskConfig(kill_switch=True)
    engine = PaperReplayEngine(ledger=ledger, risk_config=risk_cfg, dataset_registry=data_reg)

    report = engine.run_replay(qual_rec, spec, feed)

    assert report.trade_count == 0
    assert report.net_pnl == 0.0

    # Verify risk events were logged in ledger
    risk_events = ledger.get_risk_events()
    assert len(risk_events) > 0
    for ev in risk_events:
        assert ev["rule_name"] == "kill_switch"
        assert "kill switch is ACTIVE" in ev["reason"]

    # Verify all orders were rejected
    orders = ledger.get_orders()
    assert len(orders) > 0
    for ord_dict in orders:
        assert ord_dict["status"] == PaperOrderStatus.REJECTED.value
        assert "kill switch" in ord_dict["rejection_reason"]
