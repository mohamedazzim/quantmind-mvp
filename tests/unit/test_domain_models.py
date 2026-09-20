from datetime import date, datetime

import pytest

from quantmind.domain.models import (
    Bar,
    ContractSpecPeriod,
    Fill,
    FuturesContract,
    Order,
    Side,
    Trial,
)


def test_futures_contract_uses_date_effective_specification():
    contract = FuturesContract(
        symbol="NIFTY_FUT",
        expiry=date(2026, 9, 24),
        specs=(
            ContractSpecPeriod(date(2026, 1, 1), date(2026, 6, 30), 75, 0.05),
            ContractSpecPeriod(date(2026, 7, 1), None, 65, 0.05),
        ),
    )

    assert contract.spec_at(date(2026, 6, 30)).lot_size == 75
    assert contract.spec_at(date(2026, 7, 1)).lot_size == 65

    with pytest.raises(ValueError):
        contract.spec_at(date(2025, 12, 31))


def test_trial_contains_all_prd_ledger_fields():
    trial = Trial(
        trial_id="TRIAL-1",
        experiment_id="EXP-1",
        strategy_id="STRAT-1",
        dataset_version="DATA-1",
        feature_version="FEAT-1",
        parameter_set={"lookback": 21},
        seed=7,
        execution_model="next_bar_open_v1",
        cost_model="COST-1",
        slippage_model="SLIP-1",
        timestamp_started=datetime(2026, 9, 20, 10, 0),
        timestamp_completed=datetime(2026, 9, 20, 10, 1),
        result={"sharpe": 1.2},
        status="COMPLETED",
    )
    assert trial.parameter_set["lookback"] == 21
    assert trial.result["sharpe"] == 1.2


def test_core_trade_models_are_typed():
    bar = Bar(datetime(2026, 9, 20, 9, 15), "NIFTY_FUT", 100.0, 101.0, 99.0, 100.5, 1000, 200000)
    order = Order("O1", bar.timestamp, bar.contract, Side.BUY, 65)
    fill = Fill("O1", bar.timestamp, bar.contract, Side.BUY, 65, 100.5, 0.01, 2.0)

    assert bar.high >= max(bar.open, bar.close)
    assert order.quantity == 65
    assert fill.price == 100.5
