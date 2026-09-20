"""Unit tests for Paper Replay Security Boundaries (PRD v3.9).

Verifies:
1. Non-qualified strategies cannot start replay.
2. Failed or untested holdouts cannot start replay.
3. Tampered qualification records cannot start replay.
4. Strategy spec hash and strategy ID mismatches block replay.
5. Dataset version mismatches block replay.
6. Research protocol version mismatches block replay.
7. Fixture/synthetic datasets are barred from production replay.
8. Sealed FINAL_HOLDOUT partition cannot be accessed by ReplayFeed or PaperReplayEngine.
9. Malformed or arbitrary DataFrames missing required columns are rejected.
10. Verify absolute absence of broker credentials, network endpoints, and live execution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from quantmind.data.registry import DatasetKind, DatasetRecord, DatasetRegistry, DatasetZone
from quantmind.data.splits import SplitZone
from quantmind.paper.engine import PaperReplayEngine, PaperReplaySecurityError
from quantmind.paper.feed import MarketFeedSecurityError, ReplayFeed
from quantmind.paper.models import PaperOrder, PaperPosition
from quantmind.paper.risk import PaperRiskConfig, PaperRiskEngine
from quantmind.research_integrity.holdout import HoldoutState
from quantmind.research_integrity.qualification import (
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationStatus,
)
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec
from quantmind.strategy.spec import StrategySpec


def _make_sample_feed(dataset_version: str = "DS-LICENSED-2023", n: int = 10) -> ReplayFeed:
    timestamps = pd.date_range("2023-01-01 09:15", periods=n, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0 + np.arange(n),
            "high": 101.0 + np.arange(n),
            "low": 99.0 + np.arange(n),
            "close": 100.5 + np.arange(n),
        }
    )
    return ReplayFeed(df, dataset_version=dataset_version)


def _make_valid_record(
    spec: StrategySpec,
    *,
    dataset_version: str = "DS-LICENSED-2023",
    final_status: ValidationStatus = ValidationStatus.PAPER_ELIGIBLE,
    holdout_state: str = "PASSED",
    protocol_version: str = "RP-2",
    dataset_sha256: str = "dsha-licensed",
) -> StrategyQualificationRecord:
    norm_spec = normalize_strategy_spec(spec)
    strat_id = derive_strategy_id(norm_spec)
    spec_hash = hashlib.sha256(norm_spec.canonical_json().encode()).hexdigest()

    return StrategyQualificationRecord.create(
        qualification_id="QUAL-SEC-TEST",
        strategy_id=strat_id,
        strategy_spec_hash=spec_hash,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        split_manifest_version="manifest-v1",
        research_protocol_version=protocol_version,
        population_hash="pophash-test",
        effective_trial_count=12.0,
        observed_sharpe=1.75,
        dsr=0.96,
        trade_count=150,
        holdout_state=holdout_state,
        robustness_status=RobustnessStatus.PASSED,
        final_status=final_status,
    )


class TestPaperReplaySecurityBoundaries:
    def test_non_paper_eligible_status_blocked(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        feed = _make_sample_feed()
        engine = PaperReplayEngine()

        for invalid_status in [ValidationStatus.REJECTED, ValidationStatus.VALIDATION, ValidationStatus.CANDIDATE]:
            record = _make_valid_record(spec, final_status=invalid_status)
            with pytest.raises(PaperReplaySecurityError, match="requires 'PAPER_ELIGIBLE'"):
                engine.run_replay(record, spec, feed)

    def test_failed_or_untested_holdout_blocked(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        feed = _make_sample_feed()
        engine = PaperReplayEngine()

        for invalid_holdout in ["FAILED", "UNTESTED", "IN_PROGRESS"]:
            record = _make_valid_record(spec, holdout_state=invalid_holdout)
            with pytest.raises(PaperReplaySecurityError, match="requires 'PASSED'"):
                engine.run_replay(record, spec, feed)

    def test_tampered_qualification_record_blocked(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        feed = _make_sample_feed()
        engine = PaperReplayEngine()

        valid_record = _make_valid_record(spec)

        # Tamper with Sharpe ratio
        tampered_record = StrategyQualificationRecord(
            qualification_id=valid_record.qualification_id,
            strategy_id=valid_record.strategy_id,
            strategy_spec_hash=valid_record.strategy_spec_hash,
            dataset_version=valid_record.dataset_version,
            dataset_sha256=valid_record.dataset_sha256,
            split_manifest_version=valid_record.split_manifest_version,
            research_protocol_version=valid_record.research_protocol_version,
            population_hash=valid_record.population_hash,
            effective_trial_count=valid_record.effective_trial_count,
            observed_sharpe=9.99,  # Tampered!
            dsr=valid_record.dsr,
            trade_count=valid_record.trade_count,
            holdout_state=valid_record.holdout_state,
            robustness_status=valid_record.robustness_status,
            final_status=valid_record.final_status,
            reasons=valid_record.reasons,
            record_hash=valid_record.record_hash,  # Old hash!
            created_at=valid_record.created_at,
        )

        with pytest.raises(PaperReplaySecurityError, match="cryptographic digest verification"):
            engine.run_replay(tampered_record, spec, feed)

    def test_strategy_spec_hash_mismatch_blocked(self) -> None:
        spec_original = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        spec_tampered = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 20, "session_window": [0.0, 1.0]})
        record = _make_valid_record(spec_original)

        feed = _make_sample_feed()
        engine = PaperReplayEngine()

        with pytest.raises(PaperReplaySecurityError, match="strategy_id|strategy_spec_hash"):
            engine.run_replay(record, spec_tampered, feed)

    def test_strategy_id_mismatch_blocked(self) -> None:
        spec_a = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        spec_b = StrategySpec("v2", "f2", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        record = _make_valid_record(spec_a)

        feed = _make_sample_feed()
        engine = PaperReplayEngine()

        with pytest.raises(PaperReplaySecurityError, match="does not match strategy spec derived ID"):
            engine.run_replay(record, spec_b, feed)

    def test_dataset_version_mismatch_blocked(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        record = _make_valid_record(spec, dataset_version="DS-EXPECTED-2023")

        feed = _make_sample_feed(dataset_version="DS-DIFFERENT-2023")
        engine = PaperReplayEngine()

        with pytest.raises(PaperReplaySecurityError, match="does not match qualification record"):
            engine.run_replay(record, spec, feed)

    def test_protocol_version_mismatch_blocked(self) -> None:
        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 10, "session_window": [0.0, 1.0]})
        record = _make_valid_record(spec, protocol_version="RP-LEGACY")

        feed = _make_sample_feed()
        engine = PaperReplayEngine(expected_protocol_version="RP-2")

        with pytest.raises(PaperReplaySecurityError, match="research_protocol_version"):
            engine.run_replay(record, spec, feed)

    def test_fixture_synthetic_dataset_barred_from_production(self, tmp_path: Path) -> None:
        # Set up a dataset registry with a SYNTHETIC dataset
        db_path = tmp_path / "registry.db"
        registry = DatasetRegistry(db_path)
        data_csv = tmp_path / "synthetic.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 09:15", periods=5, freq="1min"),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
            }
        )
        df.to_csv(data_csv, index=False)
        rec = registry.register_file(
            version="DS-SYNTH-1",
            kind=DatasetKind.SYNTHETIC,
            path=data_csv,
            timestamp_column="timestamp",
        )
        sha = rec.sha256

        spec = StrategySpec("v1", "f1", "current_bar_momentum", {"calendar_session_bars": 5, "session_window": [0.0, 1.0]})
        record = _make_valid_record(spec, dataset_version="DS-SYNTH-1", dataset_sha256=sha)

        feed = ReplayFeed(df, dataset_version="DS-SYNTH-1")
        engine = PaperReplayEngine(dataset_registry=registry)

        with pytest.raises(PaperReplaySecurityError, match="synthetic/fixture datasets are barred from production paper replay"):
            engine.run_replay(record, spec, feed)

        # Also ReplayFeed.from_dataset_registry should reject synthetic if require_licensed=True
        with pytest.raises(MarketFeedSecurityError, match="requires LICENSED dataset"):
            ReplayFeed.from_dataset_registry(registry, "DS-SYNTH-1", require_licensed=True)

    def test_sealed_final_holdout_access_barred(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 09:15", periods=5, freq="1min"),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
            }
        )
        with pytest.raises(MarketFeedSecurityError, match="sealed FINAL_HOLDOUT zone"):
            ReplayFeed(df, split_zone=SplitZone.FINAL_HOLDOUT)

    def test_arbitrary_dataframe_missing_ohlcv_columns_rejected(self) -> None:
        # Missing 'close' and 'open'
        df_bad = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01 09:15", periods=5, freq="1min"),
                "foo": [1.0] * 5,
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            ReplayFeed(df_bad)

        # Empty dataframe
        df_empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        with pytest.raises(ValueError, match="non-empty"):
            ReplayFeed(df_empty)

    def test_no_broker_credentials_or_live_endpoints(self) -> None:
        """Adversarial check ensuring complete absence of broker execution or external routing."""
        engine = PaperReplayEngine()
        feed = _make_sample_feed()
        pos = PaperPosition("SYM", 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        risk = PaperRiskEngine()

        prohibited_attributes = [
            "api_key",
            "secret_key",
            "access_token",
            "broker_url",
            "kite",
            "zerodha",
            "interactive_brokers",
            "ibkr",
            "live_orders",
            "place_live_order",
            "route_to_market",
            "webhook",
        ]

        for obj in [engine, feed, pos, risk]:
            dir_attrs = dir(obj)
            for prohibited in prohibited_attributes:
                assert prohibited not in dir_attrs, f"Prohibited live trading attribute '{prohibited}' found on {obj}"
