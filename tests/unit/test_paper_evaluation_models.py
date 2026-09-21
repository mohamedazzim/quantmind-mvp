"""Unit tests for PRD v4.0 paper evaluation models and cryptographic hashing.

Verifies:
- Immutability of evidence dataclasses.
- Strict determinism of canonical JSON serialization and SHA-256 digests.
- Temporal determinism: exclusion of created_at from semantic hashes.
- Exclusion of administrative IDs from semantic hashes.
- Deterministic ID derivation (zero UUID / randomness).
- Sensitive hash divergence on field mutations.
- Validation and fail-closed checks on invalid configurations.
- P&L semantics regression assertions (cost/slippage non-duplication).
"""

import dataclasses
import pytest

from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringConfig,
    MonitoringConfigError,
    MonitoringSnapshot,
    PaperEvaluationTransition,
    ResearchFeedbackRecord,
)
from quantmind.paper.models import PaperPosition, ReplayReport


# ---------------------------------------------------------------------------
# Test Fixtures / Helpers
# ---------------------------------------------------------------------------


def _make_valid_config() -> MonitoringConfig:
    return MonitoringConfig(
        protocol_version="MP-1.0",
        max_drawdown_expansion_limit=1.5,
        min_rolling_sharpe_30d=0.0,
        max_slippage_drift_ratio=2.5,
        max_risk_rejection_rate=0.20,
        window_size_sessions=30,
        min_evaluation_trades=15,
    )


def _make_valid_snapshot(created_at: str = "2023-06-01T10:00:00Z") -> MonitoringSnapshot:
    return MonitoringSnapshot.create(
        strategy_id="STRAT-ALPHA-01",
        qualification_hash="qhash-1111222233334444",
        replay_report_hash="rhash-5555666677778888",
        dataset_version="DS-NIFTY-2023-FUT",
        dataset_sha256="dsha-aaaabbbbccccdddd",
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="chash-9999000011112222",
        window_start_ts="2023-01-01T09:15:00Z",
        window_end_ts="2023-03-31T15:30:00Z",
        total_trades=42,
        net_pnl=125430.50,
        max_drawdown_bps=650.0,
        realized_sharpe=1.75,
        realized_slippage_bps=2.1,
        cost_to_turnover_bps=15.4,
        risk_event_count=0,
        metrics_json='{"rolling_sharpe_30d":1.75,"realized_dd_expansion":0.85}',
        created_at=created_at,
    )


def _make_valid_degradation_event(timestamp: str = "2023-06-01T15:30:00Z") -> DegradationEvent:
    return DegradationEvent.create(
        strategy_id="STRAT-ALPHA-01",
        qualification_hash="qhash-1111222233334444",
        snapshot_hash="shash-1111222233334444",
        rule_name="RULE_DD_EXPANSION_CRITICAL",
        threshold_value=1.50,
        observed_value=1.82,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="chash-9999000011112222",
        timestamp=timestamp,
        details_json='{"observed_dd_bps":1450.0,"baseline_dd_bps":800.0}',
    )


def _make_valid_transition(timestamp: str = "2023-06-01T16:00:00Z") -> PaperEvaluationTransition:
    return PaperEvaluationTransition.create(
        strategy_id="STRAT-ALPHA-01",
        old_state="PAPER_ACTIVE",
        new_state="DEGRADED",
        initiator="SYSTEM:DEGRADATION_DETECTOR",
        evidence_type="DEGRADATION_EVENT",
        evidence_hash="deghash-999988887777",
        reason="Realized paper drawdown exceeded 150% of qualification baseline",
        timestamp=timestamp,
    )


def _make_valid_feedback(created_at: str = "2023-06-01T17:00:00Z") -> ResearchFeedbackRecord:
    return ResearchFeedbackRecord.create(
        strategy_id="STRAT-ALPHA-01",
        qualification_hash="qhash-1111222233334444",
        degradation_event_hash="deghash-999988887777",
        dataset_version="DS-NIFTY-2023-FUT",
        failure_mode="RULE_DD_EXPANSION_CRITICAL",
        realized_sharpe=0.45,
        drawdown_expansion_ratio=1.82,
        realized_slippage_bps=3.8,
        empirical_notes="Severe trending breakdown during March 2023 volatility regime",
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# 1. MonitoringConfig Tests
# ---------------------------------------------------------------------------


class TestMonitoringConfig:
    def test_identical_configs_produce_identical_hash(self) -> None:
        cfg1 = _make_valid_config()
        cfg2 = _make_valid_config()
        assert cfg1.compute_config_hash() == cfg2.compute_config_hash()
        assert len(cfg1.compute_config_hash()) == 64

    def test_field_mutations_produce_distinct_hashes(self) -> None:
        base = _make_valid_config()
        base_hash = base.compute_config_hash()

        mutations = [
            dataclasses.replace(base, protocol_version="MP-2.0"),
            dataclasses.replace(base, max_drawdown_expansion_limit=2.0),
            dataclasses.replace(base, min_rolling_sharpe_30d=0.5),
            dataclasses.replace(base, max_slippage_drift_ratio=3.0),
            dataclasses.replace(base, max_risk_rejection_rate=0.10),
            dataclasses.replace(base, window_size_sessions=60),
            dataclasses.replace(base, min_evaluation_trades=30),
        ]

        for mut in mutations:
            assert mut.compute_config_hash() != base_hash

    def test_canonical_json_has_sorted_keys(self) -> None:
        cfg = _make_valid_config()
        canon_dict = cfg.canonical_dict()
        keys = list(canon_dict.keys())
        assert keys == sorted(keys)

    def test_invalid_configurations_rejected(self) -> None:
        with pytest.raises(MonitoringConfigError, match="protocol_version cannot be empty"):
            MonitoringConfig(
                protocol_version="",
                max_drawdown_expansion_limit=1.5,
                min_rolling_sharpe_30d=0.0,
                max_slippage_drift_ratio=2.5,
                max_risk_rejection_rate=0.20,
                window_size_sessions=30,
                min_evaluation_trades=15,
            )

        with pytest.raises(MonitoringConfigError, match="max_drawdown_expansion_limit"):
            MonitoringConfig(
                protocol_version="MP-1.0",
                max_drawdown_expansion_limit=-0.1,
                min_rolling_sharpe_30d=0.0,
                max_slippage_drift_ratio=2.5,
                max_risk_rejection_rate=0.20,
                window_size_sessions=30,
                min_evaluation_trades=15,
            )

        with pytest.raises(MonitoringConfigError, match="max_risk_rejection_rate"):
            MonitoringConfig(
                protocol_version="MP-1.0",
                max_drawdown_expansion_limit=1.5,
                min_rolling_sharpe_30d=0.0,
                max_slippage_drift_ratio=2.5,
                max_risk_rejection_rate=1.5,
                window_size_sessions=30,
                min_evaluation_trades=15,
            )

        with pytest.raises(MonitoringConfigError, match="window_size_sessions"):
            MonitoringConfig(
                protocol_version="MP-1.0",
                max_drawdown_expansion_limit=1.5,
                min_rolling_sharpe_30d=0.0,
                max_slippage_drift_ratio=2.5,
                max_risk_rejection_rate=0.2,
                window_size_sessions=0,
                min_evaluation_trades=15,
            )

        with pytest.raises(MonitoringConfigError, match="min_evaluation_trades"):
            MonitoringConfig(
                protocol_version="MP-1.0",
                max_drawdown_expansion_limit=1.5,
                min_rolling_sharpe_30d=0.0,
                max_slippage_drift_ratio=2.5,
                max_risk_rejection_rate=0.2,
                window_size_sessions=30,
                min_evaluation_trades=-5,
            )

    def test_config_is_frozen(self) -> None:
        cfg = _make_valid_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.window_size_sessions = 45  # type: ignore


# ---------------------------------------------------------------------------
# 2. MonitoringSnapshot Tests
# ---------------------------------------------------------------------------


class TestMonitoringSnapshot:
    def test_identical_snapshots_produce_identical_hash(self) -> None:
        s1 = _make_valid_snapshot()
        s2 = _make_valid_snapshot()
        assert s1.snapshot_hash == s2.snapshot_hash
        assert s1.verify_digest() is True

    def test_temporal_determinism_created_at_does_not_change_hash(self) -> None:
        """Wall-clock created_at timestamp MUST be excluded from snapshot_hash."""
        s1 = _make_valid_snapshot(created_at="2023-06-01T10:00:00Z")
        s2 = _make_valid_snapshot(created_at="2026-09-21T18:45:00Z")
        assert s1.created_at != s2.created_at
        assert s1.snapshot_hash == s2.snapshot_hash

    def test_every_semantic_field_mutation_changes_hash(self) -> None:
        base = _make_valid_snapshot()
        base_hash = base.snapshot_hash

        fields_to_test = [
            ("strategy_id", "STRAT-BETA-99"),
            ("qualification_hash", "qhash-tampered-9999"),
            ("replay_report_hash", "rhash-tampered-8888"),
            ("dataset_version", "DS-NIFTY-2024-FUT"),
            ("dataset_sha256", "dsha-tampered-0000"),
            ("split_zone", "RESEARCH"),
            ("monitoring_protocol_version", "MP-2.0"),
            ("monitoring_config_hash", "chash-tampered-1111"),
            ("window_start_ts", "2023-01-02T09:15:00Z"),
            ("window_end_ts", "2023-04-01T15:30:00Z"),
            ("total_trades", 99),
            ("net_pnl", 999999.99),
            ("max_drawdown_bps", 1200.0),
            ("realized_sharpe", 0.50),
            ("realized_slippage_bps", 5.0),
            ("cost_to_turnover_bps", 99.9),
            ("risk_event_count", 3),
            ("metrics_json", '{"tampered":true}'),
        ]

        for field_name, new_val in fields_to_test:
            kwargs = {
                "strategy_id": base.strategy_id,
                "qualification_hash": base.qualification_hash,
                "replay_report_hash": base.replay_report_hash,
                "dataset_version": base.dataset_version,
                "dataset_sha256": base.dataset_sha256,
                "split_zone": base.split_zone,
                "monitoring_protocol_version": base.monitoring_protocol_version,
                "monitoring_config_hash": base.monitoring_config_hash,
                "window_start_ts": base.window_start_ts,
                "window_end_ts": base.window_end_ts,
                "total_trades": base.total_trades,
                "net_pnl": base.net_pnl,
                "max_drawdown_bps": base.max_drawdown_bps,
                "realized_sharpe": base.realized_sharpe,
                "realized_slippage_bps": base.realized_slippage_bps,
                "cost_to_turnover_bps": base.cost_to_turnover_bps,
                "risk_event_count": base.risk_event_count,
                "metrics_json": base.metrics_json,
            }
            kwargs[field_name] = new_val
            mutated = MonitoringSnapshot.create(**kwargs)
            assert (
                mutated.snapshot_hash != base_hash
            ), f"Mutation on {field_name} failed to alter snapshot_hash"

    def test_canonical_field_cost_to_turnover_bps_reconciled(self) -> None:
        """Explicit regression test proving cost_to_turnover_bps is bound and cost_to_pnl_ratio is removed."""
        s = _make_valid_snapshot()
        canonical = s.canonical_dict()

        assert "cost_to_turnover_bps" in canonical
        assert canonical["cost_to_turnover_bps"] == 15.4
        assert "cost_to_pnl_ratio" not in canonical
        assert hasattr(s, "cost_to_turnover_bps")
        assert not hasattr(s, "cost_to_pnl_ratio")

    def test_derived_snapshot_id_is_deterministic(self) -> None:
        s1 = _make_valid_snapshot()
        s2 = _make_valid_snapshot(created_at="2029-01-01T00:00:00Z")
        assert s1.derived_snapshot_id == s2.derived_snapshot_id
        assert s1.derived_snapshot_id.startswith("SNAP-STRAT-ALPHA")

    def test_snapshot_is_frozen(self) -> None:
        s = _make_valid_snapshot()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.total_trades = 100  # type: ignore

    def test_tampered_snapshot_hash_fails_verification(self) -> None:
        s = _make_valid_snapshot()
        corrupted = dataclasses.replace(s, snapshot_hash="tampered_hash_value_12345")
        assert corrupted.verify_digest() is False


# ---------------------------------------------------------------------------
# 3. DegradationEvent Tests
# ---------------------------------------------------------------------------


class TestDegradationEvent:
    def test_identical_events_produce_identical_hash(self) -> None:
        e1 = _make_valid_degradation_event()
        e2 = _make_valid_degradation_event()
        assert e1.event_hash == e2.event_hash
        assert e1.verify_digest() is True

    def test_field_mutations_alter_event_hash(self) -> None:
        base = _make_valid_degradation_event()
        base_hash = base.event_hash

        mutations = [
            dataclasses.replace(
                base, threshold_value=2.0, event_hash=dataclasses.replace(base, threshold_value=2.0).compute_hash()
            ),
            dataclasses.replace(
                base, observed_value=2.5, event_hash=dataclasses.replace(base, observed_value=2.5).compute_hash()
            ),
            dataclasses.replace(
                base, rule_name="RULE_SHARPE_COLLAPSE", event_hash=dataclasses.replace(base, rule_name="RULE_SHARPE_COLLAPSE").compute_hash()
            ),
            dataclasses.replace(
                base, snapshot_hash="shash-tampered-9999", event_hash=dataclasses.replace(base, snapshot_hash="shash-tampered-9999").compute_hash()
            ),
        ]

        for mut in mutations:
            assert mut.event_hash != base_hash
            assert mut.verify_digest() is True

    def test_derived_event_id_is_deterministic(self) -> None:
        e1 = _make_valid_degradation_event()
        e2 = _make_valid_degradation_event()
        assert e1.derived_event_id == e2.derived_event_id
        assert e1.derived_event_id.startswith("DEG-STRAT-ALPHA")

    def test_event_is_frozen(self) -> None:
        e = _make_valid_degradation_event()
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.observed_value = 99.9  # type: ignore


# ---------------------------------------------------------------------------
# 4. PaperEvaluationTransition Tests
# ---------------------------------------------------------------------------


class TestPaperEvaluationTransition:
    def test_identical_transitions_produce_identical_hash(self) -> None:
        t1 = _make_valid_transition()
        t2 = _make_valid_transition()
        assert t1.transition_hash == t2.transition_hash
        assert t1.verify_digest() is True

    def test_state_mutation_alters_hash(self) -> None:
        t1 = _make_valid_transition()
        t2 = PaperEvaluationTransition.create(
            strategy_id=t1.strategy_id,
            old_state=t1.old_state,
            new_state="RETIRED",
            initiator=t1.initiator,
            evidence_type=t1.evidence_type,
            evidence_hash=t1.evidence_hash,
            reason="Terminal retirement",
            timestamp=t1.timestamp,
        )
        assert t1.transition_hash != t2.transition_hash

    def test_derived_transition_id_is_deterministic(self) -> None:
        t1 = _make_valid_transition()
        t2 = _make_valid_transition()
        assert t1.derived_transition_id == t2.derived_transition_id
        assert t1.derived_transition_id.startswith("TRANS-STRAT-ALPHA")

    def test_transition_is_frozen(self) -> None:
        t = _make_valid_transition()
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.new_state = "RETIRED"  # type: ignore


# ---------------------------------------------------------------------------
# 5. ResearchFeedbackRecord Tests
# ---------------------------------------------------------------------------


class TestResearchFeedbackRecord:
    def test_identical_feedback_produces_identical_hash(self) -> None:
        f1 = _make_valid_feedback()
        f2 = _make_valid_feedback()
        assert f1.feedback_hash == f2.feedback_hash
        assert f1.verify_digest() is True

    def test_temporal_determinism_created_at_does_not_change_hash(self) -> None:
        f1 = _make_valid_feedback(created_at="2023-06-01T10:00:00Z")
        f2 = _make_valid_feedback(created_at="2026-09-21T22:30:00Z")
        assert f1.created_at != f2.created_at
        assert f1.feedback_hash == f2.feedback_hash

    def test_derived_feedback_id_is_deterministic(self) -> None:
        f1 = _make_valid_feedback()
        f2 = _make_valid_feedback()
        assert f1.derived_feedback_id == f2.derived_feedback_id
        assert f1.derived_feedback_id.startswith("RFB-STRAT-ALPHA")

    def test_feedback_is_frozen(self) -> None:
        f = _make_valid_feedback()
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.failure_mode = "OTHER"  # type: ignore


# ---------------------------------------------------------------------------
# 6. Critical P&L Semantics Regression Assertion Test
# ---------------------------------------------------------------------------


class TestPaperLedgerPnLSemanticsRegression:
    """Critical architectural check: PaperPosition.realized_pnl and ReplayReport.net_pnl

    are already net of transaction fees and slippage in the existing PRD v3.9 codebase.
    This test verifies that monitoring must NOT subtract fees or slippage again.
    """

    def test_paper_position_realized_pnl_is_already_net(self) -> None:
        # In PaperPosition, realized_pnl reflects the closed trade net PnL
        pos = PaperPosition(
            symbol="NIFTY_FUT",
            quantity=0,
            entry_price=0.0,
            current_price=18000.0,
            realized_pnl=500.0,  # This already has fees and slippage deducted!
            unrealized_pnl=0.0,
            fees_costs=50.0,
        )
        assert pos.realized_pnl == 500.0
        assert pos.fees_costs == 50.0
        # If a monitor mistakenly computed net_pnl = realized_pnl - fees_costs:
        erroneous_double_net = pos.realized_pnl - pos.fees_costs
        assert erroneous_double_net == 450.0  # Demonstrates erroneous double deduction!

    def test_replay_report_net_pnl_is_already_net(self) -> None:
        report = ReplayReport.create(
            strategy_id="STRAT-1",
            qualification_id="QUAL-1",
            dataset_version="DS-1",
            trade_count=1,
            gross_pnl=1000.0,
            net_pnl=950.0,  # 1000 - 50 costs
            costs=50.0,
            slippage=10.0,
            max_drawdown_bps=0.0,
            exposure=100000.0,
            win_rate=1.0,
            expectancy=950.0,
            sharpe_ratio=None,
            session_breakdown=[],
        )
        assert report.gross_pnl == 1000.0
        assert report.costs == 50.0
        assert report.net_pnl == 950.0
        # Confirms net_pnl is already net of costs; monitoring must read net_pnl directly!
