"""Unit tests for PRD v4.0 EvaluationLedger & SQLite Append-Only Evidence.

Tests:
- Schema creation (tables, columns, foreign keys, unique constraints).
- Immutability enforcement via SQL triggers (NO UPDATE, NO DELETE on all 3 tables).
- Idempotent insertion logic (first insert succeeds, identical second returns existing).
- Hash collision detection with conflicting content.
- Cryptographic hash verification and tampering rejection.
- Referential integrity (DegradationEvent -> MonitoringSnapshot).
- Deterministic read ordering (explicit ORDER BY).
- Exact type security (rejection of duck typing or subclasses).
- Persistence across connection close and reopen.
- Absence of INSERT OR REPLACE overwriting behavior.
"""

import dataclasses
import sqlite3
import pytest

from quantmind.paper.evaluation.ledger import (
    EvaluationLedger,
    EvaluationLedgerError,
    EvaluationLedgerIntegrityError,
)
from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringConfig,
    MonitoringSnapshot,
    PaperEvaluationBaseline,
    PaperEvaluationTransition,
)
from quantmind.paper.ledger import PaperLedger
from quantmind.paper.models import ReplayReport, ReplaySessionSummary


# ---------------------------------------------------------------------------
# Test Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_snapshot(
    strategy_id: str = "STRAT-ALPHA-01",
    window_start: str = "2023-01-01T09:15:00Z",
    window_end: str = "2023-01-31T15:30:00Z",
    net_pnl: float = 10000.0,
    created_at: str = "2023-02-01T10:00:00Z",
) -> MonitoringSnapshot:
    return MonitoringSnapshot.create(
        strategy_id=strategy_id,
        qualification_hash="qhash-alpha-1111",
        replay_report_hash="rhash-alpha-2222",
        dataset_version="DS-NIFTY-2023-FUT",
        dataset_sha256="dsha-alpha-3333",
        split_zone="FORWARD_PAPER",
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="chash-alpha-4444",
        window_start_ts=window_start,
        window_end_ts=window_end,
        total_trades=10,
        net_pnl=net_pnl,
        max_drawdown_bps=500.0,
        realized_sharpe=1.5,
        realized_slippage_bps=2.0,
        cost_to_turnover_bps=12.5,
        risk_event_count=0,
        metrics_json='{"rolling_sharpe":1.5}',
        created_at=created_at,
    )


def _make_degradation_event(
    snapshot_hash: str,
    strategy_id: str = "STRAT-ALPHA-01",
    timestamp: str = "2023-02-01T15:30:00Z",
    rule_name: str = "RULE_DD_EXPANSION_CRITICAL",
    baseline_replay_report_hash: str = "bhash-alpha-9999",
) -> DegradationEvent:
    return DegradationEvent.create(
        strategy_id=strategy_id,
        qualification_hash="qhash-alpha-1111",
        snapshot_hash=snapshot_hash,
        baseline_replay_report_hash=baseline_replay_report_hash,
        rule_name=rule_name,
        threshold_value=1.5,
        observed_value=1.82,
        monitoring_protocol_version="MP-1.0",
        monitoring_config_hash="chash-alpha-4444",
        timestamp=timestamp,
        details_json='{"realized_dd_bps":1450.0}',
    )


def _make_replay_report(
    strategy_id: str = "STRAT-ALPHA-01",
    qualification_id: str = "QUAL-001",
    qualification_hash: str = "qhash-alpha-1111",
    dataset_version: str = "DS-NIFTY-2026",
    dataset_sha256: str = "dsha-alpha-1234",
    split_zone: str = "FORWARD_PAPER",
    max_drawdown_bps: float = 500.0,
    slippage_bps_per_side: float = 4.0,
    execution_policy: str = "next_bar_open_v1",
    cost_schedule_hash: str = "csched-1111",
    risk_config_hash: str = "risk-1111",
    created_at: str = "2026-03-01T15:30:00Z",
) -> ReplayReport:
    sessions = (
        ReplaySessionSummary(
            session_id="SESS-001",
            start_ts="2026-03-01T09:15:00Z",
            end_ts="2026-03-01T15:30:00Z",
            trades=5,
            gross_pnl=2500.0,
            net_pnl=2200.0,
        ),
    )
    return ReplayReport.create(
        strategy_id=strategy_id,
        qualification_id=qualification_id,
        dataset_version=dataset_version,
        trade_count=50,
        gross_pnl=15000.0,
        net_pnl=12000.0,
        costs=1800.0,
        slippage=1200.0,
        max_drawdown_bps=max_drawdown_bps,
        exposure=50000.0,
        win_rate=0.55,
        expectancy=240.0,
        sharpe_ratio=1.85,
        session_breakdown=sessions,
        created_at=created_at,
        qualification_hash=qualification_hash,
        dataset_sha256=dataset_sha256,
        split_zone=split_zone,
        slippage_bps_per_side=slippage_bps_per_side,
        execution_policy=execution_policy,
        cost_schedule_hash=cost_schedule_hash,
        risk_config_hash=risk_config_hash,
    )


def _make_baseline_from_report(report: ReplayReport) -> PaperEvaluationBaseline:
    return PaperEvaluationBaseline.from_replay_report(report)


def _make_transition(
    evidence_hash: str,
    strategy_id: str = "STRAT-ALPHA-01",
    timestamp: str = "2023-02-01T16:00:00Z",
) -> PaperEvaluationTransition:
    return PaperEvaluationTransition.create(
        strategy_id=strategy_id,
        old_state="PAPER_ACTIVE",
        new_state="DEGRADED",
        initiator="SYSTEM:DEGRADATION_DETECTOR",
        evidence_type="DEGRADATION_EVENT",
        evidence_hash=evidence_hash,
        reason="Drawdown expansion breached threshold limit",
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# 1. Schema & Table Tests
# ---------------------------------------------------------------------------


class TestEvaluationLedgerSchema:
    def test_all_five_tables_and_columns_created(self) -> None:
        ledger = EvaluationLedger()
        conn = ledger._connection

        # Check table names
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert table_names == {
            "monitoring_snapshots",
            "degradation_events",
            "paper_evaluation_transitions",
            "evaluation_baselines",
            "paper_evaluation_regimes",
            "research_feedback",
        }

        # Check columns of monitoring_snapshots
        snap_cols = {
            c["name"] for c in conn.execute("PRAGMA table_info(monitoring_snapshots)").fetchall()
        }
        assert "snapshot_id" in snap_cols
        assert "snapshot_hash" in snap_cols
        assert "cost_to_turnover_bps" in snap_cols
        assert "strategy_id" in snap_cols
        assert "regime_hash" in snap_cols

        # Check columns of degradation_events
        deg_cols = {
            c["name"] for c in conn.execute("PRAGMA table_info(degradation_events)").fetchall()
        }
        assert "event_id" in deg_cols
        assert "event_hash" in deg_cols
        assert "snapshot_hash" in deg_cols
        assert "baseline_replay_report_hash" in deg_cols

        # Check columns of paper_evaluation_transitions
        trans_cols = {
            c["name"] for c in conn.execute("PRAGMA table_info(paper_evaluation_transitions)").fetchall()
        }
        assert "transition_id" in trans_cols
        assert "transition_hash" in trans_cols
        assert "evidence_hash" in trans_cols

        # Check columns of evaluation_baselines
        base_cols = {
            c["name"] for c in conn.execute("PRAGMA table_info(evaluation_baselines)").fetchall()
        }
        assert "binding_hash" in base_cols
        assert "baseline_replay_report_hash" in base_cols
        assert "strategy_id" in base_cols
        assert "qualification_hash" in base_cols
        assert "baseline_execution_policy" in base_cols
        assert "baseline_cost_schedule_hash" in base_cols
        assert "baseline_risk_config_hash" in base_cols

        # Check columns of paper_evaluation_regimes
        regime_cols = {
            c["name"] for c in conn.execute("PRAGMA table_info(paper_evaluation_regimes)").fetchall()
        }
        assert "regime_hash" in regime_cols
        assert "strategy_id" in regime_cols
        assert "qualification_hash" in regime_cols
        assert "baseline_replay_report_hash" in regime_cols
        assert "forward_dataset_version" in regime_cols
        assert "forward_dataset_sha256" in regime_cols
        assert "execution_policy" in regime_cols
        assert "cost_schedule_hash" in regime_cols
        assert "risk_config_hash" in regime_cols
        assert "monitoring_protocol_version" in regime_cols
        assert "created_at" in regime_cols

        # Check columns of research_feedback
        feedback_cols = {
            c["name"] for c in conn.execute("PRAGMA table_info(research_feedback)").fetchall()
        }
        assert "feedback_id" in feedback_cols
        assert "feedback_hash" in feedback_cols
        assert "degradation_event_hash" in feedback_cols
        assert "strategy_id" in feedback_cols
        assert "qualification_hash" in feedback_cols
        assert "dataset_version" in feedback_cols
        assert "failure_mode" in feedback_cols
        assert "realized_sharpe" in feedback_cols
        assert "drawdown_expansion_ratio" in feedback_cols
        assert "realized_slippage_bps" in feedback_cols
        assert "empirical_notes" in feedback_cols
        assert "created_at" in feedback_cols


# ---------------------------------------------------------------------------
# 2. Immutability Trigger Tests (Direct SQL UPDATE / DELETE Abort)
# ---------------------------------------------------------------------------


class TestEvaluationLedgerImmutability:
    def test_regimes_update_and_delete_aborted_by_trigger(self) -> None:
        from quantmind.paper.evaluation.models import PaperEvaluationRegime
        ledger = EvaluationLedger()
        regime = PaperEvaluationRegime.create(
            strategy_id="STRAT-001",
            qualification_hash="qual_hash_" + "0" * 54,
            baseline_replay_report_hash="rep_hash_" + "0" * 55,
            forward_dataset_version="nifty_forward_v1",
            forward_dataset_sha256="dsha_" + "0" * 59,
            execution_policy="market_order_v1",
            cost_schedule_hash="csh_" + "0" * 60,
            risk_config_hash="rch_" + "0" * 60,
            monitoring_protocol_version="MP-1.0",
        )
        ledger.register_regime(regime)

        # Attempt direct SQL UPDATE
        with pytest.raises(sqlite3.IntegrityError, match="immutable and cannot be updated"):
            ledger._connection.execute(
                "UPDATE paper_evaluation_regimes SET execution_policy = 'tampered' WHERE regime_hash = ?",
                (regime.regime_hash,),
            )

        # Attempt direct SQL DELETE
        with pytest.raises(sqlite3.IntegrityError, match="permanent and cannot be deleted"):
            ledger._connection.execute(
                "DELETE FROM paper_evaluation_regimes WHERE regime_hash = ?",
                (regime.regime_hash,),
            )

    def test_snapshots_update_and_delete_aborted_by_trigger(self) -> None:
        ledger = EvaluationLedger()
        snap = _make_snapshot()
        ledger.get_or_insert_snapshot(snap)

        # Attempt direct SQL UPDATE
        with pytest.raises(sqlite3.IntegrityError, match="immutable and cannot be updated"):
            ledger._connection.execute(
                "UPDATE monitoring_snapshots SET net_pnl = 999999.0 WHERE snapshot_id = ?",
                (snap.derived_snapshot_id,),
            )

        # Attempt direct SQL DELETE
        with pytest.raises(sqlite3.IntegrityError, match="permanent and cannot be deleted"):
            ledger._connection.execute(
                "DELETE FROM monitoring_snapshots WHERE snapshot_id = ?",
                (snap.derived_snapshot_id,),
            )

    def test_degradation_events_update_and_delete_aborted_by_trigger(self) -> None:
        ledger = EvaluationLedger()
        snap = _make_snapshot()
        ledger.get_or_insert_snapshot(snap)
        deg = _make_degradation_event(snap.snapshot_hash)
        ledger.record_degradation_event(deg)

        # Attempt direct SQL UPDATE
        with pytest.raises(sqlite3.IntegrityError, match="immutable and cannot be updated"):
            ledger._connection.execute(
                "UPDATE degradation_events SET threshold_value = 99.0 WHERE event_id = ?",
                (deg.derived_event_id,),
            )

        # Attempt direct SQL DELETE
        with pytest.raises(sqlite3.IntegrityError, match="permanent and cannot be deleted"):
            ledger._connection.execute(
                "DELETE FROM degradation_events WHERE event_id = ?",
                (deg.derived_event_id,),
            )

    def test_transitions_update_and_delete_aborted_by_trigger(self) -> None:
        ledger = EvaluationLedger()
        trans = _make_transition("some-evidence-hash")
        ledger.record_transition(trans)

        # Attempt direct SQL UPDATE
        with pytest.raises(sqlite3.IntegrityError, match="immutable and cannot be updated"):
            ledger._connection.execute(
                "UPDATE paper_evaluation_transitions SET new_state = 'PAPER_ACTIVE' WHERE transition_id = ?",
                (trans.derived_transition_id,),
            )

        # Attempt direct SQL DELETE
        with pytest.raises(sqlite3.IntegrityError, match="permanent and cannot be deleted"):
            ledger._connection.execute(
                "DELETE FROM paper_evaluation_transitions WHERE transition_id = ?",
                (trans.derived_transition_id,),
            )


# ---------------------------------------------------------------------------
# 3. Snapshot Idempotency & Hash Collision Tests
# ---------------------------------------------------------------------------


class TestSnapshotIdempotency:
    def test_first_insert_and_identical_second_insert_returns_existing(self) -> None:
        ledger = EvaluationLedger()
        snap1 = _make_snapshot()
        res1 = ledger.get_or_insert_snapshot(snap1)
        assert res1.snapshot_hash == snap1.snapshot_hash

        # Re-evaluating returns the exact existing record idempotently
        snap2 = _make_snapshot(created_at="2029-01-01T00:00:00Z")  # created_at differs, but hash identical!
        res2 = ledger.get_or_insert_snapshot(snap2)
        assert res2.snapshot_hash == snap1.snapshot_hash
        assert res2.created_at == snap1.created_at  # Returns original record's created_at

        # Verify only 1 row exists
        count = ledger._connection.execute("SELECT COUNT(*) as cnt FROM monitoring_snapshots").fetchone()["cnt"]
        assert count == 1

    def test_same_hash_with_conflicting_content_raises_integrity_error(self) -> None:
        ledger = EvaluationLedger()
        snap1 = _make_snapshot()

        # Simulate a corrupted or colliding database row already existing under snap1's hash
        ledger._connection.execute(
            """
            INSERT INTO monitoring_snapshots (
                snapshot_id, strategy_id, qualification_hash, replay_report_hash,
                dataset_version, dataset_sha256, split_zone, monitoring_protocol_version,
                monitoring_config_hash, window_start_ts, window_end_ts, total_trades,
                net_pnl, max_drawdown_bps, realized_sharpe, realized_slippage_bps,
                cost_to_turnover_bps, risk_event_count, metrics_json, created_at, snapshot_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SNAP-CORRUPT-01",
                snap1.strategy_id,
                snap1.qualification_hash,
                snap1.replay_report_hash,
                snap1.dataset_version,
                snap1.dataset_sha256,
                snap1.split_zone,
                snap1.monitoring_protocol_version,
                snap1.monitoring_config_hash,
                snap1.window_start_ts,
                snap1.window_end_ts,
                99999,  # Conflicting field value!
                snap1.net_pnl,
                snap1.max_drawdown_bps,
                snap1.realized_sharpe,
                snap1.realized_slippage_bps,
                snap1.cost_to_turnover_bps,
                snap1.risk_event_count,
                snap1.metrics_json,
                snap1.created_at,
                snap1.snapshot_hash,
            ),
        )

        with pytest.raises(EvaluationLedgerIntegrityError, match="conflicting semantic content"):
            ledger.get_or_insert_snapshot(snap1)

    def test_tampered_digest_rejected_on_insert(self) -> None:
        ledger = EvaluationLedger()
        snap = _make_snapshot()
        tampered = dataclasses.replace(snap, snapshot_hash="tampered-fake-hash-12345")
        with pytest.raises(EvaluationLedgerIntegrityError, match="digest verification"):
            ledger.get_or_insert_snapshot(tampered)


# ---------------------------------------------------------------------------
# 4. Referential Integrity Tests
# ---------------------------------------------------------------------------


class TestReferentialIntegrity:
    def test_degradation_event_with_valid_snapshot_reference_succeeds(self) -> None:
        ledger = EvaluationLedger()
        snap = _make_snapshot()
        ledger.get_or_insert_snapshot(snap)

        event = _make_degradation_event(snap.snapshot_hash)
        saved_event = ledger.record_degradation_event(event)
        assert saved_event.event_hash == event.event_hash

    def test_degradation_event_with_nonexistent_snapshot_raises_error(self) -> None:
        ledger = EvaluationLedger()
        event = _make_degradation_event("nonexistent-snapshot-hash-999")
        with pytest.raises(EvaluationLedgerIntegrityError, match="non-existent snapshot_hash"):
            ledger.record_degradation_event(event)

    def test_transition_record_succeeds(self) -> None:
        ledger = EvaluationLedger()
        trans = _make_transition("some-evidence-hash")
        saved = ledger.record_transition(trans)
        assert saved.transition_hash == trans.transition_hash

    def test_tampered_event_or_transition_rejected(self) -> None:
        ledger = EvaluationLedger()
        snap = _make_snapshot()
        ledger.get_or_insert_snapshot(snap)

        event = _make_degradation_event(snap.snapshot_hash)
        tampered_event = dataclasses.replace(event, event_hash="bad-hash")
        with pytest.raises(EvaluationLedgerIntegrityError, match="digest verification"):
            ledger.record_degradation_event(tampered_event)

        trans = _make_transition("ev-hash")
        tampered_trans = dataclasses.replace(trans, transition_hash="bad-hash")
        with pytest.raises(EvaluationLedgerIntegrityError, match="digest verification"):
            ledger.record_transition(tampered_trans)


# ---------------------------------------------------------------------------
# 5. Deterministic Query & Ordering Tests
# ---------------------------------------------------------------------------


class TestDeterministicReadAPIs:
    def test_list_snapshots_ordered_by_window_timestamps(self) -> None:
        ledger = EvaluationLedger()
        s1 = _make_snapshot(window_start="2023-01-01T09:15:00Z", window_end="2023-01-10T15:30:00Z", net_pnl=100.0)
        s2 = _make_snapshot(window_start="2023-01-11T09:15:00Z", window_end="2023-01-20T15:30:00Z", net_pnl=200.0)
        s3 = _make_snapshot(window_start="2023-01-21T09:15:00Z", window_end="2023-01-31T15:30:00Z", net_pnl=300.0)

        # Insert out of order
        ledger.get_or_insert_snapshot(s2)
        ledger.get_or_insert_snapshot(s3)
        ledger.get_or_insert_snapshot(s1)

        # Query all
        results = ledger.list_snapshots("STRAT-ALPHA-01")
        assert len(results) == 3
        assert [r.window_start_ts for r in results] == [
            "2023-01-01T09:15:00Z",
            "2023-01-11T09:15:00Z",
            "2023-01-21T09:15:00Z",
        ]

        # Query with bounds
        filtered = ledger.list_snapshots(
            "STRAT-ALPHA-01",
            window_start_ts="2023-01-10T00:00:00Z",
            window_end_ts="2023-01-25T00:00:00Z",
        )
        assert len(filtered) == 1
        assert filtered[0].window_start_ts == "2023-01-11T09:15:00Z"

    def test_list_degradation_events_ordered_by_timestamp(self) -> None:
        ledger = EvaluationLedger()
        s1 = _make_snapshot(window_start="2023-01-01T09:15:00Z", window_end="2023-01-10T15:30:00Z")
        s2 = _make_snapshot(window_start="2023-01-11T09:15:00Z", window_end="2023-01-20T15:30:00Z")
        ledger.get_or_insert_snapshot(s1)
        ledger.get_or_insert_snapshot(s2)

        e1 = _make_degradation_event(s1.snapshot_hash, timestamp="2023-01-05T12:00:00Z")
        e2 = _make_degradation_event(s2.snapshot_hash, timestamp="2023-01-15T12:00:00Z")

        # Insert in reverse order
        ledger.record_degradation_event(e2)
        ledger.record_degradation_event(e1)

        events = ledger.list_degradation_events("STRAT-ALPHA-01")
        assert len(events) == 2
        assert events[0].timestamp == "2023-01-05T12:00:00Z"
        assert events[1].timestamp == "2023-01-15T12:00:00Z"

    def test_list_transitions_ordered_by_timestamp(self) -> None:
        ledger = EvaluationLedger()
        t1 = _make_transition("ev-1", timestamp="2023-01-05T12:00:00Z")
        t2 = _make_transition("ev-2", timestamp="2023-01-15T12:00:00Z")

        ledger.record_transition(t2)
        ledger.record_transition(t1)

        transitions = ledger.list_transitions("STRAT-ALPHA-01")
        assert len(transitions) == 2
        assert transitions[0].timestamp == "2023-01-05T12:00:00Z"
        assert transitions[1].timestamp == "2023-01-15T12:00:00Z"


# ---------------------------------------------------------------------------
# 6. Type Security & Bypass Tests
# ---------------------------------------------------------------------------


class TestTypeSecurity:
    def test_duck_typed_objects_rejected(self) -> None:
        ledger = EvaluationLedger()

        class DuckSnapshot:
            snapshot_hash = "duck"

        with pytest.raises(TypeError, match="Expected MonitoringSnapshot"):
            ledger.get_or_insert_snapshot(DuckSnapshot())  # type: ignore

        class DuckEvent:
            event_hash = "duck"

        with pytest.raises(TypeError, match="Expected DegradationEvent"):
            ledger.record_degradation_event(DuckEvent())  # type: ignore

        class DuckTransition:
            transition_hash = "duck"

        with pytest.raises(TypeError, match="Expected PaperEvaluationTransition"):
            ledger.record_transition(DuckTransition())  # type: ignore


# ---------------------------------------------------------------------------
# 7. Persistence Across Reopen Tests
# ---------------------------------------------------------------------------


class TestPersistenceAcrossReopen:
    def test_data_persists_across_ledger_close_and_reopen(self, tmp_path) -> None:
        db_file = tmp_path / "evaluation.db"

        # Session 1: write records
        ledger1 = EvaluationLedger(db_file)
        snap = _make_snapshot()
        ledger1.get_or_insert_snapshot(snap)
        event = _make_degradation_event(snap.snapshot_hash)
        ledger1.record_degradation_event(event)
        trans = _make_transition(event.event_hash)
        ledger1.record_transition(trans)
        ledger1.close()

        # Session 2: reopen and verify
        ledger2 = EvaluationLedger(db_file)
        retrieved_snap = ledger2.get_snapshot(snap.snapshot_hash)
        assert retrieved_snap is not None
        assert retrieved_snap.snapshot_hash == snap.snapshot_hash
        assert retrieved_snap.cost_to_turnover_bps == snap.cost_to_turnover_bps

        retrieved_event = ledger2.get_degradation_event(event.event_hash)
        assert retrieved_event is not None
        assert retrieved_event.event_hash == event.event_hash

        retrieved_trans = ledger2.get_transition(trans.transition_hash)
        assert retrieved_trans is not None
        assert retrieved_trans.transition_hash == trans.transition_hash
        ledger2.close()


# ---------------------------------------------------------------------------
# 8. Authoritative Evaluation Baseline Ledger Tests (M4.3)
# ---------------------------------------------------------------------------


class TestEvaluationBaselineLedger:
    """Tests for authoritative PaperEvaluationBaseline persistence, uniqueness, and verification."""

    def test_two_baseline_attack_rejected(self) -> None:
        """Adversarial test: attempt to register baseline B for an already bound strategy + qualification."""
        ledger = EvaluationLedger()

        rep_a = _make_replay_report(max_drawdown_bps=500.0, slippage_bps_per_side=4.0)
        base_a = _make_baseline_from_report(rep_a)

        # Baseline A registered successfully
        reg_a = ledger.register_baseline(base_a, replay_report=rep_a)
        assert reg_a.binding_hash == base_a.binding_hash

        # Baseline B has different parameters (e.g. max_drawdown_bps) -> different report hash
        rep_b = _make_replay_report(max_drawdown_bps=350.0, slippage_bps_per_side=2.0)
        base_b = _make_baseline_from_report(rep_b)

        # Attempting to register baseline B for the SAME strategy + qualification MUST fail closed
        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="already has an authoritative baseline registered",
        ):
            ledger.register_baseline(base_b, replay_report=rep_b)

        # Verify baseline A remains the sole authoritative baseline
        canonical = ledger.get_baseline(base_a.strategy_id, base_a.qualification_hash)
        assert canonical is not None
        assert canonical.binding_hash == base_a.binding_hash
        assert canonical.baseline_replay_report_hash == rep_a.report_hash

    def test_identical_re_registration_returns_existing_baseline(self) -> None:
        """Idempotency test: registering the exact same baseline multiple times succeeds and returns existing."""
        ledger = EvaluationLedger()
        rep = _make_replay_report()
        base = _make_baseline_from_report(rep)

        reg1 = ledger.register_baseline(base, replay_report=rep)
        reg2 = ledger.register_baseline(base, replay_report=rep)

        assert reg1.binding_hash == reg2.binding_hash
        assert reg1 == reg2

    def test_missing_replay_report_rejected(self) -> None:
        """Cannot register an unverified metadata-only baseline without authoritative replay report."""
        ledger = EvaluationLedger()
        rep = _make_replay_report()
        base = _make_baseline_from_report(rep)

        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="Authoritative ReplayReport is required",
        ):
            ledger.register_baseline(base)

    def test_tampered_replay_report_rejected(self) -> None:
        """Replay report with tampered payload fails digest verification and is rejected."""
        ledger = EvaluationLedger()
        rep = _make_replay_report()
        base = _make_baseline_from_report(rep)

        tampered_rep = ReplayReport(
            **{k: v for k, v in rep.__dict__.items() if k != "gross_pnl"},
            gross_pnl=999999.0,
        )

        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="Bound ReplayReport failed digest verification",
        ):
            ledger.register_baseline(base, replay_report=tampered_rep)

    def test_mismatched_replay_report_hash_rejected(self) -> None:
        ledger = EvaluationLedger()
        rep1 = _make_replay_report(max_drawdown_bps=500.0)
        rep2 = _make_replay_report(max_drawdown_bps=600.0)
        base1 = _make_baseline_from_report(rep1)

        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="ReplayReport hash mismatch",
        ):
            ledger.register_baseline(base1, replay_report=rep2)

    def test_mismatched_strategy_id_rejected(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report(strategy_id="STRAT-ALPHA-01")
        base_wrong_strat = PaperEvaluationBaseline.create(
            strategy_id="STRAT-OTHER",
            qualification_hash=rep.qualification_hash,
            baseline_replay_report_hash=rep.report_hash,
            baseline_dataset_version=rep.dataset_version,
            baseline_dataset_sha256=rep.dataset_sha256,
            baseline_execution_policy=rep.execution_policy,
            baseline_cost_schedule_hash=rep.cost_schedule_hash,
            baseline_risk_config_hash=rep.risk_config_hash,
        )
        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="ReplayReport strategy_id mismatch",
        ):
            ledger.register_baseline(base_wrong_strat, replay_report=rep)

    def test_mismatched_qualification_hash_rejected(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report(qualification_hash="qhash-CORRECT")
        base_wrong_qual = PaperEvaluationBaseline.create(
            strategy_id=rep.strategy_id,
            qualification_hash="qhash-WRONG",
            baseline_replay_report_hash=rep.report_hash,
            baseline_dataset_version=rep.dataset_version,
            baseline_dataset_sha256=rep.dataset_sha256,
            baseline_execution_policy=rep.execution_policy,
            baseline_cost_schedule_hash=rep.cost_schedule_hash,
            baseline_risk_config_hash=rep.risk_config_hash,
        )
        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="ReplayReport qualification_hash mismatch",
        ):
            ledger.register_baseline(base_wrong_qual, replay_report=rep)

    def test_mismatched_dataset_version_rejected(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report(dataset_version="DS-V1")
        base_wrong_ds = PaperEvaluationBaseline.create(
            strategy_id=rep.strategy_id,
            qualification_hash=rep.qualification_hash,
            baseline_replay_report_hash=rep.report_hash,
            baseline_dataset_version="DS-V2",
            baseline_dataset_sha256=rep.dataset_sha256,
            baseline_execution_policy=rep.execution_policy,
            baseline_cost_schedule_hash=rep.cost_schedule_hash,
            baseline_risk_config_hash=rep.risk_config_hash,
        )
        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="ReplayReport dataset_version mismatch",
        ):
            ledger.register_baseline(base_wrong_ds, replay_report=rep)

    def test_mismatched_execution_policy_rejected(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report(execution_policy="next_bar_open_v1")
        base_wrong_pol = PaperEvaluationBaseline.create(
            strategy_id=rep.strategy_id,
            qualification_hash=rep.qualification_hash,
            baseline_replay_report_hash=rep.report_hash,
            baseline_dataset_version=rep.dataset_version,
            baseline_dataset_sha256=rep.dataset_sha256,
            baseline_execution_policy="limit_cross_v2",
            baseline_cost_schedule_hash=rep.cost_schedule_hash,
            baseline_risk_config_hash=rep.risk_config_hash,
        )
        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="ReplayReport execution_policy mismatch",
        ):
            ledger.register_baseline(base_wrong_pol, replay_report=rep)

    def test_mismatched_cost_schedule_rejected(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report(cost_schedule_hash="csched-1111")
        base_wrong_cost = PaperEvaluationBaseline.create(
            strategy_id=rep.strategy_id,
            qualification_hash=rep.qualification_hash,
            baseline_replay_report_hash=rep.report_hash,
            baseline_dataset_version=rep.dataset_version,
            baseline_dataset_sha256=rep.dataset_sha256,
            baseline_execution_policy=rep.execution_policy,
            baseline_cost_schedule_hash="csched-DIFFERENT",
            baseline_risk_config_hash=rep.risk_config_hash,
        )
        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="ReplayReport cost_schedule_hash mismatch",
        ):
            ledger.register_baseline(base_wrong_cost, replay_report=rep)

    def test_mismatched_risk_config_rejected(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report(risk_config_hash="risk-1111")
        base_wrong_risk = PaperEvaluationBaseline.create(
            strategy_id=rep.strategy_id,
            qualification_hash=rep.qualification_hash,
            baseline_replay_report_hash=rep.report_hash,
            baseline_dataset_version=rep.dataset_version,
            baseline_dataset_sha256=rep.dataset_sha256,
            baseline_execution_policy=rep.execution_policy,
            baseline_cost_schedule_hash=rep.cost_schedule_hash,
            baseline_risk_config_hash="risk-DIFFERENT",
        )
        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="ReplayReport risk_config_hash mismatch",
        ):
            ledger.register_baseline(base_wrong_risk, replay_report=rep)

    def test_tampered_baseline_binding_hash_rejected(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report()
        base = _make_baseline_from_report(rep)

        tampered_base = PaperEvaluationBaseline(
            **{k: v for k, v in base.__dict__.items() if k != "binding_hash"},
            binding_hash="tampered_binding_hash_xyz",
        )
        with pytest.raises(
            EvaluationLedgerIntegrityError,
            match="PaperEvaluationBaseline failed digest verification",
        ):
            ledger.register_baseline(tampered_base, replay_report=rep)

    def test_duck_typed_baseline_rejected(self) -> None:
        ledger = EvaluationLedger()

        class DuckBaseline:
            strategy_id = "STRAT-01"
            qualification_hash = "qhash-01"

        with pytest.raises(TypeError, match="Expected PaperEvaluationBaseline"):
            ledger.register_baseline(DuckBaseline())  # type: ignore

    def test_get_baseline_returns_canonical_or_none(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report()
        base = _make_baseline_from_report(rep)

        # Before registration
        assert ledger.get_baseline(base.strategy_id, base.qualification_hash) is None

        # After registration
        ledger.register_baseline(base, replay_report=rep)
        canonical = ledger.get_baseline(base.strategy_id, base.qualification_hash)
        assert canonical is not None
        assert canonical.binding_hash == base.binding_hash
        assert canonical.baseline_replay_report_hash == base.baseline_replay_report_hash

    def test_get_baseline_by_hash(self) -> None:
        ledger = EvaluationLedger()
        rep = _make_replay_report()
        base = _make_baseline_from_report(rep)

        assert ledger.get_baseline_by_hash(base.binding_hash) is None

        ledger.register_baseline(base, replay_report=rep)
        found = ledger.get_baseline_by_hash(base.binding_hash)
        assert found is not None
        assert found.binding_hash == base.binding_hash

    def test_registration_via_paper_ledger(self) -> None:
        """PaperLedger integration: report is recorded in PaperLedger and verified during baseline registration."""
        paper_ledger = PaperLedger()
        eval_ledger = EvaluationLedger()

        rep = _make_replay_report()
        paper_ledger.record_report(rep)

        base = _make_baseline_from_report(rep)

        # Register passing paper_ledger instead of replay_report directly
        reg = eval_ledger.register_baseline(base, paper_ledger=paper_ledger)
        assert reg.binding_hash == base.binding_hash

    def test_evaluation_baselines_immutability_triggers(self) -> None:
        """Direct SQL UPDATE and DELETE on evaluation_baselines must be blocked by SQLite triggers."""
        ledger = EvaluationLedger()
        rep = _make_replay_report()
        base = _make_baseline_from_report(rep)
        ledger.register_baseline(base, replay_report=rep)

        # Direct SQL UPDATE must fail
        with pytest.raises(sqlite3.DatabaseError, match="evaluation baselines are immutable and cannot be updated"):
            ledger._connection.execute(
                "UPDATE evaluation_baselines SET baseline_replay_report_hash = 'tampered' WHERE binding_hash = ?",
                (base.binding_hash,),
            )

        # Direct SQL DELETE must fail
        with pytest.raises(sqlite3.DatabaseError, match="evaluation baselines are permanent and cannot be deleted"):
            ledger._connection.execute(
                "DELETE FROM evaluation_baselines WHERE binding_hash = ?",
                (base.binding_hash,),
            )

    def test_baseline_persists_across_ledger_close_and_reopen(self, tmp_path) -> None:
        db_file = tmp_path / "evaluation_with_baseline.db"

        # Session 1: Register baseline
        ledger1 = EvaluationLedger(db_file)
        rep = _make_replay_report()
        base = _make_baseline_from_report(rep)
        ledger1.register_baseline(base, replay_report=rep)
        ledger1.close()

        # Session 2: Reopen and verify
        ledger2 = EvaluationLedger(db_file)
        retrieved = ledger2.get_baseline(base.strategy_id, base.qualification_hash)
        assert retrieved is not None
        assert retrieved.binding_hash == base.binding_hash
        assert retrieved.baseline_replay_report_hash == rep.report_hash
        assert retrieved.canonical_dict() == base.canonical_dict()
        ledger2.close()
