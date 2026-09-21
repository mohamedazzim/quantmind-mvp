"""Temporal Paper Evaluation Service (PRD v4.0 Milestone 5).

This module implements the authoritative orchestration service that transforms
paper execution evidence into immutable, cryptographically-bound evaluation
snapshots and degradation events.

## Service Contract

    PaperEvaluationService.evaluate_window(
        *,
        strategy_id: str,
        qualification_hash: str,
        qualification_record: StrategyQualificationRecord,
        window_start_ts: str,
        window_end_ts: str,
        monitoring_config: MonitoringConfig,
        paper_ledger: PaperLedger,
        evaluation_ledger: EvaluationLedger,
        observed_replay_report: ReplayReport | None = None,
    ) -> EvaluationResult

## Temporal Model

All PaperLedger tables use UTC ISO 8601 timestamps as authoritative evidence:
    - paper_orders:      submit_timestamp
    - paper_fills:       fill_timestamp
    - paper_positions:   timestamp
    - paper_risk_events: timestamp

The universal inclusion rule is (INCLUSIVE on both ends):
    T_start <= authoritative_timestamp <= T_cutoff

No source may bypass this policy. Events at exactly T_start or T_cutoff
are INCLUDED. Events strictly outside are EXCLUDED.

## Authoritative Baseline

The service ALWAYS loads the baseline from EvaluationLedger — it is NEVER
accepted from the external caller. If none is registered the evaluation
fails closed with PaperEvaluationServiceError.

## Observed Forward Report

The observed_replay_report represents forward paper execution evidence for the
monitored window. It is semantically SEPARATE from the baseline report.

Verified: exact type, digest, strategy_id, qualification_hash, split_zone,
and no session ending after T_cutoff (future-coverage guard).

If observed_replay_report is None the snapshot is bound to the baseline
report hash (backwards-compatible mode for windows without an explicit report).

## Opening-State Policy

For positions that existed before T_start and remain open during the window:
the service INCLUDES the earliest position snapshot within [T_start, T_cutoff].
It does NOT reconstruct pre-window state and does NOT fabricate closed trades.
P&L and drawdown are computed strictly from snapshots within the window.

## Completed Session Policy

A ReplaySessionSummary is COMPLETE for a window [T_start, T_cutoff] iff:
    session.start_ts >= T_start  AND  session.end_ts <= T_cutoff

Incomplete sessions (end_ts > T_cutoff) are rejected by the observed-report
future-coverage guard. Sessions starting before T_start are excluded from
the completed-session count used for session-based metrics.

## Idempotency

get_or_insert_snapshot: identical evidence + config + window → same
snapshot_hash → returns existing record.

record_degradation_event: same evidence → same event_hash → returns existing.

## Failure-Closed Behaviour

Every provenance check raises PaperEvaluationServiceError on mismatch.
No partial results are persisted if provenance fails.

## M6 Lifecycle Boundary

M5 MUST NOT call StrategyRegistry.transition_state().
M5 MUST NOT write to TrialLedger, EICT, or DSR.
Lifecycle governance and research feedback are Milestone 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from quantmind.paper.evaluation.detector import DegradationDetector
from quantmind.paper.evaluation.ledger import EvaluationLedger
from quantmind.paper.evaluation.metrics import (
    compute_cost_to_turnover_bps_from_fills,
    compute_peak_gross_exposure,
    compute_realized_slippage_bps_from_fills,
    compute_rejection_rate,
    compute_rolling_drawdown_bps,
    compute_rolling_sharpe,
    compute_session_returns,
    count_rejected_orders,
    count_risk_events,
)
from quantmind.paper.evaluation.models import (
    DegradationEvent,
    MonitoringConfig,
    MonitoringSnapshot,
    PaperEvaluationBaseline,
)
from quantmind.paper.ledger import PaperLedger
from quantmind.paper.models import (
    ReplayReport,
    ReplaySessionSummary,
)
from quantmind.research_integrity.qualification import (
    StrategyQualificationRecord,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Public Errors
# ---------------------------------------------------------------------------


class PaperEvaluationServiceError(RuntimeError):
    """Raised when evaluation provenance or integrity checks fail."""


class InsufficientEvaluationEvidenceError(PaperEvaluationServiceError):
    """Raised when a window contains insufficient evidence for evaluation."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable result of a single temporal evaluation window.

    Contains the persisted MonitoringSnapshot and any detected DegradationEvents.
    Both fields are authoritative evidence — already persisted in EvaluationLedger.
    """

    snapshot: MonitoringSnapshot
    degradation_events: tuple[DegradationEvent, ...]

    @property
    def has_degradation(self) -> bool:
        """Return True if at least one degradation event was detected."""
        return len(self.degradation_events) > 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PaperEvaluationService:
    """Temporal orchestration service for paper evaluation windows.

    Responsibilities:
    1. Validate all caller-supplied inputs (fail-closed provenance).
    2. Load the authoritative baseline from EvaluationLedger — never from caller.
    3. Load and verify the baseline ReplayReport from PaperLedger.
    4. Validate the observed ReplayReport (if supplied).
    5. Slice PaperLedger evidence to [T_start, T_cutoff] inclusive.
    6. Run data integrity checks on the temporal slice.
    7. Filter completed sessions from the observed report.
    8. Compute M3 pure metrics (no formulas duplicated here).
    9. Construct an immutable MonitoringSnapshot.
    10. Persist the snapshot via EvaluationLedger.get_or_insert_snapshot (idempotent).
    11. Run DegradationDetector (zero DB access in detector).
    12. Persist DegradationEvents via EvaluationLedger.record_degradation_event (idempotent).
    13. Return EvaluationResult.

    Invariants:
    - NEVER mutates PaperLedger, StrategyRegistry, TrialLedger, EICT, or DSR.
    - NEVER uses wall-clock time for semantic values.
    - NEVER calls uuid.uuid4() or random.
    - Repeated identical evaluations produce identical outputs.
    """

    def evaluate_window(
        self,
        *,
        strategy_id: str,
        qualification_hash: str,
        qualification_record: StrategyQualificationRecord,
        window_start_ts: str,
        window_end_ts: str,
        monitoring_config: MonitoringConfig,
        paper_ledger: PaperLedger,
        evaluation_ledger: EvaluationLedger,
        observed_replay_report: ReplayReport | None = None,
    ) -> EvaluationResult:
        """Evaluate a single temporal window of paper trading evidence.

        Args:
            strategy_id: Authoritative strategy identifier.
            qualification_hash: Qualification record hash binding the evaluation.
            qualification_record: Authoritative StrategyQualificationRecord.
            window_start_ts: ISO 8601 UTC inclusive window start (T_start).
            window_end_ts: ISO 8601 UTC inclusive window end (T_cutoff).
            monitoring_config: Versioned monitoring protocol configuration.
            paper_ledger: READ-ONLY access to paper trading evidence.
            evaluation_ledger: Append-only evaluation evidence ledger.
            observed_replay_report: Optional ReplayReport for the observed
                forward paper evaluation window. Semantically separate from
                the authoritative baseline report.

        Returns:
            EvaluationResult with persisted snapshot and degradation events.

        Raises:
            PaperEvaluationServiceError: On any provenance or integrity failure.
        """
        # ------------------------------------------------------------------
        # Phase 1: Fail-Closed Input Provenance Validation
        # ------------------------------------------------------------------
        self._validate_inputs(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            qualification_record=qualification_record,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            monitoring_config=monitoring_config,
            paper_ledger=paper_ledger,
            evaluation_ledger=evaluation_ledger,
            observed_replay_report=observed_replay_report,
        )

        # ------------------------------------------------------------------
        # Phase 2: Load Authoritative Baseline from EvaluationLedger
        # Callers CANNOT supply an arbitrary baseline.
        # ------------------------------------------------------------------
        baseline = self._load_baseline(
            evaluation_ledger=evaluation_ledger,
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            qualification_record=qualification_record,
        )

        # ------------------------------------------------------------------
        # Phase 3: Load and Verify Baseline ReplayReport from PaperLedger
        # ------------------------------------------------------------------
        baseline_report = self._load_baseline_report(
            paper_ledger=paper_ledger,
            baseline=baseline,
        )

        # ------------------------------------------------------------------
        # Phase 4: Validate Observed ReplayReport (future-coverage guard)
        # ------------------------------------------------------------------
        observed_report = self._validate_observed_report(
            observed_replay_report=observed_replay_report,
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            baseline=baseline,
        )

        # ------------------------------------------------------------------
        # Phase 5: Slice PaperLedger Evidence to [T_start, T_cutoff]
        # Each table uses its own authoritative timestamp column.
        # ------------------------------------------------------------------
        temporal_slice = self._slice_temporal_evidence(
            paper_ledger=paper_ledger,
            strategy_id=strategy_id,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
        )

        # ------------------------------------------------------------------
        # Phase 6: Data Integrity Checks on the Temporal Slice
        # ------------------------------------------------------------------
        self._check_slice_integrity(
            strategy_id=strategy_id,
            temporal_slice=temporal_slice,
        )

        # ------------------------------------------------------------------
        # Phase 7: Filter Completed Sessions from Observed Report
        # Only sessions fully contained in [T_start, T_cutoff] count.
        # ------------------------------------------------------------------
        completed_sessions = self._filter_completed_sessions(
            observed_report=observed_report,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
        )

        # ------------------------------------------------------------------
        # Phase 8: Compute M3 Pure Metrics — No formulas duplicated here.
        # ------------------------------------------------------------------
        metrics, metrics_json = self._compute_metrics(
            temporal_slice=temporal_slice,
            completed_sessions=completed_sessions,
            baseline_report=baseline_report,
            monitoring_config=monitoring_config,
        )

        # ------------------------------------------------------------------
        # Phase 9: Determine Observed Report Binding for Snapshot
        # The snapshot binds to the observed report hash (or baseline hash
        # when no separate observed report is supplied).
        # ------------------------------------------------------------------
        if observed_report is not None:
            obs_report_hash = observed_report.report_hash
            obs_dataset_version = observed_report.dataset_version
            obs_dataset_sha256 = observed_report.dataset_sha256
            obs_split_zone = observed_report.split_zone
        else:
            obs_report_hash = baseline.baseline_replay_report_hash
            obs_dataset_version = baseline.baseline_dataset_version
            obs_dataset_sha256 = baseline.baseline_dataset_sha256
            obs_split_zone = baseline.baseline_split_zone

        # ------------------------------------------------------------------
        # Phase 10: Construct MonitoringSnapshot
        # created_at = window_end_ts → deterministic across re-evaluations.
        # created_at is excluded from canonical_dict/snapshot_hash by design.
        # ------------------------------------------------------------------
        snapshot = MonitoringSnapshot.create(
            strategy_id=strategy_id,
            qualification_hash=qualification_hash,
            replay_report_hash=obs_report_hash,
            dataset_version=obs_dataset_version,
            dataset_sha256=obs_dataset_sha256,
            split_zone=obs_split_zone,
            monitoring_protocol_version=monitoring_config.protocol_version,
            monitoring_config_hash=monitoring_config.compute_config_hash(),
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            total_trades=metrics["total_trades"],
            net_pnl=metrics["net_pnl"],
            max_drawdown_bps=metrics["max_drawdown_bps"],
            realized_sharpe=metrics["realized_sharpe"],
            realized_slippage_bps=metrics["realized_slippage_bps"],
            cost_to_turnover_bps=metrics["cost_to_turnover_bps"],
            risk_event_count=metrics["risk_event_count"],
            metrics_json=metrics_json,
            created_at=window_end_ts,
        )

        # ------------------------------------------------------------------
        # Phase 11: Persist Snapshot (idempotent)
        # ------------------------------------------------------------------
        snapshot = evaluation_ledger.get_or_insert_snapshot(snapshot)

        # ------------------------------------------------------------------
        # Phase 12: Run DegradationDetector — zero DB access inside detector
        # ------------------------------------------------------------------
        detector = DegradationDetector(config=monitoring_config, baseline=baseline)
        degradation_events = detector.evaluate(
            snapshot=snapshot,
            qualification_record=qualification_record,
            baseline_replay_report=baseline_report,
            baseline=baseline,
            observed_replay_report=observed_report,
            submitted_orders=metrics["submitted_orders"],
            rejected_orders=metrics["rejected_orders"],
        )

        # ------------------------------------------------------------------
        # Phase 13: Persist DegradationEvents (idempotent)
        # Events are already sorted deterministically by rule_name from detector.
        # ------------------------------------------------------------------
        persisted_events: list[DegradationEvent] = []
        for event in degradation_events:
            persisted = evaluation_ledger.record_degradation_event(event)
            persisted_events.append(persisted)

        return EvaluationResult(
            snapshot=snapshot,
            degradation_events=tuple(persisted_events),
        )

    # -----------------------------------------------------------------------
    # Phase 1: Input Provenance Validation
    # -----------------------------------------------------------------------

    def _validate_inputs(
        self,
        *,
        strategy_id: str,
        qualification_hash: str,
        qualification_record: StrategyQualificationRecord,
        window_start_ts: str,
        window_end_ts: str,
        monitoring_config: MonitoringConfig,
        paper_ledger: PaperLedger,
        evaluation_ledger: EvaluationLedger,
        observed_replay_report: ReplayReport | None,
    ) -> None:
        """Fail-closed validation of all caller-supplied inputs."""
        if not strategy_id:
            raise PaperEvaluationServiceError("strategy_id cannot be empty")
        if not qualification_hash:
            raise PaperEvaluationServiceError("qualification_hash cannot be empty")
        if not window_start_ts:
            raise PaperEvaluationServiceError("window_start_ts cannot be empty")
        if not window_end_ts:
            raise PaperEvaluationServiceError("window_end_ts cannot be empty")
        if window_start_ts > window_end_ts:
            raise PaperEvaluationServiceError(
                f"window_start_ts ({window_start_ts}) must be <= window_end_ts ({window_end_ts})"
            )

        # Exact type checks (rejects duck-typed or subclassed models/ledgers)
        if type(qualification_record) is not StrategyQualificationRecord:
            raise PaperEvaluationServiceError(
                f"qualification_record must be StrategyQualificationRecord, got {type(qualification_record)}"
            )
        if type(monitoring_config) is not MonitoringConfig:
            raise PaperEvaluationServiceError(
                f"monitoring_config must be MonitoringConfig, got {type(monitoring_config)}"
            )
        if type(paper_ledger) is not PaperLedger:
            raise PaperEvaluationServiceError(
                f"paper_ledger must be PaperLedger, got {type(paper_ledger)}"
            )
        if type(evaluation_ledger) is not EvaluationLedger:
            raise PaperEvaluationServiceError(
                f"evaluation_ledger must be EvaluationLedger, got {type(evaluation_ledger)}"
            )
        if observed_replay_report is not None and type(observed_replay_report) is not ReplayReport:
            raise PaperEvaluationServiceError(
                f"observed_replay_report must be ReplayReport, got {type(observed_replay_report)}"
            )

        # Qualification record integrity
        if not qualification_record.verify_digest():
            raise PaperEvaluationServiceError(
                "Qualification record digest verification failed (tampered qualification record)"
            )
        if qualification_record.strategy_id != strategy_id:
            raise PaperEvaluationServiceError(
                f"qualification_record.strategy_id mismatch: "
                f"'{qualification_record.strategy_id}' != '{strategy_id}'"
            )
        if qualification_record.record_hash != qualification_hash:
            raise PaperEvaluationServiceError(
                f"qualification_record.record_hash mismatch: "
                f"'{qualification_record.record_hash}' != '{qualification_hash}'"
            )
        if qualification_record.final_status != ValidationStatus.PAPER_ELIGIBLE:
            raise PaperEvaluationServiceError(
                f"qualification_record.final_status must be PAPER_ELIGIBLE, "
                f"got '{qualification_record.final_status.value}'"
            )

        # Observed report preliminary checks (before loading baseline)
        if observed_replay_report is not None:
            obs = observed_replay_report
            if obs.report_hash != obs.compute_report_hash():
                raise PaperEvaluationServiceError(
                    "observed_replay_report digest verification failed (tampered replay report)"
                )
            if obs.strategy_id != strategy_id:
                raise PaperEvaluationServiceError(
                    f"observed_replay_report.strategy_id mismatch: "
                    f"'{obs.strategy_id}' != '{strategy_id}'"
                )
            if obs.qualification_hash != qualification_hash:
                raise PaperEvaluationServiceError(
                    f"observed_replay_report.qualification_hash mismatch: "
                    f"'{obs.qualification_hash}' != '{qualification_hash}'"
                )
            if obs.split_zone != "FORWARD_PAPER":
                raise PaperEvaluationServiceError(
                    f"observed_replay_report.split_zone must be 'FORWARD_PAPER', "
                    f"got '{obs.split_zone}'"
                )
            if obs.dataset_version != qualification_record.dataset_version:
                raise PaperEvaluationServiceError(
                    f"observed_replay_report.dataset_version mismatch: "
                    f"'{obs.dataset_version}' != '{qualification_record.dataset_version}'"
                )
            if obs.dataset_sha256 != qualification_record.dataset_sha256:
                raise PaperEvaluationServiceError(
                    "observed_replay_report.dataset_sha256 mismatch with qualification record"
                )

    # -----------------------------------------------------------------------
    # Phase 2: Load Authoritative Baseline
    # -----------------------------------------------------------------------

    def _load_baseline(
        self,
        *,
        evaluation_ledger: EvaluationLedger,
        strategy_id: str,
        qualification_hash: str,
        qualification_record: StrategyQualificationRecord,
    ) -> PaperEvaluationBaseline:
        """Load and verify the authoritative baseline from EvaluationLedger.

        Fails closed if no baseline is registered or digest verification fails.
        Callers cannot substitute an arbitrary baseline.
        """
        baseline = evaluation_ledger.get_baseline(strategy_id, qualification_hash)
        if baseline is None:
            raise PaperEvaluationServiceError(
                f"No authoritative baseline registered for strategy '{strategy_id}' "
                f"with qualification '{qualification_hash}'. "
                "Register a baseline via EvaluationLedger.register_baseline() before evaluation."
            )
        if type(baseline) is not PaperEvaluationBaseline:
            raise PaperEvaluationServiceError(
                f"Authoritative baseline must be PaperEvaluationBaseline, got {type(baseline)}"
            )
        if not baseline.verify_digest():
            raise PaperEvaluationServiceError(
                "Authoritative baseline binding digest verification failed (tampered baseline)"
            )
        if baseline.strategy_id != strategy_id:
            raise PaperEvaluationServiceError(
                f"Baseline strategy_id mismatch: '{baseline.strategy_id}' != '{strategy_id}'"
            )
        if baseline.qualification_hash != qualification_hash:
            raise PaperEvaluationServiceError(
                f"Baseline qualification_hash mismatch: "
                f"'{baseline.qualification_hash}' != '{qualification_hash}'"
            )
        if baseline.baseline_dataset_version != qualification_record.dataset_version:
            raise PaperEvaluationServiceError(
                f"Baseline dataset_version mismatch: "
                f"'{baseline.baseline_dataset_version}' != '{qualification_record.dataset_version}'"
            )
        if baseline.baseline_dataset_sha256 != qualification_record.dataset_sha256:
            raise PaperEvaluationServiceError(
                "Baseline dataset_sha256 mismatch with qualification record"
            )
        return baseline

    # -----------------------------------------------------------------------
    # Phase 3: Load Baseline ReplayReport
    # -----------------------------------------------------------------------

    def _load_baseline_report(
        self,
        *,
        paper_ledger: PaperLedger,
        baseline: PaperEvaluationBaseline,
    ) -> ReplayReport:
        """Load and independently verify the baseline ReplayReport from PaperLedger.

        Fails closed if the report is absent, digest-invalid, or mismatched.
        """
        report = paper_ledger.get_report(baseline.baseline_replay_report_hash)
        if report is None:
            raise PaperEvaluationServiceError(
                f"Baseline ReplayReport '{baseline.baseline_replay_report_hash}' "
                "not found in PaperLedger"
            )
        if type(report) is not ReplayReport:
            raise PaperEvaluationServiceError(
                f"Baseline ReplayReport must be ReplayReport, got {type(report)}"
            )
        if report.report_hash != report.compute_report_hash():
            raise PaperEvaluationServiceError(
                "Baseline ReplayReport digest verification failed (tampered replay report)"
            )
        if report.strategy_id != baseline.strategy_id:
            raise PaperEvaluationServiceError(
                f"Baseline report strategy_id mismatch: "
                f"'{report.strategy_id}' != '{baseline.strategy_id}'"
            )
        if report.qualification_hash != baseline.qualification_hash:
            raise PaperEvaluationServiceError(
                f"Baseline report qualification_hash mismatch: "
                f"'{report.qualification_hash}' != '{baseline.qualification_hash}'"
            )
        if report.dataset_version != baseline.baseline_dataset_version:
            raise PaperEvaluationServiceError(
                f"Baseline report dataset_version mismatch: "
                f"'{report.dataset_version}' != '{baseline.baseline_dataset_version}'"
            )
        if report.dataset_sha256 != baseline.baseline_dataset_sha256:
            raise PaperEvaluationServiceError(
                "Baseline report dataset_sha256 mismatch"
            )
        if report.execution_policy != baseline.baseline_execution_policy:
            raise PaperEvaluationServiceError(
                f"Baseline report execution_policy mismatch: "
                f"'{report.execution_policy}' != '{baseline.baseline_execution_policy}'"
            )
        if report.cost_schedule_hash != baseline.baseline_cost_schedule_hash:
            raise PaperEvaluationServiceError(
                "Baseline report cost_schedule_hash mismatch"
            )
        if report.risk_config_hash != baseline.baseline_risk_config_hash:
            raise PaperEvaluationServiceError(
                "Baseline report risk_config_hash mismatch"
            )
        return report

    # -----------------------------------------------------------------------
    # Phase 4: Validate Observed ReplayReport
    # -----------------------------------------------------------------------

    def _validate_observed_report(
        self,
        *,
        observed_replay_report: ReplayReport | None,
        strategy_id: str,
        qualification_hash: str,
        window_start_ts: str,
        window_end_ts: str,
        baseline: PaperEvaluationBaseline,
    ) -> ReplayReport | None:
        """Apply future-coverage guard: sessions must not extend beyond T_cutoff.

        Also cross-verifies execution configuration consistency with baseline,
        and ensures the report does not cover an older, disjoint window.
        """
        if observed_replay_report is None:
            return None

        # Future-coverage guard: no session may end after T_cutoff
        for sess in observed_replay_report.session_breakdown:
            if sess.end_ts > window_end_ts:
                raise PaperEvaluationServiceError(
                    f"observed_replay_report contains session '{sess.session_id}' "
                    f"ending at '{sess.end_ts}' which is after T_cutoff '{window_end_ts}'. "
                    "This report covers future observations beyond the evaluation window."
                )

        # Disjoint older-window guard: if report has sessions, not all may end before window_start_ts
        if observed_replay_report.session_breakdown and all(
            sess.end_ts < window_start_ts for sess in observed_replay_report.session_breakdown
        ):
            raise PaperEvaluationServiceError(
                f"observed_replay_report contains no sessions within or overlapping "
                f"evaluation window [{window_start_ts}, {window_end_ts}] (older report / wrong window)"
            )

        # Execution configuration consistency with authoritative baseline
        if observed_replay_report.execution_policy != baseline.baseline_execution_policy:
            raise PaperEvaluationServiceError(
                f"observed_replay_report.execution_policy mismatch: "
                f"'{observed_replay_report.execution_policy}' != '{baseline.baseline_execution_policy}'"
            )
        if observed_replay_report.cost_schedule_hash != baseline.baseline_cost_schedule_hash:
            raise PaperEvaluationServiceError(
                "observed_replay_report.cost_schedule_hash mismatch with baseline"
            )
        if observed_replay_report.risk_config_hash != baseline.baseline_risk_config_hash:
            raise PaperEvaluationServiceError(
                "observed_replay_report.risk_config_hash mismatch with baseline"
            )

        return observed_replay_report

    # -----------------------------------------------------------------------
    # Phase 5: Slice Temporal Evidence
    # -----------------------------------------------------------------------

    @dataclass(frozen=True)
    class _TemporalSlice:
        """Raw temporal slice of PaperLedger rows for a single evaluation window."""
        orders: tuple  # tuple of sqlite3.Row
        fills: tuple   # tuple of sqlite3.Row
        positions: tuple  # tuple of sqlite3.Row
        risk_events: tuple  # tuple of sqlite3.Row

    def _slice_temporal_evidence(
        self,
        *,
        paper_ledger: PaperLedger,
        strategy_id: str,
        window_start_ts: str,
        window_end_ts: str,
    ) -> "_TemporalSlice":
        """Extract PaperLedger rows within [T_start, T_cutoff] inclusive.

        Timestamp authority per table (as per ledger schema):
            paper_orders:      submit_timestamp
            paper_fills:       fill_timestamp
            paper_positions:   timestamp
            paper_risk_events: timestamp

        All results are ordered by authoritative timestamp ASC (already guaranteed
        by PaperLedger query ordering).

        PaperLedger is accessed READ-ONLY — no writes of any kind.
        """
        # Orders — filter by submit_timestamp
        all_orders = paper_ledger.get_orders(strategy_id)
        orders = tuple(
            r for r in all_orders
            if window_start_ts <= r["submit_timestamp"] <= window_end_ts
        )

        # Fills — filter by fill_timestamp
        all_fills = paper_ledger.get_fills(strategy_id)
        fills = tuple(
            r for r in all_fills
            if window_start_ts <= r["fill_timestamp"] <= window_end_ts
        )

        # Positions — use get_positions with from_ts/to_ts for efficient filtering
        positions = tuple(
            paper_ledger.get_positions(
                strategy_id,
                from_ts=window_start_ts,
                to_ts=window_end_ts,
            )
        )

        # Risk events — filter by timestamp
        all_risk = paper_ledger.get_risk_events(strategy_id)
        risk_events = tuple(
            r for r in all_risk
            if window_start_ts <= r["timestamp"] <= window_end_ts
        )

        return PaperEvaluationService._TemporalSlice(
            orders=orders,
            fills=fills,
            positions=positions,
            risk_events=risk_events,
        )

    # -----------------------------------------------------------------------
    # Phase 6: Data Integrity Checks
    # -----------------------------------------------------------------------

    def _check_slice_integrity(
        self,
        *,
        strategy_id: str,
        temporal_slice: "_TemporalSlice",
    ) -> None:
        """Validate the temporal slice for structural and provenance integrity.

        Checks performed:
        - All rows belong to the expected strategy_id
        - No duplicate order_ids, fill_ids, or risk_event_ids
        - Fills have strictly positive prices and quantities
        - Risk events have non-empty rule_name
        """
        # Strategy ID invariant
        for r in temporal_slice.orders:
            if r["strategy_id"] != strategy_id:
                raise PaperEvaluationServiceError(
                    f"Order '{r['order_id']}' strategy_id '{r['strategy_id']}' "
                    f"!= expected '{strategy_id}'"
                )
        for r in temporal_slice.fills:
            if r["strategy_id"] != strategy_id:
                raise PaperEvaluationServiceError(
                    f"Fill '{r['fill_id']}' strategy_id '{r['strategy_id']}' "
                    f"!= expected '{strategy_id}'"
                )
        for r in temporal_slice.risk_events:
            if r["strategy_id"] != strategy_id:
                raise PaperEvaluationServiceError(
                    f"RiskEvent '{r['event_id']}' strategy_id '{r['strategy_id']}' "
                    f"!= expected '{strategy_id}'"
                )

        # Duplicate identity checks
        order_ids = [r["order_id"] for r in temporal_slice.orders]
        if len(order_ids) != len(set(order_ids)):
            raise PaperEvaluationServiceError(
                "Duplicate order_ids detected in temporal slice — ledger integrity violation"
            )
        fill_ids = [r["fill_id"] for r in temporal_slice.fills]
        if len(fill_ids) != len(set(fill_ids)):
            raise PaperEvaluationServiceError(
                "Duplicate fill_ids detected in temporal slice — ledger integrity violation"
            )
        risk_ids = [r["event_id"] for r in temporal_slice.risk_events]
        if len(risk_ids) != len(set(risk_ids)):
            raise PaperEvaluationServiceError(
                "Duplicate risk event_ids detected in temporal slice — ledger integrity violation"
            )

        # Fill validity
        for r in temporal_slice.fills:
            if float(r["fill_price"]) <= 0.0:
                raise PaperEvaluationServiceError(
                    f"Fill '{r['fill_id']}' has non-positive fill_price {r['fill_price']}"
                )
            if int(r["quantity"]) <= 0:
                raise PaperEvaluationServiceError(
                    f"Fill '{r['fill_id']}' has non-positive quantity {r['quantity']}"
                )

        # Risk event validity
        for r in temporal_slice.risk_events:
            if not r["rule_name"]:
                raise PaperEvaluationServiceError(
                    f"RiskEvent '{r['event_id']}' has empty rule_name"
                )

    # -----------------------------------------------------------------------
    # Phase 7: Filter Completed Sessions
    # -----------------------------------------------------------------------

    def _filter_completed_sessions(
        self,
        *,
        observed_report: ReplayReport | None,
        window_start_ts: str,
        window_end_ts: str,
    ) -> tuple[ReplaySessionSummary, ...]:
        """Return only sessions fully contained within [T_start, T_cutoff].

        A session is COMPLETE for this window iff:
            session.start_ts >= window_start_ts
            AND session.end_ts <= window_end_ts

        Sessions starting before T_start are excluded (their evidence predates
        the evaluation window boundary). Sessions ending after T_cutoff are
        already blocked by the future-coverage guard in Phase 4.

        Returns sessions sorted chronologically by start_ts.
        """
        if observed_report is None:
            return ()
        completed = [
            s for s in observed_report.session_breakdown
            if s.start_ts >= window_start_ts and s.end_ts <= window_end_ts
        ]
        completed.sort(key=lambda s: s.start_ts)
        return tuple(completed)

    # -----------------------------------------------------------------------
    # Phase 8: Compute M3 Metrics
    # -----------------------------------------------------------------------

    def _compute_metrics(
        self,
        *,
        temporal_slice: "_TemporalSlice",
        completed_sessions: tuple[ReplaySessionSummary, ...],
        baseline_report: ReplayReport,
        monitoring_config: MonitoringConfig,
    ) -> tuple[dict[str, Any], str]:
        """Compute all M3 metrics from the validated temporal slice.

        Metric pipeline:
            PaperLedger rows (temporal slice)
                → validated typed observations
                → M3 pure metric functions (no duplication of formulas)
                → metrics dict + metrics_json string

        Returns:
            (metrics_dict, metrics_json_str)

        Metrics dict contains top-level MonitoringSnapshot fields plus
        auxiliary values (submitted_orders, rejected_orders, etc.) needed
        by the detector or for transparency.
        """
        fills = [dict(r) for r in temporal_slice.fills]
        orders = [dict(r) for r in temporal_slice.orders]
        risk_events = [dict(r) for r in temporal_slice.risk_events]
        positions = temporal_slice.positions  # kept as sqlite3.Row — only accessed by key

        lot_size: int = int(baseline_report.lot_size)
        initial_capital: float = float(baseline_report.initial_capital)

        # --- Trade count = number of fills in window ---
        total_trades = len(fills)

        # --- Net P&L from position snapshots ---
        # Take the latest realized_pnl per symbol within the window, then sum.
        # This is the authoritative cumulative realized value from the execution engine.
        net_pnl = _compute_net_pnl_from_positions(positions)

        # --- Equity curve for max drawdown ---
        # Build equity_t = initial_capital + sum(realized_pnl at time t across all symbols).
        # Group position rows by timestamp, sum realized_pnl, then build the equity curve.
        equity_curve = _build_equity_curve(positions, initial_capital)
        max_drawdown_bps = compute_rolling_drawdown_bps(equity_curve)

        # --- Execution quality metrics (M3 — no formula duplication) ---
        realized_slippage_bps = compute_realized_slippage_bps_from_fills(fills, lot_size)
        cost_to_turnover_bps = compute_cost_to_turnover_bps_from_fills(fills, lot_size)

        # --- Risk metrics ---
        risk_event_count = count_risk_events(risk_events)
        rejected_count = count_rejected_orders(orders)
        submitted_count = len(orders)
        rejection_rate_val = compute_rejection_rate(rejected_count, submitted_count)

        # --- Session-based metrics (only from completed sessions) ---
        session_net_pnls = [s.net_pnl for s in completed_sessions]
        n_completed = len(completed_sessions)

        realized_sharpe: float | None = None
        if session_net_pnls and initial_capital > 0.0:
            try:
                sess_returns = compute_session_returns(session_net_pnls, initial_capital)
                realized_sharpe = compute_rolling_sharpe(
                    sess_returns,
                    window_size=monitoring_config.window_size_sessions,
                    min_sessions=10,
                )
            except ValueError:
                realized_sharpe = None

        # --- Consecutive inactive sessions (for dropout detection) ---
        consecutive_inactive = _count_trailing_inactive_sessions(completed_sessions)

        # --- Peak gross exposure ---
        peak_exposure = compute_peak_gross_exposure(positions, lot_size)

        # --- Auxiliary metrics embedded in metrics_json ---
        # These pass through to the detector via snapshot.metrics_json.
        aux: dict[str, Any] = {
            "completed_sessions": n_completed,
            "consecutive_inactive_sessions": consecutive_inactive,
            "peak_gross_exposure": peak_exposure,
            "rejected_orders": rejected_count,
            "rejection_rate": (
                round(float(rejection_rate_val), 4)
                if rejection_rate_val is not None
                else None
            ),
            "submitted_orders": submitted_count,
        }
        metrics_json = json.dumps(aux, sort_keys=True, separators=(",", ":"))

        metrics: dict[str, Any] = {
            "total_trades": total_trades,
            "net_pnl": net_pnl,
            "max_drawdown_bps": max_drawdown_bps,
            "realized_sharpe": realized_sharpe,
            "realized_slippage_bps": realized_slippage_bps,
            "cost_to_turnover_bps": cost_to_turnover_bps,
            "risk_event_count": risk_event_count,
            "submitted_orders": submitted_count,
            "rejected_orders": rejected_count,
        }

        return metrics, metrics_json


# ---------------------------------------------------------------------------
# Module-level Pure Helpers (no side effects)
# ---------------------------------------------------------------------------


def _compute_net_pnl_from_positions(positions: tuple) -> float:
    """Compute net P&L as sum of latest realized_pnl per symbol in the window.

    Takes the most recent (highest timestamp) realized_pnl snapshot for each
    symbol, then sums across all symbols. Returns 0.0 for empty windows.

    This preserves the authoritative cumulative realized value from the
    paper execution engine without double-counting intermediate snapshots.
    """
    if not positions:
        return 0.0
    latest_pnl: dict[str, float] = {}
    latest_ts: dict[str, str] = {}
    for row in positions:
        sym = row["symbol"]
        ts = row["timestamp"]
        if sym not in latest_ts or ts > latest_ts[sym]:
            latest_ts[sym] = ts
            latest_pnl[sym] = float(row["realized_pnl"])
    return round(sum(latest_pnl.values()), 4)


def _build_equity_curve(positions: tuple, initial_capital: float) -> list[float]:
    """Build chronological equity curve from position snapshots.

    equity_t = initial_capital + sum(realized_pnl across all symbols at timestamp t)

    Timestamps are grouped; at each unique timestamp the sum of realized_pnl
    across all symbols is added to initial_capital to give the equity level.
    The resulting list is ordered chronologically (ascending timestamp).

    Returns empty list for empty windows (drawdown = 0.0).
    """
    if not positions:
        return []
    ts_to_pnl: dict[str, float] = {}
    for row in positions:
        ts = row["timestamp"]
        ts_to_pnl[ts] = ts_to_pnl.get(ts, 0.0) + float(row["realized_pnl"])
    curve = [
        initial_capital + pnl
        for ts, pnl in sorted(ts_to_pnl.items())
    ]
    return curve


def _count_trailing_inactive_sessions(
    completed_sessions: tuple[ReplaySessionSummary, ...],
) -> int:
    """Count trailing consecutive completed sessions with zero trades.

    Traverses the completed session list in reverse (most recent first).
    Stops at the first session with trades > 0. Used by the detector to
    identify RULE_TRADE_DROPOUT.

    Returns 0 if the most recent completed session has trades > 0, or if
    there are no completed sessions.
    """
    count = 0
    for sess in reversed(completed_sessions):
        if sess.trades == 0:
            count += 1
        else:
            break
    return count
