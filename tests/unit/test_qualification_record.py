"""Unit tests for StrategyQualificationRecord, QualificationLedger, and State Machine (PRD v3.8)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import sqlite3
import pytest

from quantmind.research_integrity.holdout import HoldoutState
from quantmind.research_integrity.qualification import (
    ALLOWED_VALIDATION_TRANSITIONS,
    InvalidStateTransitionError,
    PaperReplayEligibility,
    QualificationLedger,
    QualificationLedgerError,
    RobustnessReport,
    RobustnessStatus,
    StrategyQualificationRecord,
    ValidationReport,
    ValidationStatus,
    check_paper_replay_eligibility,
    validate_transition,
)


def _make_sample_record(
    qualification_id: str = "QUAL-TEST-001",
    strategy_id: str = "STRAT-ALPHA-1",
    final_status: ValidationStatus = ValidationStatus.PAPER_ELIGIBLE,
    holdout_state: str = "PASSED",
    observed_sharpe: float = 1.85,
    dsr: float = 0.965,
    trade_count: int = 120,
) -> StrategyQualificationRecord:
    return StrategyQualificationRecord.create(
        qualification_id=qualification_id,
        strategy_id=strategy_id,
        strategy_spec_hash="a" * 64,
        dataset_version="DS-NIFTY-2024",
        dataset_sha256="b" * 64,
        split_manifest_version="v1.0",
        research_protocol_version="RP-2",
        population_hash="c" * 64,
        effective_trial_count=14.0,
        observed_sharpe=observed_sharpe,
        dsr=dsr,
        trade_count=trade_count,
        holdout_state=holdout_state,
        robustness_status=RobustnessStatus.PASSED,
        final_status=final_status,
        created_at="2026-09-20T12:00:00+00:00",
        reasons=["All validation checks passed"],
    )


class TestStrategyQualificationRecord:
    def test_record_is_frozen_and_immutable(self) -> None:
        rec = _make_sample_record()
        with pytest.raises(FrozenInstanceError):
            rec.observed_sharpe = 9.99  # type: ignore[misc]

    def test_canonical_json_and_digest_verification(self) -> None:
        rec = _make_sample_record()
        assert rec.verify_digest() is True
        assert len(rec.record_hash) == 64
        assert rec.record_hash == rec.compute_record_hash()

    def test_tamper_detection(self) -> None:
        rec = _make_sample_record()
        d = rec.to_dict()
        # Tamper with Sharpe ratio
        d["observed_sharpe"] = 3.50
        tampered_rec = StrategyQualificationRecord.from_dict(d)
        assert tampered_rec.verify_digest() is False

    def test_serialization_round_trip(self) -> None:
        rec = _make_sample_record()
        d = rec.to_dict()
        restored = StrategyQualificationRecord.from_dict(d)
        assert restored == rec
        assert restored.verify_digest() is True


class TestQualificationLedger:
    def test_record_and_retrieve_qualification(self) -> None:
        ledger = QualificationLedger()
        rec = _make_sample_record()
        ledger.record_qualification(rec)

        retrieved = ledger.get(rec.qualification_id)
        assert retrieved is not None
        assert retrieved == rec
        assert retrieved.verify_digest() is True

    def test_idempotent_duplicate_record(self) -> None:
        ledger = QualificationLedger()
        rec = _make_sample_record()
        ledger.record_qualification(rec)
        # Re-recording identical record should succeed
        ledger.record_qualification(rec)
        assert ledger.get(rec.qualification_id) == rec

    def test_reject_tampered_record(self) -> None:
        ledger = QualificationLedger()
        rec = _make_sample_record()
        d = rec.to_dict()
        d["dsr"] = 0.999
        tampered = StrategyQualificationRecord.from_dict(d)

        with pytest.raises(QualificationLedgerError, match="failed cryptographic digest"):
            ledger.record_qualification(tampered)

    def test_reject_conflicting_qualification_id(self) -> None:
        ledger = QualificationLedger()
        rec1 = _make_sample_record(qualification_id="QUAL-SAME-ID", observed_sharpe=1.2)
        rec2 = _make_sample_record(qualification_id="QUAL-SAME-ID", observed_sharpe=2.4)
        ledger.record_qualification(rec1)

        with pytest.raises(QualificationLedgerError, match="Conflicting qualification record"):
            ledger.record_qualification(rec2)

    def test_sqlite_trigger_blocks_delete(self) -> None:
        ledger = QualificationLedger()
        rec = _make_sample_record()
        ledger.record_qualification(rec)

        with pytest.raises(sqlite3.IntegrityError, match="permanent and append-only"):
            ledger._connection.execute(
                "DELETE FROM strategy_qualifications WHERE qualification_id = ?",
                (rec.qualification_id,),
            )

    def test_sqlite_trigger_blocks_update(self) -> None:
        ledger = QualificationLedger()
        rec = _make_sample_record()
        ledger.record_qualification(rec)

        with pytest.raises(sqlite3.IntegrityError, match="permanent and immutable"):
            ledger._connection.execute(
                "UPDATE strategy_qualifications SET dsr = 1.0 WHERE qualification_id = ?",
                (rec.qualification_id,),
            )


    def test_find_qualifications(self) -> None:
        ledger = QualificationLedger()
        rec1 = _make_sample_record(qualification_id="Q1", strategy_id="STRAT-1", final_status=ValidationStatus.PAPER_ELIGIBLE)
        rec2 = _make_sample_record(qualification_id="Q2", strategy_id="STRAT-2", final_status=ValidationStatus.UNDERPOWERED)
        ledger.record_qualification(rec1)
        ledger.record_qualification(rec2)

        results = ledger.find_qualifications(final_status=ValidationStatus.PAPER_ELIGIBLE)
        assert len(results) == 1
        assert results[0].qualification_id == "Q1"

        strat2_results = ledger.find_qualifications(strategy_id="STRAT-2")
        assert len(strat2_results) == 1
        assert strat2_results[0].qualification_id == "Q2"


class TestValidationStateMachine:
    def test_legal_state_transitions(self) -> None:
        validate_transition(ValidationStatus.CANDIDATE, ValidationStatus.VALIDATION)
        validate_transition(ValidationStatus.VALIDATION, ValidationStatus.HOLDOUT_REQUIRED)
        validate_transition(ValidationStatus.HOLDOUT_REQUIRED, ValidationStatus.HOLDOUT_PASSED)
        validate_transition(ValidationStatus.HOLDOUT_PASSED, ValidationStatus.PAPER_ELIGIBLE)

    def test_illegal_state_transitions(self) -> None:
        # Cannot transition from PAPER_ELIGIBLE to REJECTED or VALIDATION
        with pytest.raises(InvalidStateTransitionError, match="Illegal validation status transition"):
            validate_transition(ValidationStatus.PAPER_ELIGIBLE, ValidationStatus.REJECTED)
        with pytest.raises(InvalidStateTransitionError, match="Illegal validation status transition"):
            validate_transition(ValidationStatus.PAPER_ELIGIBLE, ValidationStatus.VALIDATION)

        # Cannot jump from CANDIDATE to PAPER_ELIGIBLE
        with pytest.raises(InvalidStateTransitionError, match="Illegal validation status transition"):
            validate_transition(ValidationStatus.CANDIDATE, ValidationStatus.PAPER_ELIGIBLE)


        # Cannot jump from REJECTED to PAPER_ELIGIBLE
        with pytest.raises(InvalidStateTransitionError, match="Illegal validation status transition"):
            validate_transition(ValidationStatus.REJECTED, ValidationStatus.PAPER_ELIGIBLE)

        # Cannot jump from UNDERPOWERED to PAPER_ELIGIBLE
        with pytest.raises(InvalidStateTransitionError, match="Illegal validation status transition"):
            validate_transition(ValidationStatus.UNDERPOWERED, ValidationStatus.PAPER_ELIGIBLE)

        # REJECTED is terminal
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(ValidationStatus.REJECTED, ValidationStatus.VALIDATION)


class TestPaperReplayEligibility:
    def test_eligible_candidate(self) -> None:
        rec = _make_sample_record(final_status=ValidationStatus.PAPER_ELIGIBLE, holdout_state="PASSED")
        eligibility = check_paper_replay_eligibility(rec)
        assert eligibility.is_eligible is True
        assert eligibility.qualification_id == rec.qualification_id
        assert eligibility.qualification_hash == rec.record_hash

    def test_ineligible_when_not_paper_eligible(self) -> None:
        rec = _make_sample_record(final_status=ValidationStatus.HOLDOUT_REQUIRED, holdout_state="UNTOUCHED")
        eligibility = check_paper_replay_eligibility(rec)
        assert eligibility.is_eligible is False
        assert "requires 'PAPER_ELIGIBLE'" in eligibility.reason

    def test_ineligible_when_holdout_not_passed(self) -> None:
        rec = _make_sample_record(final_status=ValidationStatus.PAPER_ELIGIBLE, holdout_state="FAILED")
        eligibility = check_paper_replay_eligibility(rec)
        assert eligibility.is_eligible is False
        assert "requires 'PASSED'" in eligibility.reason

    def test_ineligible_when_digest_tampered(self) -> None:
        rec = _make_sample_record()
        d = rec.to_dict()
        d["final_status"] = ValidationStatus.PAPER_ELIGIBLE.value
        d["observed_sharpe"] = 5.0  # digest will not match
        tampered = StrategyQualificationRecord.from_dict(d)
        eligibility = check_paper_replay_eligibility(tampered)
        assert eligibility.is_eligible is False
        assert "verification failed" in eligibility.reason
