"""Strategy Validation Gate & Immutable Qualification Record (PRD v3.8).

This module implements:
- ValidationStatus state machine with legal transition enforcement.
- Immutable StrategyQualificationRecord with canonical JSON and SHA-256 digest.
- QualificationLedger with SQLite append-only triggers.
- RobustnessStatus and RobustnessReport.
- StrategyValidationGate consuming authoritative evidence only.
- PaperReplayEligibility interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import yaml

from quantmind.data.registry import DatasetRegistry
from quantmind.research_integrity.holdout import HoldoutManager, HoldoutState


from quantmind.research_integrity.population import ProductionPopulationQuery
from quantmind.research_integrity.statistical_validation import (
    DeflatedSharpeCalculator,
    DeflatedSharpeResult,
    EictClusterResult,
    EictCorr1Calculator,
)
from quantmind.research_integrity.trial_ledger import TrialLedger
from quantmind.strategy.spec import StrategySpec
from quantmind.strategy.compiler import derive_strategy_id, normalize_strategy_spec



# ---------------------------------------------------------------------------
# Exceptions & Status Enums
# ---------------------------------------------------------------------------


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal lifecycle or validation state transition is attempted."""


class QualificationLedgerError(RuntimeError):
    """Raised when an error occurs while writing to or querying the qualification ledger."""


class ValidationGateError(RuntimeError):
    """Raised when validation gate prerequisites or integrity checks fail."""


class ValidationStatus(str, Enum):
    """Lifecycle statuses for candidate strategy validation."""

    CANDIDATE = "CANDIDATE"
    UNDERPOWERED = "UNDERPOWERED"
    VALIDATION = "VALIDATION"
    REJECTED = "REJECTED"
    REJECTED_FINAL_HOLDOUT = "REJECTED_FINAL_HOLDOUT"
    HOLDOUT_REQUIRED = "HOLDOUT_REQUIRED"
    HOLDOUT_PASSED = "HOLDOUT_PASSED"
    PAPER_ELIGIBLE = "PAPER_ELIGIBLE"


class RobustnessStatus(str, Enum):
    """Outcome of sensitivity/perturbation robustness tests."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_TESTED = "NOT_TESTED"


# Legal transitions between validation statuses
ALLOWED_VALIDATION_TRANSITIONS: Mapping[ValidationStatus, frozenset[ValidationStatus]] = {
    ValidationStatus.CANDIDATE: frozenset(
        {
            ValidationStatus.UNDERPOWERED,
            ValidationStatus.VALIDATION,
            ValidationStatus.REJECTED,
            ValidationStatus.HOLDOUT_REQUIRED,
        }
    ),
    ValidationStatus.UNDERPOWERED: frozenset(
        {
            ValidationStatus.CANDIDATE,
            ValidationStatus.REJECTED,
        }
    ),
    ValidationStatus.VALIDATION: frozenset(
        {
            ValidationStatus.HOLDOUT_REQUIRED,
            ValidationStatus.UNDERPOWERED,
            ValidationStatus.REJECTED,
        }
    ),
    ValidationStatus.HOLDOUT_REQUIRED: frozenset(
        {
            ValidationStatus.HOLDOUT_PASSED,
            ValidationStatus.REJECTED_FINAL_HOLDOUT,
            ValidationStatus.REJECTED,
        }
    ),
    ValidationStatus.HOLDOUT_PASSED: frozenset(
        {
            ValidationStatus.PAPER_ELIGIBLE,
            ValidationStatus.REJECTED,
        }
    ),
    ValidationStatus.PAPER_ELIGIBLE: frozenset(),
    ValidationStatus.REJECTED: frozenset(),
    ValidationStatus.REJECTED_FINAL_HOLDOUT: frozenset(),
}



def validate_transition(current: ValidationStatus, target: ValidationStatus) -> None:
    """Enforce that state transitions obey the locked state machine."""
    allowed = ALLOWED_VALIDATION_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidStateTransitionError(
            f"Illegal validation status transition from {current.value} to {target.value}. "
            f"Allowed targets: {sorted(s.value for s in allowed) if allowed else 'None (terminal state)'}"
        )


# ---------------------------------------------------------------------------
# Robustness & Qualification Record Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessReport:
    """Summary of parameter sensitivity and perturbation checks."""

    status: RobustnessStatus
    details: Mapping[str, Any]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "details": dict(self.details),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StrategyQualificationRecord:
    """Immutable audit record certifying the qualification status of a candidate strategy.

    Every record is sealed with a SHA-256 digest over its canonical representation.
    """

    qualification_id: str
    strategy_id: str
    strategy_spec_hash: str
    dataset_version: str
    dataset_sha256: str
    split_manifest_version: str
    research_protocol_version: str
    population_hash: str
    effective_trial_count: float
    observed_sharpe: float
    dsr: float
    trade_count: int
    holdout_state: str
    robustness_status: RobustnessStatus
    final_status: ValidationStatus
    created_at: str
    reasons: tuple[str, ...]
    record_hash: str

    def canonical_dict(self) -> dict[str, Any]:
        """Return dict representation excluding the computed record_hash."""
        return {
            "created_at": self.created_at,
            "dataset_sha256": self.dataset_sha256,
            "dataset_version": self.dataset_version,
            "dsr": round(float(self.dsr), 8),
            "effective_trial_count": round(float(self.effective_trial_count), 8),
            "final_status": self.final_status.value,
            "holdout_state": self.holdout_state,
            "observed_sharpe": round(float(self.observed_sharpe), 8),
            "population_hash": self.population_hash,
            "qualification_id": self.qualification_id,
            "reasons": list(self.reasons),
            "research_protocol_version": self.research_protocol_version,
            "robustness_status": self.robustness_status.value,
            "split_manifest_version": self.split_manifest_version,
            "strategy_id": self.strategy_id,
            "strategy_spec_hash": self.strategy_spec_hash,
            "trade_count": int(self.trade_count),
        }

    def canonical_json(self) -> str:
        """Produce deterministic canonical JSON string."""
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_record_hash(self) -> str:
        """Compute SHA-256 digest over canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def verify_digest(self) -> bool:
        """Verify that record_hash matches the canonical content digest."""
        return self.record_hash == self.compute_record_hash()

    @classmethod
    def create(
        cls,
        *,
        qualification_id: str,
        strategy_id: str,
        strategy_spec_hash: str,
        dataset_version: str,
        dataset_sha256: str,
        split_manifest_version: str,
        research_protocol_version: str,
        population_hash: str,
        effective_trial_count: float,
        observed_sharpe: float,
        dsr: float,
        trade_count: int,
        holdout_state: str,
        robustness_status: RobustnessStatus,
        final_status: ValidationStatus,
        created_at: str | None = None,
        reasons: Sequence[str] = (),
    ) -> StrategyQualificationRecord:
        now = created_at or datetime.now(timezone.utc).isoformat()
        temp_record = cls(
            qualification_id=qualification_id,
            strategy_id=strategy_id,
            strategy_spec_hash=strategy_spec_hash,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            split_manifest_version=split_manifest_version,
            research_protocol_version=research_protocol_version,
            population_hash=population_hash,
            effective_trial_count=effective_trial_count,
            observed_sharpe=observed_sharpe,
            dsr=dsr,
            trade_count=trade_count,
            holdout_state=holdout_state,
            robustness_status=robustness_status,
            final_status=final_status,
            created_at=now,
            reasons=tuple(reasons),
            record_hash="",
        )
        digest = temp_record.compute_record_hash()
        return cls(
            qualification_id=qualification_id,
            strategy_id=strategy_id,
            strategy_spec_hash=strategy_spec_hash,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            split_manifest_version=split_manifest_version,
            research_protocol_version=research_protocol_version,
            population_hash=population_hash,
            effective_trial_count=effective_trial_count,
            observed_sharpe=observed_sharpe,
            dsr=dsr,
            trade_count=trade_count,
            holdout_state=holdout_state,
            robustness_status=robustness_status,
            final_status=final_status,
            created_at=now,
            reasons=tuple(reasons),
            record_hash=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        d = self.canonical_dict()
        d["record_hash"] = self.record_hash
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StrategyQualificationRecord:
        return cls(
            qualification_id=d["qualification_id"],
            strategy_id=d["strategy_id"],
            strategy_spec_hash=d["strategy_spec_hash"],
            dataset_version=d["dataset_version"],
            dataset_sha256=d["dataset_sha256"],
            split_manifest_version=d["split_manifest_version"],
            research_protocol_version=d["research_protocol_version"],
            population_hash=d["population_hash"],
            effective_trial_count=float(d["effective_trial_count"]),
            observed_sharpe=float(d["observed_sharpe"]),
            dsr=float(d["dsr"]),
            trade_count=int(d["trade_count"]),
            holdout_state=d["holdout_state"],
            robustness_status=RobustnessStatus(d["robustness_status"]),
            final_status=ValidationStatus(d["final_status"]),
            created_at=d["created_at"],
            reasons=tuple(d.get("reasons", ())),
            record_hash=d["record_hash"],
        )


@dataclass(frozen=True)
class ValidationReport:
    """Structured report documenting the evaluation of a candidate strategy.

    Explicitly excludes strategy rankings, relative leaderboards, or 'best strategy' claims.
    """

    record: StrategyQualificationRecord
    checks_passed: tuple[str, ...]
    checks_failed: tuple[str, ...]
    checks_warned: tuple[str, ...]
    summary_text: str


@dataclass(frozen=True)
class PaperReplayEligibility:
    """Eligibility certification for paper forward replay testing."""

    strategy_id: str
    is_eligible: bool
    qualification_id: str | None
    qualification_hash: str | None
    reason: str


def check_paper_replay_eligibility(record: StrategyQualificationRecord) -> PaperReplayEligibility:
    """Assess whether a StrategyQualificationRecord entitles the strategy to paper replay."""
    if not record.verify_digest():
        return PaperReplayEligibility(
            strategy_id=record.strategy_id,
            is_eligible=False,
            qualification_id=record.qualification_id,
            qualification_hash=record.record_hash,
            reason="Qualification record digest verification failed (record has been tampered with)",
        )

    if record.final_status != ValidationStatus.PAPER_ELIGIBLE:
        return PaperReplayEligibility(
            strategy_id=record.strategy_id,
            is_eligible=False,
            qualification_id=record.qualification_id,
            qualification_hash=record.record_hash,
            reason=f"Candidate status is '{record.final_status.value}', requires 'PAPER_ELIGIBLE'",
        )

    if record.holdout_state != HoldoutState.PASSED.value:
        return PaperReplayEligibility(
            strategy_id=record.strategy_id,
            is_eligible=False,
            qualification_id=record.qualification_id,
            qualification_hash=record.record_hash,
            reason=f"Sealed holdout state is '{record.holdout_state}', requires '{HoldoutState.PASSED.value}'",
        )

    return PaperReplayEligibility(
        strategy_id=record.strategy_id,
        is_eligible=True,
        qualification_id=record.qualification_id,
        qualification_hash=record.record_hash,
        reason="Candidate strategy meets all statistical, holdout, and sample requirements for paper replay",
    )


# ---------------------------------------------------------------------------
# Qualification Ledger
# ---------------------------------------------------------------------------


class QualificationLedger:
    """Append-only SQLite ledger storing immutable StrategyQualificationRecords."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is not None:
            self._connection = connection
        elif db_path is not None:
            self._connection = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,
            )
        else:
            self._connection = sqlite3.connect(
                ":memory:",
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,
            )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS strategy_qualifications (
                qualification_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                strategy_spec_hash TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                split_manifest_version TEXT NOT NULL,
                research_protocol_version TEXT NOT NULL,
                population_hash TEXT NOT NULL,
                effective_trial_count REAL NOT NULL,
                observed_sharpe REAL NOT NULL,
                dsr REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                holdout_state TEXT NOT NULL,
                robustness_status TEXT NOT NULL,
                final_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                record_hash TEXT NOT NULL UNIQUE,
                record_json TEXT NOT NULL
            );

            CREATE TRIGGER IF NOT EXISTS qualifications_no_delete
            BEFORE DELETE ON strategy_qualifications
            BEGIN
                SELECT RAISE(ABORT, 'qualification records are permanent and append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS qualifications_no_update
            BEFORE UPDATE ON strategy_qualifications
            BEGIN
                SELECT RAISE(ABORT, 'qualification records are permanent and immutable');
            END;
            """
        )

    def record_qualification(self, record: StrategyQualificationRecord) -> None:
        """Insert qualification record into append-only ledger."""
        if not record.verify_digest():
            raise QualificationLedgerError(
                f"Record {record.qualification_id} failed cryptographic digest verification; refusing to persist"
            )

        existing = self.get(record.qualification_id)
        if existing is not None:
            if existing.record_hash == record.record_hash:
                return  # Idempotent write of identical record
            raise QualificationLedgerError(
                f"Conflicting qualification record exists for {record.qualification_id}"
            )

        self._connection.execute(
            """
            INSERT INTO strategy_qualifications (
                qualification_id, strategy_id, strategy_spec_hash, dataset_version,
                dataset_sha256, split_manifest_version, research_protocol_version,
                population_hash, effective_trial_count, observed_sharpe, dsr,
                trade_count, holdout_state, robustness_status, final_status,
                created_at, reasons_json, record_hash, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.qualification_id,
                record.strategy_id,
                record.strategy_spec_hash,
                record.dataset_version,
                record.dataset_sha256,
                record.split_manifest_version,
                record.research_protocol_version,
                record.population_hash,
                record.effective_trial_count,
                record.observed_sharpe,
                record.dsr,
                record.trade_count,
                record.holdout_state,
                record.robustness_status.value,
                record.final_status.value,
                record.created_at,
                json.dumps(list(record.reasons)),
                record.record_hash,
                json.dumps(record.to_dict(), sort_keys=True),
            ),
        )

    def get(self, qualification_id: str) -> StrategyQualificationRecord | None:
        row = self._connection.execute(
            "SELECT record_json FROM strategy_qualifications WHERE qualification_id = ?",
            (qualification_id,),
        ).fetchone()
        if row is None:
            return None
        return StrategyQualificationRecord.from_dict(json.loads(row["record_json"]))

    def find_qualifications(
        self,
        *,
        strategy_id: str | None = None,
        final_status: ValidationStatus | None = None,
        dataset_version: str | None = None,
        research_protocol_version: str | None = None,
    ) -> list[StrategyQualificationRecord]:
        query = "SELECT record_json FROM strategy_qualifications WHERE 1=1"
        params: list[Any] = []
        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if final_status is not None:
            query += " AND final_status = ?"
            params.append(final_status.value)
        if dataset_version is not None:
            query += " AND dataset_version = ?"
            params.append(dataset_version)
        if research_protocol_version is not None:
            query += " AND research_protocol_version = ?"
            params.append(research_protocol_version)
        query += " ORDER BY created_at ASC"

        rows = self._connection.execute(query, params).fetchall()
        return [StrategyQualificationRecord.from_dict(json.loads(r["record_json"])) for r in rows]


# ---------------------------------------------------------------------------
# Strategy Validation Gate
# ---------------------------------------------------------------------------


class StrategyValidationGate:
    """Authoritative gate deciding qualification status for candidate strategies.

    Guarantees:
    - Consumes ONLY authoritative system evidence: DatasetRegistry, TrialLedger,
      HoldoutManager, ProductionPopulationQuery, EICT-CORR-1, and DSR.
    - Strictly forbids and rejects manual overrides of Sharpe, DSR, or trial count.
    - Prohibits synthetic/fixture trials from participating in production qualification.
    - Evaluates protocol-defined sample size requirements from versioned research protocol.
    - Requires sealed final holdout evaluation before certifying PAPER_ELIGIBLE.
    """

    def __init__(
        self,
        *,
        dataset_registry: DatasetRegistry,
        trial_ledger: TrialLedger,
        holdout_manager: HoldoutManager,
        population_query: ProductionPopulationQuery,
        qualification_ledger: QualificationLedger | None = None,
        protocol_path: str | Path | None = None,
        protocol_config: Mapping[str, Any] | None = None,
        eict_calculator: EictCorr1Calculator | None = None,
    ) -> None:
        self._dataset_registry = dataset_registry
        self._trial_ledger = trial_ledger
        self._holdout_manager = holdout_manager
        self._population_query = population_query
        self._qualification_ledger = qualification_ledger or QualificationLedger()
        self._eict_calculator = eict_calculator or EictCorr1Calculator()

        if protocol_config is not None:
            self._protocol_config = dict(protocol_config)
        elif protocol_path is not None:
            with open(protocol_path, "r", encoding="utf-8") as f:
                self._protocol_config = yaml.safe_load(f)
        else:
            default_path = Path("configs/research_protocol_v2.yaml")
            if default_path.exists():
                with open(default_path, "r", encoding="utf-8") as f:
                    self._protocol_config = yaml.safe_load(f)
            else:
                self._protocol_config = {}

    @property
    def qualification_ledger(self) -> QualificationLedger:
        return self._qualification_ledger

    def _get_protocol_sample_requirements(self) -> dict[str, int]:
        reqs = self._protocol_config.get("sample_requirements", {})
        return {
            "minimum_trades_total": int(reqs.get("minimum_trades_total", 300)),
            "minimum_trades_validation": int(reqs.get("minimum_trades_validation", 75)),
            "minimum_trades_final_holdout": int(reqs.get("minimum_trades_final_holdout", 50)),
            "minimum_effective_outcome_observations": int(reqs.get("minimum_effective_outcome_observations", 100)),
        }

    def evaluate_candidate(
        self,
        strategy_spec: StrategySpec,
        validation_trial_id: str,
        *,
        dataset_version: str,
        research_protocol_version: str,
        robustness_report: RobustnessReport | None = None,
        require_holdout: bool = True,
        **kwargs: Any,
    ) -> tuple[StrategyQualificationRecord, ValidationReport]:
        """Evaluate candidate strategy validation trial against authoritative evidence."""
        # 1. Reject any attempt to pass manual statistical overrides or unauthorized caller parameters
        prohibited_overrides = {
            "manual_sharpe",
            "sharpe",
            "manual_dsr",
            "dsr",
            "manual_trials",
            "manual_trial_count",
            "trial_count",
            "manual_effective_trial_count",
            "effective_trial_count",
            "manual_skew",
            "manual_kurtosis",
            "manual_holdout",
            "manual_holdout_state",
            "holdout_state",
            "manual_trade_count",
            "trade_count",
            "manual_population_hash",
            "population_hash",
            "manual_dataset_sha256",
            "dataset_sha256",
            "manual_dataset_version",
            "dataset_identity",
            "manual_dataset_identity",
        }
        if kwargs:
            keys = sorted(kwargs.keys())
            raise ValueError(
                f"Manual statistical overrides or caller-supplied parameters ({keys}) are strictly prohibited; "
                "qualification metrics must derive exclusively from authoritative system evidence."
            )



        # 2. Retrieve and verify validation trial from authoritative TrialLedger
        try:
            trial_row = self._trial_ledger.get(validation_trial_id)
        except KeyError:
            raise ValidationGateError(f"Validation trial '{validation_trial_id}' not found in TrialLedger")

        # Must be PRODUCTION mode
        if trial_row["mode"] != "PRODUCTION":
            raise ValueError(
                f"Validation gate requires mode='PRODUCTION'; trial '{validation_trial_id}' has mode='{trial_row['mode']}'"
            )

        # Must be in VALIDATION split zone
        if trial_row["split_zone"] != "VALIDATION":
            raise ValueError(
                f"Validation trial must be from split_zone='VALIDATION'; got '{trial_row['split_zone']}'"
            )

        # Must be COMPLETED
        if trial_row["status"] != "COMPLETED":
            raise ValidationGateError(
                f"Validation trial '{validation_trial_id}' has status='{trial_row['status']}'; must be 'COMPLETED'"
            )

        # Verify strategy ID matches normalized strategy spec
        norm_spec = normalize_strategy_spec(strategy_spec)
        expected_strategy_id = derive_strategy_id(norm_spec)
        if trial_row["strategy_id"] != expected_strategy_id:
            raise ValueError(
                f"Validation trial strategy_id '{trial_row['strategy_id']}' does not match "
                f"derived strategy_id '{expected_strategy_id}'"
            )

        # Verify dataset and protocol versions match trial
        if trial_row["dataset_version"] != dataset_version:
            raise ValueError(
                f"Trial dataset_version '{trial_row['dataset_version']}' does not match requested '{dataset_version}'"
            )
        if trial_row["research_protocol_version"] != research_protocol_version:
            raise ValueError(
                f"Trial protocol '{trial_row['research_protocol_version']}' does not match requested '{research_protocol_version}'"
            )

        # 3. Check DatasetRegistry and SplitManifest provenance
        dataset_record = self._dataset_registry.get(dataset_version)
        dataset_sha256 = dataset_record.sha256
        split_manifest_version = (
            dataset_record.split_manifest.manifest_version
            if dataset_record.split_manifest
            else trial_row["split_manifest_version"]
        )
        spec_hash = hashlib.sha256(norm_spec.canonical_json().encode("utf-8")).hexdigest()

        # Parse trial backtest result
        result_data: dict[str, Any] = {}
        if trial_row["result_json"]:
            result_data = json.loads(trial_row["result_json"])

        trade_count = int(result_data.get("trade_count", 0))
        net_pnl = float(result_data.get("net_pnl", 0.0))

        # 4. Load production population and run statistical verification (EICT + DSR)
        population = self._population_query.load_eligible_trials(
            dataset_version=dataset_version,
            research_protocol_version=research_protocol_version,
        )
        trial_in_pop = any(t.trial_id == validation_trial_id for t in population)
        if not trial_in_pop:
            raise ValidationGateError(
                f"Trial '{validation_trial_id}' is not in the eligible production population query. "
                "Ensure OOS return stream artifact was written and registered."
            )

        eict_result = self._eict_calculator.calculate(population)
        dsr_calculator = DeflatedSharpeCalculator(population, eict_result)
        dsr_result = dsr_calculator.calculate(validation_trial_id)

        # 5. Evaluate Protocol Sample Requirements
        sample_reqs = self._get_protocol_sample_requirements()
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        checks_warned: list[str] = []
        reasons: list[str] = []

        is_underpowered = False
        is_rejected = False
        is_holdout_rejected = False

        # Validation sample size check
        if trade_count < sample_reqs["minimum_trades_validation"]:
            checks_failed.append(
                f"Validation trade count ({trade_count}) < protocol minimum ({sample_reqs['minimum_trades_validation']})"
            )
            reasons.append(
                f"Sample size underpowered: {trade_count} trades in validation (minimum required: {sample_reqs['minimum_trades_validation']})"
            )
            is_underpowered = True
        else:
            checks_passed.append(
                f"Validation trade count ({trade_count}) >= protocol minimum ({sample_reqs['minimum_trades_validation']})"
            )

        # Effective observations check
        if dsr_result.sample_length < sample_reqs["minimum_effective_outcome_observations"]:
            checks_failed.append(
                f"Effective observations ({dsr_result.sample_length}) < protocol minimum ({sample_reqs['minimum_effective_outcome_observations']})"
            )
            reasons.append(
                f"Sample observations underpowered: {dsr_result.sample_length} (minimum required: {sample_reqs['minimum_effective_outcome_observations']})"
            )
            is_underpowered = True
        else:
            checks_passed.append(
                f"Effective observations ({dsr_result.sample_length}) >= protocol minimum ({sample_reqs['minimum_effective_outcome_observations']})"
            )

        # Net PnL check
        if net_pnl <= 0.0:
            checks_failed.append(f"Validation net PnL is non-positive: {net_pnl:.4f}")
            reasons.append(f"Validation trial failed net profitability: net PnL = {net_pnl:.4f}")
            is_rejected = True
        else:
            checks_passed.append(f"Validation net PnL is positive: {net_pnl:.4f}")

        # 6. Evaluate Robustness Report
        rob_status = RobustnessStatus.NOT_TESTED
        if robustness_report is not None:
            rob_status = robustness_report.status
            if rob_status == RobustnessStatus.FAILED:
                checks_failed.append("Robustness perturbation test failed")
                reasons.append(f"Robustness failure: {', '.join(robustness_report.reasons)}")
                is_rejected = True
            elif rob_status == RobustnessStatus.PASSED:
                checks_passed.append("Robustness perturbation test passed")
        else:
            checks_warned.append("Robustness report not provided (NOT_TESTED)")

        # 7. Check Sealed Holdout State
        holdout_state = self._holdout_manager.get_holdout_state(
            expected_strategy_id, dataset_version, research_protocol_version
        )
        if holdout_state == HoldoutState.BURNED:
            checks_failed.append("Holdout is BURNED")
            reasons.append("Holdout partition is BURNED; further evaluations prohibited")
            is_rejected = True
        elif holdout_state == HoldoutState.FAILED:
            checks_failed.append("Candidate FAILED sealed holdout evaluation")
            reasons.append("Candidate failed sealed holdout evaluation; retuning prohibited")
            is_holdout_rejected = True
        elif holdout_state == HoldoutState.PASSED:
            checks_passed.append("Sealed holdout evaluation PASSED")
        elif holdout_state == HoldoutState.UNTOUCHED:
            if require_holdout:
                checks_warned.append("Sealed holdout UNTOUCHED; holdout evaluation required before paper replay")

        # 8. Determine Final Validation Status
        if is_holdout_rejected:
            final_status = ValidationStatus.REJECTED_FINAL_HOLDOUT
        elif is_rejected:
            final_status = ValidationStatus.REJECTED
        elif is_underpowered:
            final_status = ValidationStatus.UNDERPOWERED
        else:
            # Passed all validation checks
            if holdout_state == HoldoutState.PASSED:
                final_status = ValidationStatus.PAPER_ELIGIBLE
            elif require_holdout and holdout_state == HoldoutState.UNTOUCHED:
                final_status = ValidationStatus.HOLDOUT_REQUIRED
            else:
                final_status = ValidationStatus.VALIDATION

        now_str = datetime.now(timezone.utc).isoformat()
        qualification_id = "QUAL-" + hashlib.sha256(
            f"{expected_strategy_id}:{dataset_version}:{research_protocol_version}:{validation_trial_id}:{now_str}".encode()
        ).hexdigest()[:24]

        record = StrategyQualificationRecord.create(
            qualification_id=qualification_id,
            strategy_id=expected_strategy_id,
            strategy_spec_hash=spec_hash,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            split_manifest_version=split_manifest_version,
            research_protocol_version=research_protocol_version,
            population_hash=eict_result.population_hash,
            effective_trial_count=float(eict_result.effective_trial_count),
            observed_sharpe=float(dsr_result.observed_sharpe),
            dsr=float(dsr_result.dsr),
            trade_count=trade_count,
            holdout_state=holdout_state.value,
            robustness_status=rob_status,
            final_status=final_status,
            created_at=now_str,
            reasons=reasons,
        )

        # Store in append-only qualification ledger
        self._qualification_ledger.record_qualification(record)

        summary_text = (
            f"StrategyValidationGate evaluated {expected_strategy_id} -> {final_status.value}. "
            f"Observed Sharpe: {record.observed_sharpe:.4f}, DSR: {record.dsr:.4f}, "
            f"Effective Trials: {record.effective_trial_count}, Validation Trades: {record.trade_count}, "
            f"Holdout: {record.holdout_state}."
        )

        report = ValidationReport(
            record=record,
            checks_passed=tuple(checks_passed),
            checks_failed=tuple(checks_failed),
            checks_warned=tuple(checks_warned),
            summary_text=summary_text,
        )

        return record, report
