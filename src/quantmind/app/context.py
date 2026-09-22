"""Application Context container holding singletons for core ledgers and services."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from quantmind.app.auth.service import AuthService
from quantmind.app.config import AppSettings, get_settings
from quantmind.app.jobs.orchestrator import JobOrchestrator
from quantmind.backtest.engine import BacktestEngine
from quantmind.data.registry import DatasetRegistry
from quantmind.paper.engine import PaperReplayEngine
from quantmind.paper.evaluation.feedback import ResearchFeedbackService
from quantmind.paper.evaluation.governance import PaperGovernanceService
from quantmind.paper.evaluation.ledger import EvaluationLedger
from quantmind.paper.ledger import PaperLedger
from quantmind.research_integrity.artifacts import ArtifactRegistry
from quantmind.research_integrity.harness import ResearchHarness
from quantmind.research_integrity.holdout import HoldoutManager
from quantmind.research_integrity.population import ProductionPopulationQuery
from quantmind.research_integrity.qualification import QualificationLedger, StrategyValidationGate
from quantmind.research_integrity.trial_ledger import TrialLedger
from quantmind.strategy.registry import StrategyRegistry


class AppContext:
    """Dependency injection container for the application."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

        # Ensure directories exist
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.core_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.app_db_path.parent.mkdir(parents=True, exist_ok=True)

        core_db = str(self.settings.core_db_path)

        # 1. Core authoritative ledgers
        self.dataset_registry = DatasetRegistry(core_db)
        self.strategy_registry = StrategyRegistry(core_db)
        self.trial_ledger = TrialLedger(core_db)
        self.artifact_registry = ArtifactRegistry(core_db)
        self.qualification_ledger = QualificationLedger(db_path=core_db)
        self.backtest_engine = BacktestEngine()
        self.holdout_manager = HoldoutManager(
            engine=self.backtest_engine,
            ledger=self.trial_ledger,
            dataset_registry=self.dataset_registry,
            db_path=core_db,
        )
        self.population_query = ProductionPopulationQuery(
            ledger=self.trial_ledger,
            artifact_registry=self.artifact_registry,
        )
        self.validation_gate = StrategyValidationGate(
            dataset_registry=self.dataset_registry,
            trial_ledger=self.trial_ledger,
            holdout_manager=self.holdout_manager,
            population_query=self.population_query,
            qualification_ledger=self.qualification_ledger,
        )
        self.research_harness = ResearchHarness(
            engine=self.backtest_engine,
            ledger=self.trial_ledger,
            dataset_registry=self.dataset_registry,
            holdout_manager=self.holdout_manager,
            artifact_registry=self.artifact_registry,
        )
        self.paper_ledger = PaperLedger(db_path=core_db)
        self.evaluation_ledger = EvaluationLedger(db_path=core_db)
        self.paper_replay_engine = PaperReplayEngine(
            ledger=self.paper_ledger,
            dataset_registry=self.dataset_registry,
            allow_fixture_feed=True,
        )

        # 2. Domain governance & feedback services
        self.governance_service = PaperGovernanceService(
            strategy_registry=self.strategy_registry,
            evaluation_ledger=self.evaluation_ledger,
        )
        self.feedback_service = ResearchFeedbackService(
            strategy_registry=self.strategy_registry,
            evaluation_ledger=self.evaluation_ledger,
        )

        # 3. Application-level services
        self.auth_service = AuthService(
            db_path=self.settings.app_db_path,
            secret_key=self.settings.secret_key,
            algorithm=self.settings.jwt_algorithm,
            app_env=self.settings.app_env,
        )
        self.job_orchestrator = JobOrchestrator(
            app_db_path=self.settings.app_db_path,
        )

    def get_core_connection(self) -> sqlite3.Connection:
        """Helper to get a query connection to core database."""
        conn = sqlite3.connect(str(self.settings.core_db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_app_connection(self) -> sqlite3.Connection:
        """Helper to get a query connection to application metadata database."""
        conn = sqlite3.connect(str(self.settings.app_db_path))
        conn.row_factory = sqlite3.Row
        return conn


_global_context: AppContext | None = None


def get_app_context() -> AppContext:
    """Retrieve global AppContext singleton."""
    global _global_context
    if _global_context is None:
        _global_context = AppContext()
    return _global_context


def set_app_context(ctx: AppContext) -> None:
    """Set global AppContext (useful for testing)."""
    global _global_context
    _global_context = ctx
