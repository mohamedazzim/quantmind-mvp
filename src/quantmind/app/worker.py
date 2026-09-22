"""Dedicated Background Worker Process for QuantMind (PRD v4.0 Productization)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any

from quantmind.app.config import get_settings
from quantmind.app.context import AppContext, get_app_context
from quantmind.app.jobs.models import Job, JobType
from quantmind.app.jobs.orchestrator import JobOrchestrator
from quantmind.backtest.engine import BacktestConfig
from quantmind.research_integrity.trial_ledger import ResearchBudget
from quantmind.strategy import StrategySpec, normalize_strategy_spec
from quantmind.strategy.compiler import derive_strategy_id
from quantmind.paper.evaluation.models import PaperExecutionLifecycleContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Worker] %(message)s",
)
logger = logging.getLogger("quantmind.worker")


class QuantMindWorker:
    """Independent worker processing long-running domain jobs."""

    def __init__(self, ctx: AppContext | None = None, worker_id: str | None = None) -> None:
        self.ctx = ctx or get_app_context()
        self.orchestrator = self.ctx.job_orchestrator
        self.worker_id = worker_id or f"WRK-{secrets.token_hex(4)}"
        self._running = False

    def startup(self) -> int:
        """Initialize worker and recover any stale RUNNING jobs left by previous crashes."""
        logger.info(f"Worker {self.worker_id} starting up...")
        recovered = self.orchestrator.recover_stale_jobs()
        if recovered > 0:
            logger.warning(f"Recovered {recovered} stale RUNNING jobs to FAILED status.")
        return recovered

    def process_one_job(self) -> bool:
        """Claim and execute a single queued job. Returns True if a job was processed."""
        job = self.orchestrator.claim_next_job(self.worker_id)
        if not job:
            return False

        logger.info(f"Claimed job {job.job_id} (Type: {job.job_type.value})")
        self.orchestrator.update_progress(job.job_id, 10)

        try:
            if job.job_type == JobType.RESEARCH_TRIAL:
                self._execute_research_trial(job)
            elif job.job_type == JobType.PAPER_REPLAY:
                self._execute_paper_replay(job)
            elif job.job_type == JobType.QUALIFICATION_GATE:
                self._execute_qualification_gate(job)
            else:
                raise NotImplementedError(f"Unsupported job type: {job.job_type}")
            return True
        except Exception as exc:
            logger.error(f"Job {job.job_id} failed: {exc}", exc_info=True)
            self.orchestrator.fail_job(job.job_id, f"{type(exc).__name__}: {str(exc)}")
            return True

    def _execute_research_trial(self, job: Job) -> None:
        """Execute ResearchHarness trial."""
        params = job.parameters
        spec_dict = params.get("strategy_spec")
        if not spec_dict:
            raise ValueError("Job parameters missing 'strategy_spec'")

        spec = StrategySpec(
            strategy_version=spec_dict.get("strategy_version", "1.0.0"),
            feature_version=spec_dict.get("feature_version", "1.0.0"),
            signal_name=spec_dict.get("signal_name", "current_bar_momentum"),
            parameters=spec_dict.get("parameters", {}),
        )
        norm_spec = normalize_strategy_spec(spec)
        self.orchestrator.update_progress(job.job_id, 30)

        dataset_version = params.get("dataset_version", "ds-synth-v1")
        strategy_id = derive_strategy_id(norm_spec)

        config = BacktestConfig(
            execution_model=params.get("execution_model", "next_bar_open_v1"),
            quantity=int(params.get("quantity", 1)),
            lot_size=int(params.get("lot_size", 1)),
            slippage_bps_per_side=float(params.get("slippage_bps_per_side", 0.0)),
        )
        budget = ResearchBudget(
            max_trials=int(params.get("max_trials", 100)),
            max_experiments=int(params.get("max_experiments", 50)),
            max_strategy_variants=int(params.get("max_strategy_variants", 20)),
            max_runtime_minutes=float(params.get("max_runtime_minutes", 60.0)),
        )
        research_task_id = params.get("research_task_id", f"TASK-{job.job_id}")
        split_zone = params.get("split_zone", "RESEARCH")
        protocol_version = params.get("research_protocol_version", "proto-v1")
        seed = int(params.get("seed", 42))

        # Invoke core domain research harness
        harness = self.ctx.research_harness
        bt_result = harness.run_trial(
            config=config,
            research_task_id=research_task_id,
            strategy_spec=norm_spec,
            dataset_version=dataset_version,
            split_zone=split_zone,
            research_protocol_version=protocol_version,
            seed=seed,
            budget=budget,
        )
        self.orchestrator.update_progress(job.job_id, 90)

        with self.ctx.get_core_connection() as conn:
            row = conn.execute(
                "SELECT trial_id, timestamp_started, status FROM trials WHERE strategy_id = ? ORDER BY rowid DESC LIMIT 1",
                (strategy_id,),
            ).fetchone()

        trial_id = row["trial_id"] if row else f"TRIAL-{secrets.token_hex(6)}"
        created_at = row["timestamp_started"] if row else ""

        result_payload = {
            "trial_id": trial_id,
            "strategy_id": strategy_id,
            "dataset_version": dataset_version,
            "feature_set_version": norm_spec.feature_version,
            "signal_name": norm_spec.signal_name,
            "parameters": dict(norm_spec.parameters),
            "trade_count": len(bt_result.trades),
            "gross_pnl": bt_result.gross_pnl,
            "net_pnl": bt_result.net_pnl,
            "created_at": created_at,
        }
        self.orchestrator.complete_job(
            job.job_id,
            result=result_payload,
            result_ref=trial_id,
        )
        logger.info(f"Job {job.job_id} completed successfully. Trial ID: {trial_id}")

    def _execute_paper_replay(self, job: Job) -> None:
        """Execute PaperReplayEngine replay with explicit lifecycle context."""
        params = job.parameters
        strategy_id = params.get("strategy_id")
        if not strategy_id:
            raise ValueError("Missing strategy_id for paper replay")

        record = self.ctx.strategy_registry.get_strategy(strategy_id)
        auth_context = PaperExecutionLifecycleContext(
            strategy_id=strategy_id,
            authorized_state=record.state.value,
            authorization_timestamp=record.updated_at,
            governance_transition_hash=record.transition_hash or "",
        )
        self.orchestrator.update_progress(job.job_id, 40)

        # Run replay engine
        report = self.ctx.paper_replay_engine.run_fixture_replay(
            strategy_spec=record.strategy_spec,
            lifecycle_context=auth_context,
            bars=params.get("bars", 100),
        )
        self.orchestrator.update_progress(job.job_id, 90)

        result_payload = {
            "report_hash": report.report_hash,
            "strategy_id": report.strategy_id,
            "total_bars": report.total_bars,
            "trade_count": report.trade_count,
            "final_equity": report.final_equity,
            "max_drawdown": report.max_drawdown,
        }
        self.orchestrator.complete_job(
            job.job_id,
            result=result_payload,
            result_ref=report.report_hash,
        )
        logger.info(f"Job {job.job_id} completed successfully. Report Hash: {report.report_hash}")

    def _execute_qualification_gate(self, job: Job) -> None:
        """Execute StrategyValidationGate qualification."""
        params = job.parameters
        strategy_id = params.get("strategy_id")
        trial_id = params.get("trial_id")
        if not strategy_id or not trial_id:
            raise ValueError("Missing strategy_id or trial_id for qualification")

        record = self.ctx.strategy_registry.get_strategy(strategy_id)
        self.orchestrator.update_progress(job.job_id, 40)

        # Run validation gate
        qual_record = self.ctx.validation_gate.run_validation(
            strategy_id=strategy_id,
            trial_id=trial_id,
            strategy_spec=record.strategy_spec,
        )
        self.orchestrator.update_progress(job.job_id, 90)

        result_payload = {
            "qualification_hash": qual_record.qualification_hash,
            "strategy_id": qual_record.strategy_id,
            "trial_id": qual_record.trial_id,
            "decision": qual_record.decision.value,
            "created_at": qual_record.created_at,
        }
        self.orchestrator.complete_job(
            job.job_id,
            result=result_payload,
            result_ref=qual_record.qualification_hash,
        )
        logger.info(f"Job {job.job_id} completed successfully. Qual Hash: {qual_record.qualification_hash}")

    def run_loop(self, poll_interval: float = 1.0) -> None:
        """Continuous polling loop."""
        self._running = True
        self.startup()
        logger.info(f"Worker {self.worker_id} polling for jobs (interval: {poll_interval}s)...")
        try:
            while self._running:
                processed = self.process_one_job()
                if not processed:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user, shutting down gracefully...")
        finally:
            self._running = False
            logger.info("Worker stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantMind Dedicated Job Worker")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--worker-id", default=None, help="Optional worker ID identifier")
    args = parser.parse_args()

    worker = QuantMindWorker(worker_id=args.worker_id)
    worker.run_loop(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
