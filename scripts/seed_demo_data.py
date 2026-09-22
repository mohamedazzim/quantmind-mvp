"""QuantMind Demo Data Seeding Script (PRD v4.0 Productization).

Seeds realistic, deterministic market data, licensed datasets, strategy specs,
qualification records, evaluation baselines, paper executions, and monitoring snapshots
into data/quantmind_core.db and data/quantmind_app.db when running in DEMO mode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from quantmind.app.auth.models import UserRole
from quantmind.app.config import get_settings
from quantmind.app.context import get_app_context, AppContext
from quantmind.data.registry import DatasetKind, DatasetZone
from quantmind.research_integrity.qualification import (
    StrategyQualificationRecord,
    RobustnessStatus,
    ValidationStatus,
)
from quantmind.strategy.spec import StrategySpec
from quantmind.strategy.compiler import normalize_strategy_spec, derive_strategy_id
from quantmind.strategy.registry import StrategyLifecycleState


def create_sample_market_csv(csv_path: Path) -> None:
    """Generate a clean, deterministic OHLCV market dataset."""
    timestamps = []
    # 2026-01-02 (Session 1: 09:15 to 11:15)
    ts_day1 = pd.date_range("2026-01-02 09:15:00", periods=60, freq="1min")
    # 2026-01-03 (Session 2: 09:15 to 11:15)
    ts_day2 = pd.date_range("2026-01-03 09:15:00", periods=60, freq="1min")
    all_ts = list(ts_day1) + list(ts_day2)

    np.random.seed(42)
    price = 21500.0
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for _ in all_ts:
        ret = np.random.normal(0.0002, 0.001)
        new_price = price * (1 + ret)
        high = max(price, new_price) + np.random.uniform(1.0, 5.0)
        low = min(price, new_price) - np.random.uniform(1.0, 5.0)
        opens.append(round(price, 2))
        highs.append(round(high, 2))
        lows.append(round(low, 2))
        closes.append(round(new_price, 2))
        volumes.append(float(np.random.randint(500, 3000)))
        price = new_price

    df = pd.DataFrame(
        {
            "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in all_ts],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def seed_demo_data(force: bool = False) -> None:
    settings = get_settings()
    env_name = settings.app_env.upper()
    if env_name != "DEMO" and not force:
        print(f"[-] Aborted: app_env is '{env_name}'. Use --force or set APP_ENV=DEMO to seed demo data.")
        sys.exit(1)

    print(f"[*] Seeding QuantMind Demo Environment (APP_ENV={env_name})...")
    ctx = get_app_context()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Market Data CSV & Dataset Registration
    csv_path = data_dir / "demo_market_data.csv"
    create_sample_market_csv(csv_path)

    zones = {
        "RESEARCH": DatasetZone("RESEARCH", "2026-01-02 09:15:00", "2026-01-02 09:45:00"),
        "VALIDATION": DatasetZone("VALIDATION", "2026-01-02 09:46:00", "2026-01-02 10:15:00"),
        "FINAL_HOLDOUT": DatasetZone("FINAL_HOLDOUT", "2026-01-02 10:16:00", "2026-01-02 10:45:00"),
        "FORWARD_PAPER": DatasetZone("FORWARD_PAPER", "2026-01-03 09:15:00", "2026-01-03 10:15:00"),
    }

    dataset_version = "DS-NIFTY-2026-DEMO"
    try:
        dataset_record = ctx.dataset_registry.register_file(
            version=dataset_version,
            kind=DatasetKind.LICENSED,
            path=csv_path,
            timestamp_column="timestamp",
            zones=zones,
            metadata={"exchange": "NSE", "asset": "NIFTY-DEMO", "frequency": "1m"},
        )
        print(f"[+] Registered licensed dataset: {dataset_version} (SHA256: {dataset_record.sha256[:12]}...)")
    except Exception as exc:
        print(f"[-] Dataset registration note: {exc}")
        dataset_record = ctx.dataset_registry.get(dataset_version)

    # 2. Register Strategy Spec
    spec = StrategySpec(
        strategy_version="1.0.0",
        feature_version="1.0.0",
        signal_name="current_bar_momentum",
        parameters={"calendar_session_bars": 30, "session_window": [0.0, 1.0]},
    )
    norm_spec = normalize_strategy_spec(spec)
    strat_id = derive_strategy_id(norm_spec)
    spec_hash = hashlib.sha256(norm_spec.canonical_json().encode("utf-8")).hexdigest()

    try:
        ctx.strategy_registry.register_strategy(norm_spec)
        print(f"[+] Registered strategy spec: {strat_id}")
    except Exception:
        print(f"[*] Strategy {strat_id} already registered.")

    # 3. Advance Lifecycle: IDEA -> RESEARCH -> VALIDATION
    try:
        ctx.strategy_registry.transition_state(strat_id, StrategyLifecycleState.RESEARCH, reason="Initial demo research")
        ctx.strategy_registry.transition_state(strat_id, StrategyLifecycleState.VALIDATION, reason="Entering statistical validation")
    except Exception:
        pass

    # 4. Insert Authoritative Qualification Record
    qual_id = "QUAL-DEMO-2026-001"
    now_iso = datetime.now(timezone.utc).isoformat()

    qual_rec = StrategyQualificationRecord.create(
        qualification_id=qual_id,
        strategy_id=strat_id,
        strategy_spec_hash=spec_hash,
        dataset_version=dataset_version,
        dataset_sha256=dataset_record.sha256,
        split_manifest_version="manifest-v1",
        research_protocol_version="proto-v1",
        population_hash="pop-demo-2026",
        effective_trial_count=24.5,
        observed_sharpe=2.14,
        dsr=0.982,
        trade_count=142,
        holdout_state="PASSED",
        robustness_status=RobustnessStatus.PASSED,
        final_status=ValidationStatus.PAPER_ELIGIBLE,
        reasons=["DSR threshold exceeded (0.982 > 0.95)", "Holdout verified clean"],
        created_at=now_iso,
    )
    qual_hash = qual_rec.record_hash

    try:
        ctx.qualification_ledger.record_qualification(qual_rec)
        print(f"[+] Saved qualification record {qual_id} ({qual_hash[:12]}...)")
    except Exception as exc:
        print(f"[*] Qualification record notice: {exc}")

    # 5. Transition to PAPER_ELIGIBLE
    try:
        ctx.strategy_registry.transition_state(
            strat_id,
            StrategyLifecycleState.PAPER_ELIGIBLE,
            reason="Certified by StrategyValidationGate",
            qualification_record=qual_rec,
        )
        print(f"[+] Strategy {strat_id} transitioned to PAPER_ELIGIBLE")
    except Exception as exc:
        print(f"[*] PAPER_ELIGIBLE transition notice: {exc}")

    # 6. Evaluation Baseline
    report_hash = hashlib.sha256(f"replay-demo:{strat_id}:{now_iso}".encode("utf-8")).hexdigest()
    binding_hash = hashlib.sha256(f"{strat_id}:{qual_hash}:{report_hash}".encode("utf-8")).hexdigest()

    with ctx.get_core_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO evaluation_baselines (
                strategy_id, qualification_hash, baseline_replay_report_hash,
                baseline_dataset_version, baseline_dataset_sha256, baseline_split_zone,
                baseline_execution_policy, baseline_cost_schedule_hash, baseline_risk_config_hash,
                created_at, binding_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strat_id,
                qual_hash,
                report_hash,
                dataset_version,
                dataset_record.sha256,
                "FORWARD_PAPER",
                "next_bar_open_v1",
                "cost-sched-demo-2bps",
                "risk-default-v1",
                now_iso,
                binding_hash,
            ),
        )

        # Baseline Paper Report
        report_json = json.dumps(
            {
                "strategy_id": strat_id,
                "report_hash": report_hash,
                "total_bars": 60,
                "trade_count": 28,
                "final_equity": 105420.50,
                "net_pnl": 5420.50,
                "gross_pnl": 5780.00,
                "total_costs": 359.50,
                "sharpe_ratio": 2.14,
                "max_drawdown": 0.021,
            }
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO paper_reports (
                report_hash, strategy_id, qualification_id, created_at, report_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (report_hash, strat_id, qual_id, now_iso, report_json),
        )

    # 7. Transition to PAPER_ACTIVE
    try:
        ctx.strategy_registry.transition_state(
            strat_id,
            StrategyLifecycleState.PAPER_ACTIVE,
            reason="Approved with established baseline",
        )
        print(f"[+] Strategy {strat_id} transitioned to PAPER_ACTIVE")
    except Exception as exc:
        print(f"[*] PAPER_ACTIVE transition notice: {exc}")

    # 8. Seed Monitoring Snapshots, Degradation Events, Transitions & Feedback
    snapshot_id = f"SNAP-{strat_id[6:14]}-001"
    snapshot_hash = hashlib.sha256(f"snap:{strat_id}:{now_iso}".encode("utf-8")).hexdigest()
    with ctx.get_core_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO monitoring_snapshots (
                snapshot_id, strategy_id, qualification_hash, replay_report_hash,
                dataset_version, dataset_sha256, split_zone,
                monitoring_protocol_version, monitoring_config_hash,
                window_start_ts, window_end_ts, total_trades, net_pnl,
                max_drawdown_bps, realized_sharpe, realized_slippage_bps,
                cost_to_turnover_bps, risk_event_count, metrics_json,
                created_at, snapshot_hash, regime_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                strat_id,
                qual_hash,
                report_hash,
                dataset_version,
                dataset_record.sha256,
                "FORWARD_PAPER",
                "proto-v1",
                "cfg-hash-1",
                "2026-01-03 09:15:00",
                "2026-01-03 10:15:00",
                28,
                5420.50,
                210.0,
                2.08,
                1.2,
                3.4,
                0,
                json.dumps({"win_rate": 0.64, "profit_factor": 1.82}),
                now_iso,
                snapshot_hash,
                None,
            ),
        )

        event_hash = hashlib.sha256(f"deg:{strat_id}:{now_iso}".encode("utf-8")).hexdigest()
        conn.execute(
            """
            INSERT OR IGNORE INTO degradation_events (
                event_id, strategy_id, qualification_hash, snapshot_hash,
                baseline_replay_report_hash, rule_name, threshold_value,
                observed_value, monitoring_protocol_version, monitoring_config_hash,
                timestamp, details_json, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"DEG-{event_hash[:8]}",
                strat_id,
                qual_hash,
                snapshot_hash,
                report_hash,
                "DRAWDOWN_LIMIT",
                250.0,
                210.0,
                "proto-v1",
                "cfg-hash-1",
                now_iso,
                json.dumps({"message": "Approaching warning threshold"}),
                event_hash,
            ),
        )

        # Transition record
        trans_hash = hashlib.sha256(f"trans:{strat_id}:{now_iso}".encode("utf-8")).hexdigest()
        conn.execute(
            """
            INSERT OR IGNORE INTO paper_evaluation_transitions (
                transition_id, strategy_id, old_state, new_state, initiator,
                evidence_type, evidence_hash, reason, timestamp, transition_hash,
                qualification_hash, snapshot_hash, regime_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"TRN-{strat_id[6:14]}-001",
                strat_id,
                "PAPER_ELIGIBLE",
                "PAPER_ACTIVE",
                "GOVERNANCE_SYSTEM",
                "BASELINE_BINDING",
                binding_hash,
                "Approved with established baseline",
                now_iso,
                trans_hash,
                qual_hash,
                None,
                None,
            ),
        )

        # Research Feedback Record (Domain Model Factory)
        from quantmind.paper.evaluation.models import ResearchFeedbackRecord
        fb_rec = ResearchFeedbackRecord.create(
            strategy_id=strat_id,
            qualification_hash=qual_hash,
            degradation_event_hash=event_hash,
            dataset_version=dataset_version,
            failure_mode="DRAWDOWN_LIMIT",
            realized_sharpe=2.08,
            drawdown_expansion_ratio=1.42,
            realized_slippage_bps=1.2,
            empirical_notes="Approaching warning threshold in session 2",
            created_at=now_iso,
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO research_feedback (
                feedback_id, strategy_id, qualification_hash, degradation_event_hash,
                dataset_version, failure_mode, realized_sharpe,
                drawdown_expansion_ratio, realized_slippage_bps,
                empirical_notes, created_at, feedback_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fb_rec.derived_feedback_id,
                fb_rec.strategy_id,
                fb_rec.qualification_hash,
                fb_rec.degradation_event_hash,
                fb_rec.dataset_version,
                fb_rec.failure_mode,
                fb_rec.realized_sharpe,
                fb_rec.drawdown_expansion_ratio,
                fb_rec.realized_slippage_bps,
                fb_rec.empirical_notes,
                fb_rec.created_at,
                fb_rec.feedback_hash,
            ),
        )

    # 9. Seed Paper Orders & Fills
    order_id = f"ORD-{strat_id[6:14]}-001"
    fill_id = f"FILL-{strat_id[6:14]}-001"
    with ctx.get_core_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO paper_orders (
                order_id, strategy_id, qualification_id, symbol, side,
                quantity, order_type, signal_timestamp, submit_timestamp,
                requested_price, status, rejection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                strat_id,
                qual_id,
                "NIFTY-DEMO",
                1,  # 1 for BUY
                50,
                "MARKET",
                now_iso,
                now_iso,
                21550.0,
                "FILLED",
                None,
            ),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO paper_fills (
                fill_id, order_id, strategy_id, symbol, fill_timestamp,
                fill_price, quantity, side, cost, slippage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill_id,
                order_id,
                strat_id,
                "NIFTY-DEMO",
                now_iso,
                21550.0,
                50,
                1,
                15.20,
                1.50,
            ),
        )
        conn.execute(
            """
            INSERT INTO paper_positions (
                timestamp, strategy_id, symbol, quantity, entry_price,
                current_price, realized_pnl, unrealized_pnl, fees_costs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                strat_id,
                "NIFTY-DEMO",
                50,
                21550.0,
                21595.0,
                3170.50,
                2250.0,
                15.20,
            ),
        )

    print("\n============================================================")
    print("DEMO SEEDING COMPLETED SUCCESSFULLY")
    print("============================================================")
    print(f"Strategy ID:        {strat_id} (PAPER_ACTIVE)")
    print(f"Dataset Version:    {dataset_version} (LICENSED)")
    print(f"Baseline Report:    {report_hash[:16]}...")
    print(f"Snapshot Hash:      {snapshot_hash[:16]}...")
    print("------------------------------------------------------------")
    print("Demo Credentials (APP_ENV=DEMO):")
    print("  Admin:       demo_admin      / QuantMindDemoAdmin2026!")
    print("  Researcher:  demo_researcher / QuantMindDemoResearch2026!")
    print("  Viewer:      demo_viewer     / QuantMindDemoViewer2026!")
    print("============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed QuantMind Demo Data")
    parser.add_argument("--force", action="store_true", help="Force seeding even if APP_ENV != DEMO")
    args = parser.parse_args()
    seed_demo_data(force=args.force)
