"""Audit Provenance Graph Adapter across the 5-Type Lineage Classification (PRD v4.0 Productization)."""

from __future__ import annotations

from typing import Any
from quantmind.app.context import AppContext
from quantmind.research_integrity.feedback_bridge import derive_feedback_task_id


class AuditAdapter:
    """Traverses and constructs the authoritative provenance graph classifying each edge (Type A-E)."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def get_provenance_graph(self, identifier: str) -> dict[str, Any]:
        """Given a strategy ID, record hash, or entity ID, builds the connected DAG of evidence."""
        conn = self.ctx.get_core_connection()
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        node_ids: set[str] = set()

        def add_node(
            nid: str,
            label: str,
            ntype: str,
            *,
            identifier_str: str,
            hash_val: str = "",
            timestamp: str = "",
            source: str = "QuantMind Core",
            status: str = "VERIFIED",
            details: dict[str, Any] | None = None,
        ) -> None:
            if nid not in node_ids:
                node_ids.add(nid)
                nodes.append({
                    "id": nid,
                    "label": label,
                    "type": ntype,
                    "identifier": identifier_str,
                    "hash": hash_val,
                    "timestamp": timestamp,
                    "source": source,
                    "status": status,
                    "details": details or {},
                })

        def add_edge(
            source: str,
            target: str,
            edge_type_code: str,
            label: str,
            verification_mechanism: str,
        ) -> None:
            edges.append({
                "id": f"{source}->{target}",
                "source": source,
                "target": target,
                "edge_type": f"Type {edge_type_code}",
                "relationship_type": edge_type_code,
                "label": label,
                "verification_mechanism": verification_mechanism,
            })

        try:
            # 1. Resolve strategy_id from identifier
            strategy_id: str | None = None
            strat_row = conn.execute(
                "SELECT * FROM strategies WHERE strategy_id = ?", (identifier,)
            ).fetchone()

            if strat_row:
                strategy_id = identifier
            else:
                # Try qualification hash / ID
                q = conn.execute(
                    "SELECT strategy_id FROM strategy_qualifications WHERE record_hash = ? OR qualification_id = ?",
                    (identifier, identifier),
                ).fetchone()
                if q:
                    strategy_id = q["strategy_id"]

                # Try replay report hash
                if not strategy_id:
                    rep = conn.execute(
                        "SELECT strategy_id FROM paper_reports WHERE report_hash = ?",
                        (identifier,),
                    ).fetchone()
                    if rep:
                        strategy_id = rep["strategy_id"]

                # Try regime hash
                if not strategy_id:
                    reg = conn.execute(
                        "SELECT strategy_id FROM paper_evaluation_regimes WHERE regime_hash = ?",
                        (identifier,),
                    ).fetchone()
                    if reg:
                        strategy_id = reg["strategy_id"]

                # Try snapshot hash
                if not strategy_id:
                    snap = conn.execute(
                        "SELECT strategy_id FROM monitoring_snapshots WHERE snapshot_hash = ? OR snapshot_id = ?",
                        (identifier, identifier),
                    ).fetchone()
                    if snap:
                        strategy_id = snap["strategy_id"]

                # Try degradation event hash
                if not strategy_id:
                    deg = conn.execute(
                        "SELECT strategy_id FROM degradation_events WHERE event_hash = ?",
                        (identifier,),
                    ).fetchone()
                    if deg:
                        strategy_id = deg["strategy_id"]

                # Try transition hash
                if not strategy_id:
                    trans = conn.execute(
                        "SELECT strategy_id FROM paper_evaluation_transitions WHERE transition_hash = ? OR transition_id = ?",
                        (identifier, identifier),
                    ).fetchone()
                    if trans:
                        strategy_id = trans["strategy_id"]

                # Try feedback hash
                if not strategy_id:
                    fb = conn.execute(
                        "SELECT strategy_id FROM research_feedback WHERE feedback_hash = ? OR feedback_id = ?",
                        (identifier, identifier),
                    ).fetchone()
                    if fb:
                        strategy_id = fb["strategy_id"]

            if not strategy_id:
                return {
                    "nodes": [],
                    "edges": [],
                    "error": f"No provenance record found for identifier '{identifier}'",
                }

            # Strategy Record
            strat_info = conn.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,)).fetchone()
            s_state = strat_info["state"] if strat_info else "UNKNOWN"
            s_qual_h = strat_info["qualification_hash"] if strat_info else ""
            s_ts = strat_info["updated_at"] if strat_info else ""
            spec_dict = {}
            if strat_info and "spec_json" in strat_info.keys():
                try:
                    spec_dict = json.loads(strat_info["spec_json"])
                except Exception:
                    pass
            add_node(
                strategy_id,
                f"Strategy ({s_state})",
                "STRATEGY",
                identifier_str=strategy_id,
                hash_val="",
                timestamp=s_ts,
                source="StrategyRegistry",
                status="VERIFIED",
                details={
                    "strategy_id": strategy_id,
                    "state": s_state,
                    "qualification_hash": s_qual_h,
                    "signal_name": spec_dict.get("signal_name", ""),
                },
            )

            # Qualification Records
            qual_rows = conn.execute(
                "SELECT * FROM strategy_qualifications WHERE strategy_id = ?", (strategy_id,)
            ).fetchall()
            for qr in qual_rows:
                qid = qr["qualification_id"]
                qhash = qr["record_hash"]
                add_node(
                    qhash,
                    f"Qualification ({qr['final_status']})",
                    "QUALIFICATION",
                    identifier_str=qid,
                    hash_val=qhash,
                    timestamp=qr["created_at"],
                    source="QualificationLedger",
                    status="VERIFIED",
                    details={
                        "qualification_id": qid,
                        "record_hash": qhash,
                        "dsr": float(qr["dsr"]),
                        "observed_sharpe": float(qr["observed_sharpe"]),
                        "holdout_state": qr["holdout_state"],
                        "dataset_version": qr["dataset_version"],
                    },
                )
                # Type A Edge: Qualification -> Strategy (Cryptographic Hash Binding)
                add_edge(
                    qhash,
                    strategy_id,
                    "A",
                    "Authorizes State",
                    "SHA-256 qualification_hash stored in StrategyRegistry",
                )

                # Dataset node
                ds_ver = qr["dataset_version"]
                ds_hash = qr["dataset_sha256"]
                add_node(
                    ds_hash,
                    f"Dataset: {ds_ver}",
                    "DATASET",
                    identifier_str=ds_ver,
                    hash_val=ds_hash,
                    timestamp=qr["created_at"],
                    source="DatasetRegistry",
                    status="VERIFIED",
                    details={"dataset_version": ds_ver, "dataset_sha256": ds_hash},
                )
                # Type A Edge: Dataset -> Qualification
                add_edge(
                    ds_hash,
                    qhash,
                    "A",
                    "Binds Checksum",
                    "SHA-256 dataset digest bound in QualificationRecord",
                )

            # Baselines & Replay Reports
            base_rows = conn.execute(
                "SELECT * FROM evaluation_baselines WHERE strategy_id = ?", (strategy_id,)
            ).fetchall()
            for b in base_rows:
                b_hash = b["binding_hash"]
                add_node(
                    b_hash,
                    "Baseline Binding",
                    "BASELINE",
                    identifier_str=b_hash[:16],
                    hash_val=b_hash,
                    timestamp=b["created_at"],
                    source="EvaluationLedger",
                    status="VERIFIED",
                    details={
                        "binding_hash": b_hash,
                        "baseline_dataset_version": b["baseline_dataset_version"],
                    },
                )
                parent_qual = b["qualification_hash"]
                if parent_qual in node_ids:
                    # Type A Edge: Qual -> Baseline
                    add_edge(
                        parent_qual,
                        b_hash,
                        "A",
                        "Binds Qualification",
                        "SHA-256 qualification_hash embedded in baseline binding digest",
                    )
                else:
                    # Missing parent qualification -> flag broken provenance
                    missing_id = f"MISSING-QUAL-{parent_qual[:8]}"
                    add_node(
                        missing_id,
                        f"Missing Qual ({parent_qual[:8]})",
                        "QUALIFICATION",
                        identifier_str=parent_qual,
                        hash_val=parent_qual,
                        source="Unknown",
                        status="BROKEN PROVENANCE",
                    )
                    add_edge(
                        missing_id,
                        b_hash,
                        "A",
                        "Broken Qualification Link",
                        "Referenced qualification_hash not found in ledger",
                    )

                # Replay Report
                rep_hash = b["baseline_replay_report_hash"]
                rep_row = conn.execute(
                    "SELECT * FROM paper_reports WHERE report_hash = ?", (rep_hash,)
                ).fetchone()
                if rep_row:
                    import json
                    rep_data = json.loads(rep_row["report_json"]) if rep_row["report_json"] else {}
                    add_node(
                        rep_hash,
                        f"Replay Report ({float(rep_data.get('net_pnl', 0.0)):.1f} PnL)",
                        "REPLAY_REPORT",
                        identifier_str=rep_hash[:16],
                        hash_val=rep_hash,
                        timestamp=rep_row["created_at"],
                        source="PaperLedger",
                        status="VERIFIED",
                        details={
                            "report_hash": rep_hash,
                            "net_pnl": float(rep_data.get("net_pnl", 0.0)),
                            "max_drawdown_bps": float(rep_data.get("max_drawdown_bps", 0.0)),
                        },
                    )
                    # Type B Edge: Strategy -> Replay Report (Referential FK)
                    add_edge(
                        strategy_id,
                        rep_hash,
                        "B",
                        "Executes Fixture Replay",
                        "strategy_id foreign reference executed under PaperExecutionLifecycleContext",
                    )
                    # Type A Edge: Replay Report -> Baseline (Cryptographic digest)
                    add_edge(
                        rep_hash,
                        b_hash,
                        "A",
                        "Certifies Baseline",
                        "baseline_replay_report_hash matches replay digest",
                    )

            # Regimes
            reg_rows = conn.execute(
                "SELECT * FROM paper_evaluation_regimes WHERE strategy_id = ?", (strategy_id,)
            ).fetchall()
            for r in reg_rows:
                r_hash = r["regime_hash"]
                add_node(
                    r_hash,
                    "Evaluation Regime",
                    "REGIME",
                    identifier_str=r_hash[:16],
                    hash_val=r_hash,
                    timestamp=r["created_at"],
                    source="EvaluationLedger",
                    status="VERIFIED",
                    details={
                        "regime_hash": r_hash,
                        "monitoring_protocol_version": r["monitoring_protocol_version"],
                    },
                )
                if r["baseline_replay_report_hash"] in node_ids:
                    # Type A Edge: Replay Report -> Regime
                    add_edge(
                        r["baseline_replay_report_hash"],
                        r_hash,
                        "A",
                        "Anchors Forward Regime",
                        "baseline_replay_report_hash bound in regime digest",
                    )

            # Snapshots
            snap_rows = conn.execute(
                "SELECT * FROM monitoring_snapshots WHERE strategy_id = ? ORDER BY created_at DESC LIMIT 10",
                (strategy_id,),
            ).fetchall()
            for s in snap_rows:
                s_hash = s["snapshot_hash"]
                add_node(
                    s_hash,
                    f"Snapshot (DD: {s['max_drawdown_bps']:.0f} bps)",
                    "SNAPSHOT",
                    identifier_str=s["snapshot_id"],
                    hash_val=s_hash,
                    timestamp=s["created_at"],
                    source="EvaluationLedger",
                    status="VERIFIED",
                    details={
                        "snapshot_id": s["snapshot_id"],
                        "snapshot_hash": s_hash,
                        "max_drawdown_bps": float(s["max_drawdown_bps"]),
                    },
                )
                if s["regime_hash"] in node_ids:
                    # Type A Edge: Regime -> Snapshot
                    add_edge(
                        s["regime_hash"],
                        s_hash,
                        "A",
                        "Context Bounds",
                        "regime_hash embedded in snapshot digest",
                    )

            # Degradation Events
            deg_rows = conn.execute(
                "SELECT * FROM degradation_events WHERE strategy_id = ?", (strategy_id,)
            ).fetchall()
            for d in deg_rows:
                d_hash = d["event_hash"]
                add_node(
                    d_hash,
                    f"Degradation: {d['rule_name']}",
                    "DEGRADATION",
                    identifier_str=d_hash[:16],
                    hash_val=d_hash,
                    timestamp=d["timestamp"],
                    source="EvaluationLedger",
                    status="VERIFIED",
                    details={
                        "event_hash": d_hash,
                        "rule_name": d["rule_name"],
                        "threshold_value": float(d["threshold_value"]),
                        "observed_value": float(d["observed_value"]),
                    },
                )
                if d["snapshot_hash"] in node_ids:
                    # Type A Edge: Snapshot -> Degradation Event
                    add_edge(
                        d["snapshot_hash"],
                        d_hash,
                        "A",
                        "Breaches Threshold",
                        "snapshot_hash cryptographically anchored in degradation event",
                    )

            # Governance Transitions
            trans_rows = conn.execute(
                "SELECT * FROM paper_evaluation_transitions WHERE strategy_id = ?", (strategy_id,)
            ).fetchall()
            for t in trans_rows:
                t_hash = t["transition_hash"]
                add_node(
                    t_hash,
                    f"Transition -> {t['new_state']}",
                    "TRANSITION",
                    identifier_str=t["transition_id"],
                    hash_val=t_hash,
                    timestamp=t["timestamp"],
                    source="EvaluationLedger",
                    status="VERIFIED",
                    details={
                        "transition_id": t["transition_id"],
                        "transition_hash": t_hash,
                        "old_state": t["old_state"],
                        "new_state": t["new_state"],
                        "reason": t["reason"],
                    },
                )
                ev_hash = t["evidence_hash"]
                if ev_hash in node_ids:
                    # Type A Edge: Evidence -> Transition
                    add_edge(
                        ev_hash,
                        t_hash,
                        "A",
                        "Transition Evidence",
                        f"evidence_type={t['evidence_type']} matches evidence_hash digest",
                    )
                # Type B Edge: Transition -> Strategy (Referential mutation)
                add_edge(
                    t_hash,
                    strategy_id,
                    "B",
                    "Governs Lifecycle State",
                    "Foreign key mutation in StrategyRegistry verified against EvaluationLedger",
                )

            # Research Feedback Records
            fb_rows = conn.execute(
                "SELECT * FROM research_feedback WHERE strategy_id = ?", (strategy_id,)
            ).fetchall()
            for f in fb_rows:
                f_hash = f["feedback_hash"]
                add_node(
                    f_hash,
                    f"Feedback: {f['failure_mode']}",
                    "FEEDBACK",
                    identifier_str=f["feedback_id"],
                    hash_val=f_hash,
                    timestamp=f["created_at"],
                    source="EvaluationLedger",
                    status="VERIFIED",
                    details={
                        "feedback_id": f["feedback_id"],
                        "feedback_hash": f_hash,
                        "failure_mode": f["failure_mode"],
                        "drawdown_expansion_ratio": float(f["drawdown_expansion_ratio"]),
                    },
                )
                if f["degradation_event_hash"] in node_ids:
                    # Type A Edge: Degradation -> Feedback
                    add_edge(
                        f["degradation_event_hash"],
                        f_hash,
                        "A",
                        "Feedback Capture",
                        "degradation_event_hash cryptographically bound in feedback record",
                    )

                # Virtual Feedback Task
                task_id = derive_feedback_task_id(f_hash)
                add_node(
                    task_id,
                    f"Research Task: {task_id[:16]}",
                    "RESEARCH_TASK",
                    identifier_str=task_id,
                    hash_val="",
                    timestamp=f["created_at"],
                    source="FeedbackBridge",
                    status="VERIFIED",
                    details={
                        "task_id": task_id,
                        "source_feedback_hash": f_hash,
                        "failure_mode": f["failure_mode"],
                    },
                )
                # Type C Edge: Feedback -> Task (Deterministic semantic identity)
                add_edge(
                    f_hash,
                    task_id,
                    "C",
                    "Derives Zero-Trial Task",
                    "Deterministic task_id derived from feedback_hash via derive_feedback_task_id",
                )

            return {
                "strategy_id": strategy_id,
                "root_id": strategy_id,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }
        finally:
            conn.close()
