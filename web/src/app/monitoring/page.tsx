"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { DataTable } from "@/components/ui/DataTable";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import { Activity, AlertTriangle, RefreshCw } from "lucide-react";
import clsx from "clsx";

interface DegradationRecord {
  strategy_id: string;
  rule_name: string;
  observed_value: number;
  threshold_value: number;
  event_hash: string;
  snapshot_hash: string;
  timestamp?: string;
}

interface SnapshotRecord {
  strategy_id: string;
  snapshot_hash: string;
  max_drawdown_bps: number;
  realized_sharpe?: number;
  total_trades: number;
  created_at?: string;
}

export default function MonitoringPage() {
  const [tab, setTab] = useState<"degradations" | "snapshots">("degradations");

  const { data, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["monitoring-overview"],
    queryFn: () => api.get("/monitoring"),
  });

  return (
    <AppShell
      title="Performance Monitoring & Degradation Detection"
      subtitle="Rolling evaluation windows and deterministic degradation event capture"
      action={
        <button
          type="button"
          onClick={() => refetch()}
          disabled={isRefetching}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-hover text-xs font-medium text-slate-300 hover:text-white"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefetching ? "animate-spin" : ""}`} />
          <span>Refresh</span>
        </button>
      }
    >
      {/* Tabs */}
      <div className="border-b border-border flex space-x-6 text-xs font-medium">
        <button
          type="button"
          onClick={() => setTab("degradations")}
          className={clsx(
            "pb-3 border-b-2 transition-colors flex items-center space-x-2",
            tab === "degradations"
              ? "border-primary text-primary font-semibold"
              : "border-transparent text-slate-400 hover:text-white"
          )}
        >
          <AlertTriangle className="h-4 w-4 text-amber-400" />
          <span>Degradation Events ({data?.total_degradations ?? 0})</span>
        </button>
        <button
          type="button"
          onClick={() => setTab("snapshots")}
          className={clsx(
            "pb-3 border-b-2 transition-colors flex items-center space-x-2",
            tab === "snapshots"
              ? "border-primary text-primary font-semibold"
              : "border-transparent text-slate-400 hover:text-white"
          )}
        >
          <Activity className="h-4 w-4 text-cyan-400" />
          <span>Monitoring Snapshots ({data?.total_snapshots ?? 0})</span>
        </button>
      </div>

      {tab === "degradations" && (
        <DataTable<DegradationRecord>
          data={data?.recent_degradations || []}
          keyExtractor={(item) => item.event_hash}
          emptyMessage="No degradation events detected. All monitored paper metrics remain within acceptable bounds."
          columns={[
            {
              header: "Strategy ID",
              accessorKey: "strategy_id",
              className: "font-mono font-medium text-white",
            },
            {
              header: "Breach Rule",
              accessorKey: "rule_name",
              className: "font-mono text-amber-400 font-semibold",
            },
            {
              header: "Observed",
              cell: (item) => (
                <span className="font-mono text-white font-bold">{item.observed_value}</span>
              ),
            },
            {
              header: "Threshold Limit",
              cell: (item) => (
                <span className="font-mono text-slate-400">{item.threshold_value}</span>
              ),
            },
            {
              header: "Event Digest",
              cell: (item) => <HashViewer hash={item.event_hash} length={12} />,
            },
            {
              header: "Parent Snapshot",
              cell: (item) => <HashViewer hash={item.snapshot_hash} length={10} />,
            },
            {
              header: "Timestamp",
              cell: (item) => (
                <span className="font-mono text-slate-500 text-xs">
                  {item.timestamp ? item.timestamp.slice(0, 19).replace("T", " ") : "—"}
                </span>
              ),
            },
          ]}
        />
      )}

      {tab === "snapshots" && (
        <DataTable<SnapshotRecord>
          data={data?.recent_snapshots || []}
          keyExtractor={(item) => item.snapshot_hash}
          emptyMessage="No evaluation window snapshots recorded."
          columns={[
            {
              header: "Strategy ID",
              accessorKey: "strategy_id",
              className: "font-mono font-medium text-white",
            },
            {
              header: "Snapshot Hash",
              cell: (item) => <HashViewer hash={item.snapshot_hash} length={10} />,
            },
            {
              header: "Max Drawdown",
              cell: (item) => (
                <span className="font-mono text-slate-300">
                  {Number(item.max_drawdown_bps).toFixed(0)} bps
                </span>
              ),
            },
            {
              header: "Realized Sharpe",
              cell: (item) => (
                <span className="font-mono text-slate-300">
                  {item.realized_sharpe ? Number(item.realized_sharpe).toFixed(2) : "—"}
                </span>
              ),
            },
            {
              header: "Trades",
              accessorKey: "total_trades",
              className: "font-mono text-slate-400",
            },
            {
              header: "Created",
              cell: (item) => (
                <span className="font-mono text-slate-500 text-xs">
                  {item.created_at ? item.created_at.slice(0, 19).replace("T", " ") : "—"}
                </span>
              ),
            },
          ]}
        />
      )}
    </AppShell>
  );
}
