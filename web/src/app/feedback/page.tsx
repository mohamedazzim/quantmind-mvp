"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { DataTable } from "@/components/ui/DataTable";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import { MessageSquareShare, RefreshCw } from "lucide-react";

export default function FeedbackPage() {
  const { data: records, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["feedback-records"],
    queryFn: () => api.get<any[]>("/feedback"),
  });

  return (
    <AppShell
      title="Research Feedback Bridge"
      subtitle="Empirical failure modes and performance degradation feedback (PRD v4.0 M7)"
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
      <DataTable
        data={records || []}
        keyExtractor={(item) => item.feedback_hash}
        emptyMessage={
          isLoading
            ? "Loading feedback packages..."
            : "No research feedback records generated. All strategies operating within degradation thresholds."
        }
        columns={[
          {
            header: "Strategy ID",
            accessorKey: "strategy_id",
            className: "font-mono font-medium text-white",
          },
          {
            header: "Failure Mode",
            accessorKey: "failure_mode",
            className: "font-mono text-amber-400 font-semibold",
          },
          {
            header: "Drawdown Expansion",
            cell: (item) => (
              <span className="font-mono text-white">
                {item.drawdown_expansion_ratio !== undefined
                  ? `${Number(item.drawdown_expansion_ratio).toFixed(2)}x`
                  : "—"}
              </span>
            ),
          },
          {
            header: "Slippage",
            cell: (item) => (
              <span className="font-mono text-slate-300">
                {item.realized_slippage_bps !== undefined
                  ? `${Number(item.realized_slippage_bps).toFixed(1)} bps`
                  : "—"}
              </span>
            ),
          },
          {
            header: "Feedback Digest",
            cell: (item) => <HashViewer hash={item.feedback_hash} length={10} />,
          },
          {
            header: "Degradation Event",
            cell: (item) => <HashViewer hash={item.degradation_event_hash} length={10} />,
          },
          {
            header: "Derived Task",
            cell: (item) => <HashViewer hash={item.task_id} length={10} />,
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
    </AppShell>
  );
}
