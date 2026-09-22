"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { DataTable } from "@/components/ui/DataTable";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import { CheckCircle2, RefreshCw, ShieldAlert } from "lucide-react";

export default function QualificationPage() {
  const { data: quals, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["qualifications-list"],
    queryFn: () => api.get<any[]>("/qualification"),
  });

  return (
    <AppShell
      title="Strategy Qualification"
      subtitle="Statistical qualification gates: Deflated Sharpe Ratio (DSR), EICT-CORR-1, and sealed holdout verification"
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
        data={quals || []}
        keyExtractor={(item) => item.qualification_id}
        emptyMessage={
          isLoading
            ? "Loading qualifications..."
            : "No certified qualification records registered in ledger."
        }
        columns={[
          {
            header: "Strategy ID",
            accessorKey: "strategy_id",
            className: "font-mono font-semibold text-white",
          },
          {
            header: "Record Hash",
            cell: (item) => <HashViewer hash={item.record_hash} length={12} />,
          },
          {
            header: "DSR",
            cell: (item) => (
              <span className="font-mono text-emerald-400 font-bold">
                {Number(item.dsr).toFixed(3)}
              </span>
            ),
          },
          {
            header: "Observed Sharpe",
            cell: (item) => (
              <span className="font-mono text-slate-200">
                {Number(item.observed_sharpe).toFixed(2)}
              </span>
            ),
          },
          {
            header: "Holdout State",
            cell: (item) => (
              <span
                className={`font-mono text-xs ${
                  item.holdout_state === "PASSED" ? "text-emerald-400" : "text-amber-400"
                }`}
              >
                {item.holdout_state}
              </span>
            ),
          },
          {
            header: "Gate Status",
            cell: (item) => <StatusBadge status={item.final_status} />,
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
