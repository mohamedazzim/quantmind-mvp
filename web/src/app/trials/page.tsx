"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/layout/AppShell";
import { DataTable } from "@/components/ui/DataTable";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import { RefreshCw } from "lucide-react";

export default function TrialsPage() {
  const router = useRouter();
  const { data: trials, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["trials-list"],
    queryFn: () => api.get<any[]>("/trials"),
  });

  return (
    <AppShell
      title="Trial Ledger"
      subtitle="Immutable record of research trials and multiple-testing population accounting"
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
        data={trials || []}
        keyExtractor={(item) => item.trial_id}
        onRowClick={(item) => router.push(`/trials/${item.trial_id}`)}
        emptyMessage={isLoading ? "Loading trials..." : "No research trials recorded in ledger."}
        columns={[
          {
            header: "Trial ID",
            accessorKey: "trial_id",
            className: "font-mono font-semibold text-white",
          },
          {
            header: "Strategy ID",
            cell: (item) => (
              <span className="font-mono text-slate-300">{item.strategy_id}</span>
            ),
          },
          {
            header: "Dataset / Zone",
            cell: (item) => (
              <span className="font-mono text-slate-400">
                {item.dataset_version} ({item.split_zone})
              </span>
            ),
          },
          {
            header: "Status",
            cell: (item) => <StatusBadge status={item.status} />,
          },
          {
            header: "Runtime",
            cell: (item) => (
              <span className="font-mono text-slate-400 text-xs">
                {item.actual_runtime_minutes ? `${item.actual_runtime_minutes.toFixed(2)}m` : "—"}
              </span>
            ),
          },
          {
            header: "Started",
            cell: (item) => (
              <span className="font-mono text-slate-500 text-xs">
                {item.timestamp_started ? item.timestamp_started.slice(0, 19).replace("T", " ") : "—"}
              </span>
            ),
          },
        ]}
      />
    </AppShell>
  );
}
