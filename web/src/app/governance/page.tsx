"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { DataTable } from "@/components/ui/DataTable";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import { ShieldCheck, RefreshCw } from "lucide-react";

export default function GovernancePage() {
  const { data: transitions, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["governance-transitions"],
    queryFn: () => api.get<any[]>("/governance/transitions"),
  });

  return (
    <AppShell
      title="Governance Audit Trail"
      subtitle="Immutable state transitions recorded in EvaluationLedger with cryptographic evidence bindings"
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
        data={transitions || []}
        keyExtractor={(item) => item.transition_hash}
        emptyMessage={
          isLoading
            ? "Loading governance audit trail..."
            : "No governance state transitions recorded."
        }
        columns={[
          {
            header: "Strategy ID",
            accessorKey: "strategy_id",
            className: "font-mono font-medium text-white",
          },
          {
            header: "Transition",
            cell: (item) => (
              <div className="flex items-center space-x-1.5">
                <StatusBadge status={item.old_state} />
                <span className="text-slate-500">→</span>
                <StatusBadge status={item.new_state} />
              </div>
            ),
          },
          {
            header: "Evidence Type",
            accessorKey: "evidence_type",
            className: "font-mono text-cyan-400 text-xs",
          },
          {
            header: "Evidence Hash",
            cell: (item) => <HashViewer hash={item.evidence_hash} length={10} />,
          },
          {
            header: "Transition Digest",
            cell: (item) => <HashViewer hash={item.transition_hash} length={10} />,
          },
          {
            header: "Initiator",
            accessorKey: "initiator",
            className: "font-mono text-slate-400 text-xs",
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
    </AppShell>
  );
}
