"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/layout/AppShell";
import { DataTable } from "@/components/ui/DataTable";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import clsx from "clsx";
import { RefreshCw } from "lucide-react";

const STATES = [
  "ALL",
  "IDEA",
  "RESEARCH",
  "VALIDATION",
  "REJECTED",
  "PAPER_ELIGIBLE",
  "PAPER_ACTIVE",
  "DEGRADED",
  "RETIRED",
];

export default function StrategiesPage() {
  const router = useRouter();
  const [selectedState, setSelectedState] = useState("ALL");

  const queryUrl =
    selectedState === "ALL"
      ? "/strategies"
      : `/strategies?state=${encodeURIComponent(selectedState)}`;

  const { data, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["strategies-list", selectedState],
    queryFn: () => api.get<any[]>(queryUrl),
  });

  return (
    <AppShell
      title="Strategy Registry"
      subtitle="Deterministic strategy specifications managed across the 8 lifecycle states"
      action={
        <button
          type="button"
          onClick={() => refetch()}
          disabled={isRefetching}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-hover text-xs font-medium text-slate-300 hover:text-white transition-colors"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefetching ? "animate-spin" : ""}`} />
          <span>Refresh</span>
        </button>
      }
    >
      {/* State Filter Pills */}
      <div className="flex items-center space-x-1.5 overflow-x-auto pb-2">
        {STATES.map((state) => {
          const isSelected = selectedState === state;
          return (
            <button
              key={state}
              type="button"
              onClick={() => setSelectedState(state)}
              className={clsx(
                "px-3 py-1 rounded-lg text-xs font-mono font-medium transition-colors border",
                isSelected
                  ? "bg-primary text-white border-primary"
                  : "bg-surface text-slate-400 border-border hover:bg-surface-hover hover:text-white"
              )}
            >
              {state}
            </button>
          );
        })}
      </div>

      {/* Strategies Data Table */}
      <DataTable
        data={data || []}
        keyExtractor={(item) => item.strategy_id}
        onRowClick={(item) => router.push(`/strategies/${item.strategy_id}`)}
        emptyMessage={
          isLoading
            ? "Loading strategies..."
            : `No strategies found in ${selectedState} state.`
        }
        columns={[
          {
            header: "Strategy ID",
            accessorKey: "strategy_id",
            className: "font-mono font-semibold text-white",
          },
          {
            header: "Lifecycle State",
            cell: (item) => <StatusBadge status={item.state} />,
          },
          {
            header: "Qualification Binding",
            cell: (item) => (
              <HashViewer hash={item.qualification_hash} length={12} />
            ),
          },
          {
            header: "Created At",
            cell: (item) => (
              <span className="font-mono text-slate-400 text-xs">
                {item.created_at ? item.created_at.slice(0, 19).replace("T", " ") : "—"}
              </span>
            ),
          },
          {
            header: "Updated At",
            cell: (item) => (
              <span className="font-mono text-slate-400 text-xs">
                {item.updated_at ? item.updated_at.slice(0, 19).replace("T", " ") : "—"}
              </span>
            ),
          },
        ]}
      />
    </AppShell>
  );
}
