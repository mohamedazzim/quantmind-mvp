"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { DataTable } from "@/components/ui/DataTable";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import { Database, RefreshCw, CheckCircle2 } from "lucide-react";

export default function DatasetsPage() {
  const { data: datasets, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["datasets-list"],
    queryFn: () => api.get<any[]>("/datasets"),
  });

  const { data: enums } = useQuery({
    queryKey: ["datasets-enums"],
    queryFn: () => api.get<any>("/datasets/enums"),
  });

  return (
    <AppShell
      title="Dataset Registry"
      subtitle="Versioned and checksum-verified production and research datasets"
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
      {/* Authoritative Split Zones Info Banner */}
      <div className="rounded-xl border border-border bg-surface p-5 space-y-2">
        <div className="flex items-center space-x-2 text-white text-xs font-semibold uppercase tracking-wider">
          <Database className="h-4 w-4 text-primary" />
          <span>Authoritative Core Split Zones (quantmind.data.splits.SplitZone)</span>
        </div>
        <div className="flex flex-wrap gap-2 pt-1">
          {(enums?.split_zones || ["RESEARCH", "VALIDATION", "FINAL_HOLDOUT", "FORWARD_PAPER"]).map(
            (zone: string) => (
              <span
                key={zone}
                className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-xs font-mono text-cyan-400"
              >
                {zone}
              </span>
            )
          )}
        </div>
      </div>

      {/* Datasets Table */}
      <DataTable
        data={datasets || []}
        keyExtractor={(item) => item.version}
        emptyMessage={isLoading ? "Loading dataset registry..." : "No registered datasets found."}
        columns={[
          {
            header: "Version",
            accessorKey: "version",
            className: "font-mono font-semibold text-white",
          },
          {
            header: "Kind",
            cell: (item) => (
              <span
                className={`px-2 py-0.5 rounded text-[11px] font-mono border ${
                  item.kind === "LICENSED"
                    ? "bg-emerald-950/60 text-emerald-400 border-emerald-800"
                    : "bg-slate-900 text-slate-400 border-slate-800"
                }`}
              >
                {item.kind}
              </span>
            ),
          },
          {
            header: "SHA-256 Digest",
            cell: (item) => <HashViewer hash={item.sha256} length={16} />,
          },
          {
            header: "Partitions / Zones",
            cell: (item) => (
              <div className="flex flex-wrap gap-1">
                {item.zones && item.zones.length > 0 ? (
                  item.zones.map((z: any) => (
                    <span
                      key={z.zone_name}
                      className="px-1.5 py-0.5 rounded bg-slate-950 text-[10px] font-mono text-slate-400 border border-border"
                    >
                      {z.zone_name}
                    </span>
                  ))
                ) : (
                  <span className="text-slate-600 text-xs font-mono">—</span>
                )}
              </div>
            ),
          },
          {
            header: "Registered",
            cell: (item) => (
              <span className="font-mono text-slate-400 text-xs">
                {item.registered_at ? item.registered_at.slice(0, 19).replace("T", " ") : "—"}
              </span>
            ),
          },
        ]}
      />
    </AppShell>
  );
}
