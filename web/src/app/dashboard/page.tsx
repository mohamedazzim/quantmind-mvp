"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { MetricCard } from "@/components/ui/MetricCard";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { DataTable } from "@/components/ui/DataTable";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import {
  Layers,
  Activity,
  AlertTriangle,
  TrendingUp,
  RefreshCw,
  Clock,
} from "lucide-react";

interface DegradationItem {
  event_hash: string;
  strategy_id: string;
  rule_name: string;
  observed_value: number;
  threshold_value: number;
  timestamp?: string;
}

interface TransitionItem {
  transition_hash: string;
  strategy_id: string;
  old_state: string;
  new_state: string;
  initiator: string;
  timestamp?: string;
}

export default function DashboardPage() {

  const { data, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["dashboard-kpis"],
    queryFn: () => api.get("/dashboard/kpis"),
  });

  return (
    <AppShell
      title="System Overview"
      subtitle="Real-time status across research, paper evaluation, and governance ledgers"
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
      {/* Metric Cards Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Active Paper Strategies"
          value={data?.active_strategies ?? (isLoading ? "..." : 0)}
          subtitle={`${data?.eligible_strategies ?? 0} paper eligible`}
          icon={<Layers className="h-5 w-5 text-emerald-400" />}
        />
        <MetricCard
          title="Degraded Strategies"
          value={data?.degraded_strategies ?? (isLoading ? "..." : 0)}
          subtitle={`${data?.total_degradation_events ?? 0} total breach events`}
          icon={<AlertTriangle className="h-5 w-5 text-amber-400" />}
        />
        <MetricCard
          title="Total Production Trials"
          value={data?.total_trials ?? (isLoading ? "..." : 0)}
          subtitle="Enforced research budget accounting"
          icon={<Activity className="h-5 w-5 text-blue-400" />}
        />
        <MetricCard
          title="Paper Replay Net PnL"
          value={
            data?.total_net_pnl !== undefined
              ? `$${Number(data.total_net_pnl).toFixed(2)}`
              : isLoading
              ? "..."
              : "$0.00"
          }
          subtitle={`Avg Sharpe: ${Number(data?.avg_sharpe_ratio ?? 0).toFixed(2)}`}
          icon={<TrendingUp className="h-5 w-5 text-cyan-400" />}
        />
      </div>

      {/* Two Column Layout: Recent Degradations & Recent Transitions */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Degradation Feed */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <span>Recent Degradation Breaches</span>
            </h3>
            <span className="text-xs text-slate-500 font-mono">Real-time alerts</span>
          </div>

          <DataTable<DegradationItem>
            data={data?.recent_degradations || []}
            keyExtractor={(item) => item.event_hash}
            emptyMessage="Zero active degradation breaches. All evaluation windows within bounds."
            columns={[
              {
                header: "Strategy",
                accessorKey: "strategy_id",
                className: "font-mono font-medium text-white",
              },
              {
                header: "Breach Rule",
                accessorKey: "rule_name",
              },
              {
                header: "Observed vs Limit",
                cell: (item) => (
                  <span className="font-mono text-amber-400">
                    {item.observed_value} / {item.threshold_value}
                  </span>
                ),
              },
              {
                header: "Hash",
                cell: (item) => <HashViewer hash={item.event_hash} length={8} />,
              },
            ]}
          />
        </div>

        {/* Governance Transitions */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
              <Clock className="h-4 w-4 text-primary" />
              <span>Authoritative Lifecycle Transitions</span>
            </h3>
            <span className="text-xs text-slate-500 font-mono">Immutable audit trail</span>
          </div>

          <DataTable<TransitionItem>
            data={data?.recent_transitions || []}
            keyExtractor={(item) => item.transition_hash}
            emptyMessage="No state transitions recorded."
            columns={[
              {
                header: "Strategy",
                accessorKey: "strategy_id",
                className: "font-mono font-medium text-white",
              },
              {
                header: "State Transition",
                cell: (item) => (
                  <div className="flex items-center space-x-1.5">
                    <StatusBadge status={item.old_state} />
                    <span className="text-slate-500">→</span>
                    <StatusBadge status={item.new_state} />
                  </div>
                ),
              },
              {
                header: "Initiator",
                accessorKey: "initiator",
                className: "text-slate-400 font-mono",
              },
              {
                header: "Digest",
                cell: (item) => <HashViewer hash={item.transition_hash} length={8} />,
              },
            ]}
          />
        </div>
      </div>
    </AppShell>
  );
}
