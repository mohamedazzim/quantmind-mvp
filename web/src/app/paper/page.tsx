"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { MetricCard } from "@/components/ui/MetricCard";
import { DataTable } from "@/components/ui/DataTable";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import { TrendingUp, ShieldAlert, CheckCircle, RefreshCw, Layers } from "lucide-react";
import clsx from "clsx";

export default function PaperExecutionPage() {
  const [tab, setTab] = useState<"positions" | "orders" | "fills" | "risk" | "reports">("positions");

  const { data: overview, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["paper-overview"],
    queryFn: () => api.get("/paper/overview"),
  });

  const { data: orders } = useQuery({
    queryKey: ["paper-orders"],
    queryFn: () => api.get<any[]>("/paper/orders"),
    enabled: tab === "orders",
  });

  const { data: fills } = useQuery({
    queryKey: ["paper-fills"],
    queryFn: () => api.get<any[]>("/paper/fills"),
    enabled: tab === "fills",
  });

  const { data: riskEvents } = useQuery({
    queryKey: ["paper-risk"],
    queryFn: () => api.get<any[]>("/paper/risk"),
    enabled: tab === "risk",
  });

  const { data: reports } = useQuery({
    queryKey: ["paper-reports"],
    queryFn: () => api.get<any[]>("/paper/reports"),
    enabled: tab === "reports",
  });

  return (
    <AppShell
      title="Paper Execution & Replay"
      subtitle="Deterministic simulated execution, locked next-bar-open semantics, and risk engine"
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
      {/* Top Metrics Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Active Strategies"
          value={overview?.active_strategies ?? (isLoading ? "..." : 0)}
          icon={<Layers className="h-5 w-5 text-emerald-400" />}
        />
        <MetricCard
          title="Simulated Orders"
          value={overview?.total_orders ?? (isLoading ? "..." : 0)}
          subtitle={`${overview?.total_fills ?? 0} fills executed`}
          icon={<TrendingUp className="h-5 w-5 text-blue-400" />}
        />
        <MetricCard
          title="Pre-Trade Risk Blocks"
          value={overview?.total_risk_events ?? (isLoading ? "..." : 0)}
          subtitle="Enforced by PaperRiskEngine"
          icon={<ShieldAlert className="h-5 w-5 text-amber-400" />}
        />
        <MetricCard
          title="Total Replay Net PnL"
          value={
            overview?.total_net_pnl !== undefined
              ? `$${Number(overview.total_net_pnl).toFixed(2)}`
              : "$0.00"
          }
          icon={<TrendingUp className="h-5 w-5 text-cyan-400" />}
        />
      </div>

      {/* Tabs */}
      <div className="border-b border-border flex space-x-6 text-xs font-medium">
        {[
          { key: "positions", label: "Open Positions" },
          { key: "orders", label: "Simulated Orders" },
          { key: "fills", label: "Execution Fills" },
          { key: "risk", label: "Risk Engine Events" },
          { key: "reports", label: "Replay Reports" },
        ].map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key as any)}
            className={clsx(
              "pb-3 border-b-2 transition-colors",
              tab === t.key
                ? "border-primary text-primary font-semibold"
                : "border-transparent text-slate-400 hover:text-white"
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab: Open Positions */}
      {tab === "positions" && (
        <DataTable
          data={overview?.open_positions || []}
          keyExtractor={(item) => `${item.strategy_id}-${item.symbol}`}
          emptyMessage="No open paper positions. All active strategies currently flat."
          columns={[
            {
              header: "Strategy ID",
              accessorKey: "strategy_id",
              className: "font-mono font-medium text-white",
            },
            {
              header: "Symbol",
              accessorKey: "symbol",
              className: "font-mono text-slate-300",
            },
            {
              header: "Quantity",
              cell: (item) => (
                <span className="font-mono font-semibold text-white">
                  {item.quantity}
                </span>
              ),
            },
            {
              header: "Entry Price",
              cell: (item) => (
                <span className="font-mono text-slate-400">
                  ${Number(item.entry_price).toFixed(2)}
                </span>
              ),
            },
            {
              header: "Realized PnL",
              cell: (item) => (
                <span
                  className={clsx(
                    "font-mono font-medium",
                    item.realized_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                  )}
                >
                  ${Number(item.realized_pnl).toFixed(2)}
                </span>
              ),
            },
            {
              header: "Unrealized PnL",
              cell: (item) => (
                <span
                  className={clsx(
                    "font-mono font-medium",
                    item.unrealized_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                  )}
                >
                  ${Number(item.unrealized_pnl).toFixed(2)}
                </span>
              ),
            },
          ]}
        />
      )}

      {/* Tab: Simulated Orders */}
      {tab === "orders" && (
        <DataTable
          data={orders || []}
          keyExtractor={(item) => item.order_id}
          emptyMessage="No simulated orders recorded."
          columns={[
            {
              header: "Order ID",
              accessorKey: "order_id",
              className: "font-mono font-medium text-white",
            },
            {
              header: "Strategy ID",
              accessorKey: "strategy_id",
              className: "font-mono text-slate-300",
            },
            {
              header: "Side",
              cell: (item) => (
                <span
                  className={clsx(
                    "font-mono font-semibold",
                    item.side === 1 ? "text-emerald-400" : "text-rose-400"
                  )}
                >
                  {item.side === 1 ? "BUY" : "SELL"}
                </span>
              ),
            },
            {
              header: "Quantity",
              accessorKey: "quantity",
              className: "font-mono text-white",
            },
            {
              header: "Status",
              cell: (item) => <StatusBadge status={item.status} />,
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

      {/* Tab: Execution Fills */}
      {tab === "fills" && (
        <DataTable
          data={fills || []}
          keyExtractor={(item) => item.fill_id}
          emptyMessage="No execution fills recorded."
          columns={[
            {
              header: "Fill ID",
              accessorKey: "fill_id",
              className: "font-mono font-medium text-white",
            },
            {
              header: "Order ID",
              accessorKey: "order_id",
              className: "font-mono text-slate-400",
            },
            {
              header: "Price",
              cell: (item) => (
                <span className="font-mono text-white font-medium">
                  ${Number(item.price).toFixed(2)}
                </span>
              ),
            },
            {
              header: "Quantity",
              accessorKey: "quantity",
              className: "font-mono text-slate-300",
            },
            {
              header: "Cost",
              cell: (item) => (
                <span className="font-mono text-slate-400">
                  ${Number(item.cost).toFixed(2)}
                </span>
              ),
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

      {/* Tab: Risk Engine Events */}
      {tab === "risk" && (
        <DataTable
          data={riskEvents || []}
          keyExtractor={(item) => item.event_id}
          emptyMessage="No risk block events recorded."
          columns={[
            {
              header: "Event ID",
              accessorKey: "event_id",
              className: "font-mono font-medium text-white",
            },
            {
              header: "Strategy ID",
              accessorKey: "strategy_id",
              className: "font-mono text-slate-300",
            },
            {
              header: "Rule Name",
              accessorKey: "rule_name",
              className: "text-amber-400 font-mono",
            },
            {
              header: "Reason",
              accessorKey: "reason",
              className: "text-slate-300",
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

      {/* Tab: Replay Reports */}
      {tab === "reports" && (
        <DataTable
          data={reports || []}
          keyExtractor={(item) => item.report_hash}
          emptyMessage="No deterministic replay reports recorded."
          columns={[
            {
              header: "Strategy ID",
              accessorKey: "strategy_id",
              className: "font-mono font-medium text-white",
            },
            {
              header: "Net PnL",
              cell: (item) => (
                <span
                  className={clsx(
                    "font-mono font-semibold",
                    item.net_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                  )}
                >
                  ${Number(item.net_pnl).toFixed(2)}
                </span>
              ),
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
              header: "Sharpe",
              cell: (item) => (
                <span className="font-mono text-slate-300">
                  {Number(item.sharpe_ratio).toFixed(2)}
                </span>
              ),
            },
            {
              header: "Trades",
              accessorKey: "trade_count",
              className: "font-mono text-slate-400",
            },
            {
              header: "Report Hash",
              cell: (item) => <HashViewer hash={item.report_hash} length={10} />,
            },
          ]}
        />
      )}
    </AppShell>
  );
}
