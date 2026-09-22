"use client";

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { HashViewer } from "@/components/ui/HashViewer";
import { DataTable } from "@/components/ui/DataTable";
import { api, ApiError } from "@/lib/api";
import {
  Shield,
  Play,
  Archive,
  RotateCcw,
  AlertTriangle,
  ArrowLeft,
  CheckCircle,
} from "lucide-react";
import clsx from "clsx";

export default function StrategyDetailPage() {
  const { strategyId } = useParams() as { strategyId: string };
  const router = useRouter();
  const queryClient = useQueryClient();

  const [activeTab, setActiveTab] = useState<
    "overview" | "qualification" | "baseline" | "snapshots" | "governance"
  >("overview");

  // Governance action form states
  const [retireReason, setRetireReason] = useState("");
  const [reResearchReason, setReResearchReason] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  const { data: strat, isLoading, refetch } = useQuery({
    queryKey: ["strategy-detail", strategyId],
    queryFn: () => api.get(`/strategies/${strategyId}`),
    enabled: !!strategyId,
  });

  // 1. Activate Mutation (Strictly Non-Optimistic)
  const activateMutation = useMutation({
    mutationFn: () =>
      api.post(`/strategies/${strategyId}/activate`, {
        initiator: "ui-console",
      }),
    onSuccess: (resp) => {
      setActionSuccess(resp.message || "Strategy activated to PAPER_ACTIVE.");
      setActionError(null);
      queryClient.invalidateQueries({ queryKey: ["strategy-detail", strategyId] });
      queryClient.invalidateQueries({ queryKey: ["strategies-list"] });
    },
    onError: (err: any) => {
      setActionError(err.message || "Failed to activate strategy.");
      setActionSuccess(null);
    },
  });

  // 2. Retire Mutation
  const retireMutation = useMutation({
    mutationFn: () =>
      api.post(`/strategies/${strategyId}/retire`, {
        reason: retireReason,
        initiator: "ui-console",
      }),
    onSuccess: (resp) => {
      setActionSuccess(resp.message || "Strategy retired permanently.");
      setActionError(null);
      setRetireReason("");
      queryClient.invalidateQueries({ queryKey: ["strategy-detail", strategyId] });
      queryClient.invalidateQueries({ queryKey: ["strategies-list"] });
    },
    onError: (err: any) => {
      setActionError(err.message || "Failed to retire strategy.");
      setActionSuccess(null);
    },
  });

  // 3. Re-Research Mutation
  const reResearchMutation = useMutation({
    mutationFn: () =>
      api.post(`/strategies/${strategyId}/re-research`, {
        reason: reResearchReason,
        initiator: "ui-console",
      }),
    onSuccess: (resp) => {
      setActionSuccess(resp.message || "Strategy returned to RESEARCH.");
      setActionError(null);
      setReResearchReason("");
      queryClient.invalidateQueries({ queryKey: ["strategy-detail", strategyId] });
      queryClient.invalidateQueries({ queryKey: ["strategies-list"] });
    },
    onError: (err: any) => {
      setActionError(err.message || "Failed to return strategy to research.");
      setActionSuccess(null);
    },
  });

  if (isLoading) {
    return (
      <AppShell title="Strategy Dossier">
        <div className="flex h-64 items-center justify-center text-slate-500 font-mono">
          Loading strategy dossier for {strategyId}...
        </div>
      </AppShell>
    );
  }

  if (!strat) {
    return (
      <AppShell title="Strategy Dossier">
        <div className="p-8 text-center text-rose-400">
          Strategy {strategyId} not found.
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell
      title={strategyId}
      subtitle={`Lifecycle state: ${strat.state} • Registered in authoritative StrategyRegistry`}
      action={
        <button
          type="button"
          onClick={() => router.push("/strategies")}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-hover text-xs font-medium text-slate-300 hover:text-white"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          <span>Back to Strategies</span>
        </button>
      }
    >
      {/* Dossier Header Info */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <h2 className="text-xl font-bold font-mono text-white">{strategyId}</h2>
            <StatusBadge status={strat.state} />
          </div>

          <div className="flex items-center space-x-6 text-xs text-slate-400 font-mono">
            <div>
              <span className="text-slate-500">Qualification: </span>
              <HashViewer hash={strat.qualification_hash} length={12} />
            </div>
            <div>
              <span className="text-slate-500">Updated: </span>
              <span>{strat.updated_at ? strat.updated_at.slice(0, 19).replace("T", " ") : "—"}</span>
            </div>
          </div>
        </div>

        {/* Action Messages */}
        {actionError && (
          <div className="mt-4 flex items-center space-x-2 rounded-lg border border-rose-800 bg-rose-950/40 p-3 text-xs text-rose-300">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <span>{actionError}</span>
          </div>
        )}
        {actionSuccess && (
          <div className="mt-4 flex items-center space-x-2 rounded-lg border border-emerald-800 bg-emerald-950/40 p-3 text-xs text-emerald-300">
            <CheckCircle className="h-4 w-4 shrink-0" />
            <span>{actionSuccess}</span>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-border flex space-x-6 text-xs font-medium">
        {[
          { key: "overview", label: "Overview & Specification" },
          { key: "qualification", label: "Qualification Evidence" },
          { key: "baseline", label: "Baseline & Replay" },
          { key: "snapshots", label: "Monitoring Snapshots" },
          { key: "governance", label: "Governance Commands" },
        ].map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => setActiveTab(tab.key as any)}
            className={clsx(
              "pb-3 border-b-2 transition-colors",
              activeTab === tab.key
                ? "border-primary text-primary font-semibold"
                : "border-transparent text-slate-400 hover:text-white"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab 1: Overview & Specification */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-surface p-6">
            <h3 className="text-sm font-semibold text-white mb-4">Declarative Strategy Specification</h3>
            {strat.specification ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
                <div>
                  <span className="text-slate-500">Signal Whitelist:</span>{" "}
                  <span className="text-white font-semibold">{strat.specification.signal_name}</span>
                </div>
                <div>
                  <span className="text-slate-500">Strategy Version:</span>{" "}
                  <span className="text-white">{strat.specification.strategy_version}</span>
                </div>
                <div>
                  <span className="text-slate-500">Feature Version:</span>{" "}
                  <span className="text-white">{strat.specification.feature_version}</span>
                </div>
                <div className="col-span-2 mt-2">
                  <span className="text-slate-500 block mb-1">Normalized Parameters:</span>
                  <pre className="p-3 bg-slate-950 rounded-lg border border-border text-slate-300 overflow-x-auto">
                    {JSON.stringify(strat.specification.parameters, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">No specification object bound.</p>
            )}
          </div>
        </div>
      )}

      {/* Tab 2: Qualification Evidence */}
      {activeTab === "qualification" && (
        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-surface p-6">
            <h3 className="text-sm font-semibold text-white mb-4">Certified Qualification Record</h3>
            {strat.qualification ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs font-mono">
                <div className="p-4 bg-slate-900/60 rounded-lg border border-border">
                  <span className="text-slate-500 block">Deflated Sharpe (DSR)</span>
                  <span className="text-xl font-bold text-white mt-1 block">
                    {Number(strat.qualification.dsr).toFixed(3)}
                  </span>
                </div>
                <div className="p-4 bg-slate-900/60 rounded-lg border border-border">
                  <span className="text-slate-500 block">Observed Sharpe</span>
                  <span className="text-xl font-bold text-white mt-1 block">
                    {Number(strat.qualification.observed_sharpe).toFixed(3)}
                  </span>
                </div>
                <div className="p-4 bg-slate-900/60 rounded-lg border border-border">
                  <span className="text-slate-500 block">Holdout State</span>
                  <span className="text-xl font-bold text-emerald-400 mt-1 block">
                    {strat.qualification.holdout_state}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">No qualification record bound to this strategy.</p>
            )}
          </div>
        </div>
      )}

      {/* Tab 3: Baseline & Replay */}
      {activeTab === "baseline" && (
        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-surface p-6">
            <h3 className="text-sm font-semibold text-white mb-4">Paper Evaluation Baseline</h3>
            {strat.baseline ? (
              <div className="space-y-3 text-xs font-mono">
                <div>
                  <span className="text-slate-500">Binding Hash: </span>
                  <HashViewer hash={strat.baseline.binding_hash} length={16} />
                </div>
                <div>
                  <span className="text-slate-500">Baseline Dataset Version: </span>
                  <span className="text-white">{strat.baseline.baseline_dataset_version}</span>
                </div>
                <div>
                  <span className="text-slate-500">Replay Report Hash: </span>
                  <HashViewer hash={strat.baseline.baseline_replay_report_hash} length={16} />
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">No authoritative baseline registered.</p>
            )}
          </div>
        </div>
      )}

      {/* Tab 4: Monitoring Snapshots */}
      {activeTab === "snapshots" && (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-white">Windowed Monitoring Snapshots</h3>
          <DataTable
            data={strat.recent_snapshots || []}
            keyExtractor={(item) => item.snapshot_hash}
            emptyMessage="No monitoring snapshots recorded for this strategy."
            columns={[
              {
                header: "Snapshot Hash",
                cell: (item) => <HashViewer hash={item.snapshot_hash} length={10} />,
              },
              {
                header: "Max Drawdown",
                cell: (item) => (
                  <span className="font-mono text-slate-300">
                    {item.max_drawdown_bps !== undefined
                      ? `${Number(item.max_drawdown_bps).toFixed(0)} bps`
                      : "—"}
                  </span>
                ),
              },
              {
                header: "Realized Sharpe",
                cell: (item) => (
                  <span className="font-mono text-slate-300">
                    {item.realized_sharpe ? item.realized_sharpe.toFixed(2) : "—"}
                  </span>
                ),
              },
              {
                header: "Trades",
                accessorKey: "total_trades",
                className: "font-mono text-slate-400",
              },
              {
                header: "Timestamp",
                accessorKey: "created_at",
                className: "font-mono text-slate-500",
              },
            ]}
          />
        </div>
      )}

      {/* Tab 5: Governance Commands (Non-Optimistic) */}
      {activeTab === "governance" && (
        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-surface p-6 space-y-6">
            <h3 className="text-sm font-semibold text-white">Authoritative Governance Commands</h3>
            <p className="text-xs text-slate-400">
              Only authorized commands with strict evidence prerequisites are accepted. State changes
              are executed fail-closed on the backend and recorded into the EvaluationLedger.
            </p>

            {/* 1. ACTIVATE COMMAND */}
            <div className="p-4 rounded-lg border border-border bg-slate-900/40 flex items-center justify-between">
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-white">Activate Strategy</h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Transitions from PAPER_ELIGIBLE to PAPER_ACTIVE. Requires authoritative baseline.
                </p>
              </div>
              <button
                type="button"
                onClick={() => activateMutation.mutate()}
                disabled={strat.state !== "PAPER_ELIGIBLE" || activateMutation.isPending}
                className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold disabled:opacity-40 transition-colors"
              >
                <Play className="h-3.5 w-3.5" />
                <span>{activateMutation.isPending ? "Validating Evidence..." : "Activate"}</span>
              </button>
            </div>

            {/* 2. RETIRE COMMAND */}
            <div className="p-4 rounded-lg border border-border bg-slate-900/40 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-white">Retire Strategy</h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Permanently transitions strategy to RETIRED. Requires flat position (quantity == 0).
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => retireMutation.mutate()}
                  disabled={
                    !["PAPER_ELIGIBLE", "PAPER_ACTIVE", "DEGRADED"].includes(strat.state) ||
                    !retireReason.trim() ||
                    retireMutation.isPending
                  }
                  className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-rose-700 hover:bg-rose-600 text-white text-xs font-semibold disabled:opacity-40 transition-colors"
                >
                  <Archive className="h-3.5 w-3.5" />
                  <span>{retireMutation.isPending ? "Retiring..." : "Retire Permanently"}</span>
                </button>
              </div>
              <input
                type="text"
                value={retireReason}
                onChange={(e) => setRetireReason(e.target.value)}
                placeholder="Enter mandatory retirement reason..."
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-primary"
              />
            </div>

            {/* 3. RE-RESEARCH COMMAND */}
            <div className="p-4 rounded-lg border border-border bg-slate-900/40 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-white">Re-Research Demotion</h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Demotes a DEGRADED strategy back to RESEARCH for revision. Invalidates qualification bindings.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => reResearchMutation.mutate()}
                  disabled={
                    strat.state !== "DEGRADED" ||
                    !reResearchReason.trim() ||
                    reResearchMutation.isPending
                  }
                  className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold disabled:opacity-40 transition-colors"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  <span>{reResearchMutation.isPending ? "Demoting..." : "Return to Research"}</span>
                </button>
              </div>
              <input
                type="text"
                value={reResearchReason}
                onChange={(e) => setReResearchReason(e.target.value)}
                placeholder="Enter mandatory re-research revision rationale..."
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-primary"
              />
            </div>
          </div>
        </div>
      )}
    </AppShell>
  );
}
