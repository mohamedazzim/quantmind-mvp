"use client";

import React from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import { ArrowLeft, Activity } from "lucide-react";

export default function TrialDetailPage() {
  const { trialId } = useParams() as { trialId: string };
  const router = useRouter();

  const { data: trial, isLoading } = useQuery({
    queryKey: ["trial-detail", trialId],
    queryFn: () => api.get(`/trials/${trialId}`),
    enabled: !!trialId,
  });

  if (isLoading) {
    return (
      <AppShell title="Trial Detail">
        <div className="flex h-64 items-center justify-center font-mono text-slate-500">
          Loading trial {trialId}...
        </div>
      </AppShell>
    );
  }

  if (!trial) {
    return (
      <AppShell title="Trial Detail">
        <div className="p-8 text-center text-rose-400">Trial {trialId} not found.</div>
      </AppShell>
    );
  }

  return (
    <AppShell
      title={trial.trial_id}
      subtitle={`Strategy: ${trial.strategy_id} • Status: ${trial.status}`}
      action={
        <button
          type="button"
          onClick={() => router.push("/trials")}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-hover text-xs font-medium text-slate-300 hover:text-white"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          <span>Back to Trials</span>
        </button>
      }
    >
      <div className="rounded-xl border border-border bg-surface p-6 space-y-6">
        <div className="flex items-center justify-between border-b border-border pb-4">
          <div className="flex items-center space-x-3">
            <h2 className="text-xl font-bold font-mono text-white">{trial.trial_id}</h2>
            <StatusBadge status={trial.status} />
          </div>
          <span className="text-xs font-mono text-slate-400">
            Experiment ID: {trial.experiment_id}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
          <div className="p-3 bg-slate-900/60 rounded-lg border border-border">
            <span className="text-slate-500 block">Dataset</span>
            <span className="text-white font-semibold mt-0.5 block">{trial.dataset_version}</span>
          </div>
          <div className="p-3 bg-slate-900/60 rounded-lg border border-border">
            <span className="text-slate-500 block">Split Zone</span>
            <span className="text-cyan-400 font-semibold mt-0.5 block">{trial.split_zone}</span>
          </div>
          <div className="p-3 bg-slate-900/60 rounded-lg border border-border">
            <span className="text-slate-500 block">Protocol Version</span>
            <span className="text-white font-semibold mt-0.5 block">{trial.research_protocol_version}</span>
          </div>
          <div className="p-3 bg-slate-900/60 rounded-lg border border-border">
            <span className="text-slate-500 block">Runtime</span>
            <span className="text-white font-semibold mt-0.5 block">
              {trial.actual_runtime_minutes ? `${trial.actual_runtime_minutes.toFixed(2)} min` : "—"}
            </span>
          </div>
        </div>

        {trial.result && (
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Backtest Execution Metrics
            </h3>
            <pre className="p-4 bg-slate-950 rounded-xl border border-border text-xs font-mono text-slate-300 overflow-x-auto">
              {JSON.stringify(trial.result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </AppShell>
  );
}
