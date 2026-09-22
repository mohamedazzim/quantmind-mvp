"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { DataTable } from "@/components/ui/DataTable";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import {
  FlaskConical,
  CheckCircle,
  AlertTriangle,
  Play,
  ArrowRight,
} from "lucide-react";

export default function ResearchPage() {
  const queryClient = useQueryClient();

  // 1. Fetch spec schema derived from domain compiler
  const { data: schema } = useQuery({
    queryKey: ["spec-schema"],
    queryFn: () => api.get("/strategies/spec-schema"),
  });

  // 2. Fetch feedback tasks
  const { data: tasks, isLoading: tasksLoading } = useQuery({
    queryKey: ["research-tasks"],
    queryFn: () => api.get<any[]>("/research/tasks"),
  });

  // Form State
  const [signalName, setSignalName] = useState("current_bar_momentum");
  const [sessionBars, setSessionBars] = useState<number>(100);
  const [windowStart, setWindowStart] = useState<number>(0.0);
  const [windowEnd, setWindowEnd] = useState<number>(1.0);
  const [datasetVersion, setDatasetVersion] = useState("ds-synth-v1");

  // Validation Preview State
  const [validatedCandidate, setValidatedCandidate] = useState<any | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [submitSuccess, setSubmitSuccess] = useState<string | null>(null);

  // Validate Mutation
  const validateMutation = useMutation({
    mutationFn: () =>
      api.post("/research/candidates/validate", {
        strategy_version: "1.0.0",
        feature_version: "1.0.0",
        signal_name: signalName,
        parameters: {
          calendar_session_bars: Number(sessionBars),
          session_window: [Number(windowStart), Number(windowEnd)],
        },
      }),
    onSuccess: (data) => {
      setValidatedCandidate(data);
      setValidationError(null);
      setSubmitSuccess(null);
    },
    onError: (err: any) => {
      setValidationError(err.message || "Candidate validation failed.");
      setValidatedCandidate(null);
    },
  });

  // Submit Trial Mutation
  const submitTrialMutation = useMutation({
    mutationFn: () =>
      api.post("/research/trials/submit", {
        strategy_version: "1.0.0",
        feature_version: "1.0.0",
        signal_name: signalName,
        parameters: {
          calendar_session_bars: Number(sessionBars),
          session_window: [Number(windowStart), Number(windowEnd)],
        },
        dataset_version: datasetVersion,
        split_zone: "RESEARCH",
        research_protocol_version: "proto-v1",
        seed: 42,
      }),
    onSuccess: (data) => {
      setSubmitSuccess(
        `Trial enqueued successfully! Job ID: ${data.job_id}. Worker will execute trial under research budget.`
      );
      setValidationError(null);
      queryClient.invalidateQueries({ queryKey: ["trials-list"] });
    },
    onError: (err: any) => {
      setValidationError(err.message || "Failed to submit trial.");
    },
  });

  return (
    <AppShell
      title="Research Workspace"
      subtitle="Schema-driven strategy candidate formulation and structured feedback tasks"
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Candidate Formulation Form */}
        <div className="rounded-xl border border-border bg-surface p-6 space-y-6">
          <div className="flex items-center space-x-2.5">
            <FlaskConical className="h-5 w-5 text-primary" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">
              Schema-Driven Strategy Builder
            </h3>
          </div>
          <p className="text-xs text-slate-400">
            Formulate declarative strategy candidate specs compiled strictly through domain
            whitelists. Identity is derived deterministically by the backend.
          </p>

          <div className="space-y-4 text-xs">
            <div>
              <label className="block text-slate-400 font-medium mb-1">Signal Whitelist</label>
              <select
                value={signalName}
                onChange={(e) => {
                  setSignalName(e.target.value);
                  setValidatedCandidate(null);
                }}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-white font-mono focus:border-primary focus:outline-none"
              >
                {(schema?.supported_signals || [
                  "current_bar_momentum",
                  "current_bar_mean_reversion",
                ]).map((sig: string) => (
                  <option key={sig} value={sig}>
                    {sig}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-slate-400 font-medium mb-1">
                Calendar Session Bars (int &gt; 1)
              </label>
              <input
                type="number"
                min={2}
                value={sessionBars}
                onChange={(e) => {
                  setSessionBars(parseInt(e.target.value, 10));
                  setValidatedCandidate(null);
                }}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-white font-mono focus:border-primary focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-400 font-medium mb-1">
                  Window Start (0.0 to 1.0)
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="0.95"
                  value={windowStart}
                  onChange={(e) => {
                    setWindowStart(parseFloat(e.target.value));
                    setValidatedCandidate(null);
                  }}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-white font-mono focus:border-primary focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-slate-400 font-medium mb-1">
                  Window End (0.0 to 1.0)
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0.05"
                  max="1.0"
                  value={windowEnd}
                  onChange={(e) => {
                    setWindowEnd(parseFloat(e.target.value));
                    setValidatedCandidate(null);
                  }}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-white font-mono focus:border-primary focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="block text-slate-400 font-medium mb-1">Dataset Scope</label>
              <input
                type="text"
                value={datasetVersion}
                onChange={(e) => setDatasetVersion(e.target.value)}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-white font-mono focus:border-primary focus:outline-none"
              />
            </div>

            <button
              type="button"
              onClick={() => validateMutation.mutate()}
              disabled={validateMutation.isPending}
              className="w-full py-2.5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold shadow-sm transition-colors"
            >
              {validateMutation.isPending ? "Validating with Core Compiler..." : "Preview & Validate Strategy Identity"}
            </button>
          </div>

          {/* Validation Feedback */}
          {validationError && (
            <div className="flex items-center space-x-2 rounded-lg border border-rose-800 bg-rose-950/40 p-3 text-xs text-rose-300">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{validationError}</span>
            </div>
          )}

          {submitSuccess && (
            <div className="flex items-center space-x-2 rounded-lg border border-emerald-800 bg-emerald-950/40 p-3 text-xs text-emerald-300">
              <CheckCircle className="h-4 w-4 shrink-0" />
              <span>{submitSuccess}</span>
            </div>
          )}

          {/* Validated Candidate Confirmation Block */}
          {validatedCandidate && (
            <div className="p-4 rounded-lg border border-emerald-800 bg-emerald-950/20 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-emerald-400">
                  Validated Canonical Identity
                </span>
                <span className="font-mono text-xs text-white font-bold">
                  {validatedCandidate.strategy_id}
                </span>
              </div>
              <pre className="p-2.5 bg-slate-950 rounded border border-border text-[11px] font-mono text-slate-300 overflow-x-auto">
                {validatedCandidate.canonical_spec_json}
              </pre>
              <button
                type="button"
                onClick={() => submitTrialMutation.mutate()}
                disabled={submitTrialMutation.isPending}
                className="w-full flex items-center justify-center space-x-2 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition-colors"
              >
                <Play className="h-3.5 w-3.5" />
                <span>
                  {submitTrialMutation.isPending
                    ? "Enqueueing Job..."
                    : "Confirm & Submit Trial to Worker Queue"}
                </span>
              </button>
            </div>
          )}
        </div>

        {/* Structured Research Feedback Tasks */}
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">
              Structured Feedback Tasks
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Empirical degradation feedback transformed into hypothesis tasks (PRD v4.0 M7)
            </p>
          </div>

          <DataTable
            data={tasks || []}
            keyExtractor={(item) => item.task_id}
            emptyMessage={
              tasksLoading
                ? "Loading feedback tasks..."
                : "No research feedback tasks currently generated."
            }
            columns={[
              {
                header: "Task ID",
                cell: (item) => <HashViewer hash={item.task_id} length={12} />,
              },
              {
                header: "Source Strategy",
                accessorKey: "strategy_id",
                className: "font-mono font-medium text-white",
              },
              {
                header: "Failure Mode",
                accessorKey: "failure_mode",
                className: "text-amber-400 font-mono",
              },
              {
                header: "Drawdown Ratio",
                cell: (item) => (
                  <span className="font-mono text-slate-300">
                    {item.drawdown_expansion_ratio !== undefined
                      ? `${Number(item.drawdown_expansion_ratio).toFixed(2)}x`
                      : "—"}
                  </span>
                ),
              },
            ]}
          />
        </div>
      </div>
    </AppShell>
  );
}
