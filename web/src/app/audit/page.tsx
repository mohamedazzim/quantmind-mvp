"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { HashViewer } from "@/components/ui/HashViewer";
import { api } from "@/lib/api";
import {
  GitBranch,
  Search,
  ShieldCheck,
  AlertTriangle,
  ArrowDown,
  Info,
} from "lucide-react";
import clsx from "clsx";

export default function AuditExplorerPage() {
  const [searchInput, setSearchInput] = useState("");
  const [activeIdentifier, setActiveIdentifier] = useState("");
  const [selectedNode, setSelectedNode] = useState<any | null>(null);

  const { data: graph, isLoading, isError } = useQuery({
    queryKey: ["audit-graph", activeIdentifier],
    queryFn: () => api.get(`/audit/graph/${encodeURIComponent(activeIdentifier)}`),
    enabled: !!activeIdentifier,
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      setActiveIdentifier(searchInput.trim());
      setSelectedNode(null);
    }
  };

  const getEdgeBadge = (typeCode: string) => {
    switch (typeCode) {
      case "A":
        return "bg-emerald-950/80 text-emerald-400 border-emerald-700";
      case "B":
        return "bg-blue-950/80 text-blue-400 border-blue-700";
      case "C":
        return "bg-purple-950/80 text-purple-400 border-purple-700";
      case "D":
        return "bg-amber-950/80 text-amber-400 border-amber-700";
      default:
        return "bg-slate-900 text-slate-400 border-slate-700";
    }
  };

  return (
    <AppShell
      title="Audit Provenance Explorer"
      subtitle="5-Type classified provenance graph resolving complete multi-milestone evidence lineage"
    >
      {/* Search Header */}
      <div className="rounded-xl border border-border bg-surface p-6 space-y-4">
        <form onSubmit={handleSearch} className="flex gap-3">
          <div className="relative flex-1">
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-500">
              <Search className="h-4 w-4" />
            </div>
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search by strategy_id, qualification_hash, replay_report_hash, snapshot_hash, event_hash..."
              className="w-full rounded-lg border border-border bg-background py-2.5 pl-9 pr-3 text-xs text-white placeholder-slate-500 focus:border-primary focus:outline-none"
            />
          </div>
          <button
            type="submit"
            className="px-5 py-2.5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold shadow-sm transition-colors"
          >
            Trace Provenance
          </button>
        </form>

        {/* 5-Type Edge Legend */}
        <div className="pt-2 border-t border-border flex flex-wrap items-center gap-4 text-[11px] font-mono">
          <span className="text-slate-400 font-bold uppercase tracking-wider">Edge Types:</span>
          <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded border bg-emerald-950/60 text-emerald-400 border-emerald-800">
            <span className="font-bold">Type A</span>
            <span className="text-slate-400">Cryptographic Hash</span>
          </span>
          <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded border bg-blue-950/60 text-blue-400 border-blue-800">
            <span className="font-bold">Type B</span>
            <span className="text-slate-400">Referential FK</span>
          </span>
          <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded border bg-purple-950/60 text-purple-400 border-purple-800">
            <span className="font-bold">Type C</span>
            <span className="text-slate-400">Semantic Identity</span>
          </span>
          <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded border bg-amber-950/60 text-amber-400 border-amber-800">
            <span className="font-bold">Type D</span>
            <span className="text-slate-400">Causal Timestamp</span>
          </span>
          <span className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded border bg-slate-900 text-slate-400 border-slate-700">
            <span className="font-bold">Type E</span>
            <span className="text-slate-500">Informational Ref</span>
          </span>
        </div>
      </div>

      {/* Graph Display */}
      {activeIdentifier && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Nodes & Edges Visual Hierarchy */}
          <div className="lg:col-span-2 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Connected Lineage Nodes ({graph?.nodes?.length || 0})
              </h3>
              {graph?.error && (
                <span className="text-xs text-rose-400 font-mono flex items-center space-x-1">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  <span>{graph.error}</span>
                </span>
              )}
            </div>

            {isLoading && (
              <div className="h-48 flex items-center justify-center font-mono text-xs text-slate-500 rounded-xl border border-border bg-surface">
                Tracing cryptographic provenance DAG...
              </div>
            )}

            {!isLoading && graph?.nodes?.length === 0 && (
              <div className="p-8 text-center text-xs text-slate-500 rounded-xl border border-border bg-surface font-mono">
                No provenance found matching &apos;{activeIdentifier}&apos;.
              </div>
            )}

            {!isLoading && graph?.nodes && graph.nodes.length > 0 && (
              <div className="space-y-3">
                {graph.nodes.map((node: any) => {
                  const isSelected = selectedNode?.id === node.id;
                  const isBroken = node.status === "BROKEN PROVENANCE";

                  return (
                    <div
                      key={node.id}
                      onClick={() => setSelectedNode(node)}
                      className={clsx(
                        "p-4 rounded-xl border transition-all cursor-pointer",
                        isSelected
                          ? "border-primary bg-surface-hover shadow-md"
                          : isBroken
                          ? "border-rose-800 bg-rose-950/20"
                          : "border-border bg-surface hover:border-slate-700"
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2.5">
                          <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-700 font-mono text-[10px] text-slate-300">
                            {node.type}
                          </span>
                          <h4 className="font-mono text-xs font-semibold text-white">
                            {node.label}
                          </h4>
                        </div>
                        <StatusBadge status={node.status} />
                      </div>

                      <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono text-slate-400">
                        <span className="text-slate-500">ID: {node.identifier}</span>
                        {node.hash && (
                          <div className="flex items-center space-x-1">
                            <span className="text-slate-500">Hash:</span>
                            <HashViewer hash={node.hash} length={8} />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Edge Relational Table */}
            {!isLoading && graph?.edges && graph.edges.length > 0 && (
              <div className="mt-6 space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Verified Relational Edges ({graph.edges.length})
                </h3>
                <div className="space-y-2">
                  {graph.edges.map((edge: any) => (
                    <div
                      key={edge.id}
                      className="p-3 rounded-lg border border-border bg-slate-900/40 text-xs font-mono flex items-center justify-between"
                    >
                      <div className="flex items-center space-x-3">
                        <span
                          className={clsx(
                            "px-2 py-0.5 rounded border text-[10px] font-bold",
                            getEdgeBadge(edge.relationship_type)
                          )}
                        >
                          {edge.edge_type}
                        </span>
                        <span className="text-white font-medium">{edge.label}</span>
                      </div>
                      <span className="text-slate-400 text-[11px]">
                        {edge.verification_mechanism}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Node Inspector Detail Panel */}
          <div className="space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Evidence Node Inspector
            </h3>
            <div className="p-5 rounded-xl border border-border bg-surface space-y-4 text-xs">
              {selectedNode ? (
                <>
                  <div className="flex items-center justify-between border-b border-border pb-3">
                    <span className="font-mono font-bold text-white text-sm">
                      {selectedNode.type}
                    </span>
                    <StatusBadge status={selectedNode.status} />
                  </div>
                  <div className="space-y-2 font-mono text-[11px]">
                    <div>
                      <span className="text-slate-500 block">Identifier:</span>
                      <span className="text-white break-all">{selectedNode.identifier}</span>
                    </div>
                    {selectedNode.hash && (
                      <div>
                        <span className="text-slate-500 block">Digest:</span>
                        <span className="text-slate-300 break-all">{selectedNode.hash}</span>
                      </div>
                    )}
                    <div>
                      <span className="text-slate-500 block">Source:</span>
                      <span className="text-cyan-400">{selectedNode.source}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block">Timestamp:</span>
                      <span className="text-slate-400">{selectedNode.timestamp || "—"}</span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-border">
                    <span className="text-slate-500 block font-mono text-[11px] mb-1">
                      Raw Evidence Payload:
                    </span>
                    <pre className="p-3 bg-slate-950 rounded-lg border border-border font-mono text-[11px] text-slate-300 overflow-x-auto max-h-64">
                      {JSON.stringify(selectedNode.details, null, 2)}
                    </pre>
                  </div>
                </>
              ) : (
                <div className="py-12 text-center text-slate-500 font-mono text-xs">
                  Click on any node in the provenance chain to inspect verified cryptographic
                  evidence attributes.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </AppShell>
  );
}
