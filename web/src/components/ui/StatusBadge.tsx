import React from "react";
import clsx from "clsx";

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const norm = (status || "").toUpperCase();

  let colorStyles = "bg-slate-800 text-slate-300 border-slate-700";

  switch (norm) {
    case "PAPER_ACTIVE":
    case "COMPLETED":
    case "PASSED":
    case "VERIFIED":
      colorStyles = "bg-emerald-950/60 text-emerald-400 border-emerald-800";
      break;
    case "PAPER_ELIGIBLE":
    case "VALIDATION":
      colorStyles = "bg-blue-950/60 text-blue-400 border-blue-800";
      break;
    case "RESEARCH":
    case "RUNNING":
    case "IDEA":
      colorStyles = "bg-cyan-950/60 text-cyan-400 border-cyan-800";
      break;
    case "DEGRADED":
    case "QUEUED":
      colorStyles = "bg-amber-950/60 text-amber-400 border-amber-800";
      break;
    case "RETIRED":
    case "CANCELLED":
      colorStyles = "bg-slate-900 text-slate-400 border-slate-800";
      break;
    case "REJECTED":
    case "FAILED":
    case "BROKEN PROVENANCE":
      colorStyles = "bg-rose-950/60 text-rose-400 border-rose-800";
      break;
  }

  return (
    <span
      className={clsx(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium border",
        colorStyles,
        className
      )}
    >
      {status}
    </span>
  );
}
