"use client";

import React, { useState } from "react";
import { Check, Copy } from "lucide-react";
import clsx from "clsx";

interface HashViewerProps {
  hash: string | null | undefined;
  length?: number;
  className?: string;
}

export function HashViewer({ hash, length = 12, className }: HashViewerProps) {
  const [copied, setCopied] = useState(false);

  if (!hash) {
    return <span className="text-slate-600 font-mono text-xs">—</span>;
  }

  const truncated =
    hash.length > length
      ? `${hash.slice(0, length / 2)}…${hash.slice(-length / 2)}`
      : hash;

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      title={`Click to copy full hash: ${hash}`}
      className={clsx(
        "inline-flex items-center space-x-1.5 font-mono text-xs text-slate-300 hover:text-white bg-slate-900/60 hover:bg-slate-800 px-2 py-0.5 rounded border border-border transition-colors",
        className
      )}
    >
      <span>{truncated}</span>
      {copied ? (
        <Check className="h-3 w-3 text-emerald-400" />
      ) : (
        <Copy className="h-3 w-3 text-slate-500 opacity-60 hover:opacity-100" />
      )}
    </button>
  );
}
