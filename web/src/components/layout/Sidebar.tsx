"use client";

import React from "react";
import Link from "next/navigation";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import {
  LayoutDashboard,
  Layers,
  FlaskConical,
  ListChecks,
  CheckCircle2,
  TrendingUp,
  Activity,
  ShieldCheck,
  MessageSquareShare,
  Database,
  GitBranch,
} from "lucide-react";

const NAV_ITEMS = [
  { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { label: "Strategies", href: "/strategies", icon: Layers },
  { label: "Research", href: "/research", icon: FlaskConical },
  { label: "Trials", href: "/trials", icon: ListChecks },
  { label: "Qualification", href: "/qualification", icon: CheckCircle2 },
  { label: "Paper Execution", href: "/paper", icon: TrendingUp },
  { label: "Monitoring", href: "/monitoring", icon: Activity },
  { label: "Governance", href: "/governance", icon: ShieldCheck },
  { label: "Feedback Tasks", href: "/feedback", icon: MessageSquareShare },
  { label: "Datasets", href: "/datasets", icon: Database },
  { label: "Audit Explorer", href: "/audit", icon: GitBranch },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 shrink-0 border-r border-border bg-surface flex flex-col justify-between h-screen sticky top-0">
      <div>
        <div className="flex h-16 items-center px-6 border-b border-border">
          <div className="flex items-center space-x-2.5">
            <div className="h-7 w-7 rounded bg-primary flex items-center justify-center font-bold text-white text-sm">
              QM
            </div>
            <div>
              <span className="font-bold tracking-tight text-white text-sm">QuantMind</span>
              <span className="ml-1.5 text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                v4.0
              </span>
            </div>
          </div>
        </div>

        <nav className="p-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive =
              pathname === item.href ||
              (item.href !== "/dashboard" && pathname.startsWith(item.href));

            return (
              <a
                key={item.href}
                href={item.href}
                className={clsx(
                  "flex items-center space-x-3 px-3 py-2 rounded-lg text-xs font-medium transition-colors",
                  isActive
                    ? "bg-primary text-white"
                    : "text-slate-400 hover:text-white hover:bg-surface-hover"
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span>{item.label}</span>
              </a>
            );
          })}
        </nav>
      </div>

      <div className="p-4 border-t border-border text-[11px] text-slate-500 font-mono">
        Deterministic Core • Frozen v4.0
      </div>
    </aside>
  );
}
