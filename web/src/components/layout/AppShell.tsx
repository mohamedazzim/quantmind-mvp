"use client";

import React from "react";
import { Sidebar } from "./Sidebar";
import { useAuth } from "@/lib/auth-context";
import { LogOut, User, Shield } from "lucide-react";

interface AppShellProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
}

export function AppShell({ children, title, subtitle, action }: AppShellProps) {
  const { user, logout } = useAuth();

  return (
    <div className="flex min-h-screen bg-background text-slate-100">
      <Sidebar />

      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top Header */}
        <header className="h-16 shrink-0 border-b border-border bg-surface/50 backdrop-blur px-8 flex items-center justify-between">
          <div>
            {title && <h1 className="text-lg font-bold tracking-tight text-white">{title}</h1>}
            {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
          </div>

          <div className="flex items-center space-x-4">
            {action && <div>{action}</div>}

            <div className="h-5 w-px bg-border"></div>

            {user && (
              <div className="flex items-center space-x-3">
                <div className="flex flex-col text-right">
                  <span className="text-xs font-medium text-white">{user.username}</span>
                  <span className="text-[10px] font-mono text-primary uppercase flex items-center justify-end space-x-1">
                    <Shield className="h-2.5 w-2.5" />
                    <span>{user.role}</span>
                  </span>
                </div>

                <button
                  type="button"
                  onClick={logout}
                  title="Sign out"
                  className="p-1.5 rounded-lg border border-border text-slate-400 hover:text-white hover:bg-surface-hover transition-colors"
                >
                  <LogOut className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto p-8">
          <div className="mx-auto max-w-7xl space-y-6">{children}</div>
        </main>
      </div>
    </div>
  );
}
