import React from "react";
import clsx from "clsx";

export interface Column<T = any> {
  header: string;
  accessorKey?: string;
  cell?: (item: T) => React.ReactNode;
  className?: string;
}

export interface DataTableProps<T = any> {
  data: T[];
  columns: Column<T>[];
  keyExtractor: (item: T) => string;
  onRowClick?: (item: T) => void;
  emptyMessage?: string;
  className?: string;
}

export function DataTable<T = any>({
  data,
  columns,
  keyExtractor,
  onRowClick,
  emptyMessage = "No records found.",
  className,
}: DataTableProps<T>) {
  if (!data || data.length === 0) {
    return (
      <div className="flex h-36 items-center justify-center rounded-xl border border-border bg-surface p-8 text-center text-sm text-slate-500">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div
      className={clsx(
        "overflow-x-auto rounded-xl border border-border bg-surface shadow-sm",
        className
      )}
    >
      <table className="w-full text-left text-xs">
        <thead className="border-b border-border bg-slate-900/50 uppercase tracking-wider text-slate-400 font-semibold">
          <tr>
            {columns.map((col, idx) => (
              <th key={idx} className={clsx("px-4 py-3", col.className)}>
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {data.map((item) => {
            const key = keyExtractor(item);
            return (
              <tr
                key={key}
                onClick={() => onRowClick && onRowClick(item)}
                className={clsx(
                  "transition-colors hover:bg-surface-hover/80",
                  onRowClick && "cursor-pointer"
                )}
              >
                {columns.map((col, idx) => (
                  <td key={idx} className={clsx("px-4 py-3 text-slate-300", col.className)}>
                    {col.cell
                      ? col.cell(item)
                      : col.accessorKey
                      ? String((item as Record<string, any>)[col.accessorKey] ?? "")
                      : null}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
