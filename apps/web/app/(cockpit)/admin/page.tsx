import { PageShell } from "@/components/page-shell";
import { formatCurrency, formatDateTime, getAdminData } from "@/lib/api";

export default async function AdminPage() {
  const { users, auditLogs, events, eventSummary, symbols, unresolvedSymbols, source } = await getAdminData();

  return (
    <PageShell
      eyebrow="Admin"
      title="Operational control room"
      description="User administration, audit oversight, outbox health and market-data governance for imported books."
      status={source === "api" ? "live api" : "fallback snapshot"}
    >
      <section className="grid gap-4 md:grid-cols-4">
        <div className="panel p-6">
          <p className="metric-label">Users</p>
          <p className="mt-4 font-mono text-3xl text-white">{users.length}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Audit Events</p>
          <p className="mt-4 font-mono text-3xl text-white">{auditLogs.length}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Pending Events</p>
          <p className="mt-4 font-mono text-3xl text-sand">{eventSummary.pending}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Failed Events</p>
          <p className="mt-4 font-mono text-3xl text-loss">{eventSummary.failed}</p>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="panel p-6">
          <p className="metric-label">Tracked Symbols</p>
          <p className="mt-4 font-mono text-3xl text-white">{symbols.length}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Review Required</p>
          <p className="mt-4 font-mono text-3xl text-sand">{unresolvedSymbols.length}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Market Data Coverage</p>
          <p className="mt-4 font-mono text-3xl text-white">
            {symbols.length ? `${Math.round(((symbols.length - unresolvedSymbols.length) / symbols.length) * 100)}%` : "0%"}
          </p>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
        <div className="grid gap-4">
          <div className="panel p-6">
            <p className="metric-label">User Access</p>
            <div className="mt-5 space-y-3">
              {users.map((user) => (
                <div key={user.id} className="rounded-2xl border border-line bg-black/20 p-4">
                  <div className="flex items-center justify-between">
                    <p className="text-white">{user.full_name}</p>
                    <p className="font-mono text-sm text-slate-400">{user.role}</p>
                  </div>
                  <p className="mt-2 text-sm text-slate-300">{user.email}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="panel p-6">
            <p className="metric-label">Market Data Review</p>
            <div className="mt-5 space-y-3">
              {unresolvedSymbols.length ? unresolvedSymbols.slice(0, 5).map((symbol) => (
                <div key={symbol.id} className="rounded-2xl border border-line bg-black/20 p-4">
                  <div className="flex items-center justify-between gap-4">
                    <p className="truncate text-white">{symbol.ticker}</p>
                    <p className="text-sm text-slate-400">{symbol.asset_class}</p>
                  </div>
                  <p className="mt-2 text-sm text-slate-300">
                    {symbol.market_data_enabled ? symbol.market_data_ticker ?? "missing alias" : "manual valuation only"}
                  </p>
                  <p className="mt-2 text-sm text-slate-400">
                    {symbol.position_count} position{symbol.position_count > 1 ? "s" : ""} | {formatCurrency(symbol.total_market_value, symbol.currency)}
                  </p>
                </div>
              )) : (
                <div className="rounded-2xl border border-line bg-black/20 p-4 text-sm text-slate-300">
                  All tracked symbols currently have a market-data path or an explicit manual-only status.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="panel p-6">
          <p className="metric-label">Audit Trail</p>
          <div className="mt-6 overflow-hidden rounded-2xl border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-black/20 text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Event</th>
                  <th className="px-4 py-3 font-medium">Actor</th>
                  <th className="px-4 py-3 font-medium">Time</th>
                </tr>
              </thead>
              <tbody>
                {auditLogs.map((log) => (
                  <tr key={log.id} className="border-t border-line/70">
                    <td className="px-4 py-3 text-white">{log.event_type}</td>
                    <td className="px-4 py-3 text-slate-300">{log.actor_email}</td>
                    <td className="px-4 py-3 text-slate-400">{formatDateTime(log.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="panel p-6">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="metric-label">Symbol Registry</p>
            <p className="mt-2 text-sm text-slate-400">
              Registry of imported and native symbols with their market-data routing status.
            </p>
          </div>
        </div>
        <div className="mt-6 overflow-hidden rounded-2xl border border-line">
          <table className="w-full text-left text-sm">
            <thead className="bg-black/20 text-slate-400">
              <tr>
                <th className="px-4 py-3 font-medium">Ticker</th>
                <th className="px-4 py-3 font-medium">Asset Class</th>
                <th className="px-4 py-3 font-medium">Market Data</th>
                <th className="px-4 py-3 font-medium">Positions</th>
                <th className="px-4 py-3 font-medium">Market Value</th>
              </tr>
            </thead>
            <tbody>
              {symbols.slice(0, 12).map((symbol) => (
                <tr key={symbol.id} className="border-t border-line/70">
                  <td className="px-4 py-3 font-mono text-white">{symbol.ticker}</td>
                  <td className="px-4 py-3 text-slate-300">{symbol.asset_class}</td>
                  <td className="px-4 py-3 text-slate-300">
                    {symbol.market_data_enabled ? symbol.market_data_ticker ?? "mapped to self" : "disabled"}
                  </td>
                  <td className="px-4 py-3 text-slate-300">{symbol.position_count}</td>
                  <td className="px-4 py-3 text-slate-300">
                    {formatCurrency(symbol.total_market_value, symbol.currency)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </PageShell>
  );
}
