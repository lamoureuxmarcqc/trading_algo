import { PageShell } from "@/components/page-shell";
import { formatCurrency, formatDateTime, getTradingData } from "@/lib/api";

export default async function TradingPage() {
  const { orders, fills, eventSummary, source } = await getTradingData();

  return (
    <PageShell
      eyebrow="Trading Engine"
      title="Execution workstation"
      description="Order entry, broker abstraction, pre-trade checks, partial fills and smart routing telemetry."
      status={source === "api" ? "live api" : "fallback snapshot"}
    >
      <section className="grid gap-4 md:grid-cols-3">
        <div className="panel p-6">
          <p className="metric-label">Orders</p>
          <p className="mt-4 font-mono text-3xl text-white">{orders.length.toString().padStart(2, "0")}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Fills</p>
          <p className="mt-4 font-mono text-3xl text-white">{fills.length.toString().padStart(2, "0")}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Outbox Status</p>
          <p className="mt-4 font-mono text-3xl text-gain">{eventSummary.delivered}</p>
          <p className="mt-2 text-sm text-slate-400">
            {eventSummary.pending} pending / {eventSummary.failed} failed
          </p>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <div className="panel p-6">
          <p className="metric-label">Order Blotter</p>
          <div className="mt-6 overflow-hidden rounded-2xl border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-black/20 text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Symbol</th>
                  <th className="px-4 py-3 font-medium">Side</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">Qty</th>
                  <th className="px-4 py-3 font-medium">Limit</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.id} className="border-t border-line/70">
                    <td className="px-4 py-3 font-mono text-white">{order.symbol}</td>
                    <td className="px-4 py-3 text-slate-300">{order.side}</td>
                    <td className="px-4 py-3 text-accent">{order.status}</td>
                    <td className="px-4 py-3 text-slate-300">{order.quantity}</td>
                    <td className="px-4 py-3 text-slate-300">
                      {order.limit_price ? formatCurrency(order.limit_price) : "market"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Recent Fills</p>
          <div className="mt-5 space-y-3">
            {fills.map((fill) => (
              <div key={`${fill.order_id}-${fill.filled_at}`} className="rounded-2xl border border-line bg-black/20 p-4">
                <div className="flex items-center justify-between">
                  <p className="font-mono text-white">{fill.symbol}</p>
                  <p className="text-sm text-slate-400">{fill.venue}</p>
                </div>
                <p className="mt-2 text-sm text-slate-300">
                  {fill.quantity} @ {formatCurrency(fill.price)} | {formatDateTime(fill.filled_at)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </PageShell>
  );
}
