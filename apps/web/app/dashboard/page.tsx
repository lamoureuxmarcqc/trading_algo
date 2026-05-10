import { MetricCard } from "@/components/metric-card";
import { formatCurrency, formatPercent, getDashboardData } from "@/lib/api";

export default async function DashboardPage() {
  const data = await getDashboardData();
  const metricCards: Array<{
    label: string;
    value: string;
    delta: string;
    tone: "gain" | "loss" | "neutral";
  }> = [
    {
      label: "NAV",
      value: formatCurrency(data.portfolio.nav, data.portfolio.base_currency),
      delta: `${formatPercent(data.performance.day_return, 2)} today`,
      tone: data.performance.day_return >= 0 ? "gain" : "loss"
    },
    {
      label: "Gross Exposure",
      value: formatPercent(data.portfolio.gross_exposure),
      delta: `Net ${formatPercent(data.portfolio.net_exposure)}`,
      tone: "neutral" as const
    },
    {
      label: "VaR 95",
      value: formatPercent(data.risk.var_95, 2),
      delta: `CVaR ${formatPercent(data.risk.cvar_95, 2)}`,
      tone: "neutral" as const
    },
    {
      label: "Alpha YTD",
      value: formatPercent(data.performance.alpha_vs_benchmark, 2),
      delta: `${data.regime.regime} regime`,
      tone: data.performance.alpha_vs_benchmark >= 0 ? "gain" : "loss"
    }
  ];

  const riskAlerts = [
    `Largest concentration at ${formatPercent(data.risk.concentration_risk, 1)} of gross book`,
    `Correlation risk currently reading ${formatPercent(data.risk.correlation_risk, 1)}`,
    `Regime confidence ${formatPercent(data.regime.confidence, 0)}: ${data.regime.recommendation}`
  ];

  return (
    <main className="min-h-screen px-6 py-8 md:px-10">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <section className="flex flex-col gap-3">
          <p className="metric-label">CIO Dashboard</p>
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <h1 className="text-4xl font-semibold text-white">Master portfolio command center</h1>
              <p className="mt-2 max-w-3xl text-slate-400">
                Dense by intent, built for fast judgment across NAV, exposure, signal flow, and
                operational risk.
              </p>
            </div>
            <div className="rounded-full border border-line px-4 py-2 font-mono text-sm text-slate-300">
              Mode: {data.source === "api" ? "live api" : "fallback snapshot"}
            </div>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {metricCards.map((metric) => (
            <MetricCard key={metric.label} {...metric} />
          ))}
        </section>

        <section className="grid gap-4 xl:grid-cols-[1.35fr_0.95fr]">
          <div className="panel p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="metric-label">Watchlist / Signals</p>
                <h2 className="mt-2 text-2xl font-semibold">Cross-asset opportunity tape</h2>
              </div>
              <p className="font-mono text-sm text-slate-400">
                Updated {new Date().toISOString().slice(11, 19)} UTC
              </p>
            </div>
            <div className="mt-6 overflow-hidden rounded-2xl border border-line">
              <table className="w-full text-left text-sm">
                <thead className="bg-black/20 text-slate-400">
                  <tr>
                    <th className="px-4 py-3 font-medium">Symbol</th>
                    <th className="px-4 py-3 font-medium">Price</th>
                    <th className="px-4 py-3 font-medium">Signal</th>
                    <th className="px-4 py-3 font-medium">Regime</th>
                  </tr>
                </thead>
                <tbody>
                  {data.signals.map((signal) => (
                    <tr key={signal.symbol} className="border-t border-line/70">
                      <td className="px-4 py-3 font-mono text-white">{signal.symbol}</td>
                      <td className="px-4 py-3 text-slate-300">
                        {formatCurrency(
                          data.portfolio.positions.find((position) => position.symbol === signal.symbol)
                            ?.market_price ?? 0,
                          data.portfolio.base_currency
                        )}
                      </td>
                      <td className="px-4 py-3 text-accent">
                        Buy {formatPercent(signal.buy_probability, 0)}
                      </td>
                      <td className="px-4 py-3 text-slate-300">{signal.market_regime}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel p-6">
            <p className="metric-label">Risk Radar</p>
            <h2 className="mt-2 text-2xl font-semibold">Control priorities</h2>
            <div className="mt-6 space-y-4">
              {riskAlerts.map((alert) => (
                <div key={alert} className="rounded-2xl border border-line bg-black/20 p-4 text-sm text-slate-300">
                  {alert}
                </div>
              ))}
            </div>
            <div className="mt-6 rounded-3xl border border-line bg-gradient-to-br from-accent/10 via-transparent to-gain/10 p-5">
              <p className="metric-label">Stress Panel</p>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <div>
                  <p className="text-sm text-slate-400">{data.scenarios[0]?.name ?? "2008 replay"}</p>
                  <p className="mt-2 font-mono text-2xl text-loss">
                    {formatPercent(data.scenarios[0]?.drawdown_impact ?? 0, 1)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-400">{data.scenarios[1]?.name ?? "Rates +2%"}</p>
                  <p className="mt-2 font-mono text-2xl text-loss">
                    {formatPercent(data.scenarios[1]?.drawdown_impact ?? 0, 1)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
