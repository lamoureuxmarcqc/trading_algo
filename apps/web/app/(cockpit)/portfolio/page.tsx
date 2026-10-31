import { PageShell } from "@/components/page-shell";
import { formatCurrency, formatPercent, getPortfolioData } from "@/lib/api";

export default async function PortfolioPage() {
  const { portfolio, performance, barbell, source } = await getPortfolioData();
  const livePnl = portfolio.positions.reduce((sum, position) => sum + position.unrealized_pnl, 0);

  return (
    <PageShell
      eyebrow="Portfolio Engine"
      title="Consolidated portfolio"
      description="Multi-account, multi-currency exposure tracking with tax lots, cash balances and benchmark analytics."
      status={source === "api" ? "live api" : "fallback snapshot"}
    >
      <section className="grid gap-4 lg:grid-cols-3">
        <div className="panel p-6">
          <p className="metric-label">Accounts</p>
          <p className="mt-4 font-mono text-3xl text-white">01</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Base Currency</p>
          <p className="mt-4 font-mono text-3xl text-white">{portfolio.base_currency}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Live PnL</p>
          <p className="mt-4 font-mono text-3xl text-gain">{formatCurrency(livePnl, portfolio.base_currency)}</p>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="panel p-6">
          <p className="metric-label">Portfolio Snapshot</p>
          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            <div>
              <p className="text-sm text-slate-400">NAV</p>
              <p className="mt-2 font-mono text-2xl text-white">
                {formatCurrency(portfolio.nav, portfolio.base_currency)}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Cash</p>
              <p className="mt-2 font-mono text-2xl text-white">
                {formatCurrency(portfolio.cash, portfolio.base_currency)}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Gross Exposure</p>
              <p className="mt-2 font-mono text-2xl text-white">
                {formatPercent(portfolio.gross_exposure)}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Alpha YTD</p>
              <p className="mt-2 font-mono text-2xl text-gain">
                {formatPercent(performance.alpha_vs_benchmark, 2)}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Market Data As Of</p>
              <p className="mt-2 font-mono text-lg text-white">
                {portfolio.market_data_as_of ? new Date(portfolio.market_data_as_of).toISOString().slice(11, 19) : "n/a"} UTC
              </p>
            </div>
          </div>
        </div>

        <div className="panel p-6">
          <p className="metric-label">Positions</p>
          <div className="mt-6 overflow-hidden rounded-2xl border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-black/20 text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Symbol</th>
                  <th className="px-4 py-3 font-medium">Qty</th>
                  <th className="px-4 py-3 font-medium">Market Value</th>
                  <th className="px-4 py-3 font-medium">Unrealized PnL</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.positions.map((position) => (
                  <tr key={position.symbol} className="border-t border-line/70">
                    <td className="px-4 py-3 font-mono text-white">{position.symbol}</td>
                    <td className="px-4 py-3 text-slate-300">{position.quantity}</td>
                    <td className="px-4 py-3 text-slate-300">
                      {formatCurrency(position.market_value, portfolio.base_currency)}
                    </td>
                    <td className="px-4 py-3 text-gain">
                      {formatCurrency(position.unrealized_pnl, portfolio.base_currency)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="panel p-6">
          <p className="metric-label">Barbell Allocation</p>
          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            <div>
              <p className="text-sm text-slate-400">Defensive Sleeve</p>
              <p className="mt-2 font-mono text-2xl text-white">{formatPercent(barbell.defensive_weight)}</p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Opportunistic Sleeve</p>
              <p className="mt-2 font-mono text-2xl text-white">{formatPercent(barbell.opportunistic_weight)}</p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Cash Buffer</p>
              <p className="mt-2 font-mono text-2xl text-white">{formatPercent(barbell.cash_buffer_weight)}</p>
            </div>
          </div>
          <div className="mt-6 rounded-2xl border border-line bg-black/10 p-4">
            <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Regime</p>
            <p className="mt-2 font-mono text-lg text-white">{barbell.regime}</p>
            <p className="mt-3 text-sm leading-6 text-slate-300">{barbell.rationale}</p>
          </div>
        </div>

        <div className="panel p-6">
          <p className="metric-label">Barbell Rebalance</p>
          <div className="mt-6 overflow-hidden rounded-2xl border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-black/20 text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Symbol</th>
                  <th className="px-4 py-3 font-medium">Bucket</th>
                  <th className="px-4 py-3 font-medium">Current</th>
                  <th className="px-4 py-3 font-medium">Target</th>
                  <th className="px-4 py-3 font-medium">Delta</th>
                </tr>
              </thead>
              <tbody>
                {barbell.allocations.map((allocation) => (
                  <tr key={allocation.symbol} className="border-t border-line/70 align-top">
                    <td className="px-4 py-3 font-mono text-white">{allocation.symbol}</td>
                    <td className="px-4 py-3 text-slate-300">{allocation.bucket}</td>
                    <td className="px-4 py-3 text-slate-300">{formatPercent(allocation.current_weight, 2)}</td>
                    <td className="px-4 py-3 text-white">{formatPercent(allocation.target_weight, 2)}</td>
                    <td className={allocation.delta_weight >= 0 ? "px-4 py-3 text-gain" : "px-4 py-3 text-loss"}>
                      {formatPercent(allocation.delta_weight, 2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </PageShell>
  );
}
