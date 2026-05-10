import { PageShell } from "@/components/page-shell";
import { formatCurrency, formatPercent, getResearchData } from "@/lib/api";

export default async function ResearchPage() {
  const { screener, factors, sectors, regime, source } = await getResearchData();

  return (
    <PageShell
      eyebrow="Research"
      title="Signal and factor lab"
      description="Screener, factor ranking, regime detection and portfolio allocation experiments in one place."
      status={source === "api" ? "live api" : "fallback snapshot"}
    >
      <section className="grid gap-4 md:grid-cols-3">
        <div className="panel p-6">
          <p className="metric-label">Regime</p>
          <p className="mt-4 text-2xl text-white">{regime.regime.replaceAll("_", " ")}</p>
          <p className="mt-2 text-sm text-slate-400">{regime.recommendation}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Top Signal Confidence</p>
          <p className="mt-4 text-2xl text-white">
            {screener[0] ? formatPercent(screener[0].confidence_score, 0) : "n/a"}
          </p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Best Sector Stance</p>
          <p className="mt-4 text-2xl text-white">{sectors[0]?.stance ?? "n/a"}</p>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <div className="panel p-6">
          <p className="metric-label">Research Screener</p>
          <div className="mt-6 overflow-hidden rounded-2xl border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-black/20 text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Symbol</th>
                  <th className="px-4 py-3 font-medium">Sector</th>
                  <th className="px-4 py-3 font-medium">Price</th>
                  <th className="px-4 py-3 font-medium">Buy Prob.</th>
                  <th className="px-4 py-3 font-medium">Expected Return</th>
                </tr>
              </thead>
              <tbody>
                {screener.map((idea) => (
                  <tr key={idea.symbol} className="border-t border-line/70">
                    <td className="px-4 py-3 font-mono text-white">{idea.symbol}</td>
                    <td className="px-4 py-3 text-slate-300">{idea.sector}</td>
                    <td className="px-4 py-3 text-slate-300">{formatCurrency(idea.price)}</td>
                    <td className="px-4 py-3 text-accent">{formatPercent(idea.buy_probability, 0)}</td>
                    <td className="px-4 py-3 text-gain">{formatPercent(idea.expected_return, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid gap-4">
          <div className="panel p-6">
            <p className="metric-label">Factor Ranking</p>
            <div className="mt-5 space-y-3">
              {factors.map((factor) => (
                <div key={factor.symbol} className="rounded-2xl border border-line bg-black/20 p-4">
                  <div className="flex items-center justify-between">
                    <p className="font-mono text-white">{factor.symbol}</p>
                    <p className="text-sm text-accent">{formatPercent(factor.overall_score, 0)}</p>
                  </div>
                  <p className="mt-2 text-sm text-slate-400">
                    Momentum {formatPercent(factor.momentum_score, 0)} | Quality {formatPercent(factor.quality_score, 0)} | Volatility {formatPercent(factor.volatility_score, 0)}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="panel p-6">
            <p className="metric-label">Sector Rotation</p>
            <div className="mt-5 space-y-3">
              {sectors.map((sector) => (
                <div key={sector.sector} className="rounded-2xl border border-line bg-black/20 p-4">
                  <div className="flex items-center justify-between">
                    <p className="text-white">{sector.sector}</p>
                    <p className="text-sm text-sand">{sector.stance}</p>
                  </div>
                  <p className="mt-2 text-sm text-slate-400">
                    Buy {formatPercent(sector.average_buy_probability, 0)} | Return {formatPercent(sector.average_expected_return, 1)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </PageShell>
  );
}
