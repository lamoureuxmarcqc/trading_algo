import { PageShell } from "@/components/page-shell";
import { formatCurrency, formatDateTime, formatPercent, getRiskData } from "@/lib/api";

export default async function RiskPage() {
  const { risk, scenarios, correlationMatrix, source } = await getRiskData();

  return (
    <PageShell
      eyebrow="Risk Engine"
      title="Live risk oversight"
      description="Portfolio VaR, CVaR, beta, correlation and scenario stress mapped for trader and risk officer workflows."
      status={source === "api" ? "live api" : "fallback snapshot"}
    >
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <div className="panel p-6">
          <p className="metric-label">VaR 95</p>
          <p className="mt-4 font-mono text-3xl text-loss">{formatPercent(risk.var_95, 2)}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">CVaR 95</p>
          <p className="mt-4 font-mono text-3xl text-loss">{formatPercent(risk.cvar_95, 2)}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Beta</p>
          <p className="mt-4 font-mono text-3xl text-white">{risk.beta.toFixed(2)}</p>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Drawdown</p>
          <p className="mt-4 font-mono text-3xl text-loss">{formatPercent(risk.drawdown, 2)}</p>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
        <div className="panel p-6">
          <p className="metric-label">Risk Posture</p>
          <div className="mt-6 space-y-4 text-sm text-slate-300">
            <div className="rounded-2xl border border-line bg-black/20 p-4">
              Gross exposure {formatPercent(risk.gross_exposure)} and net exposure {formatPercent(risk.net_exposure)}.
            </div>
            <div className="rounded-2xl border border-line bg-black/20 p-4">
              Concentration risk {formatPercent(risk.concentration_risk, 1)} across the current book.
            </div>
            <div className="rounded-2xl border border-line bg-black/20 p-4">
              Correlation risk running at {formatPercent(risk.correlation_risk, 1)}.
            </div>
          </div>
        </div>

        <div className="panel p-6">
          <p className="metric-label">Scenario Stress</p>
          <div className="mt-6 overflow-hidden rounded-2xl border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-black/20 text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Scenario</th>
                  <th className="px-4 py-3 font-medium">Macro</th>
                  <th className="px-4 py-3 font-medium">PnL Impact</th>
                  <th className="px-4 py-3 font-medium">Drawdown</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((scenario) => (
                  <tr key={scenario.scenario_id} className="border-t border-line/70">
                    <td className="px-4 py-3 text-white">
                      <div>{scenario.name}</div>
                      <div className="mt-1 text-xs text-slate-500">{scenario.period ?? scenario.scenario_id}</div>
                    </td>
                    <td className="px-4 py-3 text-slate-300">
                      <div className="space-y-1">
                        {(scenario.macro_context ?? []).slice(0, 2).map((metric) => (
                          <div key={`${scenario.scenario_id}-${metric.label}`} className="text-xs">
                            {metric.label}: {metric.value}
                          </div>
                        ))}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-loss">{formatCurrency(scenario.estimated_pnl_impact)}</td>
                    <td className="px-4 py-3 text-loss">{formatPercent(scenario.drawdown_impact, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
        <div className="panel p-6">
          <p className="metric-label">Historical Stress Detail</p>
          <div className="mt-5 space-y-4">
            {scenarios.slice(0, 4).map((scenario) => (
              <div key={scenario.scenario_id} className="rounded-2xl border border-line bg-black/20 p-4">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-white">{scenario.name}</p>
                    <p className="mt-1 text-sm text-slate-400">{scenario.trigger}</p>
                  </div>
                  <p className="font-mono text-loss">{formatPercent(scenario.drawdown_impact, 1)}</p>
                </div>
                <p className="mt-3 text-sm text-slate-300">{scenario.summary}</p>
                <div className="mt-3 grid gap-2 md:grid-cols-2">
                  {(scenario.portfolio_impacts ?? []).slice(0, 4).map((impact) => (
                    <div key={`${scenario.scenario_id}-${impact.bucket}`} className="rounded-xl border border-line/70 px-3 py-2 text-sm">
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-slate-300">{impact.bucket}</span>
                        <span className={impact.pnl_impact < 0 ? "text-loss" : "text-gain"}>
                          {formatCurrency(impact.pnl_impact)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel p-6">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="metric-label">Correlation Matrix</p>
              <p className="mt-2 text-sm text-slate-400">{correlationMatrix.methodology}</p>
            </div>
            <div className="text-right text-xs text-slate-500">
              {correlationMatrix.as_of ? formatDateTime(correlationMatrix.as_of) : "latest cache"}
            </div>
          </div>
          <div className="mt-6 overflow-auto rounded-2xl border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-black/20 text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Symbol</th>
                  {correlationMatrix.symbols.map((symbol) => (
                    <th key={`head-${symbol}`} className="px-4 py-3 font-medium">{symbol}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlationMatrix.symbols.map((rowSymbol, rowIndex) => (
                  <tr key={rowSymbol} className="border-t border-line/70">
                    <td className="px-4 py-3 font-mono text-white">{rowSymbol}</td>
                    {correlationMatrix.matrix[rowIndex]?.map((value, colIndex) => (
                      <td key={`${rowSymbol}-${correlationMatrix.symbols[colIndex]}`} className="px-4 py-3 text-slate-300">
                        {value.toFixed(2)}
                      </td>
                    ))}
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
