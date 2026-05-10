import { PageShell } from "@/components/page-shell";
import { formatCurrency, formatPercent, getRiskData } from "@/lib/api";

export default async function RiskPage() {
  const { risk, scenarios, source } = await getRiskData();

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

      <section className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
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
                  <th className="px-4 py-3 font-medium">PnL Impact</th>
                  <th className="px-4 py-3 font-medium">Drawdown</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((scenario) => (
                  <tr key={scenario.scenario_id} className="border-t border-line/70">
                    <td className="px-4 py-3 text-white">{scenario.name}</td>
                    <td className="px-4 py-3 text-loss">{formatCurrency(scenario.estimated_pnl_impact)}</td>
                    <td className="px-4 py-3 text-loss">{formatPercent(scenario.drawdown_impact, 1)}</td>
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
