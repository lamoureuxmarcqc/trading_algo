import { PageShell } from "@/components/page-shell";

export default function SettingsPage() {
  return (
    <PageShell
      eyebrow="Settings"
      title="Platform configuration"
      description="Environment settings, broker connectors, market data providers and secret management handoff."
    >
      <section className="grid gap-4 md:grid-cols-2">
        <div className="panel p-6">
          <p className="metric-label">Providers</p>
          <ul className="mt-6 space-y-3 text-sm text-slate-300">
            <li>Interactive Brokers</li>
            <li>Alpaca</li>
            <li>Polygon.io</li>
            <li>FRED</li>
          </ul>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Security</p>
          <ul className="mt-6 space-y-3 text-sm text-slate-300">
            <li>JWT refresh rotation</li>
            <li>AES encryption at rest</li>
            <li>Rate limits</li>
            <li>Secrets manager integration</li>
          </ul>
        </div>
      </section>
    </PageShell>
  );
}
