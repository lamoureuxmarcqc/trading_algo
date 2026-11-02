import { PageShell } from "@/components/page-shell";

export default function LoginPage() {
  return (
    <PageShell
      eyebrow="Secure Access"
      title="Operator login"
      description="Institutional access flow with MFA, SSO and role-based controls."
    >
      <section className="grid gap-4 md:grid-cols-2">
        <div className="panel p-6">
          <p className="metric-label">Primary Auth</p>
          <div className="mt-6 space-y-4">
            <div className="rounded-2xl border border-line bg-black/20 px-4 py-3 text-sm text-slate-400">
              Email / SSO provider
            </div>
            <div className="rounded-2xl border border-line bg-black/20 px-4 py-3 text-sm text-slate-400">
              Password or passkey
            </div>
            <div className="rounded-2xl border border-line bg-black/20 px-4 py-3 text-sm text-slate-400">
              MFA challenge
            </div>
          </div>
        </div>
        <div className="panel p-6">
          <p className="metric-label">Access Policies</p>
          <ul className="mt-6 space-y-3 text-sm text-slate-300">
            <li>JWT + refresh rotation</li>
            <li>Immutable audit logs</li>
            <li>Optional IP whitelisting</li>
            <li>RBAC: admin, trader, analyst, read-only, risk officer</li>
          </ul>
        </div>
      </section>
    </PageShell>
  );
}

