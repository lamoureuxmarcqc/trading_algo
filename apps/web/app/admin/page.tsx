import { PageShell } from "@/components/page-shell";
import { formatDateTime, getAdminData } from "@/lib/api";

export default async function AdminPage() {
  const { users, auditLogs, events, eventSummary, source } = await getAdminData();

  return (
    <PageShell
      eyebrow="Admin"
      title="Operational control room"
      description="User administration, RBAC, audit trail inspection and service health monitoring."
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
            <p className="metric-label">Outbox Delivery</p>
            <div className="mt-5 space-y-3">
              {events.slice(0, 4).map((event) => (
                <div key={event.id} className="rounded-2xl border border-line bg-black/20 p-4">
                  <div className="flex items-center justify-between gap-4">
                    <p className="truncate text-white">{event.event_name}</p>
                    <p className="text-sm text-slate-400">{event.delivery_status}</p>
                  </div>
                  <p className="mt-2 text-sm text-slate-300">
                    {event.aggregate_type} {event.aggregate_id.slice(0, 8)} | attempts {event.attempt_count}
                  </p>
                </div>
              ))}
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
    </PageShell>
  );
}
