type PageShellProps = {
  eyebrow: string;
  title: string;
  description: string;
  status?: string;
  children: React.ReactNode;
};

export function PageShell({ eyebrow, title, description, status, children }: PageShellProps) {
  return (
    <main className="min-h-screen px-6 py-8 md:px-10">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <section className="panel p-6 md:p-8">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="metric-label">{eyebrow}</p>
              <h1 className="mt-3 text-4xl font-semibold text-white">{title}</h1>
              <p className="mt-3 max-w-3xl text-slate-400">{description}</p>
            </div>
            {status ? (
              <div className="rounded-full border border-line px-4 py-2 font-mono text-sm text-slate-300">
                Mode: {status}
              </div>
            ) : null}
          </div>
        </section>
        {children}
      </div>
    </main>
  );
}
