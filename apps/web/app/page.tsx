import Link from "next/link";
import { ArrowRight, ShieldCheck, Signal, TrendingUp } from "lucide-react";

export default function HomePage() {
  return (
    <main className="min-h-screen px-6 py-8 md:px-10">
      <section className="panel relative overflow-hidden p-8 md:p-12">
        <div className="absolute inset-0 bg-grid bg-[size:36px_36px] opacity-20" />
        <div className="relative mx-auto flex max-w-6xl flex-col gap-10">
          <div className="flex flex-col gap-6 md:max-w-3xl">
            <p className="metric-label">Institutional Trading Platform</p>
            <h1 className="text-4xl font-semibold leading-tight text-white md:text-6xl">
              Built for disciplined execution, live risk control, and machine-assisted alpha.
            </h1>
            <p className="max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
              Family office ready on day one, hedge fund grade by design. The stack pairs your
              existing quant engine with a production API layer, hardened data models, and a premium
              cockpit for CIO, trader, and risk workflows.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 rounded-full bg-sand px-5 py-3 text-sm font-semibold text-slate-950 transition hover:translate-y-[-1px]"
              >
                Open cockpit
                <ArrowRight className="h-4 w-4" />
              </Link>
              <div className="rounded-full border border-line px-5 py-3 text-sm text-slate-300">
                FastAPI + Next.js + PostgreSQL + Redis
              </div>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-3xl border border-line bg-black/20 p-5">
              <TrendingUp className="h-6 w-6 text-accent" />
              <h2 className="mt-4 text-lg font-semibold">Execution</h2>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                OMS-ready order flow, broker abstraction, slippage and route metrics.
              </p>
            </div>
            <div className="rounded-3xl border border-line bg-black/20 p-5">
              <ShieldCheck className="h-6 w-6 text-gain" />
              <h2 className="mt-4 text-lg font-semibold">Risk</h2>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                Live VaR, drawdown, exposure, scenario stress and concentration controls.
              </p>
            </div>
            <div className="rounded-3xl border border-line bg-black/20 p-5">
              <Signal className="h-6 w-6 text-sand" />
              <h2 className="mt-4 text-lg font-semibold">Signals</h2>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                AI-backed forecasts, confidence scoring and regime-aware portfolio actions.
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

