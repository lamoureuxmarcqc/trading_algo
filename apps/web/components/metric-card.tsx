type MetricCardProps = {
  label: string;
  value: string;
  delta?: string;
  tone?: "gain" | "loss" | "neutral";
};

export function MetricCard({ label, value, delta, tone = "neutral" }: MetricCardProps) {
  const toneClass =
    tone === "gain" ? "text-gain" : tone === "loss" ? "text-loss" : "text-slate-300";

  return (
    <div className="panel p-6">
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
      {delta ? <p className={`mt-3 text-sm font-medium ${toneClass}`}>{delta}</p> : null}
    </div>
  );
}

