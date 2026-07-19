"use client";

import { Play, RefreshCcw } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";

import type {
  TradingAlgoCommand,
  TradingAlgoCommandResponse,
  TradingAlgoSymbolAnalysis
} from "@/lib/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

const commandOptions: TradingAlgoCommand[] = ["analyze", "compare", "screen"];
const periodOptions = ["6mo", "1y", "2y", "5y"];

const initialResult: TradingAlgoCommandResponse = {
  command: "compare",
  status: "idle",
  generated_at: "",
  summary: "Awaiting command.",
  analyses: [],
  errors: []
};

function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value);
}

function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function toneClass(value: string): string {
  const token = value.toLowerCase();
  if (["buy", "bullish", "ok"].includes(token)) {
    return "border-gain/40 bg-gain/10 text-gain";
  }
  if (["reduce", "bearish", "error"].includes(token)) {
    return "border-loss/40 bg-loss/10 text-loss";
  }
  return "border-line bg-black/20 text-accent";
}

function rowTone(analysis: TradingAlgoSymbolAnalysis): string {
  if (analysis.recommendation === "buy") {
    return "text-gain";
  }
  if (analysis.recommendation === "reduce") {
    return "text-loss";
  }
  return "text-slate-300";
}

export function TradingAlgoConsole() {
  const [command, setCommand] = useState<TradingAlgoCommand>("compare");
  const [symbols, setSymbols] = useState("AAPL,MSFT,NVDA");
  const [period, setPeriod] = useState("1y");
  const [maxSymbols, setMaxSymbols] = useState(8);
  const [result, setResult] = useState<TradingAlgoCommandResponse>(initialResult);
  const [isLoading, setIsLoading] = useState(false);

  const visibleSymbols = useMemo(
    () =>
      symbols
        .split(",")
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean)
        .slice(0, maxSymbols),
    [symbols, maxSymbols]
  );

  async function submitCommand(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/terminal/trading-algo`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-ID": "family-office-demo"
        },
        body: JSON.stringify({
          command,
          symbols: visibleSymbols,
          period,
          max_symbols: maxSymbols
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      setResult((await response.json()) as TradingAlgoCommandResponse);
    } catch (error) {
      setResult({
        command,
        status: "error",
        generated_at: new Date().toISOString(),
        summary: error instanceof Error ? error.message : "Trading-algo request failed.",
        analyses: [],
        errors: [error instanceof Error ? error.message : "Unknown error"]
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="panel p-6">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="metric-label">Trading Algo</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">Quant command console</h2>
        </div>
        <div className={`w-fit rounded-full border px-3 py-2 font-mono text-xs ${toneClass(result.status)}`}>
          {result.status}
        </div>
      </div>

      <form onSubmit={submitCommand} className="mt-6 grid gap-3 lg:grid-cols-[0.85fr_1.5fr_0.75fr_0.6fr_auto]">
        <label className="flex flex-col gap-2 text-sm text-slate-400">
          Command
          <select
            value={command}
            onChange={(event) => setCommand(event.target.value as TradingAlgoCommand)}
            className="rounded-lg border border-line bg-black/30 px-3 py-3 text-white outline-none focus:border-accent"
          >
            {commandOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-2 text-sm text-slate-400">
          Symbols
          <input
            value={symbols}
            onChange={(event) => setSymbols(event.target.value)}
            className="rounded-lg border border-line bg-black/30 px-3 py-3 font-mono text-white outline-none focus:border-accent"
          />
        </label>

        <label className="flex flex-col gap-2 text-sm text-slate-400">
          Period
          <select
            value={period}
            onChange={(event) => setPeriod(event.target.value)}
            className="rounded-lg border border-line bg-black/30 px-3 py-3 text-white outline-none focus:border-accent"
          >
            {periodOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-2 text-sm text-slate-400">
          Max
          <input
            type="number"
            min={1}
            max={25}
            value={maxSymbols}
            onChange={(event) => setMaxSymbols(Number(event.target.value))}
            className="rounded-lg border border-line bg-black/30 px-3 py-3 font-mono text-white outline-none focus:border-accent"
          />
        </label>

        <button
          type="submit"
          disabled={isLoading || visibleSymbols.length === 0}
          className="inline-flex min-h-[46px] items-center justify-center gap-2 rounded-lg border border-sand/40 bg-sand px-4 py-3 font-semibold text-ink disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          Run
        </button>
      </form>

      <div className="mt-5 rounded-xl border border-line bg-black/20 p-4 text-sm text-slate-300">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <p>{result.summary}</p>
          <p className="font-mono text-xs text-slate-500">
            {result.generated_at ? new Date(result.generated_at).toLocaleTimeString() : "not run"}
          </p>
        </div>
        {result.errors.length ? (
          <p className="mt-3 text-loss">{result.errors.slice(0, 2).join(" | ")}</p>
        ) : null}
      </div>

      <div className="mt-5 overflow-hidden rounded-xl border border-line">
        <table className="w-full min-w-[920px] text-left text-sm">
          <thead className="bg-black/20 text-slate-400">
            <tr>
              <th className="px-4 py-3 font-medium">Symbol</th>
              <th className="px-4 py-3 font-medium">Trend</th>
              <th className="px-4 py-3 font-medium">Rec.</th>
              <th className="px-4 py-3 font-medium">Price</th>
              <th className="px-4 py-3 font-medium">Day</th>
              <th className="px-4 py-3 font-medium">Total</th>
              <th className="px-4 py-3 font-medium">Vol 20D</th>
              <th className="px-4 py-3 font-medium">Sharpe</th>
              <th className="px-4 py-3 font-medium">VaR 95</th>
              <th className="px-4 py-3 font-medium">RSI</th>
            </tr>
          </thead>
          <tbody>
            {result.analyses.length ? (
              result.analyses.map((analysis) => (
                <tr key={analysis.symbol} className="border-t border-line/70">
                  <td className="px-4 py-3 font-mono text-white">{analysis.symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`rounded-md border px-2 py-1 text-xs ${toneClass(analysis.trend)}`}>
                      {analysis.trend}
                    </span>
                  </td>
                  <td className={`px-4 py-3 ${rowTone(analysis)}`}>{analysis.recommendation}</td>
                  <td className="px-4 py-3 text-slate-300">{formatCurrency(analysis.latest_price)}</td>
                  <td className="px-4 py-3 text-slate-300">{formatPercent(analysis.daily_return, 2)}</td>
                  <td className="px-4 py-3 text-slate-300">{formatPercent(analysis.total_return, 1)}</td>
                  <td className="px-4 py-3 text-slate-300">{formatPercent(analysis.volatility_20d, 1)}</td>
                  <td className="px-4 py-3 text-slate-300">{formatNumber(analysis.sharpe_ratio, 2)}</td>
                  <td className="px-4 py-3 text-loss">{formatPercent(analysis.var_95, 2)}</td>
                  <td className="px-4 py-3 text-slate-300">{formatNumber(analysis.rsi, 1)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="px-4 py-8 text-center text-slate-500" colSpan={10}>
                  No results
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
