export const topMetrics = [
  { label: "NAV", value: "$1.42M", delta: "+1.84% today", tone: "gain" as const },
  { label: "Gross Exposure", value: "71.4%", delta: "Within limits", tone: "neutral" as const },
  { label: "VaR 95", value: "-2.30%", delta: "Stable vs yesterday", tone: "neutral" as const },
  { label: "Alpha YTD", value: "+4.70%", delta: "Outperforming benchmark", tone: "gain" as const }
];

export const watchlist = [
  { symbol: "AAPL", price: "$191.80", signal: "Buy 67%", regime: "Risk-on" },
  { symbol: "MSFT", price: "$417.50", signal: "Buy 61%", regime: "Risk-on" },
  { symbol: "NVDA", price: "$912.00", signal: "Trim 52%", regime: "Crowded" },
  { symbol: "TLT", price: "$91.35", signal: "Hedge 58%", regime: "Rates +2%" }
];

export const riskAlerts = [
  "Single-name concentration in NVDA approaching soft limit",
  "USD factor sensitivity elevated into macro event window",
  "Liquidity score healthy across top positions"
];

