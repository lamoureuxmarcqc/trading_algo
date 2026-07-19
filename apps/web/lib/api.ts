export type DataSource = "api" | "fallback";

export type PortfolioPosition = {
  symbol: string;
  quantity: number;
  average_cost: number;
  market_price: number;
  market_value: number;
  daily_pnl: number;
  unrealized_pnl: number;
  currency: string;
};

export type PortfolioOverview = {
  portfolio_id: string;
  nav: number;
  cash: number;
  gross_exposure: number;
  net_exposure: number;
  base_currency: string;
  benchmark: string;
  market_data_as_of?: string | null;
  positions: PortfolioPosition[];
};

export type PortfolioPerformance = {
  day_return: number;
  month_return: number;
  year_return: number;
  alpha_vs_benchmark: number;
  sharpe_ratio: number;
  max_drawdown: number;
};

export type BarbellAllocationItem = {
  symbol: string;
  bucket: string;
  role: string;
  current_weight: number;
  target_weight: number;
  delta_weight: number;
  buy_probability: number;
  expected_return: number;
  confidence_score: number;
  rationale: string;
};

export type BarbellAllocationResponse = {
  generated_at: string;
  regime: string;
  defensive_weight: number;
  opportunistic_weight: number;
  cash_buffer_weight: number;
  rationale: string;
  allocations: BarbellAllocationItem[];
  rebalance_instructions: Array<{
    symbol: string;
    action: string;
    delta_weight: number;
  }>;
};

export type PortfolioRiskSnapshot = {
  var_95: number;
  cvar_95: number;
  beta: number;
  drawdown: number;
  gross_exposure: number;
  net_exposure: number;
  concentration_risk: number;
  correlation_risk: number;
};

export type RegimeResponse = {
  regime: string;
  confidence: number;
  recommendation: string;
};

export type SignalResponse = {
  symbol: string;
  buy_probability: number;
  sell_probability: number;
  volatility_forecast: number;
  confidence_score: number;
  market_regime: string;
};

export type ScenarioResult = {
  scenario_id: string;
  name: string;
  period?: string | null;
  trigger?: string | null;
  summary?: string | null;
  estimated_pnl_impact: number;
  drawdown_impact: number;
  macro_context?: Array<{ label: string; value: string }>;
  portfolio_impacts?: Array<{ bucket: string; pnl_impact: number; comment: string }>;
  shocks?: Array<{ factor: string; shock: number; contribution: number }>;
};

export type CorrelationMatrixResponse = {
  symbols: string[];
  matrix: number[][];
  as_of?: string | null;
  methodology: string;
};

export type ResearchIdea = {
  symbol: string;
  sector: string;
  price: number;
  buy_probability: number;
  expected_return: number;
  confidence_score: number;
  factor_score: number;
  market_regime: string;
};

export type FactorRank = {
  symbol: string;
  sector: string;
  momentum_score: number;
  quality_score: number;
  volatility_score: number;
  overall_score: number;
};

export type SectorRotation = {
  sector: string;
  average_buy_probability: number;
  average_expected_return: number;
  average_factor_score: number;
  stance: string;
};

export type OrderResponse = {
  id: string;
  symbol: string;
  side: string;
  status: string;
  order_type: string;
  quantity: number;
  filled_quantity: number;
  limit_price: number | null;
  stop_price: number | null;
  broker: string;
  created_at: string;
};

export type Fill = {
  order_id: string;
  symbol: string;
  quantity: number;
  price: number;
  venue: string;
  filled_at: string;
};

export type AdminUser = {
  id: string;
  email: string;
  full_name: string;
  role: string;
  mfa_enabled: boolean;
  is_active: boolean;
  created_at: string;
};

export type AuditLogEntry = {
  id: string;
  event_type: string;
  entity_type: string;
  entity_id?: string | null;
  actor_email: string;
  details?: string | null;
  created_at: string;
};

export type DomainEventEntry = {
  id: string;
  event_name: string;
  topic: string;
  event_version: number;
  tenant_id: string;
  correlation_id?: string | null;
  aggregate_type: string;
  aggregate_id: string;
  delivery_status: string;
  attempt_count: number;
  last_error?: string | null;
  dispatched_at?: string | null;
  payload: Record<string, unknown>;
  created_at: string;
};

export type DomainEventSummary = {
  pending: number;
  failed: number;
  delivered: number;
};

export type AdminSymbolEntry = {
  id: string;
  ticker: string;
  asset_class: string;
  exchange?: string | null;
  currency: string;
  market_data_ticker?: string | null;
  market_data_enabled: boolean;
  position_count: number;
  total_market_value: number;
  last_price?: number | null;
};

export type TerminalSnapshotResponse = {
  generated_at: string;
  portfolio: PortfolioOverview;
  performance: PortfolioPerformance;
  risk: PortfolioRiskSnapshot;
  regime: RegimeResponse;
  history: Array<{
    recorded_at: string;
    nav: number;
    cash: number;
    gross_exposure: number;
    net_exposure: number;
    benchmark: string;
  }>;
  signals: SignalResponse[];
  forecasts: Array<{
    symbol: string;
    forecast_horizon_days: number;
    expected_return: number;
    price_target: number;
    confidence_interval_low: number;
    confidence_interval_high: number;
    catalyst: string;
  }>;
  scenarios: ScenarioResult[];
  barbell: BarbellAllocationResponse;
  correlation_matrix: CorrelationMatrixResponse;
  position_risk: Array<{
    symbol: string;
    beta: number;
    var_95: number;
    cvar_95: number;
    liquidity_score: number;
    concentration_weight: number;
  }>;
  orders: OrderResponse[];
  fills: Fill[];
  research: ResearchIdea[];
  factors: FactorRank[];
  sectors: SectorRotation[];
  users: AdminUser[];
  audit_logs: AuditLogEntry[];
  event_summary: DomainEventSummary;
};

export type TradingAlgoCommand = "analyze" | "compare" | "screen";

export type TradingAlgoCommandRequest = {
  command: TradingAlgoCommand;
  symbols: string[];
  period: string;
  max_symbols: number;
};

export type TradingAlgoSymbolAnalysis = {
  symbol: string;
  period: string;
  rows: number;
  as_of?: string | null;
  latest_price?: number | null;
  daily_return?: number | null;
  total_return?: number | null;
  volatility_20d?: number | null;
  sharpe_ratio?: number | null;
  var_95?: number | null;
  max_drawdown?: number | null;
  rsi?: number | null;
  sma_20?: number | null;
  sma_50?: number | null;
  trend: string;
  recommendation: string;
};

export type TradingAlgoCommandResponse = {
  command: TradingAlgoCommand | string;
  status: "ok" | "partial" | "error" | string;
  generated_at: string;
  summary: string;
  analyses: TradingAlgoSymbolAnalysis[];
  errors: string[];
};

export type DashboardData = {
  portfolio: PortfolioOverview;
  performance: PortfolioPerformance;
  risk: PortfolioRiskSnapshot;
  regime: RegimeResponse;
  signals: SignalResponse[];
  scenarios: ScenarioResult[];
  source: DataSource;
};

export type PortfolioData = {
  portfolio: PortfolioOverview;
  performance: PortfolioPerformance;
  barbell: BarbellAllocationResponse;
  source: DataSource;
};

export type RiskData = {
  risk: PortfolioRiskSnapshot;
  scenarios: ScenarioResult[];
  correlationMatrix: CorrelationMatrixResponse;
  source: DataSource;
};

export type ResearchData = {
  screener: ResearchIdea[];
  factors: FactorRank[];
  sectors: SectorRotation[];
  regime: RegimeResponse;
  source: DataSource;
};

export type TradingData = {
  orders: OrderResponse[];
  fills: Fill[];
  eventSummary: DomainEventSummary;
  source: DataSource;
};

export type AdminData = {
  users: AdminUser[];
  auditLogs: AuditLogEntry[];
  events: DomainEventEntry[];
  eventSummary: DomainEventSummary;
  symbols: AdminSymbolEntry[];
  unresolvedSymbols: AdminSymbolEntry[];
  source: DataSource;
};

const API_BASE_URL =
  process.env.API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000/api/v1";

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    cache: "no-store",
    headers: {
      "X-Tenant-ID": "family-office-demo"
    }
  });

  if (!response.ok) {
    throw new Error(`API request failed for ${path}: ${response.status}`);
  }

  return (await response.json()) as T;
}

async function fetchTerminalSnapshot(): Promise<TerminalSnapshotResponse> {
  return fetchJson<TerminalSnapshotResponse>("/terminal/snapshot");
}

function fallbackDashboardData(): DashboardData {
  return {
    portfolio: {
      portfolio_id: "family-office-master",
      nav: 1415460,
      cash: 412000,
      gross_exposure: 0.7089,
      net_exposure: 0.7089,
      base_currency: "USD",
      benchmark: "SPY",
      market_data_as_of: "2026-05-09T13:30:00+00:00",
      positions: [
        {
          symbol: "AAPL",
          quantity: 1200,
          average_cost: 178.4,
          market_price: 191.8,
          market_value: 230160,
          daily_pnl: 3240,
          unrealized_pnl: 16080,
          currency: "USD"
        },
        {
          symbol: "MSFT",
          quantity: 760,
          average_cost: 404.2,
          market_price: 417.5,
          market_value: 317300,
          daily_pnl: 2432,
          unrealized_pnl: 10108,
          currency: "USD"
        },
        {
          symbol: "NVDA",
          quantity: 500,
          average_cost: 834.5,
          market_price: 912,
          market_value: 456000,
          daily_pnl: 5100,
          unrealized_pnl: 38750,
          currency: "USD"
        }
      ]
    },
    performance: {
      day_return: 0.0064,
      month_return: 0.034,
      year_return: 0.182,
      alpha_vs_benchmark: 0.047,
      sharpe_ratio: 1.74,
      max_drawdown: -0.081
    },
    risk: {
      var_95: -0.023,
      cvar_95: -0.034,
      beta: 1.08,
      drawdown: -0.081,
      gross_exposure: 0.7089,
      net_exposure: 0.7089,
      concentration_risk: 0.454,
      correlation_risk: 0.62
    },
    regime: {
      regime: "risk_on",
      confidence: 0.78,
      recommendation: "Scale into winners"
    },
    signals: [
      {
        symbol: "AAPL",
        buy_probability: 0.67,
        sell_probability: 0.19,
        volatility_forecast: 0.24,
        confidence_score: 0.81,
        market_regime: "risk_on"
      },
      {
        symbol: "MSFT",
        buy_probability: 0.67,
        sell_probability: 0.19,
        volatility_forecast: 0.24,
        confidence_score: 0.81,
        market_regime: "risk_on"
      },
      {
        symbol: "NVDA",
        buy_probability: 0.67,
        sell_probability: 0.19,
        volatility_forecast: 0.24,
        confidence_score: 0.81,
        market_regime: "risk_on"
      }
    ],
    scenarios: [
      {
        scenario_id: "1929",
        name: "Great Depression",
        period: "1929-1932",
        trigger: "Credit collapse and forced deleveraging",
        summary: "Severe growth equity and liquidity shock.",
        estimated_pnl_impact: -441000,
        drawdown_impact: -0.311,
        macro_context: [
          { label: "Inflation", value: "-2.1%" },
          { label: "Unemployment", value: "24.9%" },
          { label: "Policy rate", value: "0.6%" }
        ],
        portfolio_impacts: [
          { bucket: "NVDA", pnl_impact: -223440, comment: "High beta growth repricing." },
          { bucket: "MSFT", pnl_impact: -152304, comment: "Software multiples compress." },
          { bucket: "AAPL", pnl_impact: -112778, comment: "Consumer hardware demand shock." }
        ],
        shocks: [
          { factor: "Equities", shock: -0.44, contribution: -147000 },
          { factor: "Liquidity", shock: -0.18, contribution: -147000 },
          { factor: "Deflation", shock: -0.07, contribution: -147000 }
        ]
      },
      {
        scenario_id: "1973_oil",
        name: "Oil Shock",
        period: "1973-1974",
        trigger: "Oil embargo and inflation shock",
        summary: "Growth book reprices as energy and rates dominate.",
        estimated_pnl_impact: -219000,
        drawdown_impact: -0.155,
        macro_context: [
          { label: "Inflation", value: "8.7%" },
          { label: "Unemployment", value: "4.9%" },
          { label: "Policy rate", value: "10.8%" }
        ],
        portfolio_impacts: [],
        shocks: [
          { factor: "Oil", shock: 0.7, contribution: -73000 },
          { factor: "Rates", shock: 0.03, contribution: -73000 },
          { factor: "Equities", shock: -0.22, contribution: -73000 }
        ]
      },
      {
        scenario_id: "1989",
        name: "1989 Leverage Crack",
        period: "1989-1990",
        trigger: "Leverage unwind and growth slowdown",
        summary: "Quality growth and financial conditions tighten quickly.",
        estimated_pnl_impact: -159000,
        drawdown_impact: -0.112,
        macro_context: [
          { label: "Inflation", value: "4.8%" },
          { label: "Unemployment", value: "5.3%" },
          { label: "Policy rate", value: "8.1%" }
        ],
        portfolio_impacts: [],
        shocks: [
          { factor: "Equities", shock: -0.16, contribution: -53000 },
          { factor: "Credit", shock: -0.05, contribution: -53000 },
          { factor: "Property", shock: -0.08, contribution: -53000 }
        ]
      },
      {
        scenario_id: "2000_tech",
        name: "Tech Bubble Burst",
        period: "2000-2002",
        trigger: "Technology multiple compression",
        summary: "Semis and software dominate the drawdown profile.",
        estimated_pnl_impact: -309000,
        drawdown_impact: -0.218,
        macro_context: [
          { label: "Inflation", value: "3.4%" },
          { label: "Unemployment", value: "4.0%" },
          { label: "Policy rate", value: "6.5%" }
        ],
        portfolio_impacts: [],
        shocks: [
          { factor: "Growth", shock: -0.34, contribution: -103000 },
          { factor: "Volatility", shock: 0.38, contribution: -103000 },
          { factor: "Funding", shock: -0.06, contribution: -103000 }
        ]
      },
      {
        scenario_id: "2008",
        name: "Global Financial Crisis",
        period: "2008-2009",
        trigger: "Housing and credit collapse",
        summary: "Broad equity drawdown and widening credit spreads.",
        estimated_pnl_impact: -184000,
        drawdown_impact: -0.137
      },
      {
        scenario_id: "2020_pandemic",
        name: "Pandemic Shock",
        period: "2020",
        trigger: "Global shutdowns and liquidity scramble",
        summary: "Fast drawdown with violent policy response.",
        estimated_pnl_impact: -121500,
        drawdown_impact: -0.092
      },
      {
        scenario_id: "2022_inflation",
        name: "Inflation and Rates Shock",
        period: "2022",
        trigger: "Sticky inflation and aggressive hikes",
        summary: "Duration-sensitive book reprices hard.",
        estimated_pnl_impact: -164500,
        drawdown_impact: -0.116
      }
    ],
    source: "fallback"
  };
}

function fallbackCorrelationMatrix(): CorrelationMatrixResponse {
  return {
    symbols: ["AAPL", "MSFT", "NVDA"],
    matrix: [
      [1, 0.74, 0.69],
      [0.74, 1, 0.72],
      [0.69, 0.72, 1]
    ],
    as_of: "2026-05-14T13:30:00+00:00",
    methodology: "Fallback daily return correlation matrix over the last 1 year."
  };
}

function fallbackBarbellAllocation(): BarbellAllocationResponse {
  return {
    generated_at: "2026-05-14T13:30:00+00:00",
    regime: "risk_on",
    defensive_weight: 0.35,
    opportunistic_weight: 0.55,
    cash_buffer_weight: 0.1,
    rationale:
      "Barbell posture preserves a 10% liquidity sleeve while pairing defensive ballast with high-conviction growth winners.",
    allocations: [
      {
        symbol: "SGOV",
        bucket: "defensive",
        role: "cash_surrogate",
        current_weight: 0,
        target_weight: 0.15,
        delta_weight: 0.15,
        buy_probability: 0.52,
        expected_return: 0.02,
        confidence_score: 0.77,
        rationale: "Short-duration Treasury ballast keeps redeployment optionality high."
      },
      {
        symbol: "TLT",
        bucket: "defensive",
        role: "duration_hedge",
        current_weight: 0,
        target_weight: 0.1,
        delta_weight: 0.1,
        buy_probability: 0.5,
        expected_return: 0.03,
        confidence_score: 0.69,
        rationale: "Duration hedge for growth and liquidity shocks."
      },
      {
        symbol: "GLD",
        bucket: "defensive",
        role: "real_asset_hedge",
        current_weight: 0,
        target_weight: 0.1,
        delta_weight: 0.1,
        buy_probability: 0.54,
        expected_return: 0.04,
        confidence_score: 0.7,
        rationale: "Real-asset hedge against inflation and policy surprise."
      },
      {
        symbol: "NVDA",
        bucket: "opportunistic",
        role: "high_beta_growth",
        current_weight: 0.3222,
        target_weight: 0.2,
        delta_weight: -0.1222,
        buy_probability: 0.71,
        expected_return: 0.14,
        confidence_score: 0.82,
        rationale: "Convex AI beta lives in the upside sleeve."
      },
      {
        symbol: "MSFT",
        bucket: "opportunistic",
        role: "compounder",
        current_weight: 0.2242,
        target_weight: 0.2,
        delta_weight: -0.0242,
        buy_probability: 0.66,
        expected_return: 0.09,
        confidence_score: 0.8,
        rationale: "Quality compounder for the upside sleeve."
      },
      {
        symbol: "AAPL",
        bucket: "opportunistic",
        role: "franchise_growth",
        current_weight: 0.1626,
        target_weight: 0.15,
        delta_weight: -0.0126,
        buy_probability: 0.62,
        expected_return: 0.07,
        confidence_score: 0.76,
        rationale: "Liquid franchise growth anchor."
      },
      {
        symbol: "CASH",
        bucket: "cash",
        role: "liquidity_reserve",
        current_weight: 0.2911,
        target_weight: 0.1,
        delta_weight: -0.1911,
        buy_probability: 0.5,
        expected_return: 0,
        confidence_score: 1,
        rationale: "Liquidity reserve to absorb volatility and fund redeployment."
      }
    ],
    rebalance_instructions: [
      { symbol: "SGOV", action: "BUY", delta_weight: 0.15 },
      { symbol: "TLT", action: "BUY", delta_weight: 0.1 },
      { symbol: "GLD", action: "BUY", delta_weight: 0.1 },
      { symbol: "NVDA", action: "SELL", delta_weight: -0.1222 },
      { symbol: "MSFT", action: "SELL", delta_weight: -0.0242 },
      { symbol: "AAPL", action: "SELL", delta_weight: -0.0126 }
    ]
  };
}

function fallbackResearchData(): ResearchData {
  return {
    screener: [
      {
        symbol: "NVDA",
        sector: "Semiconductors",
        price: 912,
        buy_probability: 0.71,
        expected_return: 0.14,
        confidence_score: 0.82,
        factor_score: 0.87,
        market_regime: "risk_on"
      },
      {
        symbol: "MSFT",
        sector: "Software",
        price: 417.5,
        buy_probability: 0.66,
        expected_return: 0.09,
        confidence_score: 0.8,
        factor_score: 0.79,
        market_regime: "risk_on"
      },
      {
        symbol: "AAPL",
        sector: "Hardware",
        price: 191.8,
        buy_probability: 0.62,
        expected_return: 0.07,
        confidence_score: 0.76,
        factor_score: 0.74,
        market_regime: "risk_on"
      }
    ],
    factors: [
      {
        symbol: "NVDA",
        sector: "Semiconductors",
        momentum_score: 0.91,
        quality_score: 0.84,
        volatility_score: 0.58,
        overall_score: 0.84
      },
      {
        symbol: "MSFT",
        sector: "Software",
        momentum_score: 0.82,
        quality_score: 0.89,
        volatility_score: 0.69,
        overall_score: 0.8
      },
      {
        symbol: "AAPL",
        sector: "Hardware",
        momentum_score: 0.74,
        quality_score: 0.86,
        volatility_score: 0.72,
        overall_score: 0.77
      }
    ],
    sectors: [
      {
        sector: "Technology",
        average_buy_probability: 0.67,
        average_expected_return: 0.1,
        average_factor_score: 0.8,
        stance: "overweight"
      },
      {
        sector: "Healthcare",
        average_buy_probability: 0.54,
        average_expected_return: 0.05,
        average_factor_score: 0.63,
        stance: "neutral"
      }
    ],
    regime: {
      regime: "risk_on",
      confidence: 0.78,
      recommendation: "Scale into winners"
    },
    source: "fallback"
  };
}

function fallbackTradingData(): TradingData {
  return {
    orders: [
      {
        id: "seed-msft-1",
        symbol: "MSFT",
        side: "BUY",
        status: "filled",
        order_type: "limit",
        quantity: 100,
        filled_quantity: 100,
        limit_price: 415.2,
        stop_price: null,
        broker: "Interactive Brokers",
        created_at: "2026-05-09T09:10:00+00:00"
      }
    ],
    fills: [
      {
        order_id: "seed-msft-1",
        symbol: "MSFT",
        quantity: 100,
        price: 415.2,
        venue: "IEX",
        filled_at: "2026-05-09T09:10:01+00:00"
      }
    ],
    eventSummary: {
      pending: 0,
      failed: 0,
      delivered: 3
    },
    source: "fallback"
  };
}

function fallbackAdminData(): AdminData {
  return {
    users: [
      {
        id: "u-cio",
        email: "cio@hedgefund.local",
        full_name: "Chief Investment Officer",
        role: "admin",
        mfa_enabled: true,
        is_active: true,
        created_at: "2026-05-01T08:00:00+00:00"
      },
      {
        id: "u-risk",
        email: "risk@hedgefund.local",
        full_name: "Risk Officer",
        role: "risk_officer",
        mfa_enabled: false,
        is_active: true,
        created_at: "2026-05-01T08:05:00+00:00"
      }
    ],
    auditLogs: [
      {
        id: "audit-1",
        event_type: "com.terminal.orders.created.v1",
        entity_type: "order",
        entity_id: "seed-msft-1",
        actor_email: "cio@hedgefund.local",
        details: "BUY 100 MSFT via Interactive Brokers",
        created_at: "2026-05-09T09:10:00+00:00"
      },
      {
        id: "audit-2",
        event_type: "com.terminal.portfolio.refreshed.v1",
        entity_type: "portfolio",
        entity_id: "family-office-master",
        actor_email: "system",
        details: "Refreshed 3 positions from live market data",
        created_at: "2026-05-09T09:15:00+00:00"
      }
    ],
    events: [
      {
        id: "evt-1",
        event_name: "com.terminal.orders.created.v1",
        topic: "terminal.orders.v1",
        event_version: 1,
        tenant_id: "family-office-demo",
        correlation_id: "corr-seed-1",
        aggregate_type: "order",
        aggregate_id: "seed-msft-1",
        delivery_status: "delivered",
        attempt_count: 1,
        last_error: null,
        dispatched_at: "2026-05-09T09:10:02+00:00",
        payload: {
          symbol: "MSFT",
          status: "filled"
        },
        created_at: "2026-05-09T09:10:00+00:00"
      }
    ],
    eventSummary: {
      pending: 0,
      failed: 0,
      delivered: 3
    },
    symbols: [
      {
        id: "sym-aapl",
        ticker: "AAPL",
        asset_class: "equity",
        exchange: "NASDAQ",
        currency: "USD",
        market_data_ticker: "AAPL",
        market_data_enabled: true,
        position_count: 1,
        total_market_value: 230160,
        last_price: 191.8
      },
      {
        id: "sym-rbf",
        ticker: "RBF2011.TO",
        asset_class: "mutual_fund",
        exchange: "TSX",
        currency: "CAD",
        market_data_ticker: null,
        market_data_enabled: false,
        position_count: 3,
        total_market_value: 87453.66,
        last_price: 10
      }
    ],
    unresolvedSymbols: [
      {
        id: "sym-rbf",
        ticker: "RBF2011.TO",
        asset_class: "mutual_fund",
        exchange: "TSX",
        currency: "CAD",
        market_data_ticker: null,
        market_data_enabled: false,
        position_count: 3,
        total_market_value: 87453.66,
        last_price: 10
      }
    ],
    source: "fallback"
  };
}

export async function getDashboardData(): Promise<DashboardData> {
  try {
    const snapshot = await fetchTerminalSnapshot();

    return {
      portfolio: snapshot.portfolio,
      performance: snapshot.performance,
      risk: snapshot.risk,
      regime: snapshot.regime,
      signals: snapshot.signals,
      scenarios: snapshot.scenarios,
      source: "api"
    };
  } catch {
    return fallbackDashboardData();
  }
}

export async function getPortfolioData(): Promise<PortfolioData> {
  try {
    const snapshot = await fetchTerminalSnapshot();
    return {
      portfolio: snapshot.portfolio,
      performance: snapshot.performance,
      barbell: snapshot.barbell,
      source: "api"
    };
  } catch {
    const fallback = fallbackDashboardData();
    return {
      portfolio: fallback.portfolio,
      performance: fallback.performance,
      barbell: fallbackBarbellAllocation(),
      source: "fallback"
    };
  }
}

export async function getRiskData(): Promise<RiskData> {
  try {
    const snapshot = await fetchTerminalSnapshot();
    return {
      risk: snapshot.risk,
      scenarios: snapshot.scenarios,
      correlationMatrix: snapshot.correlation_matrix,
      source: "api"
    };
  } catch {
    const fallback = fallbackDashboardData();
    return {
      risk: fallback.risk,
      scenarios: fallback.scenarios,
      correlationMatrix: fallbackCorrelationMatrix(),
      source: "fallback"
    };
  }
}

export async function getResearchData(): Promise<ResearchData> {
  try {
    const snapshot = await fetchTerminalSnapshot();
    return {
      screener: snapshot.research,
      factors: snapshot.factors,
      sectors: snapshot.sectors,
      regime: snapshot.regime,
      source: "api"
    };
  } catch {
    return fallbackResearchData();
  }
}

export async function getTradingData(): Promise<TradingData> {
  try {
    const snapshot = await fetchTerminalSnapshot();
    return {
      orders: snapshot.orders,
      fills: snapshot.fills,
      eventSummary: snapshot.event_summary,
      source: "api"
    };
  } catch {
    return fallbackTradingData();
  }
}

export async function getAdminData(): Promise<AdminData> {
  try {
    const [snapshot, events, symbols, unresolvedSymbols] = await Promise.all([
      fetchTerminalSnapshot(),
      fetchJson<DomainEventEntry[]>("/admin/events"),
      fetchJson<AdminSymbolEntry[]>("/admin/symbols"),
      fetchJson<AdminSymbolEntry[]>("/admin/symbols?unresolved_only=true")
    ]);
    return {
      users: snapshot.users,
      auditLogs: snapshot.audit_logs,
      events,
      eventSummary: snapshot.event_summary,
      symbols,
      unresolvedSymbols,
      source: "api"
    };
  } catch {
    return fallbackAdminData();
  }
}

export function formatCurrency(value: number, currency = "USD"): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 0
  }).format(value);
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "n/a";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short"
  }).format(new Date(value));
}
