const fmtCurrency = (value, currency = "USD") =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 0
  }).format(value ?? 0);

const fmtPercent = (value, digits = 1) => `${((value ?? 0) * 100).toFixed(digits)}%`;
const fmtTime = (value) => new Date(value).toLocaleString();
const roleLabel = (value) => String(value ?? "").replaceAll("_", " ");
const fmtNullable = (value, digits = 2) => (value === null || value === undefined ? "--" : Number(value).toFixed(digits));
const terminalState = {
  filter: "",
  latestOptimization: null,
  projectedExposure: null,
  activePendingOrderId: null
};

function fmtAge(value) {
  if (!value) {
    return "not timestamped";
  }

  const timestamp = new Date(value).getTime();
  if (Number.isNaN(timestamp)) {
    return "invalid timestamp";
  }

  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 60) {
    return `${seconds}s ago`;
  }

  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }

  const hours = Math.round(minutes / 60);
  return `${hours}h ago`;
}

function timestampAgeMinutes(value) {
  const timestamp = new Date(value).getTime();
  if (!value || Number.isNaN(timestamp)) {
    return null;
  }
  return Math.max(0, (Date.now() - timestamp) / 60000);
}

function emptyRow(colspan, label) {
  return `<tr><td colspan="${colspan}" class="empty-state">${label}</td></tr>`;
}

function toneChipClass(value) {
  const token = String(value ?? "").toLowerCase();
  if (["filled", "enabled", "admin", "trader", "risk_on", "overweight", "buy", "bullish", "delivered"].includes(token)) {
    return "tone-chip gain";
  }
  if (["cancelled", "rejected", "underweight", "risk_off", "defensive", "sell", "reduce", "bearish", "failed"].includes(token)) {
    return "tone-chip loss";
  }
  return "tone-chip neutral";
}

async function fetchJson(path, options) {
  const response = await fetch(`/api/v1${path}`, options);
  if (!response.ok) {
    throw new Error(`Request failed for ${path}: ${response.status}`);
  }
  return response.json();
}

function showToast(title, message, tone = "neutral") {
  const stack = document.getElementById("toast-stack");
  const toast = document.createElement("div");
  toast.className = `toast ${tone}`;
  toast.innerHTML = `<strong>${title}</strong>${message}`;
  stack.appendChild(toast);
  window.setTimeout(() => {
    toast.remove();
  }, 5200);
}

function renderTerminalControls(snapshot) {
  const metrics = document.getElementById("terminal-health-metrics");
  const alerts = document.getElementById("terminal-alerts");
  const heroSnapshot = document.getElementById("hero-snapshot-at");
  const heroMarket = document.getElementById("hero-market-at");

  const openOrders = snapshot.orders.filter((order) => !["cancelled", "filled", "rejected"].includes(String(order.status).toLowerCase()));
  const marketAge = timestampAgeMinutes(snapshot.portfolio.market_data_as_of);
  const snapshotAge = timestampAgeMinutes(snapshot.generated_at);
  const riskWatchCount = [
    snapshot.event_summary.failed > 0,
    snapshot.event_summary.pending > 0,
    snapshot.risk.concentration_risk > 0.35,
    snapshot.risk.correlation_risk > 0.65,
    snapshot.risk.var_95 < -0.03,
    marketAge === null || marketAge > 60
  ].filter(Boolean).length;

  heroSnapshot.textContent = `Snapshot ${fmtAge(snapshot.generated_at)}`;
  heroMarket.textContent = `Market ${fmtAge(snapshot.portfolio.market_data_as_of)}`;

  metrics.innerHTML = [
    { title: "Snapshot", value: fmtAge(snapshot.generated_at), delta: fmtTime(snapshot.generated_at), tone: snapshotAge !== null && snapshotAge < 10 ? "gain" : "" },
    { title: "Market Data", value: fmtAge(snapshot.portfolio.market_data_as_of), delta: snapshot.portfolio.market_data_as_of ? fmtTime(snapshot.portfolio.market_data_as_of) : "No market timestamp", tone: marketAge !== null && marketAge <= 60 ? "gain" : "loss" },
    { title: "Open Orders", value: String(openOrders.length), delta: `${snapshot.orders.length} total orders`, tone: openOrders.length ? "neutral" : "" },
    { title: "Risk Watch", value: String(riskWatchCount), delta: "control exceptions", tone: riskWatchCount ? "loss" : "gain" }
  ]
    .map(
      (metric) => `
        <div class="panel metric-card">
          <h3>${metric.title}</h3>
          <div class="metric-value ${metric.tone}">${metric.value}</div>
          <div class="metric-delta">${metric.delta}</div>
        </div>
      `
    )
    .join("");

  const alertItems = [];
  if (snapshot.event_summary.failed > 0) {
    alertItems.push({
      tone: "loss",
      title: "Outbox replay required",
      body: `${snapshot.event_summary.failed} failed event(s) need investigation before downstream state is trusted.`
    });
  }
  if (snapshot.event_summary.pending > 0) {
    alertItems.push({
      tone: "warn",
      title: "Outbox backlog",
      body: `${snapshot.event_summary.pending} event(s) are still pending delivery.`
    });
  }
  if (marketAge === null || marketAge > 60) {
    alertItems.push({
      tone: "warn",
      title: "Market data freshness",
      body: marketAge === null ? "The portfolio snapshot has no market data timestamp." : `Market data is ${fmtAge(snapshot.portfolio.market_data_as_of)}.`
    });
  }
  if (snapshot.risk.concentration_risk > 0.35) {
    alertItems.push({
      tone: "warn",
      title: "Concentration watch",
      body: `Concentration risk is ${fmtPercent(snapshot.risk.concentration_risk, 1)} across ${snapshot.portfolio.positions.length} position(s).`
    });
  }
  if (snapshot.risk.correlation_risk > 0.65) {
    alertItems.push({
      tone: "warn",
      title: "Correlation watch",
      body: `Portfolio correlation risk is ${fmtPercent(snapshot.risk.correlation_risk, 1)}.`
    });
  }
  if (snapshot.risk.var_95 < -0.03) {
    alertItems.push({
      tone: "loss",
      title: "VaR threshold",
      body: `VaR 95 is ${fmtPercent(snapshot.risk.var_95, 2)}.`
    });
  }
  if (!alertItems.length) {
    alertItems.push({
      tone: "gain",
      title: "No active exceptions",
      body: "Snapshot, market data and event delivery are inside the terminal guardrails."
    });
  }

  alerts.innerHTML = alertItems
    .slice(0, 4)
    .map((item) => `<div class="alert-card ${item.tone}"><strong>${item.title}</strong>${item.body}</div>`)
    .join("");
}

function applyTerminalFilter(query) {
  terminalState.filter = String(query ?? "").trim().toLowerCase();
  const rows = [...document.querySelectorAll("tbody tr")].filter((row) => !row.querySelector(".empty-state"));
  const cards = [...document.querySelectorAll(".risk-card")];
  const items = [...rows, ...cards];

  let visible = 0;
  for (const item of items) {
    const matches = !terminalState.filter || item.textContent.toLowerCase().includes(terminalState.filter);
    item.hidden = !matches;
    if (matches) {
      visible += 1;
    }
  }

  const counter = document.getElementById("terminal-filter-count");
  counter.textContent = terminalState.filter ? `${visible}/${items.length} items visible` : "No filter";
}

function renderMetrics({ portfolio, performance, risk, regime }, projection = null) {
  const grossExposure = projection?.projected_gross_exposure ?? portfolio.gross_exposure;
  const netExposure = projection?.projected_net_exposure ?? portfolio.net_exposure;
  const projectedCash = projection?.projected_cash;
  const navDelta = projection
    ? `Cash ${fmtCurrency(projectedCash ?? portfolio.cash, portfolio.base_currency)} projected`
    : `${fmtPercent(performance.day_return, 2)} today`;

  const metrics = [
    {
      title: "NAV",
      value: fmtCurrency(portfolio.nav, portfolio.base_currency),
      delta: navDelta,
      tone: performance.day_return >= 0 ? "gain" : "loss"
    },
    {
      title: projection ? "Gross Exposure (Proj.)" : "Gross Exposure",
      value: fmtPercent(grossExposure),
      delta: projection ? `Net ${fmtPercent(netExposure)} projected` : `Net ${fmtPercent(netExposure)}`,
      tone: projection ? "neutral" : ""
    },
    {
      title: "VaR 95",
      value: fmtPercent(risk.var_95, 2),
      delta: `CVaR ${fmtPercent(risk.cvar_95, 2)}`,
      tone: "loss"
    },
    {
      title: "Regime",
      value: regime.regime,
      delta: regime.recommendation,
      tone: "gain"
    }
  ];

  const container = document.getElementById("metrics");
  container.innerHTML = metrics
    .map(
      (metric) => `
        <div class="panel metric-card">
          <h3>${metric.title}</h3>
          <div class="metric-value ${metric.tone}">${metric.value}</div>
          <div class="metric-delta">${metric.delta}</div>
        </div>
      `
    )
    .join("");
}

function renderSignals({ signals, forecasts, portfolio }) {
  const body = document.getElementById("signals-table");
  if (!signals.length) {
    body.innerHTML = emptyRow(6, "No signals available.");
    return;
  }

  const forecastMap = new Map(forecasts.map((forecast) => [forecast.symbol, forecast]));
  body.innerHTML = signals
    .map((signal) => {
      const position = portfolio.positions.find((item) => item.symbol === signal.symbol);
      const forecast = forecastMap.get(signal.symbol);
      return `
        <tr>
          <td>${signal.symbol}</td>
          <td>${fmtCurrency(position?.market_price ?? 0, portfolio.base_currency)}</td>
          <td class="${signal.buy_probability >= 0.5 ? "gain" : "loss"}">${fmtPercent(signal.buy_probability, 0)}</td>
          <td>${fmtCurrency(forecast?.price_target ?? 0, portfolio.base_currency)}</td>
          <td>${fmtPercent(signal.confidence_score, 0)}</td>
          <td><span class="${toneChipClass(signal.market_regime)}">${roleLabel(signal.market_regime)}</span></td>
        </tr>
      `;
    })
    .join("");
}

function renderPositions(portfolio) {
  const body = document.getElementById("positions-table");
  if (!portfolio.positions.length) {
    body.innerHTML = emptyRow(4, "No positions loaded.");
    return;
  }
  body.innerHTML = portfolio.positions
    .map(
      (position) => `
        <tr>
          <td>${position.symbol}</td>
          <td>${position.quantity}</td>
          <td>${fmtCurrency(position.market_value, portfolio.base_currency)}</td>
          <td class="${position.unrealized_pnl >= 0 ? "gain" : "loss"}">
            ${fmtCurrency(position.unrealized_pnl, portfolio.base_currency)}
          </td>
        </tr>
      `
    )
    .join("");
}

function renderScenarios(scenarios) {
  const stack = document.getElementById("risk-stack");
  stack.innerHTML = scenarios
    .map((scenario) => {
      const primaryMacro = scenario.macro_context?.[0];
      return `
        <div class="risk-card">
          <div class="micro">${scenario.scenario_id}</div>
          <h3>${scenario.name}</h3>
          <div class="impact loss">${fmtPercent(scenario.drawdown_impact, 1)}</div>
          <div class="metric-delta">PnL impact ${fmtCurrency(scenario.estimated_pnl_impact)}</div>
          <div class="metric-delta">${scenario.period ?? ""}</div>
          <div class="metric-delta">${primaryMacro ? `${primaryMacro.label}: ${primaryMacro.value}` : ""}</div>
        </div>
      `;
    })
    .join("");
}

function renderBarbell(barbell) {
  const regime = document.getElementById("barbell-regime");
  const metrics = document.getElementById("barbell-metrics");
  const rationale = document.getElementById("barbell-rationale");
  const table = document.getElementById("barbell-table");

  regime.textContent = roleLabel(barbell.regime);
  rationale.textContent = barbell.rationale;
  metrics.innerHTML = [
    {
      title: "Defensive",
      value: fmtPercent(barbell.defensive_weight),
      tone: ""
    },
    {
      title: "Opportunistic",
      value: fmtPercent(barbell.opportunistic_weight),
      tone: ""
    },
    {
      title: "Cash Buffer",
      value: fmtPercent(barbell.cash_buffer_weight),
      tone: ""
    }
  ]
    .map(
      (metric) => `
        <div class="panel metric-card">
          <h3>${metric.title}</h3>
          <div class="metric-value ${metric.tone}">${metric.value}</div>
        </div>
      `
    )
    .join("");

  if (!barbell.allocations.length) {
    table.innerHTML = emptyRow(6, "No barbell allocation available.");
    return;
  }

  table.innerHTML = barbell.allocations
    .map(
      (item) => `
        <tr>
          <td>${item.symbol}</td>
          <td><span class="${toneChipClass(item.bucket)}">${roleLabel(item.bucket)}</span></td>
          <td>${roleLabel(item.role)}</td>
          <td>${fmtPercent(item.current_weight, 2)}</td>
          <td>${fmtPercent(item.target_weight, 2)}</td>
          <td class="${item.delta_weight >= 0 ? "gain" : "loss"}">${fmtPercent(item.delta_weight, 2)}</td>
        </tr>
      `
    )
    .join("");
}

function renderRebalanceProjection(payload) {
  const metrics = document.getElementById("rebalance-projection-metrics");
  if (!payload || payload.projected_cash === null || payload.projected_cash === undefined) {
    metrics.innerHTML = "";
    return;
  }

  metrics.innerHTML = [
    {
      title: "Projected Cash",
      value: fmtCurrency(payload.projected_cash),
      delta: `Buffer ${fmtPercent(payload.projected_cash_weight, 1)}`,
      tone: payload.projected_cash_weight >= 0.15 ? "gain" : "loss"
    },
    {
      title: "Projected Gross",
      value: fmtPercent(payload.projected_gross_exposure, 1),
      delta: "after pending rebalance",
      tone: ""
    },
    {
      title: "Projected Net",
      value: fmtPercent(payload.projected_net_exposure, 1),
      delta: `${payload.orders?.length ?? 0} order(s) staged`,
      tone: "neutral"
    }
  ]
    .map(
      (metric) => `
        <div class="panel metric-card">
          <h3>${metric.title}</h3>
          <div class="metric-value ${metric.tone}">${metric.value}</div>
          <div class="metric-delta">${metric.delta}</div>
        </div>
      `
    )
    .join("");
}

function isOpenOrder(order) {
  return !["cancelled", "filled", "rejected"].includes(String(order.status).toLowerCase());
}

function populateOrderForm(order) {
  if (!order) {
    return;
  }
  document.getElementById("order-symbol").value = order.symbol ?? "";
  document.getElementById("order-side").value = String(order.side ?? "BUY").toUpperCase();
  document.getElementById("order-type").value = order.order_type ?? "limit";
  document.getElementById("order-quantity").value = order.quantity ?? "";
  document.getElementById("order-limit-price").value = order.limit_price ?? "";
  document.getElementById("order-stop-price").value = order.stop_price ?? "";
  document.getElementById("order-broker").value = order.broker ?? "paper";
  document.getElementById("order-strategy-tag").value = order.strategy_tag ?? "manual";
  terminalState.activePendingOrderId = order.id ?? null;
  renderPendingOrdersQueue(window.__latestOrders ?? []);
}

function renderPendingOrdersQueue(orders) {
  window.__latestOrders = orders;
  const body = document.getElementById("pending-orders-table");
  const counter = document.getElementById("pending-orders-count");
  const pending = orders.filter(isOpenOrder);
  counter.textContent = `${pending.length} pending`;

  if (!pending.length) {
    body.innerHTML = emptyRow(7, "No pending orders in queue.");
    return;
  }

  body.innerHTML = pending
    .map(
      (order) => `
        <tr class="pending-order-row ${terminalState.activePendingOrderId === order.id ? "is-active" : ""}">
          <td>${order.symbol}</td>
          <td><span class="${toneChipClass(order.side)}">${order.side}</span></td>
          <td>${order.order_type}</td>
          <td>${order.quantity}</td>
          <td>${order.limit_price ?? "--"}</td>
          <td>${order.strategy_tag ?? "--"}</td>
          <td>
            <button class="link-button" type="button" onclick="populateOrderForm(window.__latestOrders.find((item) => item.id === '${order.id}'))">
              Load
            </button>
          </td>
        </tr>
      `
    )
    .join("");
}

function applyProjectedExposure(projection) {
  terminalState.projectedExposure = projection ?? null;
  if (window.__latestSnapshot) {
    renderMetrics(window.__latestSnapshot, terminalState.projectedExposure);
  }
}

async function rebalanceBarbellPortfolio() {
  const result = document.getElementById("order-result");
  result.textContent = "Generating barbell rebalance orders...";

  try {
    const response = await fetchJson("/portfolio/barbell/rebalance", { method: "POST" });
    terminalState.projectedExposure = {
      projected_cash: response.projected_cash,
      projected_cash_weight: response.projected_cash_weight,
      projected_gross_exposure: response.projected_gross_exposure,
      projected_net_exposure: response.projected_net_exposure
    };
    await loadTerminal(false);
    renderRebalanceProjection(response);
    applyProjectedExposure(terminalState.projectedExposure);
    if (response.orders?.length) {
      populateOrderForm(response.orders[0]);
    }
    result.textContent = JSON.stringify(
      {
        generated_at: response.generated_at,
        orders: response.orders,
        notes: response.notes
      },
      null,
      2
    );
    showToast(
      "Barbell rebalance staged",
      `${response.orders.length} pending limit order(s) generated while preserving the 15% cash buffer.`,
      response.orders.length ? "gain" : "warn"
    );
  } catch (error) {
    result.textContent = String(error);
    showToast("Rebalance failed", String(error), "loss");
  }
}

function linePoints(items, key, min, max, width, height) {
  const range = Math.max(max - min, 1);
  return items
    .map((item, index) => {
      const x = (index / Math.max(items.length - 1, 1)) * width;
      const y = height - ((item[key] - min) / range) * (height - 34) - 17;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function renderMonteCarloResult(payload) {
  const metrics = document.getElementById("monte-carlo-metrics");
  const result = document.getElementById("monte-carlo-result");
  const chart = document.getElementById("monte-carlo-chart");
  const trajectory = payload.trajectory ?? [];

  metrics.innerHTML = [
    { title: "Expected Return", value: fmtPercent(payload.expected_annual_return, 1), delta: "1-year simulated", tone: payload.expected_annual_return >= 0 ? "gain" : "loss" },
    { title: "Sharpe", value: fmtNullable(payload.simulated_sharpe_ratio, 2), delta: `${payload.n_paths} paths`, tone: payload.simulated_sharpe_ratio >= 1 ? "gain" : "neutral" },
    { title: "VaR 95", value: fmtPercent(payload.var_95, 2), delta: `CVaR ${fmtPercent(payload.cvar_95, 2)}`, tone: "loss" }
  ]
    .map(
      (metric) => `
        <div class="panel metric-card">
          <h3>${metric.title}</h3>
          <div class="metric-value ${metric.tone}">${metric.value}</div>
          <div class="metric-delta">${metric.delta}</div>
        </div>
      `
    )
    .join("");

  if (!trajectory.length) {
    chart.innerHTML = "";
    result.textContent = "Simulation returned no trajectory.";
    return;
  }

  const values = trajectory.flatMap((item) => [item.p5_nav, item.p50_nav, item.p95_nav]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = 720;
  const height = 260;
  const p5 = linePoints(trajectory, "p5_nav", min, max, width, height);
  const p50 = linePoints(trajectory, "p50_nav", min, max, width, height);
  const p95 = linePoints(trajectory, "p95_nav", min, max, width, height);
  const band = `${p95} ${p5.split(" ").reverse().join(" ")}`;
  chart.innerHTML = `
    <polygon class="simulation-band" points="${band}"></polygon>
    <polyline class="sparkline-grid" points="0,225 ${width},225"></polyline>
    <polyline class="simulation-line p95" points="${p95}"></polyline>
    <polyline class="simulation-line p50" points="${p50}"></polyline>
    <polyline class="simulation-line p5" points="${p5}"></polyline>
    <text class="chart-label" x="12" y="24">P95</text>
    <text class="chart-label" x="12" y="48">P50</text>
    <text class="chart-label" x="12" y="72">P5</text>
  `;
  result.textContent = JSON.stringify(
    {
      generated_at: payload.generated_at,
      symbols: payload.symbols,
      methodology: payload.methodology
    },
    null,
    2
  );
}

async function runMonteCarloSimulation() {
  const result = document.getElementById("monte-carlo-result");
  result.textContent = "Running 1,000-path Monte Carlo simulation...";

  const proposedWeights = terminalState.latestOptimization?.targets?.reduce((weights, target) => {
    weights[target.symbol] = target.recommended_weight;
    return weights;
  }, {});

  try {
    const payload = { n_paths: 1000, horizon_days: 252 };
    if (proposedWeights && Object.keys(proposedWeights).length) {
      payload.proposed_weights = proposedWeights;
    }
    const response = await fetchJson("/simulate/monte-carlo", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    renderMonteCarloResult(response);
    const weightNote = payload.proposed_weights ? " using optimized allocation weights" : "";
    showToast("Monte Carlo complete", `Projected ${response.horizon_days} trading days across ${response.n_paths} paths${weightNote}.`, "gain");
  } catch (error) {
    result.textContent = String(error);
    showToast("Simulation failed", String(error), "loss");
  }
}

function renderOptimizationResult(payload) {
  terminalState.latestOptimization = payload;
  const metrics = document.getElementById("optimization-metrics");
  const table = document.getElementById("optimization-table");
  const result = document.getElementById("optimization-result");

  metrics.innerHTML = [
    { title: "Return", value: fmtPercent(payload.expected_annual_return, 1), delta: "optimized annual", tone: payload.expected_annual_return >= 0 ? "gain" : "loss" },
    { title: "Volatility", value: fmtPercent(payload.expected_annual_volatility, 1), delta: "annualized", tone: "" },
    { title: "VaR 95", value: fmtPercent(payload.var_95, 2), delta: `Sharpe ${fmtNullable(payload.simulated_sharpe_ratio, 2)}`, tone: payload.var_95 >= -0.015 ? "gain" : "loss" }
  ]
    .map(
      (metric) => `
        <div class="panel metric-card">
          <h3>${metric.title}</h3>
          <div class="metric-value ${metric.tone}">${metric.value}</div>
          <div class="metric-delta">${metric.delta}</div>
        </div>
      `
    )
    .join("");

  table.innerHTML = payload.targets?.length
    ? payload.targets
        .map(
          (item) => `
            <tr>
              <td>${item.symbol}</td>
              <td><span class="${toneChipClass(item.bucket)}">${roleLabel(item.bucket)}</span></td>
              <td>${fmtPercent(item.current_weight, 1)}</td>
              <td>${fmtPercent(item.recommended_weight, 1)}</td>
              <td class="${item.delta_weight >= 0 ? "gain" : "loss"}">${fmtPercent(item.delta_weight, 1)}</td>
              <td>${fmtPercent(item.expected_return, 1)}</td>
            </tr>
          `
        )
        .join("")
    : emptyRow(6, "No optimized targets returned.");

  result.textContent = JSON.stringify(
    {
      generated_at: payload.generated_at,
      status: payload.status,
      objective: payload.objective,
      notes: payload.notes
    },
    null,
    2
  );
  applyTerminalFilter(document.getElementById("terminal-filter").value);
}

async function runAllocationOptimization(apply = false) {
  const result = document.getElementById("optimization-result");
  result.textContent = apply ? "Applying optimized allocation to barbell targets..." : "Optimizing allocation...";

  try {
    const response = await fetchJson(
      apply ? "/portfolio/optimize-allocation/apply" : "/portfolio/optimize-allocation",
      { method: "POST" }
    );
    renderOptimizationResult(response);
    if (apply) {
      renderBarbell(response.barbell);
      await loadTerminal(false);
    }
    showToast(
      apply ? "Optimized targets applied" : "Optimization complete",
      `Sharpe ${fmtNullable(response.simulated_sharpe_ratio, 2)}, VaR ${fmtPercent(response.var_95, 2)}.`,
      response.var_95 >= -0.015 ? "gain" : "warn"
    );
  } catch (error) {
    result.textContent = String(error);
    showToast("Optimization failed", String(error), "loss");
  }
}

function renderEventSummary(summary, generatedAt) {
  const metrics = document.getElementById("event-summary-metrics");
  const note = document.getElementById("event-summary-note");
  const timestamp = document.getElementById("snapshot-generated-at");

  timestamp.textContent = generatedAt ? `Snapshot ${fmtTime(generatedAt)}` : "--";
  metrics.innerHTML = [
    { title: "Delivered", value: String(summary.delivered ?? 0), tone: "gain" },
    { title: "Pending", value: String(summary.pending ?? 0), tone: "" },
    { title: "Failed", value: String(summary.failed ?? 0), tone: "loss" }
  ]
    .map(
      (metric) => `
        <div class="panel metric-card">
          <h3>${metric.title}</h3>
          <div class="metric-value ${metric.tone}">${metric.value}</div>
        </div>
      `
    )
    .join("");

  note.textContent =
    summary.failed > 0
      ? `Outbox attention required: ${summary.failed} failed event(s) need investigation or replay.`
      : `Outbox healthy: ${summary.delivered} delivered, ${summary.pending} pending.`;
}

function renderCorrelationMatrix(correlation) {
  const head = document.getElementById("correlation-head");
  const body = document.getElementById("correlation-table");

  if (!correlation || !correlation.symbols?.length) {
    head.innerHTML = "";
    body.innerHTML = emptyRow(1, "No correlation matrix available.");
    return;
  }

  head.innerHTML = `
    <tr>
      <th>Symbol</th>
      ${correlation.symbols.map((symbol) => `<th>${symbol}</th>`).join("")}
    </tr>
  `;

  body.innerHTML = correlation.symbols
    .map(
      (rowSymbol, rowIndex) => `
        <tr>
          <td>${rowSymbol}</td>
          ${correlation.matrix[rowIndex]
            .map((value) => `<td>${Number(value).toFixed(2)}</td>`)
            .join("")}
        </tr>
      `
    )
    .join("");
}

function renderHistory(history, portfolio) {
  const body = document.getElementById("history-table");
  const recent = history.slice(-8).reverse();
  body.innerHTML = recent.length
    ? recent
        .map(
          (point) => `
            <tr>
              <td>${fmtTime(point.recorded_at)}</td>
              <td>${fmtCurrency(point.nav, portfolio.base_currency)}</td>
              <td>${fmtCurrency(point.cash, portfolio.base_currency)}</td>
              <td>${fmtPercent(point.gross_exposure, 1)}</td>
            </tr>
          `
        )
        .join("")
    : emptyRow(4, "No NAV snapshots yet.");

  const svg = document.getElementById("nav-sparkline");
  if (!history.length) {
    svg.innerHTML = "";
    return;
  }

  const values = history.map((point) => point.nav);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1);
  const width = 720;
  const height = 220;

  const linePoints = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - ((value - min) / range) * (height - 30) - 15;
      return `${x},${y}`;
    })
    .join(" ");

  const areaPoints = `0,${height} ${linePoints} ${width},${height}`;
  svg.innerHTML = `
    <defs>
      <linearGradient id="navFill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="rgba(125, 211, 252, 0.45)"></stop>
        <stop offset="100%" stop-color="rgba(125, 211, 252, 0.03)"></stop>
      </linearGradient>
    </defs>
    <polyline class="sparkline-grid" points="0,190 ${width},190"></polyline>
    <polygon class="sparkline-area" points="${areaPoints}"></polygon>
    <polyline class="sparkline-line" points="${linePoints}"></polyline>
  `;
}

function renderPositionRisk(items) {
  const body = document.getElementById("position-risk-table");
  if (!items.length) {
    body.innerHTML = emptyRow(6, "No position risk available.");
    return;
  }
  body.innerHTML = items
    .map(
      (item) => `
        <tr>
          <td>${item.symbol}</td>
          <td>${Number(item.beta).toFixed(2)}</td>
          <td class="loss">${fmtPercent(item.var_95, 2)}</td>
          <td class="loss">${fmtPercent(item.cvar_95, 2)}</td>
          <td>${fmtPercent(item.liquidity_score, 0)}</td>
          <td>${fmtPercent(item.concentration_weight, 1)}</td>
        </tr>
      `
    )
    .join("");
}

function renderOrders(orders) {
  renderPendingOrdersQueue(orders);
  const body = document.getElementById("orders-table");
  if (!orders.length) {
    body.innerHTML = emptyRow(8, "No orders found.");
    return;
  }
  body.innerHTML = orders
    .map(
      (order) => `
        <tr>
          <td>${order.id}</td>
          <td>${order.symbol}</td>
          <td><span class="${toneChipClass(order.side)}">${order.side}</span></td>
          <td><span class="${toneChipClass(order.status)}">${roleLabel(order.status)}</span></td>
          <td>${order.order_type}</td>
          <td>${order.quantity}</td>
          <td>${order.broker}</td>
          <td>
            <button
              class="link-button"
              onclick="cancelOrder('${order.id}')"
              ${order.status === "cancelled" || order.status === "filled" ? "disabled" : ""}
            >
              Cancel
            </button>
          </td>
        </tr>
      `
    )
    .join("");
}

function renderFills(fills) {
  const body = document.getElementById("fills-table");
  if (!fills.length) {
    body.innerHTML = emptyRow(6, "No fills booked yet.");
    return;
  }
  body.innerHTML = fills
    .map(
      (fill) => `
        <tr>
          <td>${fill.order_id}</td>
          <td>${fill.symbol}</td>
          <td>${fill.quantity}</td>
          <td>${fmtCurrency(fill.price)}</td>
          <td>${fill.venue}</td>
          <td>${fmtTime(fill.filled_at)}</td>
        </tr>
      `
    )
    .join("");
}

function renderResearch(items) {
  const body = document.getElementById("research-table");
  if (!items.length) {
    body.innerHTML = emptyRow(6, "No research ideas available.");
    return;
  }
  body.innerHTML = items
    .map(
      (item) => `
        <tr>
          <td>${item.symbol}</td>
          <td>${item.sector}</td>
          <td class="${item.buy_probability >= 0.5 ? "gain" : "loss"}">${fmtPercent(item.buy_probability, 0)}</td>
          <td class="${item.expected_return >= 0 ? "gain" : "loss"}">${fmtPercent(item.expected_return, 1)}</td>
          <td>${fmtPercent(item.factor_score, 0)}</td>
          <td><span class="${toneChipClass(item.market_regime)}">${roleLabel(item.market_regime)}</span></td>
        </tr>
      `
    )
    .join("");
}

function renderSectors(items) {
  const body = document.getElementById("sector-table");
  if (!items.length) {
    body.innerHTML = emptyRow(5, "No sector rotation signal available.");
    return;
  }
  body.innerHTML = items
    .map(
      (item) => `
        <tr>
          <td>${item.sector}</td>
          <td class="${item.average_buy_probability >= 0.5 ? "gain" : "loss"}">${fmtPercent(item.average_buy_probability, 0)}</td>
          <td class="${item.average_expected_return >= 0 ? "gain" : "loss"}">${fmtPercent(item.average_expected_return, 1)}</td>
          <td>${fmtPercent(item.average_factor_score, 0)}</td>
          <td><span class="${toneChipClass(item.stance)}">${roleLabel(item.stance)}</span></td>
        </tr>
      `
    )
    .join("");
}

function renderFactors(items) {
  const body = document.getElementById("factors-table");
  if (!items.length) {
    body.innerHTML = emptyRow(6, "No factor ranking available.");
    return;
  }
  body.innerHTML = items
    .map(
      (item) => `
        <tr>
          <td>${item.symbol}</td>
          <td>${item.sector}</td>
          <td>${fmtPercent(item.momentum_score, 0)}</td>
          <td>${fmtPercent(item.quality_score, 0)}</td>
          <td>${fmtPercent(item.volatility_score, 0)}</td>
          <td class="gain">${fmtPercent(item.overall_score, 0)}</td>
        </tr>
      `
    )
    .join("");
}

function renderUsers(users) {
  const body = document.getElementById("users-table");
  if (!users.length) {
    body.innerHTML = emptyRow(4, "No users configured.");
    return;
  }
  body.innerHTML = users
    .map(
      (user) => `
        <tr>
          <td>${user.full_name}</td>
          <td><span class="${toneChipClass(user.role)}">${roleLabel(user.role)}</span></td>
          <td>${user.email}</td>
          <td><span class="${toneChipClass(user.mfa_enabled ? "enabled" : "off")}">${user.mfa_enabled ? "Enabled" : "Off"}</span></td>
        </tr>
      `
    )
    .join("");
}

function renderAudit(logs) {
  const body = document.getElementById("audit-table");
  if (!logs.length) {
    body.innerHTML = emptyRow(4, "No audit activity yet.");
    return;
  }
  body.innerHTML = logs
    .map(
      (log) => `
        <tr>
          <td>${fmtTime(log.created_at)}</td>
          <td><span class="tone-chip neutral">${log.event_type}</span></td>
          <td>${log.actor_email}</td>
          <td>${log.details ?? ""}</td>
        </tr>
      `
    )
    .join("");
}

function renderTradingAlgoResult(payload) {
  const summary = document.getElementById("algo-summary");
  const result = document.getElementById("algo-result");
  const table = document.getElementById("algo-table");

  summary.textContent = `${payload.status} / ${payload.command}`;
  result.textContent = JSON.stringify(
    {
      generated_at: payload.generated_at,
      summary: payload.summary,
      errors: payload.errors
    },
    null,
    2
  );

  if (!payload.analyses?.length) {
    table.innerHTML = emptyRow(10, "No trading-algo results returned.");
    return;
  }

  table.innerHTML = payload.analyses
    .map(
      (item) => `
        <tr>
          <td>${item.symbol}</td>
          <td><span class="${toneChipClass(item.trend)}">${roleLabel(item.trend)}</span></td>
          <td><span class="${toneChipClass(item.recommendation)}">${roleLabel(item.recommendation)}</span></td>
          <td>${item.latest_price === null ? "--" : fmtCurrency(item.latest_price)}</td>
          <td class="${(item.daily_return ?? 0) >= 0 ? "gain" : "loss"}">${item.daily_return === null ? "--" : fmtPercent(item.daily_return, 2)}</td>
          <td class="${(item.total_return ?? 0) >= 0 ? "gain" : "loss"}">${item.total_return === null ? "--" : fmtPercent(item.total_return, 1)}</td>
          <td>${item.volatility_20d === null ? "--" : fmtPercent(item.volatility_20d, 1)}</td>
          <td>${fmtNullable(item.sharpe_ratio, 2)}</td>
          <td class="loss">${item.var_95 === null ? "--" : fmtPercent(item.var_95, 2)}</td>
          <td>${fmtNullable(item.rsi, 1)}</td>
        </tr>
      `
    )
    .join("");

  applyTerminalFilter(document.getElementById("terminal-filter").value);
}

async function runTradingAlgoCommand(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  const result = document.getElementById("algo-result");
  const summary = document.getElementById("algo-summary");
  const symbols = String(data.get("symbols") ?? "")
    .split(",")
    .map((symbol) => symbol.trim())
    .filter(Boolean);

  const payload = {
    command: data.get("command"),
    symbols,
    period: data.get("period"),
    max_symbols: Number(data.get("max_symbols") || 8)
  };

  summary.textContent = "Running";
  result.textContent = "Running trading-algo command...";

  try {
    const response = await fetchJson("/terminal/trading-algo", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    renderTradingAlgoResult(response);
    showToast("Trading algo complete", response.summary, response.status === "error" ? "loss" : "gain");
  } catch (error) {
    summary.textContent = "Error";
    result.textContent = String(error);
    showToast("Trading algo failed", String(error), "loss");
  }
}

async function submitOrder(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  const payload = {
    symbol: data.get("symbol"),
    side: data.get("side"),
    order_type: data.get("order_type"),
    quantity: Number(data.get("quantity")),
    limit_price: data.get("limit_price") ? Number(data.get("limit_price")) : null,
    stop_price: data.get("stop_price") ? Number(data.get("stop_price")) : null,
    broker: data.get("broker"),
    strategy_tag: data.get("strategy_tag")
  };

  const result = document.getElementById("order-result");
  result.textContent = "Submitting order...";

  try {
    const response = await fetchJson("/orders", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    result.textContent = JSON.stringify(response, null, 2);
    showToast("Order submitted", `${response.side} ${response.quantity} ${response.symbol} (${response.status}).`, "gain");
    await loadTerminal(false);
  } catch (error) {
    result.textContent = String(error);
    showToast("Order failed", String(error), "loss");
  }
}

async function cancelOrder(orderId) {
  const result = document.getElementById("order-result");
  result.textContent = `Cancelling order ${orderId}...`;

  try {
    const response = await fetch(`/api/v1/orders/${orderId}`, { method: "DELETE" });
    if (!response.ok) {
      throw new Error(`Cancel failed for ${orderId}: ${response.status}`);
    }
    const payload = await response.json();
    result.textContent = JSON.stringify(payload, null, 2);
    showToast("Order cancelled", `Order ${orderId} cancelled.`, "warn");
    await loadTerminal(false);
  } catch (error) {
    result.textContent = String(error);
    showToast("Cancel failed", String(error), "loss");
  }
}

async function refreshPortfolioPrices() {
  const result = document.getElementById("order-result");
  result.textContent = "Refreshing market prices...";

  try {
    const payload = await fetchJson("/portfolio/refresh", { method: "POST" });
    result.textContent = JSON.stringify(payload, null, 2);
    showToast("Market data refreshed", `${payload.positions_updated ?? 0} position(s) updated.`, "gain");
    await loadTerminal(false);
  } catch (error) {
    result.textContent = String(error);
    showToast("Refresh failed", String(error), "loss");
  }
}

function renderMarketTimestamp(portfolio) {
  const updatedAt = document.getElementById("updated-at");
  if (portfolio.market_data_as_of) {
    const asOf = new Date(portfolio.market_data_as_of);
    if (!Number.isNaN(asOf.getTime())) {
      updatedAt.textContent = `Market ${asOf.toISOString().slice(11, 19)} UTC`;
      return;
    }
  }
  updatedAt.textContent = `${new Date().toISOString().slice(11, 19)} UTC`;
}

async function loadTerminal(runRefresh = true) {
  const badge = document.getElementById("connection-badge");
  try {
    if (runRefresh) {
      try {
        await fetchJson("/portfolio/refresh", { method: "POST" });
      } catch (error) {
        document.getElementById("order-result").textContent = `Market refresh warning: ${String(error)}`;
      }
    }

    const snapshot = await fetchJson("/terminal/snapshot");
    window.__latestSnapshot = snapshot;

    renderTerminalControls(snapshot);
    renderMetrics(snapshot, terminalState.projectedExposure);
    renderHistory(snapshot.history, snapshot.portfolio);
    renderSignals({
      signals: snapshot.signals,
      forecasts: snapshot.forecasts,
      portfolio: snapshot.portfolio
    });
    renderPositions(snapshot.portfolio);
    renderScenarios(snapshot.scenarios);
    renderBarbell(snapshot.barbell);
    renderEventSummary(snapshot.event_summary, snapshot.generated_at);
    renderPositionRisk(snapshot.position_risk);
    renderCorrelationMatrix(snapshot.correlation_matrix);
    renderOrders(snapshot.orders);
    renderFills(snapshot.fills);
    renderResearch(snapshot.research);
    renderFactors(snapshot.factors);
    renderSectors(snapshot.sectors);
    renderUsers(snapshot.users);
    renderAudit(snapshot.audit_logs);
    applyTerminalFilter(document.getElementById("terminal-filter").value);

    badge.textContent = "Connected";
    badge.classList.remove("loss");
    badge.classList.add("gain");
    renderMarketTimestamp(snapshot.portfolio);
  } catch (error) {
    badge.textContent = "API unavailable";
    badge.classList.remove("gain");
    badge.classList.add("loss");
    document.getElementById("order-result").textContent = String(error);
  }
}

window.populateOrderForm = populateOrderForm;
document.getElementById("algo-form").addEventListener("submit", runTradingAlgoCommand);
document.getElementById("order-form").addEventListener("submit", submitOrder);
document.getElementById("refresh-button").addEventListener("click", refreshPortfolioPrices);
document.getElementById("rebalance-barbell-button").addEventListener("click", rebalanceBarbellPortfolio);
document.getElementById("run-monte-carlo-button").addEventListener("click", runMonteCarloSimulation);
document.getElementById("optimize-allocation-button").addEventListener("click", () => runAllocationOptimization(false));
document.getElementById("apply-optimized-allocation-button").addEventListener("click", () => runAllocationOptimization(true));
document.getElementById("terminal-filter").addEventListener("input", (event) => {
  applyTerminalFilter(event.currentTarget.value);
});
document.getElementById("terminal-filter-clear").addEventListener("click", () => {
  const input = document.getElementById("terminal-filter");
  input.value = "";
  applyTerminalFilter("");
  input.focus();
});
loadTerminal();
