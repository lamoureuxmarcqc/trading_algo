const fmtCurrency = (value, currency = "USD") =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 0
  }).format(value ?? 0);

const fmtPercent = (value, digits = 1) => `${((value ?? 0) * 100).toFixed(digits)}%`;
const fmtTime = (value) => new Date(value).toLocaleString();
const roleLabel = (value) => String(value ?? "").replaceAll("_", " ");

function emptyRow(colspan, label) {
  return `<tr><td colspan="${colspan}" class="empty-state">${label}</td></tr>`;
}

function toneChipClass(value) {
  const token = String(value ?? "").toLowerCase();
  if (["filled", "enabled", "admin", "trader", "risk_on", "overweight", "buy", "delivered"].includes(token)) {
    return "tone-chip gain";
  }
  if (["cancelled", "rejected", "underweight", "risk_off", "defensive", "sell", "failed"].includes(token)) {
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

function renderMetrics({ portfolio, performance, risk, regime }) {
  const metrics = [
    {
      title: "NAV",
      value: fmtCurrency(portfolio.nav, portfolio.base_currency),
      delta: `${fmtPercent(performance.day_return, 2)} today`,
      tone: performance.day_return >= 0 ? "gain" : "loss"
    },
    {
      title: "Gross Exposure",
      value: fmtPercent(portfolio.gross_exposure),
      delta: `Net ${fmtPercent(portfolio.net_exposure)}`,
      tone: ""
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
    await loadTerminal();
  } catch (error) {
    result.textContent = String(error);
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
    await loadTerminal();
  } catch (error) {
    result.textContent = String(error);
  }
}

async function refreshPortfolioPrices() {
  const result = document.getElementById("order-result");
  result.textContent = "Refreshing market prices...";

  try {
    const payload = await fetchJson("/portfolio/refresh", { method: "POST" });
    result.textContent = JSON.stringify(payload, null, 2);
    await loadTerminal(false);
  } catch (error) {
    result.textContent = String(error);
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

    renderMetrics(snapshot);
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

document.getElementById("order-form").addEventListener("submit", submitOrder);
document.getElementById("refresh-button").addEventListener("click", refreshPortfolioPrices);
loadTerminal();
