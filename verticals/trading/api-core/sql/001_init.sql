CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    mfa_secret TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, role_id)
);

CREATE TABLE IF NOT EXISTS role_permissions (
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    refresh_token_hash TEXT NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_user_id UUID REFERENCES users(id),
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS symbols (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker TEXT NOT NULL UNIQUE,
    asset_class TEXT NOT NULL,
    exchange TEXT,
    market_data_ticker TEXT,
    market_data_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    currency TEXT NOT NULL DEFAULT 'USD',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prices_1m (
    symbol_id UUID NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    open NUMERIC(18,6) NOT NULL,
    high NUMERIC(18,6) NOT NULL,
    low NUMERIC(18,6) NOT NULL,
    close NUMERIC(18,6) NOT NULL,
    volume BIGINT NOT NULL,
    PRIMARY KEY (symbol_id, ts)
);

CREATE TABLE IF NOT EXISTS prices_daily (
    symbol_id UUID NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    trading_day DATE NOT NULL,
    open NUMERIC(18,6) NOT NULL,
    high NUMERIC(18,6) NOT NULL,
    low NUMERIC(18,6) NOT NULL,
    close NUMERIC(18,6) NOT NULL,
    adjusted_close NUMERIC(18,6),
    volume BIGINT NOT NULL,
    PRIMARY KEY (symbol_id, trading_day)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol_id UUID NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    market_cap NUMERIC(20,2),
    pe_ratio NUMERIC(18,6),
    pb_ratio NUMERIC(18,6),
    revenue_growth NUMERIC(18,6),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (symbol_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS news (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol_id UUID REFERENCES symbols(id) ON DELETE SET NULL,
    headline TEXT NOT NULL,
    source TEXT NOT NULL,
    sentiment NUMERIC(8,4),
    url TEXT,
    published_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS economic_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_code TEXT NOT NULL,
    country_code TEXT NOT NULL,
    title TEXT NOT NULL,
    scheduled_at TIMESTAMPTZ NOT NULL,
    importance SMALLINT NOT NULL DEFAULT 1,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    broker_name TEXT NOT NULL,
    account_number TEXT NOT NULL UNIQUE,
    account_type TEXT NOT NULL,
    base_currency TEXT NOT NULL DEFAULT 'USD',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cash_balances (
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    currency TEXT NOT NULL,
    balance NUMERIC(20,2) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, currency)
);

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    symbol_id UUID NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    quantity NUMERIC(20,6) NOT NULL,
    average_cost NUMERIC(18,6) NOT NULL,
    market_price NUMERIC(18,6),
    market_value NUMERIC(20,2),
    unrealized_pnl NUMERIC(20,2),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, symbol_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    symbol_id UUID REFERENCES symbols(id) ON DELETE SET NULL,
    transaction_type TEXT NOT NULL,
    quantity NUMERIC(20,6),
    price NUMERIC(18,6),
    fees NUMERIC(18,6) NOT NULL DEFAULT 0,
    currency TEXT NOT NULL DEFAULT 'USD',
    trade_date TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS fx_rates (
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    rate NUMERIC(18,8) NOT NULL,
    PRIMARY KEY (base_currency, quote_currency, ts)
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD'
);

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    symbol_id UUID NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    tif TEXT NOT NULL DEFAULT 'DAY',
    quantity NUMERIC(20,6) NOT NULL,
    limit_price NUMERIC(18,6),
    stop_price NUMERIC(18,6),
    status TEXT NOT NULL DEFAULT 'pending',
    broker_order_id TEXT,
    strategy_tag TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_idempotency_tenant_operation_key UNIQUE (tenant_id, operation, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_idempotency_keys_tenant_operation
    ON idempotency_keys (tenant_id, operation);

CREATE TABLE IF NOT EXISTS event_outbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_name TEXT NOT NULL,
    topic TEXT NOT NULL,
    event_version INTEGER NOT NULL DEFAULT 1,
    tenant_id TEXT NOT NULL DEFAULT 'public',
    correlation_id TEXT,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    dispatched_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_outbox_topic_created_at
    ON event_outbox (topic, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_event_outbox_delivery_status
    ON event_outbox (delivery_status, created_at ASC);

CREATE TABLE IF NOT EXISTS fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    venue TEXT NOT NULL,
    quantity NUMERIC(20,6) NOT NULL,
    price NUMERIC(18,6) NOT NULL,
    fees NUMERIC(18,6) NOT NULL DEFAULT 0,
    filled_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS routes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    broker_name TEXT NOT NULL,
    destination TEXT,
    route_status TEXT NOT NULL,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS execution_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    slippage_bps NUMERIC(12,4),
    arrival_price NUMERIC(18,6),
    vwap_price NUMERIC(18,6),
    participation_rate NUMERIC(18,6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE corporate_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol_id UUID NOT NULL REFERENCES symbols(id),
    action_type TEXT NOT NULL, -- 'DIVIDEND', 'STOCK_SPLIT', 'MERGER', 'RIGHTS_OFFERING'
    ex_date DATE NOT NULL,
    record_date DATE,
    payment_date DATE,
    details JSONB NOT NULL, -- e.g., {"dividend_per_share": 0.5, "currency": "USD"}
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE restrictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    restriction_type TEXT NOT NULL, -- 'POSITION_LIMIT', 'SECTOR_CAP', 'BLACKLIST'
    entity_type TEXT NOT NULL, -- 'ACCOUNT', 'USER', 'PORTFOLIO'
    entity_id UUID NOT NULL,
    symbol_id UUID REFERENCES symbols(id),
    max_qty NUMERIC(20,6),
    max_notional NUMERIC(20,2),
    effective_from DATE NOT NULL,
    effective_to DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE watchlists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    owner_id UUID REFERENCES users(id),
    is_institutional BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE watchlist_items (
    watchlist_id UUID REFERENCES watchlists(id) ON DELETE CASCADE,
    symbol_id UUID REFERENCES symbols(id),
    added_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (watchlist_id, symbol_id)
);

CREATE TABLE risk_metrics_daily (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id TEXT NOT NULL, -- e.g., 'family-office-master'
    metric_date DATE NOT NULL,
    var_95 NUMERIC(20,4),       -- 1-day VaR at 95%
    var_99 NUMERIC(20,4),
    expected_shortfall NUMERIC(20,4),
    beta_to_spx NUMERIC(10,6),
    sharpe_ratio NUMERIC(10,6),
    max_drawdown NUMERIC(10,6),
    stress_test_label TEXT,      -- e.g., '2008_CRISIS'
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE scenario_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    shocks JSONB NOT NULL,       -- e.g., {"EQUITY": -0.30, "RATES": +0.02}
    created_by UUID REFERENCES users(id),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE fx_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(id),
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    forward_rate NUMERIC(18,8),
    maturity_date DATE,
    notional NUMERIC(20,2),
    status TEXT DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'EXPIRED'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE ledger_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(id),
    transaction_id UUID REFERENCES transactions(id),
    entry_type TEXT NOT NULL, -- 'TRADE_COST', 'FEE', 'DIVIDEND', 'INTEREST', 'REALIZED_PNL'
    amount NUMERIC(20,6),
    currency TEXT NOT NULL,
    book_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE benchmark_constituents (
    benchmark_id UUID NOT NULL REFERENCES benchmarks(id),
    symbol_id UUID NOT NULL REFERENCES symbols(id),
    weight NUMERIC(18,8) NOT NULL,
    effective_from DATE NOT NULL,
    effective_to DATE,
    PRIMARY KEY (benchmark_id, symbol_id, effective_from)
);

CREATE TABLE performance_attribution (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id TEXT NOT NULL,
    benchmark_id UUID NOT NULL REFERENCES benchmarks(id),
    as_of_date DATE NOT NULL,
    allocation_effect NUMERIC(18,8),
    selection_effect NUMERIC(18,8),
    interaction_effect NUMERIC(18,8),
    total_excess_return NUMERIC(18,8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);


INSERT INTO roles (name, description)
VALUES
    ('admin', 'Platform administrator'),
    ('trader', 'Execution and OMS access'),
    ('analyst', 'Research and signal analysis'),
    ('read-only', 'Read only reporting access'),
    ('risk_officer', 'Risk and control oversight')
ON CONFLICT (name) DO NOTHING;
