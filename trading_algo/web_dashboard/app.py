# trading_algo/web_dashboard/app.py
import os
import sys
from pathlib import Path

# Ensure package imports work when running the app module directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output
import dash
import plotly.graph_objects as go
import datetime
import threading
import json
import time
import logging

# External stylesheet (Bootstrap) for a professional look
EXTERNAL_STYLESHEETS = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
]

# Import existing layouts and callback registrars
from trading_algo.web_dashboard.layouts.portfolio_layout import portfolio_layout
from trading_algo.web_dashboard.layouts.symbol_layout import symbol_layout
from trading_algo.web_dashboard.layouts.market_layout import market_layout
from trading_algo.callbacks.portfolio_callbacks import register_portfolio_callbacks
from trading_algo.callbacks.symbol_callbacks import register_symbol_callbacks
from trading_algo.callbacks.market_callbacks import register_market_callbacks

# Data helpers
from trading_algo.data.data_extraction import StockDataExtractor, MacroDataExtractor
from trading_algo.screening.actions_sp500 import get_sp500_symbols
from trading_algo import settings

logger = logging.getLogger(__name__)

# App factory
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=EXTERNAL_STYLESHEETS,
)
server = app.server  # for gunicorn/WSGI deployments
app.title = "Trading Algo — Portfolio & Market Dashboard"

# Simple on-disk cache path (scheduler writes, callbacks read)
MARKET_CACHE_FILE = getattr(settings, "MARKET_CACHE_FILE", os.path.join(os.getcwd(), "cache", "market_overview.json"))
MARKET_CACHE_LOCK = threading.Lock()

# Top-level layout: header + market overview + tabs + tab-content
app.layout = html.Div(
    [
        # Header
        html.Nav(
            className="navbar navbar-expand-lg navbar-dark bg-dark",
            children=[
                html.Div(
                    className="container-fluid",
                    children=[
                        html.A(
                            "Trading Algo — Portfolio Dashboard",
                            className="navbar-brand",
                            href="#",
                            style={"fontWeight": "700"},
                        ),
                        html.Div(
                            className="d-flex",
                            children=[
                                html.Div(
                                    id="header-last-update",
                                    className="me-3 text-white",
                                    children="Dernière mise à jour: --",
                                ),
                                html.Div(
                                    id="header-market-labels",
                                    className="text-white",
                                    children=[],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
        html.Div(className="container-fluid p-3", children=[
            # Market overview cards
            html.Div(className="row", children=[
                html.Div(className="col-12 mb-2", children=[
                    html.Div(id="market-summary", className="d-flex flex-row gap-2 flex-wrap")
                ])
            ]),
            # Tabs for Portfolio / Symbol analysis and embedded dashboards
            dcc.Tabs(
                id="main-tabs",
                value="tab-portfolio",  # change default if you want market shown by default
                children=[
                    dcc.Tab(label="Portefeuille global", value="tab-portfolio"),
                    dcc.Tab(label="Analyse par symbole", value="tab-symbol"),
                    dcc.Tab(label="Vue: Indicateurs & Risques", value="tab-market"),  # <-- Market tab
                ],
            ),
            html.Div(id="tab-content", className="mt-3"),
            # Hidden stores and intervals
            dcc.Store(id="market-data-store"),
            dcc.Interval(id="market-refresh-interval", interval=60 * 1000, n_intervals=0),  # every minute
            dcc.Interval(id="market-cache-check-interval", interval=30 * 1000, n_intervals=0),  # heartbeat for UI
        ]),
        # Footer
        html.Footer(
            className="footer mt-auto py-2 bg-light",
            children=html.Div(
                className="container",
                children=html.Small(
                    f"© Trading Algo — {datetime.date.today().year} — Built for professional portfolio management",
                    style={"color": "#333"},
                ),
            ),
        ),
    ]
)


# Render tab content using existing layout functions
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-portfolio":
        return portfolio_layout()
    if tab == "tab-symbol":
        return symbol_layout()
    if tab == "tab-market":
        # return the market layout that contains the Graph and debug area
        return market_layout()
    return html.Div()  # safe fallback


# Background refresh job: compute sector_perf and top_movers and write JSON cache
def _safe_dump_cache(payload: dict):
    os.makedirs(os.path.dirname(MARKET_CACHE_FILE), exist_ok=True)
    with MARKET_CACHE_LOCK:
        try:
            with open(MARKET_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Failed writing market cache: {e}")


def refresh_market_cache():
    """
    Fetch a lightweight market snapshot and persist to disk. This runs in background.
    Also compute simple SMA crossover trading signals for sampled symbols and
    optionally write the payload to Redis (if REDIS_URL is set).
    """
    try:
        logger.info("Market cache refresh started")
        extractor = StockDataExtractor()
        macro = MacroDataExtractor().get_all_macro_data()

        # basic indices snapshot
        indices = {}
        for symbol in ['^GSPC', '^IXIC', '^DJI', '^FTSE', '^GDAXI', '^GSPTSE']:
            try:
                df = extractor.get_historical_data(symbol=symbol, period='3mo', interval='1d')
                if isinstance(df, dict) or df is None:
                    continue
                closes = df['Close'].dropna().iloc[-60:].tolist()
                dates = [str(d.date()) for d in df.index[-60:]]
                indices[symbol] = {'dates': dates, 'closes': closes}
            except Exception as e:
                logger.debug(f"index fetch error {symbol}: {e}")

        # Sector perf & top movers: sample S&P500 (or fallback) and compute 1M perf
        top_gainers = []
        top_losers = []
        sector_perf = []
        symbol_signals = {}  # mapping symbol -> {'last_signal': int, 'signals': [{'date':..., 'signal':...}, ...]}

        try:
            symbols = []
            try:
                symbols = get_sp500_symbols() or []
            except Exception:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'BAC', 'XOM', 'CVX']

            max_symbols = getattr(settings, "MARKET_SAMPLE_SYMBOLS", 120)
            symbols = symbols[:max_symbols]
            perf_list = []
            sector_map = {}
            for s in symbols:
                try:
                    df = extractor.get_historical_data(symbol=s, period='1mo', interval='1d')
                    if df is None or df.empty or 'Close' not in df.columns:
                        continue
                    closes = df['Close'].dropna()
                    if len(closes) < 2:
                        continue

                    # compute 1M perf
                    perf = (float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0]) * 100.0

                    # attempt to get sector from overview
                    try:
                        overview = extractor.get_fundamental_data_fallback(s)
                        sector = overview.get('profile', {}).get('sector') or "Unknown"
                    except Exception:
                        sector = "Unknown"

                    perf_entry = {'symbol': s, 'perf': perf, 'sector': sector}
                    perf_list.append(perf_entry)
                    sector_map.setdefault(sector, []).append(perf)

                    # --- Compute simple SMA crossover signals ---
                    try:
                        close_series = df['Close'].dropna()
                        sma20 = close_series.rolling(20, min_periods=1).mean()
                        sma50 = close_series.rolling(50, min_periods=1).mean()

                        # detect crossovers: buy when sma20 > sma50 and previous sma20 <= sma50
                        signals = []
                        prev_state = None
                        for idx in range(1, len(close_series)):
                            cur = sma20.iloc[idx] > sma50.iloc[idx]
                            prev = sma20.iloc[idx - 1] > sma50.iloc[idx - 1]
                            if cur and not prev:
                                signals.append({'date': str(close_series.index[idx].date()), 'signal': 1})
                            elif not cur and prev:
                                signals.append({'date': str(close_series.index[idx].date()), 'signal': -1})
                        last_signal = signals[-1]['signal'] if signals else 0
                        symbol_signals[s] = {'last_signal': int(last_signal), 'signals': signals}
                    except Exception as e:
                        # non-critical
                        symbol_signals[s] = {'last_signal': 0, 'signals': []}
                        logger.debug(f"Signal computation failed for {s}: {e}")

                except Exception as e:
                    logger.debug(f"symbol sample error {s}: {e}")
                # small pause to be nice to remote endpoints
                time.sleep(0.03)

            # top movers: sort by perf
            top_sorted = sorted(perf_list, key=lambda x: x['perf'], reverse=True)
            top_gainers = top_sorted[:10]
            top_losers = top_sorted[-10:][::-1]
            # sector perf average
            sector_perf = [{'sector': k, 'perf': float(sum(v) / max(1, len(v)))} for k, v in sector_map.items()]

        except Exception as e:
            logger.exception(f"Error computing sector/top movers: {e}")
            top_gainers, top_losers, sector_perf = [], [], []

        payload = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "indices": indices,
            "sector_perf": sector_perf,
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "symbol_signals": symbol_signals,
            "macro": macro
        }

        # persist to file
        _safe_dump_cache(payload)

        # also write to Redis if configured (for multi-worker setups)
        try:
            redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_URI")
            if redis_url:
                try:
                    import redis as _redis
                    r = _redis.from_url(redis_url)
                    ttl = getattr(settings, "MARKET_CACHE_TTL", 1800)
                    r.set("market_overview", json.dumps(payload), ex=int(ttl))
                    logger.debug("Market cache written to Redis")
                except Exception as e:
                    logger.exception(f"Failed writing market cache to Redis: {e}")
        except Exception:
            # ignore redis errors
            pass

        logger.info("Market cache refresh completed")
    except Exception as e:
        logger.exception(f"Market refresh job failed: {e}")


# Start scheduler (APScheduler if present, else fallback thread)
def _start_scheduler():
    interval_min = getattr(settings, "MARKET_REFRESH_INTERVAL_MIN", 10)
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        sched = BackgroundScheduler()
        sched.add_job(refresh_market_cache, 'interval', minutes=interval_min, id='market_cache_refresh', next_run_time=datetime.datetime.now())
        sched.start()
        logger.info(f"APScheduler started for market cache refresh every {interval_min} minutes")
    except Exception:
        # fallback simple thread loop
        def _loop():
            while True:
                try:
                    refresh_market_cache()
                except Exception:
                    logger.exception("Background market refresh failed")
                time.sleep(interval_min * 60)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        logger.info(f"Background thread started for market cache refresh every {interval_min} minutes")

# Register existing callbacks defined in separate modules
register_portfolio_callbacks(app)
register_symbol_callbacks(app)
register_market_callbacks(app)

# Start the cache scheduler at import time (so it runs in deployed server processes)
_start_scheduler()

# Run server
if __name__ == "__main__":
    # Initialize logging centrally if available (best-effort)
    try:
        from trading_algo.logging_config import init_logging

        init_logging(level=os.getenv("LOG_LEVEL", None), logfile=os.getenv("LOG_FILE", None))
    except Exception:
        pass

    # Use debug mode controlled by env var for deployment safety
    debug_flag = os.getenv("DASH_DEBUG", "False").lower() in ("1", "true", "yes")
    app.run_server(debug=debug_flag, port=int(os.getenv("PORT", 8050)), host="0.0.0.0")

# --- new: header + market cards updater with cache-age and optional Redis ---
import json
import os
import datetime
import time
from dash import html

try:
    import redis
except Exception:
    redis = None

def _read_market_cache_from_redis(redis_url: str):
    try:
        if redis is None:
            return None
        r = redis.from_url(redis_url)
        raw = r.get("market_overview")
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None

def _read_market_cache_from_file(path: str):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@app.callback(
    Output("header-last-update", "children"),
    Output("market-summary", "children"),
    Input("market-cache-check-interval", "n_intervals"),
)
def update_header_and_cards_from_cache(n_intervals):
    """
    Update header timestamp + top cards from cache.
    Uses Redis if REDIS_URL env var is set (recommended for multi-worker deployments),
    otherwise reads the file at settings.MARKET_CACHE_FILE.
    Also computes cache age in seconds and shows it in the header.
    """
    # Try Redis first if configured
    redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_URI")
    cache = None
    if redis_url:
        cache = _read_market_cache_from_redis(redis_url)
    # fallback to file
    if cache is None:
        cache_path = getattr(settings, "MARKET_CACHE_FILE", os.path.join(os.getcwd(), "cache", "market_overview.json"))
        cache = _read_market_cache_from_file(cache_path)

    # default UI when no cache
    if not cache:
        return "Dernière mise à jour: -- (cache absent)", [html.Div("Market data unavailable", className="text-muted")]

    # timestamp handling
    ts = cache.get("timestamp") or cache.get("timestamp_utc") or cache.get("timestamp_iso")
    age_s = None
    last_str = "--"
    if ts:
        try:
            # accept ISO or epoch
            if isinstance(ts, (int, float)):
                dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
            else:
                dt = datetime.datetime.fromisoformat(str(ts))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            age_s = int((now - dt).total_seconds())
            # pretty local time
            last_str = dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            last_str = str(ts)

    age_display = f"{age_s}s" if age_s is not None else "n/a"
    header_text = f"Dernière mise à jour: {last_str} — cache age: {age_display}"

    # build cards from cached indices if possible
    cards = []
    indices = cache.get("indices", {}) or {}
    display_map = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones",
        "^FTSE": "FTSE",
        "^GDAXI": "DAX",
        "^GSPTSE": "TSX"
    }
    for sym, friendly in display_map.items():
        d = indices.get(sym)
        if not d:
            continue
        closes = d.get("closes", []) or []
        if not closes:
            continue
        last_price = closes[-1]
        change = None
        if len(closes) >= 2 and closes[-2] != 0:
            change = (closes[-1] - closes[-2]) / closes[-2] * 100.0
        color = "success" if (change is not None and change > 0) else "danger" if (change is not None and change < 0) else "secondary"
        card = html.Div(
            className="card p-2 market-card",
            style={"minWidth": "150px", "marginRight": "8px"},
            children=[
                html.Div(friendly, className="text-muted small"),
                html.Div(f"{last_price:,.2f}", className=f"h5 text-{color}"),
                html.Div(f"{change:+.2f}%" if change is not None else "", className="small"),
            ],
        )
        cards.append(card)

    if not cards:
        cards = [html.Div("Market snapshot available (open Market tab)", className="text-muted")]

    return header_text, cards

# optional: write to Redis too if REDIS_URL set
redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_URI")
if redis_url and redis is not None:
    try:
        r = redis.from_url(redis_url)
        r.set("market_overview", json.dumps(payload), ex=getattr(settings, "MARKET_CACHE_TTL", 1800))
    except Exception:
        logger.exception("Failed to write market cache to Redis")