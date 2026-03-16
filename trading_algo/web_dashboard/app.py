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
import settings

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
                value="tab-portfolio",
                children=[
                    dcc.Tab(label="Portefeuille global", value="tab-portfolio"),
                    dcc.Tab(label="Analyse par symbole", value="tab-symbol"),
                    dcc.Tab(label="Vue: Indicateurs & Risques", value="tab-market")
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
        # use the dedicated market layout
        return market_layout()


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
    Best-effort: limits number of symbols fetched to avoid overloading yfinance.
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
                # keep only last 60 points as simple arrays to serialize
                closes = df['Close'].dropna().iloc[-60:].tolist()
                dates = [str(d.date()) for d in df.index[-60:]]
                indices[symbol] = {'dates': dates, 'closes': closes}
            except Exception as e:
                logger.debug(f"index fetch error {symbol}: {e}")

        # Sector perf & top movers: sample S&P500 (or fallback) and compute 1M perf
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
                    perf = (float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0]) * 100.0
                    # attempt to get sector from overview
                    try:
                        overview = extractor.get_fundamental_data_fallback(s)
                        sector = overview.get('profile', {}).get('sector') or "Unknown"
                    except Exception:
                        sector = "Unknown"
                    perf_list.append({'symbol': s, 'perf': perf, 'sector': sector})
                    # accumulate per-sector
                    sector_map.setdefault(sector, []).append(perf)
                except Exception as e:
                    logger.debug(f"symbol sample error {s}: {e}")
                # small pause to be nice to remote endpoints
                time.sleep(0.05)
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
            "macro": macro
        }
        _safe_dump_cache(payload)
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