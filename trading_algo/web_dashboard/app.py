import os
import sys
import logging
import threading
import json
import asyncio
import signal
import importlib
from typing import Any, Dict, Optional, List
from datetime import datetime as dt_class
from pathlib import Path

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from apscheduler.schedulers.background import BackgroundScheduler

# --- CONFIGURATION DES CHEMINS ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading_algo.web_dashboard.layouts.portfolio_layout import portfolio_layout
from trading_algo.web_dashboard.layouts.symbol_layout import symbol_layout
from trading_algo.web_dashboard.layouts.market_layout import market_layout
from trading_algo import settings
from trading_algo.logging_config import init_logging

# --- MODULES DE GESTION ---
try:
    from trading_algo.market.market_manager import MarketManager
except Exception as e:
    MarketManager = None
    logging.getLogger(__name__).exception("Impossible de charger MarketManager: %s", e)

# Logging
init_logging(level=os.getenv("LOG_LEVEL", None), logfile=os.getenv("LOG_FILE", None))
logger = logging.getLogger(__name__)

# --- CACHE & MANAGER (inchangés) ---
class MarketCacheService:
    def __init__(self):
        self.file_path = getattr(settings, "MARKET_CACHE_FILE", os.path.join(os.getcwd(), "cache", "market_overview.json"))
        self.redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_URI")
        self.lock = threading.Lock()
        self._redis_client = None

        if self.redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
                logger.info("Redis cache enabled for market overview")
            except Exception as e:
                logger.warning(f"Redis client init failed, falling back to file cache: {e}")
                self._redis_client = None

    def get_data(self) -> Dict[str, Any]:
        if self._redis_client:
            try:
                raw = self._redis_client.get("market_overview")
                if raw:
                    try:
                        return json.loads(raw)
                    except Exception as e:
                        logger.warning(f"Failed to decode JSON from Redis payload: {e}")
            except Exception as e:
                logger.warning(f"Redis fetch failed: {e}")

        if os.path.exists(self.file_path):
            with self.lock:
                try:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data if isinstance(data, dict) else {}
                except Exception as e:
                    logger.warning(f"Failed to read/parse cache file {self.file_path}: {e}")
                    return {}
        return {}

    def save_data(self, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with self.lock:
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"File save failed: {e}")


cache_service = MarketCacheService()
manager = MarketManager(cache_service) if MarketManager is not None else None

# --- DASH APP ---
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
)
server = app.server
app.title = "Trading Algo — Professional Terminal"

# health route
@server.route("/health")
def _health():
    return "OK", 200

# Shared wrapper to keep tab panels visually consistent.
def _tab_panel(child: Any) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(child, className="p-0"),
        className="border-0 shadow-sm",
    )

# layout
app.layout = html.Div([
    dbc.NavbarSimple(
        children=[
            html.Span(id="header-last-update", className="nav-link text-light font-monospace small"),
        ],
        brand="📊 TRADING ALGO TERMINAL",
        brand_href="#",
        color="dark",
        dark=True,
        fluid=True,
        className="mb-0 shadow-sm"
    ),

    dbc.Container(fluid=True, className="mt-3", children=[
        dbc.Row(id="market-summary", className="mb-3 g-2 flex-nowrap overflow-auto pb-2"),

        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-portfolio",
            className="nav-justified shadow-sm",
            children=[
                dbc.Tab(
                    _tab_panel(portfolio_layout()),
                    label="📈 Portefeuille",
                    tab_id="tab-portfolio",
                    label_class_name="fw-bold",
                ),
                dbc.Tab(
                    _tab_panel(symbol_layout()),
                    label="🔍 Analyse Symbole",
                    tab_id="tab-symbol",
                    label_class_name="fw-bold",
                ),
                dbc.Tab(
                    _tab_panel(market_layout()),
                    label="🌍 Macro & Risques",
                    tab_id="tab-market",
                    label_class_name="fw-bold",
                ),
            ],
        ),

        dcc.Store(id="market-data-store"),
        dcc.Interval(id="market-cache-check-interval", interval=30 * 1000, n_intervals=0),
        dcc.Location(id='url', refresh=False),
    ])
], style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# Keep the full tab tree available for Dash validation and callback wiring.
app.validation_layout = app.layout

# --- global data callback (inchangé) ---
@app.callback(
    [Output("header-last-update", "children"),
     Output("market-summary", "children"),
     Output("market-data-store", "data")],
    [Input("market-cache-check-interval", "n_intervals")]
)
def update_ui_from_cache(_):
    cache = cache_service.get_data()
    if not cache:
        return "⚠️ Sync Offline", [dbc.Col(html.P("Chargement du cache...", className="text-muted small"))], None

    ts = cache.get("timestamp", "N/A")
    try:
        dt = dt_class.fromisoformat(ts.replace('Z', '+00:00')) if isinstance(ts, str) else None
        if dt:
            time_str = dt.strftime("%H:%M:%S")
            status_text = f"Flux Direct: {time_str}"
        else:
            status_text = f"Flux: {ts}"
    except Exception:
        status_text = f"Flux: {ts}"

    indices_raw = cache.get("indices", {}) or {}
    display_map = {
        "^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones",
        "^GSPTSE": "TSX", "^VIX": "VIX", "GC=F": "OR",
        "CADUSD=X": "CAD/USD", "BTC-USD": "BTC"
    }

    cards: List[Any] = []
    for sym, label in display_map.items():
        if sym in indices_raw:
            data = indices_raw[sym]
            closes = data.get('closes', []) or []
            if len(closes) >= 2:
                try:
                    price, prev = float(closes[-1]), float(closes[-2])
                    change = ((price - prev) / prev) * 100 if prev else 0.0
                    color = "success" if change >= 0 else "danger"
                    icon = "▲" if change >= 0 else "▼"
                    cards.append(dbc.Col(
                        dbc.Card(className="shadow-sm border-0", children=[
                            dbc.CardBody([
                                html.Div(label, className="text-muted small fw-bold"),
                                html.Div([
                                    html.Span(f"{price:,.2f} ", className="fw-bold"),
                                    html.Span(f"{icon} {abs(change):.2f}%", className=f"small text-{color}")
                                ])
                            ], className="p-2")
                        ]), width="auto"
                    ))
                except Exception as e:
                    logger.debug("Skipping index %s due to parsing error: %s", sym, e)

    if not cards:
        cards = [dbc.Col(html.P("Aucune donnée d'indice disponible", className="text-muted small"))]

    return status_text, cards, cache

# --- REGISTER BUSINESS CALLBACKS (PROTECT AGAINST FAILURES) ---
def safe_register(module_path: str, register_name: str, app_instance: Dash) -> None:
    try:
        module = importlib.import_module(module_path)
        register_func = getattr(module, register_name)
        register_func(app_instance)
        logger.info("Registered callbacks: %s.%s", module_path, register_name)
    except Exception as e:
        logger.exception("Failed to register callbacks %s.%s: %s", module_path, register_name, e)

# register callbacks with protection
safe_register("trading_algo.callbacks.portfolio_callbacks", "register_portfolio_callbacks", app)
safe_register("trading_algo.callbacks.symbol_callbacks", "register_symbol_callbacks", app)
safe_register("trading_algo.callbacks.market_callbacks", "register_market_callbacks", app)

# --- SCHEDULER & SIGNAL HANDLERS (unchanged) ---
_scheduler: Optional[BackgroundScheduler] = None

def _start_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.info("Scheduler déjà démarré.")
        return
    if manager is None:
        logger.warning("Scheduler non démarré: MarketManager indisponible.")
        return

    interval_min = getattr(settings, "MARKET_REFRESH_INTERVAL_MIN", 10)
    sched = BackgroundScheduler()

    def refresh_task():
        logger.info("Démarrage du cycle de rafraîchissement marché...")
        try:
            asyncio.run(manager.refresh_all_market_data())
        except Exception as e:
            logger.exception("Erreur lors du refresh marché: %s", e)
        finally:
            logger.info("Cycle de rafraîchissement terminé.")

    sched.add_job(refresh_task, 'interval', minutes=interval_min, next_run_time=dt_class.now())
    sched.start()
    _scheduler = sched
    logger.info("Background Scheduler démarré (Intervalle: %s min)", interval_min)

def _stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        try:
            _scheduler.shutdown(wait=False)
            logger.info("Background Scheduler arrêté.")
        except Exception as e:
            logger.exception("Erreur lors de l'arrêt du scheduler: %s", e)
        finally:
            _scheduler = None

def _register_signal_handlers() -> None:
    def _handle_signal(signum, frame):
        logger.info("Received signal %s, shutting down scheduler...", signum)
        _stop_scheduler()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            logger.debug("Signal %s handler not registered.", sig)

# --- MAIN ---
if __name__ == "__main__":
    debug_mode = bool(getattr(settings, "DEBUG", False))
    werkzeug_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    try:
        if werkzeug_child or not debug_mode:
            _start_scheduler()
            _register_signal_handlers()

        host = getattr(settings, "WEB_HOST", "127.0.0.1")
        port = int(getattr(settings, "WEB_PORT", 8050))
        use_debug = bool(getattr(settings, "DEBUG", True))

        logger.info("Starting Dash server on %s:%s (debug=%s)", host, port, use_debug)
        app.run(debug=use_debug, host=host, port=port)
    except Exception as e:
        logger.exception("Unhandled exception in main: %s", e)
    finally:
        _stop_scheduler()
