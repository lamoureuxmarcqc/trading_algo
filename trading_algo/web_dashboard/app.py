import os
import sys
import logging
import threading
import json
import asyncio
import signal
import importlib
from functools import lru_cache
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

# =========================================================
# 🔹 CACHE SERVICE AMÉLIORÉ (avec fallback et validation)
# =========================================================
class MarketCacheService:
    """Service de cache pour les données de marché, avec Redis ou fichier."""
    def __init__(self):
        self.file_path = getattr(settings, "MARKET_CACHE_FILE", os.path.join(os.getcwd(), "cache", "market_overview.json"))
        self.redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_URI")
        self.lock = threading.RLock()  # RLock pour éviter les deadlocks
        self._redis_client = None
        self._cache_ttl = getattr(settings, "CACHE_TTL_SECONDS", 300)  # TTL par défaut 5 min

        if self.redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
                # Test de connexion
                self._redis_client.ping()
                logger.info("Redis cache enabled for market overview")
            except Exception as e:
                logger.warning(f"Redis client init failed, falling back to file cache: {e}")
                self._redis_client = None

    def get_data(self) -> Dict[str, Any]:
        """Récupère les données du cache, avec validation de la structure minimale."""
        data = None
        if self._redis_client:
            try:
                raw = self._redis_client.get("market_overview")
                if raw:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from Redis: {e}")
            except Exception as e:
                logger.warning(f"Redis fetch failed: {e}")

        # Fallback fichier
        if data is None and os.path.exists(self.file_path):
            with self.lock:
                try:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read cache file: {e}")

        # Validation : doit être un dict avec au moins "indices"
        if not isinstance(data, dict):
            logger.warning("Cache data is not a dict, returning empty")
            return {}
        if "indices" not in data:
            data["indices"] = {}
        return data

    def save_data(self, payload: Dict[str, Any]) -> None:
        """Sauvegarde les données dans le cache (fichier et Redis si disponible)."""
        if not isinstance(payload, dict):
            logger.error("save_data called with non-dict payload")
            return

        # Sauvegarde fichier
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with self.lock:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"File save failed: {e}")

        # Sauvegarde Redis
        if self._redis_client:
            try:
                self._redis_client.setex("market_overview", self._cache_ttl, json.dumps(payload))
            except Exception as e:
                logger.warning(f"Redis save failed: {e}")


cache_service = MarketCacheService()
manager = MarketManager(cache_service) if MarketManager is not None else None


# =========================================================
# 🔹 DASH APP
# =========================================================
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.FLATLY,          # ou FLATLY, BOOTSTRAP, etc.
        dbc.icons.BOOTSTRAP,
        "/assets/custom.css"        # votre CSS externe
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.title = "Trading Algo — Professional Terminal"

# Route santé
@server.route("/health")
def _health():
    return "OK", 200

# Helper pour les panels d'onglets (style cohérent)
def _tab_panel(child: Any) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(child, className="p-0"),
        className="border-0 shadow-sm bg-white",
    )

# Layout principal (avec CSS interne pour forcer la couleur des onglets)
app.layout = html.Div([

    dbc.NavbarSimple(
        children=[
            html.Span(id="header-last-update", className="nav-link text-light font-monospace small"),
            dbc.NavItem(dbc.NavLink("Documentation", href="#", id="doc-link", className="text-light")),
        ],
        brand="📊 TRADING ALGO TERMINAL",
        brand_href="#",
        color="dark",
        dark=True,
        fluid=True,
        className="mb-0 shadow-sm"
    ),
    dbc.Container(fluid=True, className="mt-3", children=[
        dbc.Row(id="market-summary", className="mb-3 g-2 flex-nowrap overflow-auto pb-2", style={"minHeight": "90px"}),
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
        dbc.Toast(
            id="error-toast",
            header="Erreur",
            icon="danger",
            duration=4000,
            is_open=False,
            style={"position": "fixed", "top": 70, "right": 10, "width": 350, "zIndex": 1000},
        ),
    ])
], style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# Pour la validation des callbacks (Dash >= 2.0)
app.validation_layout = app.layout

# =========================================================
# 🔹 CALLBACK GLOBAL DE MISE À JOUR DU RÉSUMÉ MARCHÉ
# =========================================================
@app.callback(
    [Output("header-last-update", "children"),
     Output("market-summary", "children"),
     Output("market-data-store", "data"),
     Output("error-toast", "is_open"),
     Output("error-toast", "children")],
    [Input("market-cache-check-interval", "n_intervals")],
    prevent_initial_call=False,
)
def update_ui_from_cache(n_intervals):
    """Met à jour le bandeau des indices et le store de données."""
    cache = cache_service.get_data()
    error_msg = None
    if not cache:
        error_msg = "Impossible de charger les données de marché. Vérifiez la connexion."
        logger.warning(error_msg)
        return "⚠️ Sync Offline", [dbc.Col(html.P(error_msg, className="text-muted small"))], None, True, error_msg

    # Timestamp
    ts = cache.get("timestamp", "N/A")
    try:
        # Gestion du format ISO avec Z
        if isinstance(ts, str):
            ts_clean = ts.replace('Z', '+00:00')
            dt = dt_class.fromisoformat(ts_clean)
            time_str = dt.strftime("%H:%M:%S")
            status_text = f"📡 Dernière mise à jour: {time_str}"
        else:
            status_text = f"📡 Cache: {ts}"
    except Exception:
        status_text = f"📡 Cache: {ts}"

    indices_raw = cache.get("indices", {}) or {}
    # Mapping des symboles vers des noms lisibles et éventuels formateurs
    display_map = {
        "^GSPC": ("S&P 500", "{:,.2f}"),
        "^IXIC": ("NASDAQ", "{:,.2f}"),
        "^DJI": ("Dow Jones", "{:,.0f}"),
        "^GSPTSE": ("TSX", "{:,.2f}"),
        "^VIX": ("VIX", "{:.2f}"),
        "GC=F": ("OR", "{:.2f}"),
        "CADUSD=X": ("CAD/USD", "{:.4f}"),
        "BTC-USD": ("BTC", "{:,.0f}")
    }

    cards = []
    for sym, (label, fmt) in display_map.items():
        if sym in indices_raw:
            data = indices_raw[sym]
            closes = data.get('closes', [])
            if isinstance(closes, list) and len(closes) >= 2:
                try:
                    price = float(closes[-1])
                    prev = float(closes[-2])
                    change = ((price - prev) / prev) * 100 if prev != 0 else 0.0
                    color = "success" if change >= 0 else "danger"
                    icon = "▲" if change >= 0 else "▼"
                    cards.append(dbc.Col(
                        dbc.Card(
                            className="shadow-sm border-0 h-100",
                            children=[
                                dbc.CardBody([
                                    html.Div(label, className="text-muted small fw-bold"),
                                    html.Div([
                                        html.Span(fmt.format(price), className="fw-bold"),
                                        html.Span(f" {icon} {abs(change):.2f}%", className=f"small text-{color}")
                                    ])
                                ], className="p-2")
                            ]
                        ),
                        width="auto",
                        className="mb-2"
                    ))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Erreur parsing {sym}: {e}")
                    continue

    if not cards:
        cards = [dbc.Col(html.P("Aucune donnée d'indice disponible", className="text-muted small"))]

    return status_text, cards, cache, False, ""

# =========================================================
# 🔹 ENREGISTREMENT SÉCURISÉ DES CALLBACKS MÉTIER
# =========================================================
def safe_register(module_path: str, register_name: str, app_instance: Dash) -> None:
    """Importe et exécute la fonction d'enregistrement des callbacks avec gestion d'erreur."""
    try:
        module = importlib.import_module(module_path)
        register_func = getattr(module, register_name)
        register_func(app_instance)
        logger.info("Registered callbacks: %s.%s", module_path, register_name)
    except ImportError as e:
        logger.error("Module %s not found: %s", module_path, e)
    except AttributeError:
        logger.error("Function %s not found in %s", register_name, module_path)
    except Exception as e:
        logger.exception("Failed to register callbacks %s.%s: %s", module_path, register_name, e)

# Enregistrement des callbacks
safe_register("trading_algo.callbacks.portfolio_callbacks", "register_portfolio_callbacks", app)
safe_register("trading_algo.callbacks.symbol_callbacks", "register_symbol_callbacks", app)
safe_register("trading_algo.callbacks.market_callbacks", "register_market_callbacks", app)

# =========================================================
# 🔹 SCHEDULER ET GESTION DES SIGNAUX (ROBUSTE)
# =========================================================
_scheduler: Optional[BackgroundScheduler] = None

def _start_scheduler() -> None:
    """Démarre le scheduler en arrière-plan pour rafraîchir les données de marché."""
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.info("Scheduler déjà démarré.")
        return
    if manager is None:
        logger.warning("Scheduler non démarré: MarketManager indisponible.")
        return

    interval_min = getattr(settings, "MARKET_REFRESH_INTERVAL_MIN", 10)
    sched = BackgroundScheduler(timezone="UTC")  # Définir un fuseau pour éviter les warnings

    def refresh_task():
        logger.info("Démarrage du cycle de rafraîchissement marché...")
        try:
            # Utiliser asyncio.run() mais attention aux event loops existants
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(manager.refresh_all_market_data())
            finally:
                loop.close()
        except Exception as e:
            logger.exception("Erreur lors du refresh marché: %s", e)
        finally:
            logger.info("Cycle de rafraîchissement terminé.")

    sched.add_job(refresh_task, 'interval', minutes=interval_min, next_run_time=dt_class.now())
    sched.start()
    _scheduler = sched
    logger.info("Background Scheduler démarré (Intervalle: %s min)", interval_min)

def _stop_scheduler() -> None:
    """Arrête proprement le scheduler."""
    global _scheduler
    if _scheduler:
        try:
            _scheduler.shutdown(wait=True, timeout=10)
            logger.info("Background Scheduler arrêté.")
        except Exception as e:
            logger.exception("Erreur lors de l'arrêt du scheduler: %s", e)
        finally:
            _scheduler = None

def _register_signal_handlers() -> None:
    """Enregistre les handlers pour SIGINT et SIGTERM."""
    def _handle_signal(signum, frame):
        logger.info("Received signal %s, shutting down scheduler...", signum)
        _stop_scheduler()
        # Option: quitter proprement
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception as e:
            logger.debug("Signal %s handler not registered: %s", sig, e)

# =========================================================
# 🔹 MAIN
# =========================================================
if __name__ == "__main__":
    debug_mode = getattr(settings, "DEBUG", False)
    # En mode debug avec rechargement automatique, le scheduler ne doit être démarré qu'une fois
    # Werkzeug lance deux processus : le parent et l'enfant. On démarre uniquement dans l'enfant.
    werkzeug_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    try:
        if werkzeug_child or not debug_mode:
            _start_scheduler()
            _register_signal_handlers()
        else:
            logger.info("Mode debug avec rechargement actif : scheduler démarré uniquement dans le processus enfant.")

        host = getattr(settings, "WEB_HOST", "127.0.0.1")
        port = int(getattr(settings, "WEB_PORT", 8050))
        use_debug = getattr(settings, "DEBUG", True)

        logger.info("Starting Dash server on %s:%s (debug=%s)", host, port, use_debug)
        app.run(debug=use_debug, host=host, port=port)
    except Exception as e:
        logger.exception("Unhandled exception in main: %s", e)
    finally:
        _stop_scheduler()