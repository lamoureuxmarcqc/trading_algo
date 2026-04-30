# trading_algo/callbacks/market_callbacks.py
import os
import json
import logging
import datetime
import functools
from typing import Optional, Any, Dict

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, html, dcc, no_update, callback_context
import dash_bootstrap_components as dbc

from trading_algo.visualization.market_dashboard import MarketDashboard
from trading_algo.market.market_manager import MarketManager
from trading_algo import settings

logger = logging.getLogger(__name__)

# Instance unique du gestionnaire de marché (cache fichier par défaut)
_manager = MarketManager(cache_service=None)


def log_exceptions(default_return=None):
    """Décorateur pour capturer et logger les exceptions dans les callbacks."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Erreur dans {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


def _build_market_health_cards(health_data: Dict[str, Any]) -> dbc.Row:
    """Construit les cartes Bootstrap pour l'affichage de la santé du marché."""
    score = health_data.get("score", 50)
    label = health_data.get("label", "Neutral")
    vix = health_data.get("vix")
    put_call = health_data.get("put_call_ratio")
    adv_decl = health_data.get("advance_decline")
    indices = health_data.get("indices", {})

    # Couleur selon le score
    color_class = "success" if score >= 70 else "danger" if score <= 30 else "warning"

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Santé du marché", className="small text-muted"),
            html.H4(f"{score:.0f}/100", className=f"mb-0 text-{color_class}"),
            html.Div(label, className="small text-muted")
        ]), className="shadow-sm h-100"), width=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("VIX", className="small text-muted"),
            html.H5(f"{vix if vix is not None else 'N/A'}", className="mb-0"),
            html.Div("Volatilité", className="small text-muted")
        ]), className="shadow-sm h-100"), width=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Put/Call", className="small text-muted"),
            html.H5(f"{put_call if put_call is not None else 'N/A'}", className="mb-0"),
            html.Div("Signal contrarien", className="small text-muted")
        ]), className="shadow-sm h-100"), width=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Avance/Déclin", className="small text-muted"),
            html.H5(f"{adv_decl if adv_decl is not None else 'N/A'}", className="mb-0"),
            html.Div("Largeur du marché", className="small text-muted")
        ]), className="shadow-sm h-100"), width=2),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Indices (S&P / NASDAQ / Dow)", className="small text-muted"),
            html.H5([
                html.Span(f"{indices.get('sp500', 'N/A'):+.1f}%" if isinstance(indices.get('sp500'), (int, float)) else indices.get('sp500', 'N/A'),
                          className="text-success" if isinstance(indices.get('sp500'), (int, float)) and indices.get('sp500', 0) > 0 else "text-danger"),
                html.Span(" | "),
                html.Span(f"{indices.get('nasdaq', 'N/A'):+.1f}%" if isinstance(indices.get('nasdaq'), (int, float)) else indices.get('nasdaq', 'N/A'),
                          className="text-success" if isinstance(indices.get('nasdaq'), (int, float)) and indices.get('nasdaq', 0) > 0 else "text-danger"),
                html.Span(" | "),
                html.Span(f"{indices.get('dow', 'N/A'):+.1f}%" if isinstance(indices.get('dow'), (int, float)) else indices.get('dow', 'N/A'),
                          className="text-success" if isinstance(indices.get('dow'), (int, float)) and indices.get('dow', 0) > 0 else "text-danger")
            ], className="mb-0"),
            html.Div("Performance quotidienne", className="small text-muted")
        ]), className="shadow-sm h-100"), width=4),
    ], className="g-2 mb-3")


def register_market_callbacks(app):
    """
    Enregistre les callbacks pour l'onglet 'Marché' :
      - Mise à jour du graphique principal (indices, secteurs, top movers)
      - Export Excel des données
      - Filtre secteur dynamique
      - Indicateurs de santé du marché
    """

    # =========================================================
    # 1. GRAPHIQUE PRINCIPAL (macro + sectoriel)
    # =========================================================
    @app.callback(
        [Output("market-overview-fig", "figure"),
         Output("market-cache-raw", "children")],
        [Input("market-data-store", "data"),
         Input("market-refresh-btn", "n_clicks"),
         Input("market-period-dropdown", "value"),
         Input("market-sector-filter", "value")],
        [State("market-auto-refresh", "value")],
        prevent_initial_call=False
    )
    @log_exceptions(default_return=(go.Figure(), html.Div("Erreur de chargement", className="text-danger")))
    def update_market_overview(store_data, n_clicks, period, sector_filter, auto_refresh_val):
        """
        Construit la vue d'ensemble du marché :
          - Indices (courbes)
          - Performance sectorielle (barres)
          - Top movers (tableau)
          - Données macro (si disponibles)
        """
        # Valeurs par défaut
        period = period or "1M"
        sector_filter = sector_filter or "ALL"

        # Construction des données agrégées via le MarketManager
        overview = _manager.build_overview(
            store_data=store_data,
            period_label=period,
            sector_filter=sector_filter
        )

        # Vérification minimale
        if not overview or (not overview.get("indices") and overview.get("sector_perf", pd.DataFrame()).empty):
            return go.Figure(), html.Div("Aucune donnée disponible", className="text-muted")

        # Transfert des données vers la couche de visualisation
        dashboard = MarketDashboard()
        dashboard.load_data(
            indices=overview.get("indices", {}),
            sector_perf=overview.get("sector_perf", pd.DataFrame()),
            top_movers=overview.get("top_movers", pd.DataFrame()),
            macro=overview.get("macro", {}),
            period_label=overview.get("period_label", "1M")
        )
        fig = dashboard.create_figure()

        # Vue debug (contenu brut du cache, tronqué pour l'affichage)
        raw_cache = _manager.read_cache(store_data) or {}
        raw_json = json.dumps(raw_cache, indent=2, default=str)
        if len(raw_json) > 5000:
            raw_json = raw_json[:5000] + "\n... (tronqué)"
        debug_view = html.Pre(raw_json, style={"fontSize": "10px", "maxHeight": "200px", "overflow": "auto"})

        return fig, debug_view

    # =========================================================
    # 2. EXPORT EXCEL DU RAPPORT DE MARCHÉ
    # =========================================================
    @app.callback(
        Output("market-download", "data"),
        Input("market-download-btn", "n_clicks"),
        State("market-sector-filter", "value"),
        prevent_initial_call=True
    )
    @log_exceptions(default_return=no_update)
    def download_market_report(n_clicks, sector_filter):
        """Génère et télécharge un fichier Excel contenant toutes les données marché."""
        if not n_clicks:
            return no_update

        bytes_data = _manager.export_market_report(sector_filter=sector_filter)
        if not bytes_data:
            logger.warning("Aucune donnée à exporter")
            return no_update

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        return dcc.send_bytes(bytes_data, filename=f"trading_market_report_{timestamp}.xlsx")

    # =========================================================
    # 3. FILTRE SECTEUR DYNAMIQUE (liste déroulante)
    # =========================================================
    @app.callback(
        Output("market-sector-filter", "options"),
        Input("market-data-store", "data"),
    )
    @log_exceptions(default_return=[{"label": "Tous les secteurs", "value": "ALL"}])
    def update_sector_dropdown(store_data):
        """Extrait la liste des secteurs présents dans le cache pour le filtre."""
        cache = _manager.read_cache(store_data)
        if not cache:
            return [{"label": "Tous les secteurs", "value": "ALL"}]

        sectors = set()
        for key in ("top_gainers", "top_losers"):
            items = cache.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("sector"):
                        sectors.add(item["sector"])

        options = [{"label": "Tous les secteurs", "value": "ALL"}]
        options += [{"label": s, "value": s} for s in sorted(sectors) if s]
        return options

    # =========================================================
    # 4. INDICATEURS DE SANTÉ DU MARCHÉ (cartes)
    # =========================================================
    @app.callback(
        Output("market-health", "children"),
        [Input("market-data-store", "data"),
         Input("market-refresh-btn", "n_clicks")]
    )
    @log_exceptions(default_return=html.Div("Indisponible", className="text-muted"))
    def update_market_health(store_data, n_clicks):
        """
        Affiche un résumé de la santé du marché sous forme de cartes Bootstrap :
          - Score global (0-100)
          - VIX, Put/Call, Advance/Decline
          - Performance des indices principaux
        """
        health = _manager.get_market_health(cache_override=store_data)
        if not health:
            return html.Div("Données de santé non disponibles", className="text-warning")

        return _build_market_health_cards(health)