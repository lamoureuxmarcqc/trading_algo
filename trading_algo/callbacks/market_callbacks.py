import os
import json
import logging
import datetime
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, html, dcc, no_update
import dash_bootstrap_components as dbc

# Imports internes
from trading_algo.visualization.market_dashboard import MarketDashboard
from trading_algo.market.market_manager import MarketManager
from trading_algo import settings

logger = logging.getLogger(__name__)

# Instantiate a manager (uses file cache fallback if no cache service provided)
manager = MarketManager(cache_service=None)


def register_market_callbacks(app):

    # 1. MISE À JOUR DE LA VUE MACRO ET SECTORIELLE
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
    def update_market_overview(store_data, n_clicks, period, sector_filter, auto_refresh_val):
        try:
            # Business logic: build the overview via manager
            overview = manager.build_overview(store_data, period_label=period, sector_filter=sector_filter)

            # If no data, return placeholder
            if not overview or (not overview.get("indices") and overview.get("sector_perf").empty and overview.get("top_movers").empty):
                return go.Figure(), html.Div("En attente de données...", className="text-muted")

            md = MarketDashboard()
            md.load_data(
                indices=overview.get("indices", {}),
                sector_perf=overview.get("sector_perf", pd.DataFrame()),
                top_movers=overview.get("top_movers", pd.DataFrame()),
                macro=overview.get("macro", {}),
                period_label=overview.get("period_label", "1M")
            )
            fig = md.create_figure()

            # Debug view (trimmed)
            raw_pretty = json.dumps(manager.read_cache(store_data) or {}, indent=2)
            debug_view = html.Pre(raw_pretty[:5000] + ("..." if len(raw_pretty) > 5000 else ""), style={"fontSize": "10px"})

            return fig, debug_view

        except Exception as e:
            logger.exception("Erreur update_market_overview")
            return go.Figure(), html.Pre(f"Erreur système : {str(e)}", className="text-danger")

    # 2. EXPORT EXCEL DES DONNÉES DE MARCHÉ (délégué au manager)
    @app.callback(
        Output("market-download", "data"),
        Input("market-download-btn", "n_clicks"),
        State("market-sector-filter", "value"),
        prevent_initial_call=True
    )
    def download_market_data(n_clicks, sector_filter):
        try:
            bytes_data = manager.export_market_report(sector_filter=sector_filter)
            if not bytes_data:
                return no_update
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            return dcc.send_bytes(bytes_data, filename=f"trading_market_report_{ts}.xlsx")
        except Exception:
            logger.exception("Erreur lors de l'export Excel")
            return no_update

    # 3. POPULATION DYNAMIQUE DU FILTRE SECTEUR (utilise manager.read_cache)
    @app.callback(
        Output("market-sector-filter", "options"),
        Input("market-data-store", "data"),
    )
    def update_sector_dropdown(store_data):
        try:
            cache = manager.read_cache(store_data)
            if not cache:
                return [{"label": "Tous les secteurs", "value": "ALL"}]

            sectors = set()
            combined = cache.get("top_gainers", []) + cache.get("top_losers", [])
            for item in combined:
                if isinstance(item, dict) and item.get("sector"):
                    sectors.add(item["sector"])

            options = [{"label": "Tous les secteurs", "value": "ALL"}]
            options += [{"label": s, "value": s} for s in sorted(list(sectors)) if s]
            return options
        except Exception:
            logger.exception("Failed to populate sector dropdown")
            return [{"label": "Tous les secteurs", "value": "ALL"}]

    # NOUVEAU : résumé de la santé du marché
    @app.callback(
        Output("market-health", "children"),
        [Input("market-data-store", "data"),
         Input("market-refresh-btn", "n_clicks")]
    )
    def update_market_health(store_data, n_clicks):
        try:
            health = manager.get_market_health(cache_override=store_data)
            score = health.get("score", 50)
            label = health.get("label", "Neutral")
            desc = health.get("description", "")
            vix = health.get("vix")
            put_call = health.get("put_call_ratio")
            adv = health.get("advance_decline")
            indices = health.get("indices", {})

            # Créer de petites cartes pour un aperçu rapide
            cards = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Santé du marché", className="small text-muted"),
                    html.H4(f"{score:.0f}/100", className="mb-0"),
                    html.Div(label, className="small text-muted")
                ]), className="shadow-sm"), width=2),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("VIX", className="small text-muted"),
                    html.H5(f"{vix if vix is not None else 'N/A'}", className="mb-0"),
                    html.Div("Volatilité", className="small text-muted")
                ]), className="shadow-sm"), width=2),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Put/Call", className="small text-muted"),
                    html.H5(f"{put_call if put_call is not None else 'N/A'}", className="mb-0"),
                    html.Div("Signal contrarien", className="small text-muted")
                ]), className="shadow-sm"), width=2),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Avance/Déclin", className="small text-muted"),
                    html.H5(f"{adv if adv is not None else 'N/A'}", className="mb-0"),
                    html.Div("Largeur", className="small text-muted")
                ]), className="shadow-sm"), width=2),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Indices (SP / NAS / DJ)", className="small text-muted"),
                    html.H5(f"{indices.get('sp500', 'N/A')}, {indices.get('nasdaq', 'N/A')}, {indices.get('dow', 'N/A')}", className="mb-0"),
                    html.Div("Perf.", className="small text-muted")
                ]), className="shadow-sm"), width=4),
            ], className="g-2 mb-2")

            return cards

        except Exception as e:
            logger.exception("Failed to build market health UI: %s", e)
            return html.Div("N/A", className="text-muted")