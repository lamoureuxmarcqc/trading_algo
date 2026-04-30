# trading_algo/callbacks/portfolio_callbacks.py
import logging
from typing import Optional, Dict, Any

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, html, dcc
from dash.exceptions import PreventUpdate

from trading_algo.data.data_extraction import StockDataExtractor
from trading_algo.portfolio.portfoliomanager import PortfolioManager
from trading_algo.visualization.portfoliodashboard import PortfolioDashboard

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_portfolio_callbacks(app):
    """Enregistre tous les callbacks Dash pour la gestion de portefeuille."""
    manager = PortfolioManager(StockDataExtractor)

    # =========================================================
    # 🔹 1. Rafraîchissement de la liste des portefeuilles
    # =========================================================
    @app.callback(
        Output("portfolio-selector", "options"),
        Input("portfolio-refresh-timer", "n_intervals"),
    )
    def refresh_portfolio_list(_):
        """Met à jour le dropdown avec les noms des portefeuilles disponibles."""
        try:
            portfolios = manager.list_portfolios()
            return [{"label": p, "value": p} for p in portfolios]
        except Exception as e:
            logger.error(f"Erreur refresh_portfolio_list : {e}")
            return []

    # =========================================================
    # 🔹 2. Chargement d’un portefeuille (stockage dans dcc.Store)
    # =========================================================
    @app.callback(
        Output("portfolio-data-store", "data"),
        Input("load-portfolio-btn", "n_clicks"),
        State("portfolio-selector", "value"),
        prevent_initial_call=True,
    )
    def load_portfolio(n_clicks, portfolio_name):
        """Charge le portefeuille sélectionné et retourne ses métriques légères."""
        if not n_clicks or not portfolio_name:
            raise PreventUpdate

        try:
            portfolio = manager.load_portfolio(portfolio_name)
            if not portfolio:
                return {
                    "portfolio_name": portfolio_name,
                    "analysis": {},
                    "status": "load_error",
                }

            # Analyse rapide (sans calculs lourds de risque long terme)
            analysis = manager.analyze_portfolio(include_risk=False)
            return {
                "portfolio_name": portfolio_name,
                "analysis": analysis,
                "status": "loaded",
            }
        except Exception as e:
            logger.exception(f"Erreur load_portfolio : {e}")
            return {
                "portfolio_name": portfolio_name,
                "analysis": {},
                "status": "exception",
            }

    # =========================================================
    # 🔹 3. Mise à jour des KPIs (valeur, rendement, volatilité, Sharpe)
    # =========================================================
    @app.callback(
        Output("kpi-total-value", "children"),
        Output("kpi-return", "children"),
        Output("kpi-volatility", "children"),
        Output("kpi-sharpe", "children"),
        Input("portfolio-data-store", "data"),
    )
    def update_kpis(data):
        if not data or "analysis" not in data:
            return "---", "---", "---", "---"

        try:
            analysis = data["analysis"]
            perf = analysis.get("performance", {})
            risk = analysis.get("risk_metrics", {})

            total_value = perf.get("total_value", 0)
            pnl_pct = perf.get("total_pnl_pct", 0)
            volatility = risk.get("volatility", 0)
            sharpe = risk.get("sharpe_ratio", 0)

            return (
                f"{total_value:,.0f} $",
                f"{pnl_pct:.2f} %",
                f"{volatility:.2%}" if volatility else "N/A",
                f"{sharpe:.2f}" if sharpe else "N/A",
            )
        except Exception as e:
            logger.error(f"Erreur update_kpis : {e}")
            return "Erreur", "Erreur", "Erreur", "Erreur"

    # =========================================================
    # 🔹 4. Graphique principal (vue stratégique)
    # =========================================================
    @app.callback(
        Output("portfolio-main-dashboard", "figure"),
        Input("portfolio-data-store", "data"),
    )
    def update_main_chart(data):
        """Affiche le rapport stratégique (courbes de performance, allocation)."""
        if not data or "portfolio_name" not in data:
            return go.Figure()

        try:
            p_name = data["portfolio_name"]
            portfolio = manager.load_portfolio(p_name)
            if not portfolio:
                return go.Figure()

            dashboard = PortfolioDashboard(portfolio, manager)
            macro = {}
            indices = pd.DataFrame()
            fundamentals = {}
            return dashboard.create_strategic_report(macro, indices, fundamentals)
        except Exception as e:
            logger.exception("Erreur update_main_chart")
            return go.Figure()

    # =========================================================
    # 🔹 5. Tableau des positions
    # =========================================================
    @app.callback(
        Output("positions-table", "children"),
        Input("portfolio-data-store", "data"),
    )
    def update_positions(data):
        """Affiche le tableau des positions avec valeurs de marché."""
        if not data or "portfolio_name" not in data:
            return dbc.Alert("Sélectionnez un portefeuille pour voir les positions.", color="info")

        try:
            p_name = data["portfolio_name"]
            portfolio = manager.load_portfolio(p_name)
            if not portfolio:
                return dbc.Alert("Erreur lors du chargement du portefeuille.", color="danger")

            dashboard = PortfolioDashboard(portfolio, manager)

            # Récupération des prix (depuis l’analyse ou en direct)
            prices = data.get("analysis", {}).get("market_prices")
            if not prices:
                try:
                    tickers = list(portfolio.positions.keys())
                    prices = manager.get_market_prices(tickers)
                except Exception:
                    prices = {}

            return dashboard.render_positions_table(prices)
        except Exception as e:
            logger.exception("Erreur update_positions")
            return dbc.Alert(f"Erreur technique : {e}", color="danger")

    # =========================================================
    # 🔹 6. Simulation Monte Carlo (déclenchée par bouton)
    # =========================================================
    @app.callback(
        Output("monte-carlo-container", "children"),
        Input("run-monte-carlo-btn", "n_clicks"),
        State("portfolio-data-store", "data"),
        prevent_initial_call=True,
    )
    def run_monte_carlo_callback(n_clicks, data):
        if not n_clicks:
            raise PreventUpdate

        if not data or "portfolio_name" not in data:
            return dbc.Alert("Sélectionnez un portefeuille avant de lancer la simulation.", color="warning")

        try:
            p_name = data["portfolio_name"]
            portfolio = manager.load_portfolio(p_name)
            if not portfolio:
                return dbc.Alert("Impossible de charger le portefeuille.", color="danger")

            sim_result = manager.run_monte_carlo_simulation(n_simulations=500, timeframe=252)
            dashboard = PortfolioDashboard(portfolio, manager)
            return dashboard.render_monte_carlo_results(sim_result)
        except Exception as e:
            logger.exception("Erreur Monte Carlo")
            return dbc.Alert(f"Simulation échouée : {e}", color="danger")

    # =========================================================
    # 🔹 7. Commutation des onglets (4 vues)
    # =========================================================
    @app.callback(
        Output("strat-view", "style"),
        Output("tech-view", "style"),
        Output("alloc-view", "style"),
        Output("opt-view", "style"),
        Input("card-tabs", "active_tab"),
    )
    def switch_tabs(active_tab):
        hidden = {"display": "none"}
        visible = {"display": "block"}

        mapping = {
            "tab-strat": (visible, hidden, hidden, hidden),
            "tab-tech": (hidden, visible, hidden, hidden),
            "tab-alloc": (hidden, hidden, visible, hidden),
            "tab-opt": (hidden, hidden, hidden, visible),
        }
        return mapping.get(active_tab, (visible, hidden, hidden, hidden))

    # =========================================================
    # 🔹 8. Graphique d’analyse technique
    # =========================================================
    @app.callback(
        Output("technical-analysis-graph", "figure"),
        Input("portfolio-data-store", "data"),
    )
    def update_technical_graph(data):
        if not data or "portfolio_name" not in data:
            return go.Figure()

        try:
            p_name = data["portfolio_name"]
            portfolio = manager.load_portfolio(p_name)
            if not portfolio:
                return go.Figure()

            dashboard = PortfolioDashboard(portfolio, manager)
            fig = dashboard.create_visual_report()
            return fig if fig is not None else go.Figure()
        except Exception as e:
            logger.exception("Erreur update_technical_graph")
            return go.Figure()

    # =========================================================
    # 🔹 9. Graphique d’allocation (camembert)
    # =========================================================
    @app.callback(
        Output("allocation-graph", "figure"),
        Input("portfolio-data-store", "data"),
    )
    def update_allocation_graph(data):
        if not data or "portfolio_name" not in data:
            return go.Figure()

        try:
            p_name = data["portfolio_name"]
            portfolio = manager.load_portfolio(p_name)
            if not portfolio:
                return go.Figure()

            prices = data.get("analysis", {}).get("market_prices")
            alloc = portfolio.get_allocation(prices)
            if not alloc:
                return go.Figure()

            labels = list(alloc.keys())
            values = [float(v) for v in alloc.values()]
            fig = go.Figure(
                data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent")]
            )
            fig.update_layout(title="Allocation du Portefeuille", template="plotly_white", height=420)
            return fig
        except Exception as e:
            logger.exception("Erreur update_allocation_graph")
            return go.Figure()

    # =========================================================
    # 🔹 10. Métriques de risque (affichage unique)
    # =========================================================
    @app.callback(
        Output("risk-metrics-div", "children"),
        Input("portfolio-data-store", "data"),
    )
    def update_risk(data):
        """Affiche les indicateurs de risque (Sharpe, VaR, volatilité)."""
        if not data:
            return dbc.Alert("Aucune donnée risque disponible.", color="secondary")

        try:
            risk = data.get("analysis", {}).get("risk_metrics", {})
            # Utilisation de composants Bootstrap pour un affichage propre
            return dbc.ListGroup(
                [
                    dbc.ListGroupItem(f"📈 Sharpe : {risk.get('sharpe_ratio', 'N/A')}"),
                    dbc.ListGroupItem(f"⚠️ VaR (95%) : {risk.get('value_at_risk', 'N/A')}"),
                    dbc.ListGroupItem(f"📉 Volatilité : {risk.get('volatility', 'N/A')}"),
                ]
            )
        except Exception as e:
            logger.error(f"Erreur update_risk : {e}")
            return dbc.Alert("Erreur de calcul des risques.", color="danger")