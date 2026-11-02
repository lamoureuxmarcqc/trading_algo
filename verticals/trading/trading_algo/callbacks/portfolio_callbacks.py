# trading_algo/callbacks/portfolio_callbacks.py
import logging
from typing import Optional, Dict, Any

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, dash_table, html, dcc
from dash.exceptions import PreventUpdate

from trading_algo.data.data_extraction import StockDataExtractor
from trading_algo.portfolio.portfoliomanager import PortfolioManager
from trading_algo.visualization.portfoliodashboard import PortfolioDashboard
from trading_algo.portfolio.portfolio import Position
import json

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
            analysis = manager.analyze_portfolio(include_risk=True)
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
            risk = data.get("analysis", {}).get("risk_metrics", {}) or {}

            # Sharpe
            sharpe = risk.get("sharpe_ratio")
            sharpe_display = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "N/A"

            # VaR 95%
            var95 = risk.get("var_95")
            if isinstance(var95, (int, float)):
                # var is typically negative (loss). Display as percentage with sign.
                var_display = f"{var95:.2%}"
            else:
                var_display = "N/A"

            # Volatility (annualized)
            vol = risk.get("volatility")
            vol_display = f"{vol:.2%}" if isinstance(vol, (int, float)) else "N/A"

            return dbc.ListGroup(
                [
                    dbc.ListGroupItem(f"📈 Sharpe : {sharpe_display}"),
                    dbc.ListGroupItem(f"⚠️ VaR (95%) : {var_display}"),
                    dbc.ListGroupItem(f"📉 Volatilité : {vol_display}"),
                ]
            )
        except Exception as e:
            logger.error(f"Erreur update_risk : {e}")
            return dbc.Alert("Erreur de calcul des risques.", color="danger")

    # =========================================================
    # 🔹 11. OPTIMISATION (déclenchée par bouton) -> affiche courbe interactive + ordres
    # =========================================================
    @app.callback(
        Output("optimization-results-container", "children"),
        Input("run-optimization-btn", "n_clicks"),
        State("portfolio-data-store", "data"),
        State("opt-tickers", "value"),
        State("opt-bond-ticker", "value"),
        State("opt-horizon", "value"),
        State("opt-rebalance-freq", "value"),
        State("opt-transaction-cost", "value"),
        State("opt-capital", "value"),
        prevent_initial_call=True,
    )
    def run_optimization_callback(n_clicks, data, tickers_input, bond_ticker, horizon, rebalance_freq, tx_cost, capital):
        if not n_clicks:
            raise PreventUpdate

        try:
            # Determine portfolio context
            portfolio_loaded = False
            portfolio = None
            if data and data.get("portfolio_name"):
                p_name = data["portfolio_name"]
                portfolio = manager.load_portfolio(p_name)
                portfolio_loaded = bool(portfolio)

            # If tickers provided manually, create a temporary portfolio with zero positions
            tickers_list = []
            if tickers_input and isinstance(tickers_input, str) and tickers_input.strip():
                tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

            if not portfolio and tickers_list:
                tmp_name = f"tmp_opt_{int(pd.Timestamp.now().timestamp())}"
                portfolio = manager.create_portfolio(tmp_name, float(capital or 100000))
                for t in tickers_list:
                    portfolio.positions[t] = Position(t, 0.0, 0.0)

            if not portfolio:
                return dbc.Alert("Aucun portefeuille chargé et aucun ticker fourni.", color="warning")

            # Normalize inputs
            try:
                hist_period = f"{int(horizon)}y"
            except Exception:
                hist_period = "5y"
            tx_cost = float(tx_cost) if tx_cost is not None else 0.001

            # Run optimization
            res = manager.optimize_current_portfolio(
                include_bonds=bool(bond_ticker),
                bond_ticker=bond_ticker if bond_ticker else None,
                history_period=hist_period,
                rebalance_freq=rebalance_freq or "ME",
                transaction_cost=tx_cost
            )

            if not res or res.get("error"):
                msg = res.get("error") if isinstance(res, dict) else "Erreur optimisation"
                return dbc.Alert(f"Optimisation échouée: {msg}", color="danger")

            # Build interactive backtest figure (line + range slider + range selector)
            back = res.get("backtest", {}) or {}
            series = back.get("series")
            fig = go.Figure()
            if series is not None:
                try:
                    # If series is a pandas Series, use index; if not, try to coerce
                    if isinstance(series, pd.Series):
                        x = series.index
                        y = series.values
                    elif isinstance(series, (list, pd.Series)):
                        # if list of values without dates, create numeric index
                        y = list(series)
                        x = list(range(len(y)))
                    else:
                        # try pandas
                        try:
                            s = pd.Series(series)
                            x = s.index
                            y = s.values
                        except Exception:
                            x = list(range(len(series))) if hasattr(series, "__len__") else [0]
                            y = series if hasattr(series, "__len__") else [series]

                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Backtest Value", line=dict(color="royalblue")))
                    fig.update_layout(
                        title="Backtest (Portefeuille optimisé)",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        template="plotly_white",
                        height=420,
                        hovermode="x unified",
                        xaxis=dict(rangeselector=dict(buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(step="all")
                        ])),
                            rangeslider=dict(visible=True),
                            type="date" if hasattr(x[0], "year") else "linear"
                        )
                    )
                except Exception as e:
                    logger.debug("Erreur génération figure backtest: %s", e)
                    fig = go.Figure()

            # Orders: propose orders based on last optimization
            orders_res = manager.apply_optimized_allocation(threshold=0.005)
            orders = orders_res.get("orders", []) if isinstance(orders_res, dict) else []
            orders_df = pd.DataFrame(orders) if orders else pd.DataFrame(columns=["ticker", "action", "target_weight", "current_weight", "delta_weight", "notional"])

            # Build interactive orders bar chart (notional)
            orders_fig = go.Figure()
            if not orders_df.empty:
                try:
                    orders_sorted = orders_df.sort_values("notional", ascending=False)
                    orders_fig.add_trace(go.Bar(
                        x=orders_sorted["ticker"],
                        y=orders_sorted["notional"],
                        marker_color=[ 'green' if a == "BUY" else 'red' for a in orders_sorted["action"] ],
                        text=[f"{n:,.0f}$" for n in orders_sorted["notional"]],
                        hovertemplate="%{x}<br>Notional: %{text}<extra></extra>"
                    ))
                    orders_fig.update_layout(title="Proposed Orders (notional)", template="plotly_white", height=300, yaxis_title="Notional ($)")
                except Exception as e:
                    logger.debug("Erreur génération orders_fig: %s", e)
                    orders_fig = go.Figure()

            # DataTable for order details
            table = dash_table.DataTable(
                data=orders_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in orders_df.columns],
                page_size=8,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "6px"},
                style_header={"fontWeight": "bold"},
            )

            # Metrics list
            metrics = back.get("metrics", {}) or {}
            metrics_list = [html.Li(f"{k}: {v}") for k, v in metrics.items()]

            # Compose final layout: left = interactive backtest, right = orders + table + apply button
            return html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig, config={"responsive": True}), md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Pondérations optimales"),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li(f"{t}: {w:.2%}") for t, w in sorted(res.get("optimal_weights", {}).items(), key=lambda x: x[1], reverse=True)
                                ], style={"maxHeight": "220px", "overflow": "auto"})
                            ])
                        ], className="mb-3"),
                        dbc.Card([
                            dbc.CardHeader("Ordres proposés"),
                            dbc.CardBody([
                                table,
                                dbc.Button("Appliquer l'allocation (simulé)", id="apply-opt-allocation-btn", color="warning", className="mt-2 w-100"),
                                html.Div(id="apply-opt-result", className="mt-2")
                            ])
                        ], className="mb-3"),
                        dbc.Card([
                            dbc.CardHeader("Backtest metrics"),
                            dbc.CardBody(html.Ul(metrics_list))
                        ], className="mb-0"),
                        html.Hr(),
                        dcc.Graph(figure=orders_fig)
                    ], md=4)
                ])
            ], className="mt-2")

        except Exception as e:
            logger.exception("Erreur run_optimization_callback: %s", e)
            return dbc.Alert(f"Erreur interne lors de l'optimisation: {e}", color="danger")

    # Optional: callback to handle apply allocation button (simulation only)
    @app.callback(
        Output("apply-opt-result", "children"),
        Input("apply-opt-allocation-btn", "n_clicks"),
        State("portfolio-data-store", "data"),
        prevent_initial_call=True
    )
    def apply_optimization_simulation(n_clicks, data):
        if not n_clicks:
            raise PreventUpdate
        try:
            # Use last optimization to show simulation of applying orders (no execution)
            opt = getattr(manager, "_last_optimization", None)
            if not opt:
                return dbc.Alert("Aucune optimisation disponible. Lancez d'abord l'optimisation.", color="warning")
            applied = manager.apply_optimized_allocation(threshold=0.005)
            orders = applied.get("orders", [])
            if not orders:
                return dbc.Alert("Aucune action requise (portefeuille déjà aligné)", color="info")
            # Return simple summary
            total_notional = sum(abs(o.get("notional", 0.0)) for o in orders)
            return html.Div([
                html.P(f"{len(orders)} ordres simulés. Notional total: {total_notional:,.0f} $"),
                dbc.ListGroup([dbc.ListGroupItem(f"{o['action']} {o['ticker']}: {o['notional']:,.0f} $") for o in orders])
            ])
        except Exception as e:
            logger.exception("Erreur apply_optimization_simulation: %s", e)
            return dbc.Alert(f"Erreur lors de l'application simulée: {e}", color="danger")