from dash import Input, Output, State, callback_context, dash_table, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from trading_algo.portfolio.portfoliomanager import PortfolioManager
from trading_algo.visualization.portfoliodashboard import PortfolioDashboard
from trading_algo.data.data_extraction import StockDataExtractor


def register_portfolio_callbacks(app):

    manager = PortfolioManager(StockDataExtractor)

    # =========================================================
    # 🔹 LOAD PORTFOLIO LIST
    # =========================================================
    @app.callback(
        Output("portfolio-selector", "options"),
        Input("portfolio-refresh-timer", "n_intervals")
    )
    def refresh_portfolio_list(_):
        portfolios = manager.list_portfolios()
        return [{"label": p, "value": p} for p in portfolios]

    # =========================================================
    # 🔹 LOAD PORTFOLIO DATA
    # =========================================================
    @app.callback(
        Output("portfolio-data-store", "data"),
        Input("load-portfolio-btn", "n_clicks"),
        State("portfolio-selector", "value"),
        prevent_initial_call=True
    )
    def load_portfolio(n_clicks, portfolio_name):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        if trigger != "load-portfolio-btn" or not n_clicks:
            raise PreventUpdate

        if not portfolio_name:
            return {
                "portfolio_name": None,
                "analysis": {},
                "status": "missing_selection",
            }

        portfolio = manager.load_portfolio(portfolio_name)
        if not portfolio:
            return {
                "portfolio_name": portfolio_name,
                "analysis": {},
                "status": "load_error",
            }

        # Keep the web tab responsive: fetch only the lightweight snapshot here.
        # Expensive long-horizon risk calculations are deferred.
        analysis = manager.analyze_portfolio(include_risk=False)

        return {
            "portfolio_name": portfolio_name,
            "analysis": analysis,
            "status": "loaded",
        }

    # =========================================================
    # 🔹 KPI UPDATE
    # =========================================================
    @app.callback(
        Output("kpi-total-value", "children"),
        Output("kpi-return", "children"),
        Output("kpi-volatility", "children"),
        Output("kpi-sharpe", "children"),
        Input("portfolio-data-store", "data")
    )
    def update_kpis(data):

        if not data or "analysis" not in data:
            return "---", "---", "---", "---"

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
            f"{sharpe:.2f}" if sharpe else "N/A"
        )

    # =========================================================
    # 🔹 MAIN GRAPH (STRATEGIC) - SÉCURISÉ
    # =========================================================
    @app.callback(
        Output("portfolio-main-dashboard", "figure"),
        Input("portfolio-data-store", "data")
    )
    def update_main_chart(data):
        # 1. Vérification stricte
        if not data or "portfolio_name" not in data:
            return go.Figure()  # Retourne une figure vide plutôt que {}

        # 2. Re-charger l'objet localement pour éviter les conflits entre utilisateurs
        p_name = data["portfolio_name"]
        portfolio = manager.load_portfolio(p_name)

        if not portfolio:
            return go.Figure()

        dashboard = PortfolioDashboard(portfolio, manager)

        # Données mock (à remplacer par tes vrais flux plus tard)
        macro = {}
        indices = pd.DataFrame()
        fundamentals = {}

        return dashboard.create_strategic_report(macro, indices, fundamentals)

    # =========================================================
    # 🔹 POSITIONS TABLE - SÉCURISÉ (UNIQUE CALLBACK)
    # =========================================================
    @app.callback(
        Output("positions-table", "children"),
        Input("portfolio-data-store", "data")
    )
    def update_positions(data):
        """
        Single callback responsible for rendering the positions table.
        Consolidated logic from previous duplicate callbacks.
        """
        if not data or "portfolio_name" not in data:
            return dbc.Alert("Sélectionnez un portefeuille pour voir les positions.", color="info")

        p_name = data["portfolio_name"]
        portfolio = manager.load_portfolio(p_name)

        if not portfolio:
            return dbc.Alert("Erreur lors du chargement du portefeuille.", color="danger")

        prices = data.get("analysis", {}).get("market_prices", {})
        df = portfolio.get_summary(prices)

        if df is None or df.empty:
            return dbc.Alert("Aucune position ouverte.", color="warning")

        return dbc.Table.from_dataframe(
            df,
            striped=True,
            bordered=True,
            hover=True,
            size="sm",
            responsive=True
        )

    # =========================================================
    # 🔥 🔥 🔥 CRITICAL FIX — TAB SWITCHING
    # =========================================================
    @app.callback(
        Output("strat-view", "style"),
        Output("tech-view", "style"),
        Input("card-tabs", "active_tab")
    )
    def switch_tabs(active_tab):

        if active_tab == "tab-tech":
            return {"display": "none"}, {"display": "block"}

        return {"display": "block"}, {"display": "none"}

    # =========================================================
    # 🔹 RISK METRICS PANEL
    # =========================================================
    @app.callback(
        Output("risk-metrics-div", "children"),
        Input("portfolio-data-store", "data")
    )
    def update_risk(data):

        if not data:
            return "N/A"

        risk = data.get("analysis", {}).get("risk_metrics", {})

        return dbc.ListGroup([
            dbc.ListGroupItem(f"Sharpe: {risk.get('sharpe_ratio', 'N/A')}"),
            dbc.ListGroupItem(f"VaR: {risk.get('value_at_risk', 'N/A')}"),
            dbc.ListGroupItem(f"Vol: {risk.get('volatility', 'N/A')}"),
        ])


# =========================================================
# HELPERS
# =========================================================

def _create_data_table(df):

    if df is None or df.empty:
        return html.Div("Aucune position")

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"},
        style_header={"fontWeight": "bold"},
    )


def _render_risk_cards(risk):

    if not risk:
        return html.Div("No risk data")

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Sharpe"),
            html.H4(f"{risk.get('sharpe_ratio', 0):.2f}")
        ]))),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Volatility"),
            html.H4(f"{risk.get('volatility', 0)*100:.2f}%")
        ]))),
    ])


def _empty_dashboard(error=None):

    fig = go.Figure()

    msg = error if error else "Aucun portefeuille chargé"

    return (
        fig,
        fig,
        msg,
        "---",
        "---",
        "---",
        html.Div(),
        html.Div(),
        {}
    )
