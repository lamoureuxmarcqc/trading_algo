import logging
from dash import Input, Output, State, html, dcc
import dash_bootstrap_components as dbc

# Imports internes
from trading_algo import settings
from trading_algo.visualization.symbol_dashboard import AdvancedTradingDashboard

logger = logging.getLogger(__name__)

def register_symbol_callbacks(app):
    """
    Callback léger : délègue la logique au SymbolManager et
    utilise uniquement la couche de visualisation pour construire la figure.
    """
    @app.callback(
        Output('symbol-analysis-results', 'children'),
        Input('analyze-btn', 'n_clicks'),
        State('symbol-input', 'value'),
        State('period-dropdown', 'value'),
        prevent_initial_call=True
    )
    def analyze_symbol(n_clicks, symbol, period):
        if not symbol or n_clicks == 0:
            return dbc.Alert("Veuillez entrer un symbole boursier valide (ex: AAPL, BTC-USD).", color="warning", className="mt-3")

        symbol = symbol.strip().upper()

        # Appel au manager (logique métier)
        try:
            from trading_algo.stock.stock_manager import symbol_manager
        except Exception as e:
            logger.exception("Cannot import SymbolManager: %s", e)
            return dbc.Alert("Erreur interne: manager indisponible.", color="danger", className="mt-3")

        resp = symbol_manager.analyze_symbol(symbol, period)
        if not resp or 'error' in resp:
            msg = resp.get('error') if isinstance(resp, dict) else "Analyse introuvable."
            return dbc.Alert(f"❌ Erreur : {msg}", color="danger", className="mt-3")

        results = resp.get('results', {}) or {}
        tech_df = resp.get('technical')
        pred_df = resp.get('predictions')

        if tech_df is None or tech_df.empty:
            return dbc.Alert(f"⚠️ Les indicateurs techniques pour {symbol} n'ont pas pu être générés.", color="warning")

        # Build figure via visualization layer
        try:
            td = AdvancedTradingDashboard(symbol)
            td.load_data(
                technical_data=tech_df,
                predictions_df=pred_df,
                risk_metrics=results.get('risk_metrics', {})
            )
            td.score = results.get('trading_score', 5.0)
            td.recommendation = results.get('recommendation', "NEUTRE")

            fig = td.create_full_dashboard()
            if fig is None:
                return dbc.Alert("Erreur lors de la génération du tableau de bord visuel.", color="danger", className="mt-3")

            return html.Div([
                _build_summary_cards(results),
                dbc.Card([
                    dbc.CardHeader(html.H5(f"Analyse Stratégique Complète - {symbol}", className="mb-0")),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=fig,
                            config={'displayModeBar': True, 'scrollZoom': True},
                            style={"height": "1200px"}
                        )
                    ])
                ], className="shadow-sm border-0 mb-5")
            ], className="mt-4 animated fadeIn")

        except Exception as e:
            logger.exception("Visualization error for %s: %s", symbol, e)
            return dbc.Alert(f"💥 Une erreur interne est survenue lors de la visualisation : {str(e)}", color="danger", className="mt-3")


def _build_summary_cards(results):
    score = results.get('trading_score', 0)
    rec = results.get('recommendation', 'N/A')

    color_map = {
        "ACHAT FORT": "success", "ACHAT": "success",
        "VENTE": "danger", "VENTE FORTE": "danger",
        "NEUTRE": "warning"
    }
    badge_color = color_map.get(rec, "secondary")

    return dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Trading Score / 10", className="text-muted small"),
                    html.H3(f"{score:.1f}", className=f"text-{badge_color} mb-0"),
                    dbc.Progress(value=score * 10, color=badge_color, className="mt-2", style={"height": "4px"})
                ])
            ], className="shadow-sm border-0"), width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Recommandation", className="text-muted small"),
                    dbc.Badge(rec, color=badge_color, className="fs-5 w-100 p-2")
                ])
            ], className="shadow-sm border-0"), width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Prix Actuel", className="text-muted small"),
                    html.H3(f"{results.get('current_price', 0):,.2f} $", className="mb-0")
                ])
            ], className="shadow-sm border-0"), width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Confiance IA", className="text-muted small"),
                    html.H3(f"{results.get('model_confidence', 0)*100:.1f}%", className="mb-0 text-info")
                ])
            ], className="shadow-sm border-0"), width=3
        ),
    ], className="g-3 mb-4")

