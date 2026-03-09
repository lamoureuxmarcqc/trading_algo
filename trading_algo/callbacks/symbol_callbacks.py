# trading_algo/web_dashboard/callbacks/symbol_callbacks.py
from dash import Input, Output, State, callback, html, dcc
import plotly.graph_objects as go
from trading_algo.models.stockpredictor import StockPredictor
from trading_algo.models.stockmodeltrain import StockModelTrain
from dash import html, dcc


def register_symbol_callbacks(app):
    
    @app.callback(
        Output('symbol-analysis-results', 'children'),
        Input('analyze-btn', 'n_clicks'),
        State('symbol-input', 'value'),
        State('period-dropdown', 'value')
    )
    def analyze_symbol(n_clicks, symbol, period):
        if not symbol:
            return html.Div("Veuillez entrer un symbole.")
        
        try:
            # Utiliser StockPredictor si disponible, sinon StockModelTrain
            try:
                predictor = StockPredictor(symbol, period)
                results = predictor.analyze_stock_advanced()
            except:
                predictor = StockModelTrain(symbol, period)
                results = predictor.analyze_model_stock()
            
            if 'error' in results:
                return html.Div(f"Erreur: {results['error']}")
            
            # Construire l'affichage
            return html.Div([
                html.H3(f"Résultats pour {symbol}"),
                html.P(f"Prix actuel: ${results['current_price']:.2f}"),
                html.P(f"Score: {results['trading_score']}/10"),
                html.P(f"Recommandation: {results['recommendation']}"),
                
                html.H4("Prédictions"),
                html.Ul([
                    html.Li(f"{k}: ${v:.2f}") for k, v in results.get('predictions', {}).items() if v
                ]),
                
                html.H4("Métriques de risque"),
                html.Pre(str(results.get('risk_metrics', {}))),
                
                dcc.Graph(
                    figure=go.Figure()
                    # Ici vous pouvez intégrer un graphique des prix historiques + prédictions
                )
            ])
        except Exception as e:
            return html.Div(f"Exception: {str(e)}")