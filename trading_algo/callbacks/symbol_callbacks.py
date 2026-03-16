# trading_algo/web_dashboard/callbacks/symbol_callbacks.py
from dash import Input, Output, State, html, dcc
import plotly.graph_objects as go
from trading_algo.models.stockpredictor import StockPredictor
from trading_algo.models.stockmodeltrain import StockModelTrain
from trading_algo.visualization.dashboard import TradingDashboard
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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
            # Prefer StockPredictor if available, otherwise fallback to StockModelTrain
            predictor = None
            results = {}
            try:
                predictor = StockPredictor(symbol, period)
                # If predictor supports advanced analysis use it
                if hasattr(predictor, "analyze_stock_advanced"):
                    results = predictor.analyze_stock_advanced()
                else:
                    results = predictor.analyze()
            except Exception:
                predictor = StockModelTrain(symbol, period)
                # Ensure data fetched before analysis
                if not predictor.fetch_data():
                    return html.Div(f"Impossible de récupérer les données pour {symbol}.")
                results = predictor.analyze_model_stock() if hasattr(predictor, "analyze_model_stock") else {}

            if not results or 'error' in results:
                # Fallback textual display
                err = results.get('error', 'Analyse indisponible')
                return html.Div(f"Erreur: {err}")

            # Prepare technical data
            technical = None
            if 'technical' in results and isinstance(results['technical'], pd.DataFrame):
                technical = results['technical']
            elif hasattr(predictor, "features"):
                technical = getattr(predictor, "features", None)
            else:
                technical = pd.DataFrame()

            # Prepare predictions_df from results['predictions'] if possible
            predictions_df = pd.DataFrame()
            preds = results.get('predictions') or {}
            if preds:
                rows = []
                now = datetime.now()
                for key, val in preds.items():
                    # key like '1d','5d' etc.
                    try:
                        if isinstance(key, str) and key.endswith('d'):
                            days = int(key[:-1])
                        else:
                            days = int(key)
                    except Exception:
                        # skip malformed keys
                        continue
                    date = (now + timedelta(days=days)).date()
                    rows.append({'date': pd.to_datetime(date), 'Predicted_Close': float(val) if val is not None else None, 'horizon': key})
                if rows:
                    predictions_df = pd.DataFrame(rows).set_index('date')

            # Gather risk metrics, score and recommendation from results
            risk_metrics = results.get('risk_metrics') or getattr(predictor, 'risk_metrics', {}) or {}
            score = results.get('trading_score') or getattr(predictor, 'trading_score', 5.0)
            recommendation = results.get('recommendation') or getattr(predictor, 'recommendation', "NEUTRE")

            # Training metrics (if predictor trained a model recently)
            training_metrics = None
            # Try several possible attributes where training info might be stored
            training_metrics = getattr(predictor, 'training_metrics', None) or getattr(predictor, 'analysis_results', {}).get('training_metrics') if hasattr(predictor, 'analysis_results') else None
            # Also try 'model_metrics' or 'validation_metrics'
            if not training_metrics:
                training_metrics = getattr(predictor, 'model_metrics', None) or getattr(predictor, 'validation_metrics', None)

            # Prediction examples: try to extract from results or build from recent history
            prediction_examples = results.get('prediction_examples') or []
            if not prediction_examples and hasattr(predictor, 'data') and getattr(predictor, 'data') is not None:
                # Build a small list using predicted points vs actual recent close if available
                try:
                    recent_close = float(predictor.data['Close'].iloc[-1])
                    for horizon, pred in preds.items():
                        prediction_examples.append({'horizon': horizon, 'predicted': float(pred) if pred is not None else None, 'actual': recent_close})
                except Exception:
                    pass

            # Build dashboard
            dd = TradingDashboard(symbol.upper(), results.get('current_price', 0.0))
            dd.load_data(
                overview=results.get('overview', {}),
                technical_data=technical if technical is not None else pd.DataFrame(),
                predictions_df=predictions_df,
                macro_data=results.get('macro', None) or getattr(predictor, 'macro_data', None),
                market_sentiment=results.get('market_sentiment', None) or getattr(predictor, 'market_sentiment', None),
                score=score,
                recommendation=recommendation,
                risk_metrics=risk_metrics,
                training_metrics=training_metrics,
                prediction_examples=prediction_examples
            )

            # Create figure (and save html/png) and embed in Dash via dcc.Graph
            fig = dd.create_main_dashboard(save_path="dashboards")
            if fig is None:
                # fallback textual summary
                return html.Div([
                    html.H3(f"Résultats pour {symbol}"),
                    html.P(f"Prix actuel: ${results.get('current_price', 0.0):.2f}"),
                    html.Pre(str(results))
                ])

            # Return graph and a compact summary beneath
            quick = dd.create_quick_overview()
            quick_table = dcc.Graph(figure=go.Figure(data=[go.Table(
                header=dict(values=list(quick.columns)),
                cells=dict(values=[quick[col] for col in quick.columns])
            )])) if quick is not None else html.Div()

            return html.Div([
                dcc.Graph(figure=fig),
                html.Hr(),
                html.Div([
                    html.Div(quick_table)
                ])
            ])

        except Exception as e:
            logger.exception(f"Exception during symbol analysis: {e}")
            return html.Div(f"Exception: {str(e)}")