"""
Module: trading_algo.visualization.dashboard
Trading dashboard and helpers (complete, self-contained TradingDashboard with helpers).
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import settings
from plotly.subplots import make_subplots

# Constants
SMA_PERIODS = [20, 50, 200]
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ATR_HIGH_THRESHOLD = 3.0
ATR_MODERATE_THRESHOLD = 1.5


class TradingDashboard:
    """Main class to build trading dashboards for a single symbol."""

    def __init__(self, symbol: str, current_price: float = 0.0):
        self.symbol = symbol
        self.current_price = current_price
        self.overview: Dict[str, Any] = {}
        self.technical_data: pd.DataFrame = pd.DataFrame()
        self.predictions_df: pd.DataFrame = pd.DataFrame()
        self.macro_data: Dict[str, Any] = {}
        self.market_sentiment: Dict[str, Any] = {}
        self.risk_metrics: Dict[str, Any] = {}
        self.score: float = 5.0
        self.recommendation: str = "NEUTRE"
        self.training_metrics: Optional[Dict[str, Any]] = None
        self.prediction_examples: Optional[List[Dict[str, Any]]] = None
        self.logger = logging.getLogger(__name__)

    def load_data(self,
                  overview: Dict[str, Any],
                  technical_data: pd.DataFrame,
                  predictions_df: pd.DataFrame,
                  macro_data: Optional[Dict[str, Any]] = None,
                  market_sentiment: Optional[Dict[str, Any]] = None,
                  score: Optional[float] = None,
                  recommendation: Optional[str] = None,
                  risk_metrics: Optional[Dict[str, Any]] = None,
                  training_metrics: Optional[Dict[str, Any]] = None,
                  prediction_examples: Optional[List[Dict[str, Any]]] = None):
        """Load all inputs for the dashboard."""
        self.overview = overview or {}
        self.technical_data = technical_data.copy() if technical_data is not None else pd.DataFrame()
        self.predictions_df = predictions_df.copy() if predictions_df is not None else pd.DataFrame()
        self.macro_data = macro_data or {}
        self.market_sentiment = market_sentiment or {}
        self.risk_metrics = risk_metrics or {}
        self.training_metrics = training_metrics
        self.prediction_examples = prediction_examples or []

        if score is not None:
            self.score = float(score)
        if recommendation is not None:
            self.recommendation = recommendation
        else:
            self._calculate_score_and_recommendation()

        self.logger.info(f"Données chargées pour {self.symbol}")

    def _calculate_score_and_recommendation(self):
        """Heuristic score based on technicals, risk ratios and training metrics."""
        try:
            tech_score = 5.0
            df = self.technical_data
            if df is not None and not df.empty:
                last = df.iloc[-1]
                if 'RSI' in last and not pd.isna(last['RSI']):
                    rsi = last['RSI']
                    if rsi < RSI_OVERSOLD:
                        tech_score += 2
                    elif rsi > RSI_OVERBOUGHT:
                        tech_score -= 2
                if 'MACD' in last and 'MACD_Signal' in last and not pd.isna(last['MACD']) and not pd.isna(last['MACD_Signal']):
                    if last['MACD'] > last['MACD_Signal']:
                        tech_score += 1
                if 'SMA_50' in last and 'SMA_200' in last and not pd.isna(last['SMA_50']) and not pd.isna(last['SMA_200']):
                    if last['SMA_50'] > last['SMA_200']:
                        tech_score += 1

            if self.risk_metrics:
                sr = self.risk_metrics.get('sharpe_ratio')
                if sr is not None:
                    tech_score += max(min(sr, 2), -2)
                beta = self.risk_metrics.get('beta')
                if beta is not None:
                    if beta > 1.5:
                        tech_score -= 1
                    elif beta < 0.5:
                        tech_score += 1

            if self.training_metrics and isinstance(self.training_metrics, dict):
                val = self.training_metrics.get('validation', {})
                t1 = val.get('Target_Close_1d') if isinstance(val, dict) else None
                if t1 and isinstance(t1, dict) and 'mae' in t1:
                    mae = t1['mae']
                    tech_score += max(-2, min(2, (50 - mae) / 50))

            self.score = min(10, max(1, tech_score))
            if self.score >= 7.5:
                self.recommendation = "FORT ACHAT 🟢"
            elif self.score >= 6.0:
                self.recommendation = "ACHAT 🟢"
            elif self.score >= 5.0:
                self.recommendation = "MAINTENIR 🟡"
            elif self.score >= 4.0:
                self.recommendation = "VENTE LÉGÈRE 🟠"
            else:
                self.recommendation = "VENTE 🔴"
        except Exception as e:
            self.logger.error(f"Erreur calcul score: {e}")
            self.score = 5.0
            self.recommendation = "NEUTRE ⚪"

    # -------------------------
    # Chart helper methods
    # -------------------------
    def _add_main_price_chart(self, fig, row: int, col: int):
        df = self.technical_data
        if df is None or df.empty or 'Close' not in df.columns:
            self.logger.warning("Pas de données de prix pour le graphique principal")
            return

        fig.add_trace(
            go.Scattergl(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Prix de clôture',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{x|%d %b %Y}<br>Prix: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )

        if all(c in df.columns for c in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(go.Scattergl(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash')), row=row, col=col)
            fig.add_trace(go.Scattergl(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='rgba(0,255,0,0.3)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(0,255,0,0.1)'), row=row, col=col)

        last_price = df['Close'].iloc[-1]
        fig.add_hline(y=last_price, line_dash="dot", line_color="blue", opacity=0.5,
                      annotation_text=f"Actuel: ${last_price:.2f}", annotation_position="bottom right", row=row, col=col)

    def _add_trading_score_gauge(self, fig, row: int, col: int):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.score,
                title={'text': "Score Trading", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 4], 'color': "red"},
                        {'range': [4, 6], 'color': "yellow"},
                        {'range': [6, 8], 'color': "lightgreen"},
                        {'range': [8, 10], 'color': "green"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': self.score}
                }
            ),
            row=row, col=col
        )

    def _add_market_sentiment_indicator(self, fig, row: int, col: int):
        sentiment_score = self.market_sentiment.get('overall_sentiment', {}).get('score', 50) if self.market_sentiment else 50
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=sentiment_score,
                title={'text': "Sentiment Marché", 'font': {'size': 14}},
                gauge={'shape': "bullet", 'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 40], 'color': "red"}, {'range': [40, 60], 'color': "yellow"}, {'range': [60, 100], 'color': "green"}], 'threshold': {'line': {'color': "black", 'width': 2}, 'value': sentiment_score}}
            ),
            row=row, col=col
        )

    def _add_rsi_chart(self, fig, row: int, col: int):
        if 'RSI' not in self.technical_data.columns:
            self.logger.warning("Colonne RSI manquante")
            return
        df = self.technical_data
        x = df.index
        fig.add_trace(go.Scattergl(x=x, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple', width=2),
                                   hovertemplate='%{x|%d %b %Y}<br>RSI: %{y:.1f}<extra></extra>'), row=row, col=col)
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[RSI_OVERBOUGHT, RSI_OVERBOUGHT], mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[RSI_OVERSOLD, RSI_OVERSOLD], mode='lines', line=dict(color='green', dash='dash'), showlegend=False), row=row, col=col)

    def _add_macd_chart(self, fig, row: int, col: int):
        df = self.technical_data
        if not all(c in df.columns for c in ['MACD', 'MACD_Signal']):
            self.logger.warning("Colonnes MACD manquantes")
            return
        fig.add_trace(go.Scattergl(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue', width=2)), row=row, col=col)
        fig.add_trace(go.Scattergl(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red', width=1.5)), row=row, col=col)
        if 'MACD_Histogram' in df.columns:
            colors = ['green' if v >= 0 else 'red' for v in df['MACD_Histogram']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], marker_color=colors, opacity=0.6, name='MACD Hist'), row=row, col=col)

    def _add_volume_chart(self, fig, row: int, col: int):
        if 'Volume' not in self.technical_data.columns:
            self.logger.warning("Colonne Volume manquante")
            return
        df = self.technical_data
        colors = ['green' if i == 0 or df['Close'].iloc[i] >= df['Close'].iloc[i-1] else 'red' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.7, name='Volume'), row=row, col=col)

    def _add_trend_chart(self, fig, row: int, col: int):
        df = self.technical_data
        if df is None or df.empty or 'Close' not in df.columns:
            return
        fig.add_trace(go.Scattergl(x=df.index, y=df['Close'], mode='lines', name='Prix', line=dict(color='black', width=1), opacity=0.5), row=row, col=col)
        for ma_period in SMA_PERIODS:
            col_name = f"SMA_{ma_period}"
            if col_name in df.columns:
                fig.add_trace(go.Scattergl(x=df.index, y=df[col_name], mode='lines', name=f"SMA {ma_period}"), row=row, col=col)

    def _add_volatility_chart(self, fig, row: int, col: int):
        if 'ATR' not in self.technical_data.columns:
            self.logger.warning("Colonne ATR manquante")
            return
        df = self.technical_data
        fig.add_trace(go.Scattergl(x=df.index, y=df['ATR'], mode='lines', name='ATR', line=dict(color='orange', width=2)), row=row, col=col)

    def _add_predictions_chart(self, fig, row: int, col: int):
        if self.predictions_df is None or self.predictions_df.empty or 'Predicted_Close' not in self.predictions_df.columns:
            return
        future_dates = self.predictions_df.index
        future_prices = self.predictions_df['Predicted_Close'].values
        fig.add_trace(go.Scattergl(x=future_dates, y=future_prices, mode='lines+markers', name='Prévisions IA', line=dict(color='red', dash='dash')), row=row, col=col)
        if 'CI_Lower' in self.predictions_df.columns and 'CI_Upper' in self.predictions_df.columns:
            fig.add_trace(go.Scattergl(x=future_dates, y=self.predictions_df['CI_Upper'], mode='lines', line=dict(color='rgba(255,0,0,0.3)')), row=row, col=col)
            fig.add_trace(go.Scattergl(x=future_dates, y=self.predictions_df['CI_Lower'], mode='lines', fill='tonexty', fillcolor='rgba(0,255,0,0.1)'), row=row, col=col)

    def _add_drawdown_chart(self, fig, row: int, col: int):
        if self.technical_data is None or 'Close' not in self.technical_data.columns:
            return
        df = self.technical_data
        rolling_max = df['Close'].cummax()
        drawdown = (df['Close'] - rolling_max) / rolling_max * 100
        fig.add_trace(go.Scattergl(x=df.index, y=drawdown, mode='lines', name='Drawdown (%)', line=dict(color='red', width=2)), row=row, col=col)
        fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', line=dict(dash='dash', color='black'), showlegend=False), row=row, col=col)

    def _add_sharpe_gauge(self, fig, row: int, col: int):
        sr = self.risk_metrics.get('sharpe_ratio') if self.risk_metrics else None
        if sr is None:
            return
        fig.add_trace(go.Indicator(mode="gauge+number", value=sr, title={'text': "Sharpe Ratio"}, gauge={'axis': {'range': [-2, 3]}}), row=row, col=col)

    def _add_var_gauge(self, fig, row: int, col: int):
        if 'value_at_risk' not in self.risk_metrics:
            return
        var = abs(self.risk_metrics['value_at_risk'] * 100)
        fig.add_trace(go.Indicator(mode="gauge+number", value=var, title={'text': "VaR (95%) %"}, gauge={'axis': {'range': [0, 20]}}), row=row, col=col)

    # -------------------------
    # Summary table helpers
    # -------------------------
    def _add_technical_summary(self, rows: List[List]):
        df = self.technical_data
        last = df.iloc[-1] if df is not None and not df.empty else {}
        rows.append(["Prix actuel", f"${self.current_price:.2f}", "-"])
        if isinstance(last, (pd.Series, dict)) and 'RSI' in last:
            rsi = last['RSI']
            rows.append(["RSI (14)", f"{rsi:.1f}", "Survente" if rsi < RSI_OVERSOLD else "Surachat" if rsi > RSI_OVERBOUGHT else "Neutre"])
        # add other short items...
        rows.append(["--- RISQUE ---", "---", "---"])

    def _add_risk_summary(self, rows: List[List]):
        if not self.risk_metrics:
            return
        if 'sharpe_ratio' in self.risk_metrics:
            rows.append(["Sharpe Ratio", f"{self.risk_metrics['sharpe_ratio']:.2f}", ""])
        if 'beta' in self.risk_metrics:
            rows.append(["Bêta", f"{self.risk_metrics['beta']:.2f}", ""])

    def _add_training_metrics(self, rows: List[List]):
        if not self.training_metrics:
            return
        rows.append(["--- ENTRAÎNEMENT MODELE ---", "---", "---"])
        ttime = self.training_metrics.get('training_time_s')
        if ttime:
            rows.append(["Temps entraînement", f"{ttime:.1f}s", ""])
        val = self.training_metrics.get('validation', {})
        if isinstance(val, dict):
            for k, v in val.items():
                mae = v.get('mae') if isinstance(v, dict) else None
                if mae is not None:
                    rows.append([f"MAE {k}", f"{mae:.2f}", ""])

    def _add_prediction_examples(self, rows: List[List], max_examples: int = 3):
        if not self.prediction_examples:
            return
        rows.append(["--- PRÉDICTIONS EXEMPLES ---", "---", "---"])
        for ex in self.prediction_examples[:max_examples]:
            pred = ex.get('predicted')
            act = ex.get('actual')
            rows.append([ex.get('horizon', ''), f"${pred:.2f}" if pred is not None else "N/A", f"Réal: ${act:.2f}" if act is not None else "N/A"])

    def _add_economic_summary(self, rows: List[List]):
        if not self.macro_data:
            return
        econ = self.macro_data.get('economic_indicators', {}) if isinstance(self.macro_data, dict) else {}
        if econ:
            rows.append(["--- ÉCONOMIE ---", "---", "---"])
            for name, data in econ.items():
                if isinstance(data, dict):
                    val = data.get('value', '')
                    rows.append([name, str(val), ""])

    def _add_summary_table(self, fig, row: int, col: int):
        rows: List[List] = []
        self._add_technical_summary(rows)
        self._add_risk_summary(rows)
        self._add_training_metrics(rows)
        self._add_prediction_examples(rows)
        self._add_economic_summary(rows)
        rows.append(["Score trading", f"{self.score}/10", self.recommendation])
        headers = ["Indicateur", "Valeur", "Interprétation"]
        fig.add_trace(go.Table(header=dict(values=headers, fill_color='paleturquoise', align='left'), cells=dict(values=list(zip(*rows)), fill_color='lavender', align='left')), row=row, col=col)

    # -------------------------
    # Layout and render
    # -------------------------
    def _update_layout(self, fig: go.Figure):
        fig.update_layout(title=dict(text=f'Tableau de Bord Trading - {self.symbol}', x=0.5), height=1400, template='plotly_white', hovermode='x unified', margin=dict(l=50, r=50, t=100, b=50))

    def _save_dashboard(self, fig: go.Figure, save_path: str):
        try:
            os.makedirs(save_path, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = f"{save_path}/{self.symbol}_dashboard_{timestamp}.html"
            fig.write_html(html_file, config={'responsive': True})
            png_file = f"{save_path}/{self.symbol}_dashboard_{timestamp}.png"
            fig.write_image(png_file, width=1600, height=900, scale=2)
            meta = {'symbol': self.symbol, 'timestamp': timestamp, 'files': {'html': html_file, 'png': png_file}}
            with open(f"{save_path}/{self.symbol}_metadata_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            self.logger.info(f"Dashboard saved: {html_file}")
        except Exception as e:
            self.logger.error(f"Error saving dashboard: {e}")

    def create_main_dashboard(self, save_path: str = "dashboards") -> Optional[go.Figure]:
        if self.technical_data is None or self.technical_data.empty:
            self.logger.error("Données techniques manquantes")
            return None

        try:
            fig = make_subplots(
                rows=6, cols=3,
                specs=[
                    [{'type': 'xy', 'rowspan': 2, 'colspan': 2}, None, {'type': 'indicator'}],
                    [None, None, {'type': 'indicator'}],
                    [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                    [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                    [{'type': 'xy'}, {'type': 'indicator'}, {'type': 'indicator'}],
                    [{'type': 'table', 'colspan': 3}, None, None]
                ],
                subplot_titles=(
                    f'{self.symbol} - Prix et Indicateurs',
                    'Score Trading',
                    'Sentiment Marché',
                    'RSI (14 jours)',
                    'MACD',
                    'Volume',
                    'Moyennes Mobiles',
                    'Volatilité (ATR)',
                    'Prévisions IA',
                    'Drawdown Historique',
                    'Sharpe Ratio',
                    'VaR (95%)'
                ),
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )

            # add charts (each helper may log on error but should exist)
            try:
                self._add_main_price_chart(fig, row=1, col=1)
                self._add_trading_score_gauge(fig, row=1, col=3)
                self._add_market_sentiment_indicator(fig, row=2, col=3)
                self._add_rsi_chart(fig, row=3, col=1)
                self._add_macd_chart(fig, row=3, col=2)
                self._add_volume_chart(fig, row=3, col=3)
                self._add_trend_chart(fig, row=4, col=1)
                self._add_volatility_chart(fig, row=4, col=2)
                self._add_predictions_chart(fig, row=4, col=3)
                self._add_drawdown_chart(fig, row=5, col=1)
                self._add_sharpe_gauge(fig, row=5, col=2)
                self._add_var_gauge(fig, row=5, col=3)
                self._add_summary_table(fig, row=6, col=1)
            except Exception as e:
                self.logger.error(f"Error adding chart: {e}", exc_info=True)

            self._update_layout(fig)
            self._save_dashboard(fig, save_path)
            return fig
        except Exception as e:
            self.logger.error(f"Erreur création dashboard: {e}", exc_info=True)
            return None

    def create_quick_overview(self) -> Optional[pd.DataFrame]:
        if not self.overview:
            return None
        items = [
            ('Symbole', self.symbol),
            ('Prix Actuel', f"${self.current_price:.2f}"),
            ('Score', f"{self.score:.1f}/10"),
            ('Recommandation', self.recommendation),
            ('Nom', self.overview.get('name', 'N/A')),
            ('Market Cap', self.overview.get('market_cap', 'N/A')),
            ('P/E Ratio', self.overview.get('pe_ratio', 'N/A')),
            ('Dividende', self.overview.get('dividend_yield', 'N/A'))
        ]
        return pd.DataFrame([{'Métrique': k, 'Valeur': v} for k, v in items])

class MiniDashboard:
    """Dashboard minimal pour affichage rapide"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
    
    def create_compact_view(self, data: pd.DataFrame, predictions_df: pd.DataFrame) -> go.Figure:
        """Crée une vue compacte du dashboard"""
        fig = go.Figure()
        if data is not None and not data.empty and 'Close' in data.columns:
            fig.add_trace(go.Scattergl(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Prix historique',
                line=dict(color='blue', width=2)
            ))
        if predictions_df is not None and not predictions_df.empty and 'Predicted_Close' in predictions_df.columns:
            fig.add_trace(go.Scattergl(
                x=predictions_df.index,
                y=predictions_df['Predicted_Close'],
                mode='lines+markers',
                name='Prévisions IA',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            ))
        fig.update_layout(
            title=f'{self.symbol} - Vue Rapide',
            xaxis_title='Date',
            yaxis_title='Prix ($)',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        return fig

def create_comparison_dashboard(symbols: List[str], data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Crée un dashboard de comparaison entre plusieurs actions"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Relative', 'Volatilité Comparée',
                       'RSI Comparé', 'Volume Comparé'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set3
    
    for idx, symbol in enumerate(symbols):
        if symbol in data_dict:
            data = data_dict[symbol]
            color = colors[idx % len(colors)]
            
            if 'Close' in data.columns and len(data) > 0:
                normalized = (data['Close'] / data['Close'].iloc[0] * 100)
                fig.add_trace(
                    go.Scattergl(x=data.index, y=normalized, name=symbol, line=dict(color=color)),
                    row=1, col=1
                )
            
            if 'ATR' in data.columns:
                fig.add_trace(
                    go.Scattergl(x=data.index, y=data['ATR'], name=symbol, line=dict(color=color), showlegend=False),
                    row=1, col=2
                )
            
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scattergl(x=data.index, y=data['RSI'], name=symbol, line=dict(color=color), showlegend=False),
                    row=2, col=1
                )
            
            if 'Volume' in data.columns:
                volume_normalized = data['Volume'] / data['Volume'].max() * 100
                fig.add_trace(
                    go.Bar(x=data.index, y=volume_normalized, name=symbol, marker_color=color,
                           opacity=0.6, showlegend=False),
                    row=2, col=2
                )
    
    fig.update_layout(
        title='Comparaison Multi-Actions',
        height=1000,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    def _add_sharpe_gauge(self, fig, row: int, col: int):
        """Ajoute un gauge pour le Sharpe Ratio avec plages colorées."""
        sr = self.risk_metrics.get('sharpe_ratio') if self.risk_metrics else None
        if sr is None:
            return

        # Gauge color bands:
        # [-2, 0): red, [0,1): yellow, [1,2): lightgreen, [2,3]: green
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sr,
                title={'text': "Sharpe Ratio", 'font': {'size': 14}},
                number={'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [-2, 3], 'tickwidth': 1, 'dtick': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-2, 0], 'color': "#d9534f"},   # red
                        {'range': [0, 1], 'color': "#f0ad4e"},    # yellow/orange
                        {'range': [1, 2], 'color': "#5cb85c"},    # green
                        {'range': [2, 3], 'color': "#2b7a2b"}     # darker green
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': sr
                    }
                },
            ),
            row=row, col=col
        )

    def _add_var_gauge(self, fig, row: int, col: int):
        """Ajoute un gauge pour la Value at Risk (VaR %) avec plages colorées."""
        if 'value_at_risk' not in self.risk_metrics:
            return
        var = abs(self.risk_metrics['value_at_risk'] * 100)  # en %
        # Define axis max adaptively but cap to 50 for display sanity
        axis_max = max(10, min(50, (var * 3) if var > 0 else 10))

        # Color bands: [0-2]: dark green, (2-5): light green, (5-10): yellow, (10-axis_max]: red
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=var,
                title={'text': "VaR (95%) %", 'font': {'size': 14}},
                number={'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [0, axis_max], 'tickwidth': 1},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 2], 'color': "#2b7a2b"},     # dark green
                        {'range': [2, 5], 'color': "#5cb85c"},     # light green
                        {'range': [5, 10], 'color': "#f0ad4e"},    # yellow/orange
                        {'range': [10, axis_max], 'color': "#d9534f"}  # red
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': var
                    }
                }
            ),
            row=row, col=col
        )

    def _add_risk_legend_rows(self, rows: List[List]):
        """
        Append human-readable legend rows (emoji color chips) to summary table rows.
        This keeps the legend visible in the summary table (compatible with static HTML export).
        """
        try:
            rows.append(["--- LÉGENDE - Sharpe Ratio ---", "", ""])
            rows.append(["Sharpe < 0", "Perf. faible / Risque élevé", "🔴"])
            rows.append(["0 ≤ Sharpe < 1", "Acceptable", "🟡"])
            rows.append(["1 ≤ Sharpe < 2", "Bon", "🟢"])
            rows.append(["Sharpe ≥ 2", "Excellent", "🟢"])

            rows.append(["--- LÉGENDE - VaR (95%) ---", "", ""])
            rows.append(["VaR ≤ 2%", "Risque faible", "🟢"])
            rows.append(["2% < VaR ≤ 5%", "Risque modéré", "🟢"])
            rows.append(["5% < VaR ≤ 10%", "Risque élevé", "🟡"])
            rows.append(["VaR > 10%", "Risque très élevé", "🔴"])
        except Exception:
            # never fail dashboard construction because of legend
            pass

    def _add_summary_table(self, fig, row, col):
        """Ajoute un tableau récapitulatif des indicateurs et métriques de risque"""
        if self.technical_data is None or self.technical_data.empty:
            self.logger.warning("Pas de données techniques pour le tableau récapitulatif")
            return

        all_rows = []
        self._add_technical_summary(all_rows)
        self._add_risk_summary(all_rows)
        # Inject training metrics and prediction examples into the summary table
        self._add_training_metrics(all_rows)
        self._add_prediction_examples(all_rows)
        self._add_economic_summary(all_rows)

        # Add the color legend rows here so they appear in the summary table
        self._add_risk_legend_rows(all_rows)

        all_rows.append(["Score trading", f"{self.score}/10", self.recommendation])

        headers = ["Indicateur", "Valeur", "Interprétation"]
        fig.add_trace(
            go.Table(
                header=dict(values=headers, fill_color='paleturquoise', align='left'),
                cells=dict(values=list(zip(*all_rows)), fill_color='lavender', align='left')
            ),
            row=row, col=col
        )

    # Insert these helper methods inside the TradingDashboard class (e.g. near other private helpers)

    def _format_color_square(self, color: str, size: int = 12) -> str:
        """Return an HTML colored square using a span (renders in Plotly annotations)."""
        return f'<span style="display:inline-block;width:{size}px;height:{size}px;background:{color};margin-right:6px;border-radius:2px;"></span>'

    def _add_gauge_legend_annotation(self, fig, x_paper: float, y_paper: float, entries: list, title: str = ""):
        """
        Add a compact horizontal legend near gauges using paper coordinates.
        x_paper, y_paper are in [0,1] (figure paper coordinates).
        entries: list of (color, label) to show.
        """
        try:
            # Build HTML content: title + colored squares with labels
            parts = []
            if title:
                parts.append(f"<b>{title}</b><br>")
            for color, label in entries:
                parts.append(f"{self._format_color_square(color, size=12)}{label}")
            html = "&nbsp;&nbsp;".join(parts)
            fig.add_annotation(
                x=x_paper, y=y_paper,
                xref="paper", yref="paper",
                text=html,
                showarrow=False,
                align="left",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=0,
                bgcolor="rgba(255,255,255,0.0)",
                font={"size": 11},
            )
        except Exception as e:
            self.logger.debug(f"Could not add gauge legend annotation: {e}")

    # Replace existing _add_sharpe_gauge and _add_var_gauge implementations with these improved versions:

    def _add_sharpe_gauge(self, fig, row: int, col: int):
        """Ajoute un gauge pour le Sharpe Ratio avec plages colorées et mini-legend graphique."""
        sr = self.risk_metrics.get('sharpe_ratio') if self.risk_metrics else None
        if sr is None:
            return

        # Use settings bands/colors if available, fallback to defaults
        bands = getattr(settings, "SHARPE_BANDS", [(-2,0),(0,1),(1,2),(2,3)])
        colors = getattr(settings, "SHARPE_COLORS", ["#d9534f","#f0ad4e","#5cb85c","#2b7a2b"])
        labels = getattr(settings, "SHARPE_LABELS", ["< 0", "0–1", "1–2", "≥2"])
        axis_min = bands[0][0]
        axis_max = bands[-1][1]

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sr,
                title={'text': "Sharpe Ratio", 'font': {'size': 14}},
                number={'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [axis_min, axis_max], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [{'range': [b[0], b[1]], 'color': c} for b, c in zip(bands, colors)],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': sr
                    }
                },
            ),
            row=row, col=col
        )

        # Add mini legend near the gauge (paper coords tuned for this dashboard)
        # these coords may be adjusted if your layout changes
        legend_entries = list(zip(colors, labels))
        self._add_gauge_legend_annotation(fig, x_paper=0.72, y_paper=0.34, entries=legend_entries, title="Sharpe")

    def _add_var_gauge(self, fig, row: int, col: int):
        """Ajoute un gauge pour la Value at Risk (VaR %) avec plages colorées et mini-legend graphique."""
        if 'value_at_risk' not in self.risk_metrics:
            return
        var = abs(self.risk_metrics['value_at_risk'] * 100)  # en %
        bands = getattr(settings, "VAR_BANDS", [(0,2),(2,5),(5,10),(10,100)])
        colors = getattr(settings, "VAR_COLORS", ["#2b7a2b","#5cb85c","#f0ad4e","#d9534f"])
        labels = getattr(settings, "VAR_LABELS", ["≤2%","2–5%","5–10%",">10%"])
        axis_min = bands[0][0]
        axis_max = bands[-1][1] if bands[-1][1] > 0 else max(10, var*3)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=var,
                title={'text': "VaR (95%) %", 'font': {'size': 14}},
                number={'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [axis_min, axis_max], 'tickwidth': 1},
                    'bar': {'color': "darkred"},
                    'steps': [{'range': [b[0], b[1]], 'color': c} for b, c in zip(bands, colors)],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': var
                    }
                }
            ),
            row=row, col=col
        )

        legend_entries = list(zip(colors, labels))
        # Place legend to the right of VaR gauge
        self._add_gauge_legend_annotation(fig, x_paper=0.88, y_paper=0.34, entries=legend_entries, title="VaR (95%)")
    return fig