"""
Module UI pour dashboard professionnel d'évaluation d'actions.
Contient uniquement la couche de visualisation.
Toute la logique métier doit être assurée par un module externe (ex: stock_manager).
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_algo import settings

logger = logging.getLogger(__name__)

# Constantes d'affichage (seuils pour interprétation visuelle)
SMA_PERIODS = getattr(settings, "SMA_PERIODS", [20, 50, 200])
RSI_OVERBOUGHT = getattr(settings, "RSI_OVERBOUGHT", 70)
RSI_OVERSOLD = getattr(settings, "RSI_OVERSOLD", 30)


class AdvancedTradingDashboard:
    """
    Dashboard principal pour l'analyse d'une action unique.
    Nom compatible avec l'appel depuis main.py.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.logger = logging.getLogger(__name__)

        # Données injectées
        self.technical_data: pd.DataFrame = pd.DataFrame()
        self.predictions_df: pd.DataFrame = pd.DataFrame()
        self.risk_metrics: Dict[str, Any] = {}
        self.overview: Dict[str, Any] = {}
        self.macro_data: Dict[str, Any] = {}
        self.market_sentiment: Dict[str, Any] = {}
        self.training_metrics: Optional[Dict[str, Any]] = None
        self.prediction_examples: Optional[List[Dict[str, Any]]] = None

        # Métriques d'affichage
        self.score: float = 5.0
        self.recommendation: str = "NEUTRE"
        self.current_price: float = 0.0

    def load_data(self,
                  technical_data: Optional[pd.DataFrame] = None,
                  predictions_df: Optional[pd.DataFrame] = None,
                  risk_metrics: Optional[Dict[str, Any]] = None,
                  overview: Optional[Dict[str, Any]] = None,
                  macro_data: Optional[Dict[str, Any]] = None,
                  market_sentiment: Optional[Dict[str, Any]] = None,
                  score: Optional[float] = None,
                  recommendation: Optional[str] = None,
                  training_metrics: Optional[Dict[str, Any]] = None,
                  prediction_examples: Optional[List[Dict[str, Any]]] = None,
                  # Support d'appels positionnels (fallback)
                  *args, **kwargs) -> None:
        """
        Injecte toutes les données nécessaires.
        Accepte aussi les appels positionnels pour la rétrocompatibilité.
        """
        # Gestion des appels positionnels (ex: load_data(tech_df, preds_df, risk))
        if technical_data is None and len(args) > 0:
            technical_data = args[0]
        if predictions_df is None and len(args) > 1:
            predictions_df = args[1]
        if risk_metrics is None and len(args) > 2:
            risk_metrics = args[2]

        self.technical_data = technical_data.copy() if technical_data is not None else pd.DataFrame()
        self.predictions_df = predictions_df.copy() if predictions_df is not None else pd.DataFrame()
        self.risk_metrics = risk_metrics or {}
        self.overview = overview or {}
        self.macro_data = macro_data or {}
        self.market_sentiment = market_sentiment or {}
        self.training_metrics = training_metrics
        self.prediction_examples = prediction_examples or []

        if self.technical_data is not None and not self.technical_data.empty:
            try:
                self.current_price = float(self.technical_data["Close"].iloc[-1])
            except Exception:
                self.current_price = 0.0

        if score is not None:
            self.score = float(score)
        if recommendation is not None:
            self.recommendation = recommendation

        self.logger.info(f"Données chargées pour {self.symbol}")

    # --------------------------------------------------------------------------
    # Méthodes de construction des graphiques (UI pure)
    # --------------------------------------------------------------------------

    def _add_main_price_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Graphique principal du prix avec bandes de Bollinger et signaux d'achat/vente."""
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

        # Bandes de Bollinger
        if all(c in df.columns for c in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scattergl(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper',
                             line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash')),
                row=row, col=col
            )
            fig.add_trace(
                go.Scattergl(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower',
                             line=dict(color='rgba(0,255,0,0.3)', width=1, dash='dash'),
                             fill='tonexty', fillcolor='rgba(0,255,0,0.1)'),
                row=row, col=col
            )

        # Prix actuel
        last_price = df['Close'].iloc[-1]
        fig.add_hline(
            y=last_price, line_dash="dot", line_color="blue", opacity=0.5,
            annotation_text=f"Actuel: ${last_price:.2f}", annotation_position="bottom right",
            row=row, col=col
        )

        # Signaux d'achat/vente si présents
        try:
            if 'Signal' in df.columns:
                buys = df[df['Signal'] == 1]['Close']
                sells = df[df['Signal'] == -1]['Close']
                if not buys.empty:
                    fig.add_trace(
                        go.Scattergl(x=buys.index, y=buys.values, mode='markers',
                                     marker=dict(symbol='triangle-up', color='green', size=10),
                                     name='Signal Achat',
                                     hovertemplate='Achat %{y:.2f}<br>%{x|%d %b %Y}<extra></extra>'),
                        row=row, col=col
                    )
                if not sells.empty:
                    fig.add_trace(
                        go.Scattergl(x=sells.index, y=sells.values, mode='markers',
                                     marker=dict(symbol='triangle-down', color='red', size=10),
                                     name='Signal Vente',
                                     hovertemplate='Vente %{y:.2f}<br>%{x|%d %b %Y}<extra></extra>'),
                        row=row, col=col
                    )
        except Exception as e:
            self.logger.debug(f"Impossible d'ajouter les signaux: {e}")

    def _add_trading_score_gauge(self, fig: go.Figure, row: int, col: int) -> None:
        """Jauge du score de trading (0 à 10)."""
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

    def _add_market_sentiment_indicator(self, fig: go.Figure, row: int, col: int) -> None:
        """Indicateur de sentiment de marché (score 0-100)."""
        sentiment_score = self.market_sentiment.get('overall_sentiment', {}).get('score', 50) if self.market_sentiment else 50
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=sentiment_score,
                title={'text': "Sentiment Marché", 'font': {'size': 14}},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "green"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 2}, 'value': sentiment_score}
                }
            ),
            row=row, col=col
        )

    def _add_rsi_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Graphique RSI avec seuils de surachat/survente."""
        if 'RSI' not in self.technical_data.columns:
            self.logger.warning("Colonne RSI manquante")
            return
        df = self.technical_data
        x = df.index
        fig.add_trace(
            go.Scattergl(x=x, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple', width=2),
                         hovertemplate='%{x|%d %b %Y}<br>RSI: %{y:.1f}<extra></extra>'),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=[x[0], x[-1]], y=[RSI_OVERBOUGHT, RSI_OVERBOUGHT], mode='lines',
                       line=dict(color='red', dash='dash'), showlegend=False),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=[x[0], x[-1]], y=[RSI_OVERSOLD, RSI_OVERSOLD], mode='lines',
                       line=dict(color='green', dash='dash'), showlegend=False),
            row=row, col=col
        )

    def _add_macd_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Graphique MACD (ligne, signal, histogramme)."""
        df = self.technical_data
        if not all(c in df.columns for c in ['MACD', 'MACD_Signal']):
            self.logger.warning("Colonnes MACD manquantes")
            return
        fig.add_trace(
            go.Scattergl(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue', width=2)),
            row=row, col=col
        )
        fig.add_trace(
            go.Scattergl(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red', width=1.5)),
            row=row, col=col
        )
        if 'MACD_Histogram' in df.columns:
            colors = ['green' if v >= 0 else 'red' for v in df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACD_Histogram'], marker_color=colors, opacity=0.6, name='MACD Hist'),
                row=row, col=col
            )

    def _add_volume_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Graphique des volumes (couleur selon variation du prix)."""
        if 'Volume' not in self.technical_data.columns:
            self.logger.warning("Colonne Volume manquante")
            return
        df = self.technical_data
        colors = []
        for i in range(len(df)):
            if i == 0:
                colors.append("green")
            else:
                colors.append("green" if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else "red")
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.7, name='Volume'),
            row=row, col=col
        )

    def _add_trend_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Graphique des moyennes mobiles."""
        df = self.technical_data
        if df is None or df.empty or 'Close' not in df.columns:
            return
        fig.add_trace(
            go.Scattergl(x=df.index, y=df['Close'], mode='lines', name='Prix', line=dict(color='black', width=1), opacity=0.5),
            row=row, col=col
        )
        for period in SMA_PERIODS:
            col_name = f"SMA_{period}"
            if col_name in df.columns:
                fig.add_trace(
                    go.Scattergl(x=df.index, y=df[col_name], mode='lines', name=f"SMA {period}"),
                    row=row, col=col
                )

    def _add_volatility_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Graphique de l'ATR (Average True Range)."""
        if 'ATR' not in self.technical_data.columns:
            self.logger.warning("Colonne ATR manquante")
            return
        df = self.technical_data
        fig.add_trace(
            go.Scattergl(x=df.index, y=df['ATR'], mode='lines', name='ATR', line=dict(color='orange', width=2)),
            row=row, col=col
        )

    def _add_predictions_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Prévisions IA (lignes + intervalles de confiance)."""
        if self.predictions_df is None or self.predictions_df.empty or 'Predicted_Close' not in self.predictions_df.columns:
            return
        df_pred = self.predictions_df
        fig.add_trace(
            go.Scattergl(x=df_pred.index, y=df_pred['Predicted_Close'], mode='lines+markers',
                         name='Prévisions IA', line=dict(color='red', dash='dash')),
            row=row, col=col
        )
        if 'CI_Lower' in df_pred.columns and 'CI_Upper' in df_pred.columns:
            fig.add_trace(
                go.Scattergl(x=df_pred.index, y=df_pred['CI_Upper'], mode='lines',
                             line=dict(color='rgba(255,0,0,0.3)'), showlegend=False),
                row=row, col=col
            )
            fig.add_trace(
                go.Scattergl(x=df_pred.index, y=df_pred['CI_Lower'], mode='lines',
                             fill='tonexty', fillcolor='rgba(0,255,0,0.1)', showlegend=False),
                row=row, col=col
            )

    def _add_drawdown_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Drawdown historique (en pourcentage)."""
        if self.technical_data is None or 'Close' not in self.technical_data.columns:
            return
        df = self.technical_data
        rolling_max = df['Close'].cummax()
        drawdown = (df['Close'] - rolling_max) / rolling_max * 100
        fig.add_trace(
            go.Scattergl(x=df.index, y=drawdown, mode='lines', name='Drawdown (%)', line=dict(color='red', width=2)),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=[0]*len(df), mode='lines', line=dict(dash='dash', color='black'), showlegend=False),
            row=row, col=col
        )

    def _add_sharpe_gauge(self, fig: go.Figure, row: int, col: int) -> None:
        """Jauge du ratio de Sharpe avec plages colorées et légende."""
        sr = self.risk_metrics.get('sharpe_ratio') if self.risk_metrics else None
        if sr is None:
            return

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
                    'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': sr}
                },
            ),
            row=row, col=col
        )
        legend_entries = list(zip(colors, labels))
        self._add_gauge_legend_annotation(fig, x_paper=0.72, y_paper=0.34, entries=legend_entries, title="Sharpe")

    def _add_var_gauge(self, fig: go.Figure, row: int, col: int) -> None:
        """Jauge de la Value at Risk (VaR) avec plages colorées et légende."""
        var = abs(self.risk_metrics.get('value_at_risk', 0.0)) * 100.0 if self.risk_metrics else None
        if var is None:
            return

        bands = getattr(settings, "VAR_BANDS", [(0,2),(2,5),(5,10),(10,100)])
        colors = getattr(settings, "VAR_COLORS", ["#2b7a2b","#5cb85c","#f0ad4e","#d9534f"])
        labels = getattr(settings, "VAR_LABELS", ["≤2%","2–5%","5–10%",">10%"])
        axis_max = bands[-1][1] if bands[-1][1] > 0 else max(10, var*3)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=var,
                title={'text': "VaR (95%) %", 'font': {'size': 14}},
                number={'valueformat': '.2f', 'suffix': '%'},
                gauge={
                    'axis': {'range': [0, axis_max], 'tickwidth': 1},
                    'bar': {'color': "darkred"},
                    'steps': [{'range': [b[0], b[1]], 'color': c} for b, c in zip(bands, colors)],
                    'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': var}
                },
            ),
            row=row, col=col
        )
        legend_entries = list(zip(colors, labels))
        self._add_gauge_legend_annotation(fig, x_paper=0.88, y_paper=0.34, entries=legend_entries, title="VaR (95%)")

    def _format_color_square(self, color: str, size: int = 12) -> str:
        """Renvoie un carré coloré HTML pour les légendes."""
        return f'<span style="display:inline-block;width:{size}px;height:{size}px;background:{color};margin-right:6px;border-radius:2px;"></span>'

    def _add_gauge_legend_annotation(self, fig: go.Figure, x_paper: float, y_paper: float,
                                     entries: List[tuple], title: str = "") -> None:
        """Ajoute une annotation de légende pour les jauges."""
        try:
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
            self.logger.debug(f"Impossible d'ajouter la légende: {e}")

    # --------------------------------------------------------------------------
    # Tableau récapitulatif
    # --------------------------------------------------------------------------

    def _add_technical_summary(self, rows: List[List]) -> None:
        """Ajoute les lignes techniques au tableau."""
        df = self.technical_data
        if df is None or df.empty:
            return
        last = df.iloc[-1]
        rows.append(["Prix actuel", f"${self.current_price:.2f}", "-"])
        if 'RSI' in last:
            rsi = last['RSI']
            interp = "Survente" if rsi < RSI_OVERSOLD else "Surachat" if rsi > RSI_OVERBOUGHT else "Neutre"
            rows.append(["RSI (14)", f"{rsi:.1f}", interp])
        if 'MACD' in last and 'MACD_Signal' in last:
            if last['MACD'] > last['MACD_Signal']:
                rows.append(["MACD", "Signal haussier", "🟢"])
            else:
                rows.append(["MACD", "Signal baissier", "🔴"])
        rows.append(["--- RISQUE ---", "---", "---"])

    def _add_risk_summary(self, rows: List[List]) -> None:
        """Ajoute les métriques de risque au tableau."""
        if not self.risk_metrics:
            return
        if 'sharpe_ratio' in self.risk_metrics:
            rows.append(["Sharpe Ratio", f"{self.risk_metrics['sharpe_ratio']:.2f}", ""])
        if 'beta' in self.risk_metrics:
            rows.append(["Bêta", f"{self.risk_metrics['beta']:.2f}", ""])
        if 'value_at_risk' in self.risk_metrics:
            var_pct = abs(self.risk_metrics['value_at_risk'] * 100)
            rows.append(["VaR (95%)", f"{var_pct:.2f}%", ""])

    def _add_training_metrics(self, rows: List[List]) -> None:
        """Ajoute les métriques d'entraînement du modèle."""
        if not self.training_metrics:
            return
        rows.append(["--- ENTRAÎNEMENT MODÈLE ---", "---", "---"])
        ttime = self.training_metrics.get('training_time_s')
        if ttime:
            rows.append(["Temps entraînement", f"{ttime:.1f}s", ""])
        val = self.training_metrics.get('validation', {})
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, dict) and 'mae' in v:
                    rows.append([f"MAE {k}", f"{v['mae']:.2f}", ""])

    def _add_prediction_examples(self, rows: List[List], max_examples: int = 3) -> None:
        """Ajoute des exemples de prédictions."""
        if not self.prediction_examples:
            return
        rows.append(["--- PRÉDICTIONS EXEMPLES ---", "---", "---"])
        for ex in self.prediction_examples[:max_examples]:
            pred = ex.get('predicted')
            act = ex.get('actual')
            rows.append([
                ex.get('horizon', ''),
                f"${pred:.2f}" if pred is not None else "N/A",
                f"Réel: ${act:.2f}" if act is not None else "N/A"
            ])

    def _add_economic_summary(self, rows: List[List]) -> None:
        """Ajoute les indicateurs économiques."""
        if not self.macro_data:
            return
        econ = self.macro_data.get('economic_indicators', {}) if isinstance(self.macro_data, dict) else {}
        if econ:
            rows.append(["--- ÉCONOMIE ---", "---", "---"])
            for name, data in econ.items():
                if isinstance(data, dict):
                    val = data.get('value', '')
                    rows.append([name, str(val), ""])

    def _add_risk_legend_rows(self, rows: List[List]) -> None:
        """Ajoute une légende textuelle pour les métriques de risque dans le tableau."""
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
            pass

    def _add_summary_table(self, fig: go.Figure, row: int, col: int) -> None:
        """Construit et ajoute le tableau récapitulatif complet."""
        if self.technical_data is None or self.technical_data.empty:
            self.logger.warning("Pas de données techniques pour le tableau récapitulatif")
            return

        all_rows: List[List] = []
        self._add_technical_summary(all_rows)
        self._add_risk_summary(all_rows)
        self._add_training_metrics(all_rows)
        self._add_prediction_examples(all_rows)
        self._add_economic_summary(all_rows)
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

    # --------------------------------------------------------------------------
    # Construction et sauvegarde du dashboard complet
    # --------------------------------------------------------------------------

    def create_full_dashboard(self) -> go.Figure:
        """Appelée par main.py (via dash.create_full_dashboard())."""
        return self.create_main_dashboard(save_path=None)

    def create_dashboard(self) -> go.Figure:
        """Alias pour compatibilité."""
        return self.create_full_dashboard()

    def create_main_dashboard(self, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """Construit le dashboard complet."""
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

            # Ajout de tous les sous-graphiques
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

            fig.update_layout(
                title=dict(text=f'Tableau de Bord Trading - {self.symbol}', x=0.5),
                height=1400,
                template='plotly_white',
                hovermode='x unified',
                margin=dict(l=50, r=50, t=100, b=50)
            )

            if save_path:
                self._save_dashboard(fig, save_path)

            return fig
        except Exception as e:
            self.logger.error(f"Erreur création dashboard: {e}", exc_info=True)
            return None

    def _save_dashboard(self, fig: go.Figure, save_path: str) -> None:
        """Sauvegarde le dashboard au format HTML, PNG et métadonnées JSON."""
        try:
            os.makedirs(save_path, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = f"{save_path}/{self.symbol}_dashboard_{timestamp}.html"
            fig.write_html(html_file, config={'responsive': True})
            png_file = f"{save_path}/{self.symbol}_dashboard_{timestamp}.png"
            fig.write_image(png_file, width=1600, height=900, scale=2)
            meta = {
                'symbol': self.symbol,
                'timestamp': timestamp,
                'files': {'html': html_file, 'png': png_file}
            }
            with open(f"{save_path}/{self.symbol}_metadata_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            self.logger.info(f"Dashboard sauvegardé : {html_file}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du dashboard: {e}")

    # --------------------------------------------------------------------------
    # Vue rapide (aperçu sous forme de tableau)
    # --------------------------------------------------------------------------

    def create_quick_overview(self) -> Optional[pd.DataFrame]:
        """Retourne un aperçu rapide sous forme de DataFrame."""
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
            ('Dividende', self.overview.get('dividend_yield', 'N/A')),
        ]
        return pd.DataFrame([{'Métrique': k, 'Valeur': v} for k, v in items])


class MiniDashboard:
    """Dashboard minimal pour un affichage rapide (prix + prévisions)."""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.logger = logging.getLogger(__name__)

    def create_compact_view(self, data: pd.DataFrame, predictions_df: pd.DataFrame) -> go.Figure:
        """Crée une vue compacte avec le prix historique et les prévisions."""
        fig = go.Figure()
        if data is not None and not data.empty and 'Close' in data.columns:
            fig.add_trace(
                go.Scattergl(x=data.index, y=data['Close'], mode='lines', name='Prix historique',
                             line=dict(color='blue', width=2))
            )
        if predictions_df is not None and not predictions_df.empty and 'Predicted_Close' in predictions_df.columns:
            fig.add_trace(
                go.Scattergl(x=predictions_df.index, y=predictions_df['Predicted_Close'], mode='lines+markers',
                             name='Prévisions IA', line=dict(color='red', width=2, dash='dash'), marker=dict(size=4))
            )
        fig.update_layout(
            title=f'{self.symbol} - Vue Rapide',
            xaxis_title='Date',
            yaxis_title='Prix ($)',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        return fig


# --------------------------------------------------------------------------
# Dashboard de comparaison multi-actions
# --------------------------------------------------------------------------

def create_comparison_dashboard(symbols: List[str], data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Crée un dashboard de comparaison entre plusieurs actions.
    data_dict : dictionnaire {symbole: DataFrame contenant au moins Close, ATR, RSI, Volume}
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Relative', 'Volatilité Comparée (ATR)', 'RSI Comparé', 'Volume Normalisé'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    colors = px.colors.qualitative.Set3
    for idx, symbol in enumerate(symbols):
        df = data_dict.get(symbol)
        if df is None or df.empty:
            continue
        color = colors[idx % len(colors)]

        # Performance relative (normalisée à 100)
        if 'Close' in df.columns and len(df) > 0:
            normalized = df['Close'] / df['Close'].iloc[0] * 100
            fig.add_trace(
                go.Scattergl(x=df.index, y=normalized, name=symbol, line=dict(color=color)),
                row=1, col=1
            )

        # Volatilité (ATR)
        if 'ATR' in df.columns:
            fig.add_trace(
                go.Scattergl(x=df.index, y=df['ATR'], name=symbol, line=dict(color=color), showlegend=False),
                row=1, col=2
            )

        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scattergl(x=df.index, y=df['RSI'], name=symbol, line=dict(color=color), showlegend=False),
                row=2, col=1
            )

        # Volume normalisé
        if 'Volume' in df.columns:
            volume_norm = df['Volume'] / df['Volume'].max() * 100
            fig.add_trace(
                go.Bar(x=df.index, y=volume_norm, name=symbol, marker_color=color, opacity=0.6, showlegend=False),
                row=2, col=2
            )

    fig.update_layout(
        title='Comparaison Multi-Actions',
        height=1000,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    return fig