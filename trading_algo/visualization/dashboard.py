"""
Module de tableau de bord interactif pour l'analyse d'actions
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import json

# Constantes
SMA_PERIODS = [20, 50, 200]
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ATR_HIGH_THRESHOLD = 3.0
ATR_MODERATE_THRESHOLD = 1.5

class TradingDashboard:
    """Classe principale pour la création de tableaux de bord trading"""
    
    def __init__(self, symbol: str, current_price: float):
        self.symbol = symbol
        self.current_price = current_price
        self.overview = None
        self.technical_data = None
        self.predictions_df = None
        self.macro_data = None
        self.market_sentiment = None
        self.score = 5.0
        self.recommendation = "NEUTRE"
        self.risk_metrics = None
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self,
                  overview: Dict[str, Any],
                  technical_data: pd.DataFrame,
                  predictions_df: pd.DataFrame,
                  macro_data: Optional[Dict[str, Any]] = None,
                  market_sentiment: Optional[Dict[str, Any]] = None,
                  score: Optional[float] = None,
                  recommendation: Optional[str] = None,
                  risk_metrics: Optional[Dict[str, Any]] = None):
        """Charge les données nécessaires pour le dashboard"""
        self.overview = overview
        self.technical_data = technical_data
        self.predictions_df = predictions_df
        self.macro_data = macro_data or {}
        self.market_sentiment = market_sentiment or {}
        self.risk_metrics = risk_metrics or {}
        if score is not None and recommendation is not None:
            self.score = score
            self.recommendation = recommendation
        else:
            self._calculate_score_and_recommendation()
    
        self.logger.info(f"Données chargées pour {self.symbol}")
    
    def _calculate_score_and_recommendation(self):
        """Calcule le score et la recommandation basés sur les données"""
        try:
            tech_score = 5.0
            if self.technical_data is not None and not self.technical_data.empty:
                last_row = self.technical_data.iloc[-1]
                if 'RSI' in last_row:
                    rsi = last_row['RSI']
                    if rsi < RSI_OVERSOLD:
                        tech_score += 2
                    elif rsi > RSI_OVERBOUGHT:
                        tech_score -= 2
                if 'MACD' in last_row and 'MACD_Signal' in last_row:
                    if last_row['MACD'] > last_row['MACD_Signal']:
                        tech_score += 1
                if 'SMA_50' in last_row and 'SMA_200' in last_row:
                    if last_row['SMA_50'] > last_row['SMA_200']:
                        tech_score += 1
            
            # Intégration des métriques de risque dans le score
            if self.risk_metrics:
                if 'sharpe_ratio' in self.risk_metrics:
                    sr = self.risk_metrics['sharpe_ratio']
                    tech_score += max(min(sr, 2), -2)  # Ajustement basé sur Sharpe
                if 'beta' in self.risk_metrics:
                    beta = self.risk_metrics['beta']
                    if beta > 1.5:
                        tech_score -= 1  # Pénalité pour volatilité élevée
                    elif beta < 0.5:
                        tech_score += 1  # Bonus pour stabilité
            
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
    
    def _add_technical_summary(self, all_rows):
        """Ajoute la section des indicateurs techniques au tableau récapitulatif"""
        df = self.technical_data
        last_row = df.iloc[-1] if df is not None and not df.empty else {}
        
        tech_rows = []
        tech_rows.append(["Prix actuel", f"${self.current_price:.2f}", "-"])
        
        if 'RSI' in last_row:
            rsi = last_row['RSI']
            interpretation = "Surachat ⚠️" if rsi > RSI_OVERBOUGHT else "Survente ✅" if rsi < RSI_OVERSOLD else "Neutre"
            tech_rows.append(["RSI (14)", f"{rsi:.1f}", interpretation])
        
        if 'MACD' in last_row and 'MACD_Signal' in last_row:
            macd = last_row['MACD']
            signal = last_row['MACD_Signal']
            status = "Haussier 📈" if macd > signal else "Baissier 📉"
            tech_rows.append(["MACD", f"{macd:.2f}", status])
        
        if 'SMA_50' in last_row and 'SMA_200' in last_row:
            sma_50 = last_row['SMA_50']
            sma_200 = last_row['SMA_200']
            cross = "Croisement doré ✅" if sma_50 > sma_200 else "Croisement mortel ⚠️"
            tech_rows.append(["SMA 50/200", f"{sma_50:.2f}/{sma_200:.2f}", cross])
        
        if 'ATR' in last_row:
            atr_pct = (last_row['ATR'] / self.current_price) * 100
            vol_status = ("Élevée ⚠️" if atr_pct > ATR_HIGH_THRESHOLD 
                         else "Modérée ⚖️" if atr_pct > ATR_MODERATE_THRESHOLD 
                         else "Faible ✅")
            tech_rows.append(["Volatilité (ATR%)", f"{atr_pct:.1f}%", vol_status])
        
        all_rows.extend(tech_rows)
        all_rows.append(["--- RISQUE ---", "---", "---"])
    
    def _add_risk_summary(self, all_rows):
        """Ajoute la section des métriques de risque au tableau récapitulatif"""
        risk_rows = []
        if self.risk_metrics:
            if 'sharpe_ratio' in self.risk_metrics:
                sr = self.risk_metrics['sharpe_ratio']
                sr_status = "Bon ✅" if sr > 1 else "Médiocre ⚠️" if sr > 0 else "Négatif 🔴"
                risk_rows.append(["Sharpe Ratio", f"{sr:.2f}", sr_status])
            
            if 'max_drawdown' in self.risk_metrics:
                mdd = self.risk_metrics['max_drawdown'] * 100
                mdd_status = ("Élevé ⚠️" if abs(mdd) > 20 
                             else "Modéré ⚖️" if abs(mdd) > 10 
                             else "Faible ✅")
                risk_rows.append(["Max Drawdown", f"{mdd:.1f}%", mdd_status])
            
            if 'value_at_risk' in self.risk_metrics:
                var = self.risk_metrics['value_at_risk'] * 100
                var_status = "Élevé ⚠️" if abs(var) > 5 else "Modéré ⚖️"
                risk_rows.append(["VaR (95%)", f"{var:.1f}%", var_status])
            
            if 'beta' in self.risk_metrics:
                beta = self.risk_metrics['beta']
                beta_status = "Aggressif 📈" if beta > 1 else "Défensif 🛡️" if beta < 1 else "Neutre ⚖️"
                risk_rows.append(["Bêta", f"{beta:.2f}", beta_status])
            
            if 'stop_loss_levels' in self.risk_metrics:
                for horizon, stop in self.risk_metrics['stop_loss_levels'].items():
                    risk_rows.append([f"Stop {horizon}", f"${stop:.2f}", ""])
            if 'take_profit_levels' in self.risk_metrics:
                for horizon, tp in self.risk_metrics['take_profit_levels'].items():
                    risk_rows.append([f"Target {horizon}", f"${tp:.2f}", ""])
            if 'risk_reward_ratios' in self.risk_metrics:
                for horizon, rr in self.risk_metrics['risk_reward_ratios'].items():
                    rr_status = "Favorable ✅" if rr > 2 else "Neutre ⚖️" if rr > 1 else "Risqué ⚠️"
                    risk_rows.append([f"R/R {horizon}", f"{rr:.2f}", rr_status])
            if 'suggested_position_sizes' in self.risk_metrics:
                for horizon, size in self.risk_metrics['suggested_position_sizes'].items():
                    risk_rows.append([f"Position {horizon}", f"{size:.0f} actions", ""])
        
        all_rows.extend(risk_rows)
    
    def _add_economic_summary(self, all_rows):
        """Ajoute la section des indicateurs économiques au tableau récapitulatif"""
        econ_rows = []
        if self.macro_data and 'economic_indicators' in self.macro_data:
            econ = self.macro_data['economic_indicators']
            for name, data in econ.items():
                if isinstance(data, dict) and 'value' in data:
                    value = data['value']
                    unit = data.get('unit', '')
                    if isinstance(value, (int, float)):
                        value_str = f"{value:,.2f} {unit}".strip()
                    else:
                        value_str = f"{value} {unit}".strip()
                    econ_rows.append([name, value_str, ""])
        if econ_rows:
            all_rows.append(["--- ÉCONOMIE ---", "---", "---"])
            all_rows.extend(econ_rows)
    
    def _add_summary_table(self, fig, row, col):
        """Ajoute un tableau récapitulatif des indicateurs et métriques de risque"""
        if self.technical_data is None or self.technical_data.empty:
            self.logger.warning("Pas de données techniques pour le tableau récapitulatif")
            return
        
        all_rows = []
        self._add_technical_summary(all_rows)
        self._add_risk_summary(all_rows)
        self._add_economic_summary(all_rows)
        all_rows.append(["Score trading", f"{self.score}/10", self.recommendation])
        
        headers = ["Indicateur", "Valeur", "Interprétation"]
        fig.add_trace(
            go.Table(
                header=dict(values=headers, fill_color='paleturquoise', align='left'),
                cells=dict(values=list(zip(*all_rows)), fill_color='lavender', align='left')
            ),
            row=row, col=col
        )
    
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
            
            # Ajout des graphiques avec gestion d'erreur locale
            try:
                self._add_main_price_chart(fig, row=1, col=1)
            except Exception as e:
                self.logger.error(f"Erreur ajout graphique principal: {e}")
            
            try:
                self._add_trading_score_gauge(fig, row=1, col=3)
            except Exception as e:
                self.logger.error(f"Erreur ajout score gauge: {e}")
            
            try:
                self._add_market_sentiment_indicator(fig, row=2, col=3)
            except Exception as e:
                self.logger.error(f"Erreur ajout sentiment: {e}")
            
            try:
                self._add_rsi_chart(fig, row=3, col=1)
            except Exception as e:
                self.logger.error(f"Erreur ajout RSI: {e}")
            
            try:
                self._add_macd_chart(fig, row=3, col=2)
            except Exception as e:
                self.logger.error(f"Erreur ajout MACD: {e}")
            
            try:
                self._add_volume_chart(fig, row=3, col=3)
            except Exception as e:
                self.logger.error(f"Erreur ajout volume: {e}")
            
            try:
                self._add_trend_chart(fig, row=4, col=1)
            except Exception as e:
                self.logger.error(f"Erreur ajout tendance: {e}")
            
            try:
                self._add_volatility_chart(fig, row=4, col=2)
            except Exception as e:
                self.logger.error(f"Erreur ajout volatilité: {e}")
            
            try:
                self._add_predictions_chart(fig, row=4, col=3)
            except Exception as e:
                self.logger.error(f"Erreur ajout prédictions: {e}")
            
            try:
                self._add_drawdown_chart(fig, row=5, col=1)
            except Exception as e:
                self.logger.error(f"Erreur ajout drawdown: {e}")
            
            try:
                self._add_sharpe_gauge(fig, row=5, col=2)
            except Exception as e:
                self.logger.error(f"Erreur ajout sharpe: {e}")
            
            try:
                self._add_var_gauge(fig, row=5, col=3)
            except Exception as e:
                self.logger.error(f"Erreur ajout VaR: {e}")
            
            try:
                self._add_summary_table(fig, row=6, col=1)
            except Exception as e:
                self.logger.error(f"Erreur ajout tableau récapitulatif: {e}")
            
            self._update_layout(fig)
            self._save_dashboard(fig, save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Erreur création dashboard: {e}", exc_info=True)
            return None
    
    def _add_main_price_chart(self, fig, row: int, col: int):
        df = self.technical_data
        if df is None or df.empty or 'Close' not in df.columns:
            self.logger.warning("Pas de données de prix pour le graphique principal")
            return
        
        # Utiliser Scattergl pour de meilleures performances
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
        
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scattergl(
                    x=df.index,
                    y=df['BB_Upper'],
                    mode='lines',
                    name='Bande supérieure',
                    line=dict(color='rgba(255, 0, 0, 0.3)', width=1, dash='dash'),
                    showlegend=True
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scattergl(
                    x=df.index,
                    y=df['BB_Lower'],
                    mode='lines',
                    name='Bande inférieure',
                    line=dict(color='rgba(0, 255, 0, 0.3)', width=1, dash='dash'),
                    showlegend=True,
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.1)'
                ),
                row=row, col=col
            )
        
        last_price = df['Close'].iloc[-1]
        fig.add_hline(
            y=last_price,
            line_dash="dot",
            line_color="blue",
            opacity=0.5,
            annotation_text=f"Actuel: ${last_price:.2f}",
            annotation_position="bottom right",
            row=row, col=col
        )
        
        # Annotation automatique pour signaux de croisement de bandes
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            if len(df) > 1:
                last_close = df['Close'].iloc[-1]
                last_upper = df['BB_Upper'].iloc[-1]
                last_lower = df['BB_Lower'].iloc[-1]
                if last_close > last_upper:
                    fig.add_annotation(
                        x=df.index[-1],
                        y=last_close,
                        text="Prix > bande supérieure (surachat potentiel)",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40,
                        row=row, col=col
                    )
                elif last_close < last_lower:
                    fig.add_annotation(
                        x=df.index[-1],
                        y=last_close,
                        text="Prix < bande inférieure (survente potentielle)",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=40,
                        row=row, col=col
                    )
    
    def _add_trading_score_gauge(self, fig, row: int, col: int):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.score,
                title={'text': "Score Trading", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 10], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 4], 'color': "red"},
                        {'range': [4, 6], 'color': "yellow"},
                        {'range': [6, 8], 'color': "lightgreen"},
                        {'range': [8, 10], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': self.score
                    }
                }
            ),
            row=row, col=col
        )
    
    def _add_market_sentiment_indicator(self, fig, row: int, col: int):
        sentiment_score = self.market_sentiment.get('overall_sentiment', {}).get('score', 50)
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=sentiment_score,
                title={'text': "Sentiment Marché", 'font': {'size': 14}},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [0, 100]},
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': sentiment_score
                    },
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "green"}
                    ]
                }
            ),
            row=row, col=col
        )
    
    def _add_rsi_chart(self, fig, row: int, col: int):
        if 'RSI' not in self.technical_data.columns:
            self.logger.warning("Colonne RSI manquante")
            return
        df = self.technical_data
        x = df.index
        
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2),
                hovertemplate='%{x|%d %b %Y}<br>RSI: %{y:.1f}<extra></extra>'
            ),
            row=row, col=col
        )
        # Lignes horizontales
        fig.add_trace(
            go.Scatter(x=[x[0], x[-1]], y=[RSI_OVERBOUGHT, RSI_OVERBOUGHT], mode='lines',
                       line=dict(color='red', dash='dash'), showlegend=False, hoverinfo='none'),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=[x[0], x[-1]], y=[RSI_OVERSOLD, RSI_OVERSOLD], mode='lines',
                       line=dict(color='green', dash='dash'), showlegend=False, hoverinfo='none'),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=[x[0], x[-1]], y=[50, 50], mode='lines',
                       line=dict(color='gray', dash='dot'), showlegend=False, hoverinfo='none'),
            row=row, col=col
        )
        
        # Annotation si RSI en zone extrême
        last_rsi = df['RSI'].iloc[-1]
        if last_rsi > RSI_OVERBOUGHT:
            fig.add_annotation(
                x=x[-1],
                y=last_rsi,
                text="RSI surachat",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
                row=row, col=col
            )
        elif last_rsi < RSI_OVERSOLD:
            fig.add_annotation(
                x=x[-1],
                y=last_rsi,
                text="RSI survente",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=20,
                row=row, col=col
            )
    
    def _add_macd_chart(self, fig, row: int, col: int):
        df = self.technical_data
        if not all(col in df.columns for col in ['MACD', 'MACD_Signal']):
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
            colors = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogramme MACD',
                       marker_color=colors, opacity=0.6),
                row=row, col=col
            )
        
        # Annotation pour croisement MACD
        if len(df) > 1:
            last_macd = df['MACD'].iloc[-1]
            last_signal = df['MACD_Signal'].iloc[-1]
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['MACD_Signal'].iloc[-2]
            if prev_macd < prev_signal and last_macd > last_signal:
                fig.add_annotation(
                    x=df.index[-1],
                    y=last_macd,
                    text="Croisement haussier MACD",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-20,
                    row=row, col=col
                )
            elif prev_macd > prev_signal and last_macd < last_signal:
                fig.add_annotation(
                    x=df.index[-1],
                    y=last_macd,
                    text="Croisement baissier MACD",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=20,
                    row=row, col=col
                )
    
    def _add_volume_chart(self, fig, row: int, col: int):
        if 'Volume' not in self.technical_data.columns:
            self.logger.warning("Colonne Volume manquante")
            return
        
        df = self.technical_data
        colors = []
        for i in range(len(df)):
            if i == 0:
                colors.append('gray')
            else:
                colors.append('green' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else 'red')
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.7,
                   hovertemplate='%{x|%d %b %Y}<br>Volume: %{y:,.0f}<extra></extra>'),
            row=row, col=col
        )
        if 'Volume_SMA' in df.columns:
            fig.add_trace(
                go.Scattergl(x=df.index, y=df['Volume_SMA'], mode='lines', name='Volume Moyen (20j)',
                           line=dict(color='orange', width=1.5)),
                row=row, col=col
            )
    
    def _add_trend_chart(self, fig, row: int, col: int):
        df = self.technical_data
        if df is None or df.empty or 'Close' not in df.columns:
            return
        
        fig.add_trace(
            go.Scattergl(x=df.index, y=df['Close'], mode='lines', name='Prix', line=dict(color='black', width=1), opacity=0.5),
            row=row, col=col
        )
        for ma_period in SMA_PERIODS:
            col_name = f'SMA_{ma_period}'
            if col_name in df.columns:
                fig.add_trace(
                    go.Scattergl(x=df.index, y=df[col_name], mode='lines', name=f'SMA {ma_period}',
                               line=dict(color=px.colors.qualitative.Set1[SMA_PERIODS.index(ma_period) % len(px.colors.qualitative.Set1)], width=1.5)),
                    row=row, col=col
                )
        
        # Annotation pour croisement SMA
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns and len(df) > 1:
            last_50 = df['SMA_50'].iloc[-1]
            last_200 = df['SMA_200'].iloc[-1]
            prev_50 = df['SMA_50'].iloc[-2]
            prev_200 = df['SMA_200'].iloc[-2]
            if prev_50 < prev_200 and last_50 > last_200:
                fig.add_annotation(
                    x=df.index[-1],
                    y=last_50,
                    text="Croisement doré (50>200)",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-20,
                    row=row, col=col
                )
            elif prev_50 > prev_200 and last_50 < last_200:
                fig.add_annotation(
                    x=df.index[-1],
                    y=last_50,
                    text="Croisement mortel (50<200)",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=20,
                    row=row, col=col
                )
    
    def _add_volatility_chart(self, fig, row: int, col: int):
        if 'ATR' not in self.technical_data.columns:
            self.logger.warning("Colonne ATR manquante")
            return
        df = self.technical_data
        fig.add_trace(
            go.Scattergl(
                x=df.index,
                y=df['ATR'],
                mode='lines',
                name='ATR',
                line=dict(color='orange', width=2),
                hovertemplate='%{x|%d %b %Y}<br>ATR: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_predictions_chart(self, fig, row: int, col: int):
        if self.predictions_df is None or self.predictions_df.empty:
            return
        if 'Predicted_Close' not in self.predictions_df.columns:
            return
        
        future_dates = self.predictions_df.index
        future_prices = self.predictions_df['Predicted_Close'].values
        
        fig.add_trace(
            go.Scattergl(
                x=future_dates,
                y=future_prices,
                mode='lines+markers',
                name='Prévisions IA',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4),
                hovertemplate='%{x|%d %b %Y}<br>Prévision: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Ajout des intervalles de confiance si disponibles
        if 'CI_Lower' in self.predictions_df.columns and 'CI_Upper' in self.predictions_df.columns:
            fig.add_trace(
                go.Scattergl(
                    x=future_dates,
                    y=self.predictions_df['CI_Upper'],
                    mode='lines',
                    name='CI Supérieur',
                    line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                    showlegend=True
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scattergl(
                    x=future_dates,
                    y=self.predictions_df['CI_Lower'],
                    mode='lines',
                    name='CI Inférieur',
                    line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.1)',
                    showlegend=True
                ),
                row=row, col=col
            )
        
        if len(future_dates) > 0:
            fig.add_shape(
                type="line",
                x0=future_dates[0],
                x1=future_dates[-1],
                y0=self.current_price,
                y1=self.current_price,
                line=dict(dash="dot", color="blue", width=1),
                row=row, col=col
            )
            fig.add_annotation(
                x=future_dates[-1],
                y=self.current_price,
                text=f"Actuel: ${self.current_price:.2f}",
                showarrow=False,
                yshift=10,
                font=dict(size=10),
                row=row, col=col
            )
    
    def _add_drawdown_chart(self, fig, row: int, col: int):
        """Ajoute un graphique de drawdown historique"""
        if self.technical_data is None or 'Close' not in self.technical_data.columns:
            return
        df = self.technical_data
        rolling_max = df['Close'].cummax()
        drawdown = (df['Close'] - rolling_max) / rolling_max * 100
        fig.add_trace(
            go.Scattergl(
                x=df.index,
                y=drawdown,
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red', width=2),
                hovertemplate='%{x|%d %b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
        # Ligne de référence à zéro
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[0] * len(df),
                mode='lines',
                line=dict(dash='dash', color='black', width=1),
                name='Zero Line',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        if 'max_drawdown' in self.risk_metrics:
            max_dd = self.risk_metrics['max_drawdown'] * 100
            # Ligne horizontale pour le max drawdown
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=[max_dd] * len(df),
                    mode='lines',
                    line=dict(dash='dash', color='darkred', width=1),
                    name=f'Max DD: {max_dd:.1f}%',
                    showlegend=True,
                    hoverinfo='y',
                    hovertemplate=f'Max Drawdown: {max_dd:.1f}%<extra></extra>'
                ),
                row=row, col=col
            )
            # Annotation pour indiquer la valeur
            fig.add_annotation(
                x=df.index[-1],
                y=max_dd,
                text=f"Max: {max_dd:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='darkred',
                ax=0,
                ay=-30,
                row=row, col=col
            )

    def _add_sharpe_gauge(self, fig, row: int, col: int):
        """Ajoute un gauge pour le Sharpe Ratio"""
        if 'sharpe_ratio' not in self.risk_metrics:
            return
        sr = self.risk_metrics['sharpe_ratio']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sr,
                title={'text': "Sharpe Ratio", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [-2, 3]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-2, 0], 'color': "red"},
                        {'range': [0, 1], 'color': "yellow"},
                        {'range': [1, 2], 'color': "lightgreen"},
                        {'range': [2, 3], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': sr
                    }
                }
            ),
            row=row, col=col
        )

    def _add_var_gauge(self, fig, row: int, col: int):
        """Ajoute un gauge pour la Value at Risk"""
        if 'value_at_risk' not in self.risk_metrics:
            return
        var = abs(self.risk_metrics['value_at_risk'] * 100)  # Convertir en % positif pour affichage
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=var,
                title={'text': "VaR (95%) %", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 20]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 5], 'color': "green"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 15], 'color': "orange"},
                        {'range': [15, 20], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': var
                    }
                }
            ),
            row=row, col=col
        )    
    def _update_layout(self, fig):
        try:
            fig.update_layout(
                title=dict(
                    text=f'Tableau de Bord Trading - {self.symbol}',
                    font=dict(size=24, color='darkblue', family="Arial, sans-serif"),
                    x=0.5,
                    y=0.98
                ),
                height=1600,
                showlegend=True,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=100, b=50),
                plot_bgcolor='rgba(240, 240, 240, 0.5)'
            )
            
            # Définition des titres d'axes spécifiques
            axis_titles = {
                (1, 1): {"x": "Date", "y": "Prix ($)"},
                (3, 1): {"x": "Date", "y": "RSI"},
                (3, 2): {"x": "Date", "y": "MACD"},
                (3, 3): {"x": "Date", "y": "Volume"},
                (4, 1): {"x": "Date", "y": "Prix ($)"},
                (4, 2): {"x": "Date", "y": "ATR ($)"},
                (4, 3): {"x": "Date", "y": "Prix ($)"},
                (5, 1): {"x": "Date", "y": "Drawdown (%)"},
            }
            
            for (r, c), titles in axis_titles.items():
                fig.update_xaxes(title_text=titles["x"], tickangle=45, gridcolor='lightgray', showgrid=True, row=r, col=c)
                fig.update_yaxes(title_text=titles["y"], gridcolor='lightgray', showgrid=True, row=r, col=c)
                
        except Exception as e:
            self.logger.error(f"Erreur mise à jour du layout: {e}")
    
    def _save_dashboard(self, fig, save_path: str):
        try:
            os.makedirs(save_path, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            html_file = f"{save_path}/{self.symbol}_dashboard_{timestamp}.html"
            # Rendre le dashboard responsive
            fig.write_html(html_file, config={'responsive': True})
            
            png_file = f"{save_path}/{self.symbol}_dashboard_{timestamp}.png"
            fig.write_image(png_file, width=1600, height=900, scale=2)
            
            metadata = {
                'symbol': self.symbol,
                'timestamp': timestamp,
                'current_price': self.current_price,
                'score': self.score,
                'recommendation': self.recommendation,
                'files': {
                    'html': html_file,
                    'png': png_file
                }
            }
            
            metadata_file = f"{save_path}/{self.symbol}_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Dashboard sauvegardé: {html_file}")
            self.logger.info(f"Image sauvegardée: {png_file}")
            self.logger.info(f"Métadonnées sauvegardées: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde dashboard: {e}")
    
    def create_quick_overview(self) -> Optional[pd.DataFrame]:
        if not self.overview:
            return None
        overview_data = []
        key_info = [
            ('Symbole', self.symbol),
            ('Prix Actuel', f"${self.current_price:.2f}"),
            ('Score', f"{self.score:.1f}/10"),
            ('Recommandation', self.recommendation),
            ('Nom', self.overview.get('name', 'N/A')),
            ('Market Cap', self.overview.get('market_cap', 'N/A')),
            ('P/E Ratio', self.overview.get('pe_ratio', 'N/A')),
            ('Dividende', self.overview.get('dividend_yield', 'N/A')),
            ('Secteur', self.overview.get('sector', 'N/A')),
            ('52W High', f"${float(self.overview.get('52_week_high', 0)):.2f}" if self.overview.get('52_week_high') != 'N/A' else 'N/A'),
            ('52W Low', f"${float(self.overview.get('52_week_low', 0)):.2f}" if self.overview.get('52_week_low') != 'N/A' else 'N/A')
        ]
        for label, value in key_info:
            overview_data.append({'Métrique': label, 'Valeur': value})
        return pd.DataFrame(overview_data)
    
    def create_technical_summary(self) -> Optional[pd.DataFrame]:
        if self.technical_data is None or self.technical_data.empty:
            return None
        df = self.technical_data
        last_row = df.iloc[-1]
        summary = []
        if 'RSI' in last_row and not pd.isna(last_row['RSI']):
            rsi = last_row['RSI']
            rsi_status = "SURACHAT ⚠️" if rsi > RSI_OVERBOUGHT else "SURVENTE ✅" if rsi < RSI_OVERSOLD else "NEUTRE ⚖️"
            summary.append(('RSI (14j)', f"{rsi:.1f}", rsi_status))
        if 'MACD' in last_row and 'MACD_Signal' in last_row:
            macd = last_row['MACD']
            signal = last_row['MACD_Signal']
            if not pd.isna(macd) and not pd.isna(signal):
                macd_status = "HAUSSIER 📈" if macd > signal else "BAISSIER 📉"
                summary.append(('MACD', f"{macd:.4f}", macd_status))
        if 'SMA_50' in last_row and 'SMA_200' in last_row:
            sma_50 = last_row['SMA_50']
            sma_200 = last_row['SMA_200']
            if not pd.isna(sma_50) and not pd.isna(sma_200):
                ma_status = "CROISEMENT DORÉ ✅" if sma_50 > sma_200 else "CROISEMENT MORTEL ⚠️"
                summary.append(('SMA 50/200', f"{sma_50:.2f}/{sma_200:.2f}", ma_status))
        if 'ATR' in last_row and not pd.isna(last_row['ATR']):
            atr = last_row['ATR']
            atr_pct = (atr / self.current_price) * 100 if self.current_price > 0 else 0
            vol_status = ("ÉLEVÉ ⚠️" if atr_pct > ATR_HIGH_THRESHOLD 
                         else "MODÉRÉ ⚖️" if atr_pct > ATR_MODERATE_THRESHOLD 
                         else "FAIBLE ✅")
            summary.append(('Volatilité (ATR)', f"{atr_pct:.1f}%", vol_status))
        return pd.DataFrame(summary, columns=['Indicateur', 'Valeur', 'Statut'])

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
    return fig