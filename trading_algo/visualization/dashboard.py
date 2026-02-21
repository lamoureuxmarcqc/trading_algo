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
        self.score = score
        self.recommendation = recommendation

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
                    if rsi < 30:
                        tech_score += 2
                    elif rsi > 70:
                        tech_score -= 2
                if 'MACD' in last_row and 'MACD_Signal' in last_row:
                    if last_row['MACD'] > last_row['MACD_Signal']:
                        tech_score += 1
                if 'SMA_50' in last_row and 'SMA_200' in last_row:
                    if last_row['SMA_50'] > last_row['SMA_200']:
                        tech_score += 1
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

    def _add_summary_table(self, fig, row, col):
        """Ajoute un tableau récapitulatif des indicateurs et métriques de risque"""
        try:
            df = self.technical_data
            last_row = df.iloc[-1] if df is not None and not df.empty else {}
            
            # Section indicateurs techniques
            tech_rows = []
            tech_rows.append(["Prix actuel", f"${self.current_price:.2f}", "-"])
            
            if 'RSI' in last_row:
                rsi = last_row['RSI']
                interpretation = "Surachat ⚠️" if rsi > 70 else "Survente ✅" if rsi < 30 else "Neutre ⚖️"
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
                vol_status = "Élevée ⚠️" if atr_pct > 3 else "Modérée ⚖️" if atr_pct > 1.5 else "Faible ✅"
                tech_rows.append(["Volatilité (ATR%)", f"{atr_pct:.1f}%", vol_status])
            
            # Section métriques de risque
            risk_rows = []
            if self.risk_metrics:
                if 'sharpe_ratio' in self.risk_metrics:
                    sr = self.risk_metrics['sharpe_ratio']
                    sr_status = "Bon" if sr > 1 else "Médiocre" if sr > 0 else "Négatif"
                    risk_rows.append(["Sharpe Ratio", f"{sr:.2f}", sr_status])
                
                if 'max_drawdown' in self.risk_metrics:
                    mdd = self.risk_metrics['max_drawdown'] * 100
                    risk_rows.append(["Max Drawdown", f"{mdd:.1f}%", ""])
                
                if 'atr_stop' in self.risk_metrics and self.risk_metrics['atr_stop']:
                    stop = self.risk_metrics['atr_stop']
                    risk_rows.append(["Stop ATR (2x)", f"${stop:.2f}", ""])
                
                if 'stop_loss_levels' in self.risk_metrics:
                    for horizon, stop in self.risk_metrics['stop_loss_levels'].items():
                        risk_rows.append([f"Stop {horizon}", f"${stop:.2f}", ""])
                if 'take_profit_levels' in self.risk_metrics:
                    for horizon, tp in self.risk_metrics['take_profit_levels'].items():
                        risk_rows.append([f"Target {horizon}", f"${tp:.2f}", ""])
            
            # --- NOUVELLE SECTION : Indicateurs économiques ---
            econ_rows = []
            if self.macro_data and 'economic_indicators' in self.macro_data:
                econ = self.macro_data['economic_indicators']
                for name, data in econ.items():
                    if isinstance(data, dict) and 'value' in data:
                        value = data['value']
                        unit = data.get('unit', '')
                        # Formater joliment
                        if isinstance(value, (int, float)):
                            value_str = f"{value:,.2f} {unit}".strip()
                        else:
                            value_str = f"{value} {unit}".strip()
                        econ_rows.append([name, value_str, ""])
            # --- NOUVELLE SECTION : Indicateurs économiques ---
            econ_rows = []
            if self.macro_data and 'economic_indicators' in self.macro_data:
                econ = self.macro_data['economic_indicators']
                for name, data in econ.items():
                    if isinstance(data, dict) and 'value' in data:
                        value = data['value']
                        unit = data.get('unit', '')
                        # Formater joliment
                        if isinstance(value, (int, float)):
                            value_str = f"{value:,.2f} {unit}".strip()
                        else:
                            value_str = f"{value} {unit}".strip()
                        econ_rows.append([name, value_str, ""])
        
            # Fusion des sections
            all_rows = tech_rows + [["---", "---", "---"]] + risk_rows
            if econ_rows:
                all_rows += [["--- ÉCONOMIE ---", "", ""]] + econ_rows
            all_rows.append(["Score trading", f"{self.score}/10", self.recommendation])
        
            headers = ["Indicateur", "Valeur", "Interprétation"]
            fig.add_trace(
                go.Table(
                    header=dict(values=headers, fill_color='paleturquoise', align='left'),
                    cells=dict(values=list(zip(*all_rows)), fill_color='lavender', align='left')
                ),
                row=row, col=col
            )
        except Exception as e:
            self.logger.error(f"Erreur ajout table de sommaire: {e}")

    def create_main_dashboard(self, save_path: str = "dashboards") -> Optional[go.Figure]:
        """Crée le tableau de bord principal interactif"""
        if self.technical_data is None or self.technical_data.empty:
            self.logger.error("Données techniques manquantes")
            return None
        
        try:
            fig = make_subplots(
                rows=5, cols=3,
                specs=[
                    [{'type': 'scatter', 'rowspan': 2, 'colspan': 2}, None, {'type': 'indicator'}],
                    [None, None, {'type': 'indicator'}],
                    [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}],
                    [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
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
                    'Prévisions IA'
                ),
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            
            self._add_main_price_chart(fig, row=1, col=1)
            self._add_trading_score_gauge(fig, row=1, col=3)
            self._add_market_sentiment_indicator(fig, row=2, col=3)
            self._add_rsi_chart(fig, row=3, col=1)
            self._add_macd_chart(fig, row=3, col=2)
            self._add_volume_chart(fig, row=3, col=3)
            self._add_trend_chart(fig, row=4, col=1)
            self._add_volatility_chart(fig, row=4, col=2)
            self._add_predictions_chart(fig, row=4, col=3)
            self._add_summary_table(fig, row=5, col=1)
            
            self._update_layout(fig)
            self._save_dashboard(fig, save_path)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Erreur création dashboard: {e}", exc_info=True)
            return None
    
    def _add_main_price_chart(self, fig, row: int, col: int):
        df = self.technical_data
        fig.add_trace(
            go.Scatter(
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
                go.Scatter(
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
                go.Scatter(
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
            return
        df = self.technical_data
        x = df.index
        fig.add_trace(
            go.Scatter(
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
            go.Scatter(x=[x[0], x[-1]], y=[70, 70], mode='lines',
                       line=dict(color='red', dash='dash'), showlegend=False, hoverinfo='none'),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=[x[0], x[-1]], y=[30, 30], mode='lines',
                       line=dict(color='green', dash='dash'), showlegend=False, hoverinfo='none'),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=[x[0], x[-1]], y=[50, 50], mode='lines',
                       line=dict(color='gray', dash='dot'), showlegend=False, hoverinfo='none'),
            row=row, col=col
        )
    
    def _add_macd_chart(self, fig, row: int, col: int):
        if all(col in self.technical_data.columns for col in ['MACD', 'MACD_Signal']):
            df = self.technical_data
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue', width=2)),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red', width=1.5)),
                row=row, col=col
            )
            if 'MACD_Histogram' in df.columns:
                colors = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
                fig.add_trace(
                    go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogramme MACD',
                           marker_color=colors, opacity=0.6),
                    row=row, col=col
                )
    
    def _add_volume_chart(self, fig, row: int, col: int):
        if 'Volume' in self.technical_data.columns:
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
                    go.Scatter(x=df.index, y=df['Volume_SMA'], mode='lines', name='Volume Moyen (20j)',
                               line=dict(color='orange', width=1.5)),
                    row=row, col=col
                )
    
    def _add_trend_chart(self, fig, row: int, col: int):
        df = self.technical_data
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Prix', line=dict(color='black', width=1), opacity=0.5),
            row=row, col=col
        )
        for ma_period, color in [(20, 'blue'), (50, 'red'), (200, 'green')]:
            col_name = f'SMA_{ma_period}'
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col_name], mode='lines', name=f'SMA {ma_period}',
                               line=dict(color=color, width=1.5)),
                    row=row, col=col
                )
    
    def _add_volatility_chart(self, fig, row: int, col: int):
        if 'ATR' in self.technical_data.columns:
            df = self.technical_data
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR',
                           line=dict(color='orange', width=2),
                           hovertemplate='%{x|%d %b %Y}<br>ATR: %{y:.2f}<extra></extra>'),
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
            go.Scatter(
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

    def _update_layout(self, fig):
        try:
            fig.update_layout(
                title=dict(
                    text=f'Tableau de Bord Trading - {self.symbol}',
                    font=dict(size=24, color='darkblue', family="Arial, sans-serif"),
                    x=0.5,
                    y=0.98
                ),
                height=1400,
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
            
            subplots_with_axes = [
                (1, 1, True),
                (3, 1, True),
                (3, 2, True),
                (3, 3, True),
                (4, 1, True),
                (4, 2, True),
                (4, 3, True),
            ]
            
            for row, col, has_title in subplots_with_axes:
                fig.update_xaxes(
                    title_text="Date" if has_title else None,
                    tickangle=45,
                    gridcolor='lightgray',
                    showgrid=True,
                    row=row, col=col
                )
                fig.update_yaxes(
                    gridcolor='lightgray',
                    showgrid=True,
                    row=row, col=col
                )
        except Exception as e:
            self.logger.error(f"Erreur mise à jour du layout: {e}")
    
    def _save_dashboard(self, fig, save_path: str):
        try:
            os.makedirs(save_path, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            html_file = f"{save_path}/{self.symbol}_dashboard_{timestamp}.html"
            fig.write_html(html_file)
            
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
            rsi_status = "SURACHAT ⚠️" if rsi > 70 else "SURVENTE ✅" if rsi < 30 else "NEUTRE ⚖️"
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
            vol_status = "ÉLEVÉ ⚠️" if atr_pct > 3 else "MODÉRÉ ⚖️" if atr_pct > 1.5 else "FAIBLE ✅"
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
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Prix historique',
            line=dict(color='blue', width=2)
        ))
        if predictions_df is not None and not predictions_df.empty and 'Predicted_Close' in predictions_df.columns:
            fig.add_trace(go.Scatter(
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
                    go.Scatter(x=data.index, y=normalized, name=symbol, line=dict(color=color)),
                    row=1, col=1
                )
            
            if 'ATR' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['ATR'], name=symbol, line=dict(color=color), showlegend=False),
                    row=1, col=2
                )
            
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['RSI'], name=symbol, line=dict(color=color), showlegend=False),
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
        height=800,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    return fig