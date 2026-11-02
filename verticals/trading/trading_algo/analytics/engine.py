import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Dict, Optional, Any

# On importe les réglages centralisés
from trading_algo import settings

class AdvancedTradingDashboard:
    def __init__(self, symbol: str = "ASSET"):
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)

    def _get_status_color(self, value: float, bands: list, colors: list) -> str:
        """Détermine la couleur selon les réglages du settings.py"""
        for i, (low, high) in enumerate(bands):
            if low <= value < high:
                return colors[i]
        return colors[-1] if value >= bands[-1][1] else colors[0]

    def create_full_dashboard(self, data: pd.DataFrame, risk_metrics: dict):
        if data is None or data.empty:
            return go.Figure().update_layout(title="Données manquantes")

        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]],
            vertical_spacing=0.1,
            subplot_titles=(f"Analyse {self.symbol}", "Sharpe Ratio", "VaR (95%)", "Synthèse Risque")
        )

        # 1. Prix
        fig.add_trace(go.Scattergl(x=data.index, y=data['Close'], name="Prix"), row=1, col=1)

        # 2. Jauge Sharpe (Utilise settings.SHARPE_BANDS et COLORS)
        sr = risk_metrics.get('sharpe_ratio', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=sr,
            gauge={
                'axis': {'range': [settings.SHARPE_BANDS[0][0], settings.SHARPE_BANDS[-1][1]]},
                'bar': {'color': "white"},
                'steps': [{'range': b, 'color': c} for b, c in zip(settings.SHARPE_BANDS, settings.SHARPE_COLORS)]
            }
        ), row=2, col=1)

        # 3. Jauge VaR (Utilise settings.VAR_BANDS et COLORS)
        var_val = abs(risk_metrics.get('value_at_risk', 0) * 100)
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=var_val,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, settings.VAR_BANDS[-1][1]]},
                'bar': {'color': "white"},
                'steps': [{'range': b, 'color': c} for b, c in zip(settings.VAR_BANDS, settings.VAR_COLORS)]
            }
        ), row=2, col=2)

        # 4. Tableau (Utilise les labels de settings.py)
        sr_color = self._get_status_color(sr, settings.SHARPE_BANDS, settings.SHARPE_COLORS)
        var_color = self._get_status_color(var_val, settings.VAR_BANDS, settings.VAR_COLORS)

        summary_table = go.Table(
            header=dict(values=["Indicateur", "Valeur", "Statut"], fill_color='#2c3e50', font=dict(color='white')),
            cells=dict(values=[
                ["Sharpe Ratio", "VaR (95%)", "Volatilité"],
                [f"{sr:.2f}", f"{var_val:.2f}%", f"{risk_metrics.get('volatility', 0)*100:.1f}%"],
                [f"Color: {sr_color}", f"Color: {var_color}", "N/A"]
            ])
        )
        fig.add_trace(summary_table, row=3, col=1)

        fig.update_layout(height=1000, template='plotly_white')
        return fig