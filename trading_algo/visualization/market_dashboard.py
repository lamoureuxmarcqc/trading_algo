from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class MarketDashboard:
    def __init__(self):
        self.indices: Dict[str, pd.DataFrame] = {}
        self.sector_perf: pd.DataFrame = pd.DataFrame()
        self.top_movers: pd.DataFrame = pd.DataFrame()
        self.commodities: Dict[str, Any] = {}
        self.currencies: Dict[str, Any] = {}
        self.macro: Dict[str, Any] = {}
        self.period_label = "1M"

    def load_data(self, **kwargs):
        """Charge les données et gère les valeurs par défaut."""
        self.indices = kwargs.get('indices', {})
        self.sector_perf = kwargs.get('sector_perf', pd.DataFrame())
        self.top_movers = kwargs.get('top_movers', pd.DataFrame())
        self.commodities = kwargs.get('commodities', {})
        self.currencies = kwargs.get('currencies', {})
        self.macro = kwargs.get('macro', {})
        self.period_label = kwargs.get('period_label', "1M")
        
        # Nettoyage automatique des données vides
        if isinstance(self.sector_perf, list):
            self.sector_perf = pd.DataFrame(self.sector_perf)
            
        logger.info(f"Market Dashboard prêt : {len(self.indices)} indices chargés.")

    def _get_color(self, val: float) -> str:
        # Palette harmonisée avec votre dashboard de symboles
        return "#2b7a2b" if val >= 0 else "#d9534f"

    def create_figure(self) -> go.Figure:
        """Crée une vue 'Global Market Overview' 2x2"""
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.55, 0.45],
            row_heights=[0.5, 0.5],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            specs=[
                [{"type": "xy"}, {"type": "treemap"}],
                [{"type": "table"}, {"type": "table"}]
            ],
            subplot_titles=(
                f"📈 Comparaison Indices ({self.period_label})", 
                "📊 Force Relative des Secteurs", 
                "🔥 Top 10 Movers (S&P 500)", 
                "🌐 Macro, FX & Commodities"
            )
        )

        # 1. Panel Indices : Performance Relative (Base 100)
        # On normalise pour comparer des indices à prix très différents (ex: BTC vs SP500)
        for name, df in self.indices.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                norm_price = (df['Close'] / df['Close'].iloc[0]) * 100
                change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=norm_price, name=name,
                    line=dict(width=2),
                    hovertemplate=f"<b>{name}</b>: %{{y:.2f}} (Perf: {change:+.2f}%)<extra></extra>"
                ), row=1, col=1)

        # 2. Treemap Secteurs
        if not self.sector_perf.empty:
            fig.add_trace(go.Treemap(
                labels=self.sector_perf['sector'],
                parents=[""] * len(self.sector_perf),
                values=[1] * len(self.sector_perf),
                marker=dict(
                    colors=self.sector_perf['perf'], 
                    colorscale='RdYlGn', 
                    cmid=0,
                    showscale=True,
                    colorbar=dict(title="Perf %", thickness=15, x=1.05)
                ),
                text=self.sector_perf['perf'].apply(lambda x: f"{x:+.2f}%"),
                textinfo="label+text",
                hoverinfo="label+text+percent parent"
            ), row=1, col=2)

        # 3. Table Top Movers (Actions individuelles)
        movers_headers = ["Symbole", "Perf %", "Secteur"]
        movers_data = [self.top_movers['symbol'], 
                       self.top_movers['perf'].map('{:+.2f}%'.format),
                       self.top_movers['sector']] if not self.top_movers.empty else [[], [], []]

        fig.add_trace(go.Table(
            header=dict(values=[f"<b>{h}</b>" for h in movers_headers],
                        fill_color='#2c3e50', font=dict(color='white', size=12), align='left'),
            cells=dict(values=movers_data,
                       fill_color='#f8f9fa', font=dict(color='black', size=11), align='left', height=28)
        ), row=2, col=1)

        # 4. Table Macro Consolidée
        all_rows = self._prepare_macro_rows()
        
        fig.add_trace(go.Table(
            header=dict(values=["<b>Indicateur</b>", "<b>Valeur</b>", "<b>Variation</b>"],
                        fill_color='#34495e', font=dict(color='white', size=12), align='left'),
            cells=dict(values=list(zip(*all_rows)),
                       fill_color='#f8f9fa', font=dict(color='black', size=11), align='left', height=28)
        ), row=2, col=2)

        # Layout Final
        fig.update_layout(
            height=900,
            template='plotly_white', # On passe en blanc pour matcher le fond de vos onglets Dash
            margin=dict(t=80, b=40, l=40, r=40),
            legend=dict(orientation="h", y=1.08, x=0)
        )
        return fig

    def _prepare_macro_rows(self) -> List[List[str]]:
        rows = []
        # Fusion des sources (Macro, FX, Commodities)
        data_sources = [
            ("📊 Macro", self.macro),
            ("💱 FX", self.currencies),
            ("📦 Comm.", self.commodities)
        ]
        
        for prefix, source in data_sources:
            for k, v in source.items():
                if isinstance(v, dict):
                    val = v.get('price', v.get('value', v.get('rate', 'N/A')))
                    chg = v.get('change', '-')
                    rows.append([f"{prefix} {k}", str(val), str(chg)])
                else:
                    rows.append([f"{prefix} {k}", str(v), "-"])
        
        return rows if rows else [["Aucune donnée", "-", "-"]]