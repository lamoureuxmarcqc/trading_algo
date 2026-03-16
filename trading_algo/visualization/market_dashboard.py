# python trading_algo/visualization/market_dashboard.py
"""
Market-level dashboard helper returning a Plotly figure summarizing global market state.
Place this file at trading_algo/visualization/market_dashboard.py
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_algo.data.data_extraction import StockDataExtractor, MacroDataExtractor

logger = logging.getLogger(__name__)


class MarketDashboard:
    """
    Builds a consolidated market overview figure.
    Usage:
      md = MarketDashboard()
      md.load_data(...)          # supply dicts / DataFrames from your extractors
      fig = md.create_figure()
    """
    def __init__(self):
        self.indices: Dict[str, pd.DataFrame] = {}
        self.sector_perf: Optional[pd.DataFrame] = None
        self.top_movers: Optional[pd.DataFrame] = None
        self.macro: Dict[str, Any] = {}
        self.commodities: Dict[str, Any] = {}
        self.currencies: Dict[str, Any] = {}
        self.period_label = "1M"
        self.logger = logging.getLogger(__name__)

    def load_data(self,
                  indices: Dict[str, pd.DataFrame] = None,
                  sector_perf: pd.DataFrame = None,
                  top_movers: pd.DataFrame = None,
                  macro: Dict[str, Any] = None,
                  commodities: Dict[str, Any] = None,
                  currencies: Dict[str, Any] = None,
                  period_label: str = "1M"):
        """Load pre-fetched dataframes/dicts (fetching lives in data_extraction or scheduled job)."""
        self.indices = indices or {}
        self.sector_perf = sector_perf
        self.top_movers = top_movers
        self.macro = macro or {}
        self.commodities = commodities or {}
        self.currencies = currencies or {}
        self.period_label = period_label
        self.logger.info("Market data loaded for dashboard")

    # ---- small helpers to build primitives ----
    def _build_indices_panel(self):
        """Return a small horizontal subplot with index name + sparkline + pct change."""
        traces = []
        rows = []
        for name, df in self.indices.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            last = df['Close'].iloc[-1]
            pct = (df['Close'].pct_change().iloc[-1]) * 100 if len(df) > 1 else 0.0
            spark = go.Scattergl(x=df.index, y=df['Close'], mode='lines', line={'width': 1}, showlegend=False, hoverinfo='skip')
            traces.append((name, spark, last, pct))
        return traces

    def _sector_heatmap_trace(self):
        """Create a sector heatmap from self.sector_perf (expects columns: sector, perf)."""
        if self.sector_perf is None or self.sector_perf.empty:
            return None
        df = self.sector_perf.copy()
        # pivot into grid by grouping heuristically (simple lexicographic)
        df['row'] = (pd.Categorical(df['sector']).codes // 6)
        df['col'] = (pd.Categorical(df['sector']).codes % 6)
        z = df.pivot(index='row', columns='col', values='perf')
        text = df.pivot(index='row', columns='col', values='sector')
        heat = go.Heatmap(z=z.values, x=list(z.columns), y=list(z.index), text=text.values, colorscale='RdYlGn', reversescale=False, showscale=True)
        return heat

    def _top_movers_table(self, n: int = 10):
        if self.top_movers is None or self.top_movers.empty:
            return None
        df = self.top_movers.head(n)
        header = dict(values=list(df.columns), fill_color='paleturquoise')
        cells = dict(values=[df[col].tolist() for col in df.columns], fill_color='lavender')
        return go.Table(header=header, cells=cells)

    # ---- main figure assembly ----
    def create_figure(self) -> go.Figure:
        """Compose the market overview figure (multi-panel)."""
        fig = make_subplots(
            rows=3, cols=3,
            specs=[
                [{"type": "xy", "colspan": 2}, {"type": "xy", "rowspan": 2}, {"type": "domain"}],
                [None, None, {"type": "table"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            ],
            subplot_titles=("Indices", "Sector Heatmap", "Top Movers",
                            "", "", "Commodities & FX")
        )

        # Indices panel: stack small sparklines vertically inside one subplot via traces offset
        traces = self._build_indices_panel()
        row_idx, col_idx = 1, 1
        y_offset = 0
        for name, spark, last, pct in traces:
            fig.add_trace(spark, row=row_idx, col=col_idx)
            fig.add_annotation(xref='paper', yref='paper',
                               x=0.04, y=0.95 - y_offset,
                               text=f"<b>{name}</b> {last:.2f} ({pct:+.2f}%)",
                               showarrow=False, font={'size':10})
            y_offset += 0.08

        # Sector heatmap
        heat = self._sector_heatmap_trace()
        if heat is not None:
            fig.add_trace(heat, row=1, col=2)

        # Top movers
        table = self._top_movers_table()
        if table is not None:
            fig.add_trace(table, row=1, col=3)

        # Commodities & FX summary
        # create small traces text annotations for each commodity
        cm_x = 3; cm_row = 3
        ypos = 0.95
        for name, val in (self.commodities or {}).items():
            fig.add_annotation(xref='paper', yref='paper', x=0.74, y=ypos, text=f"{name}: {val.get('price', 'N/A')}", showarrow=False, font={'size':11})
            ypos -= 0.06
        for name, val in (self.currencies or {}).items():
            fig.add_annotation(xref='paper', yref='paper', x=0.74, y=ypos, text=f"{name}: {val.get('rate', 'N/A')}", showarrow=False, font={'size':11})
            ypos -= 0.06

        fig.update_layout(height=900, showlegend=False, template='plotly_white', title_text=f"Market Overview — {self.period_label}")
        return fig
