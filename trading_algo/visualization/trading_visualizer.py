import logging
from typing import List, Dict, Optional, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Use central settings where applicable
from trading_algo import settings

logger = logging.getLogger(__name__)


class AdvancedTradingDashboard:
    """Dashboard consolidé : mini-views, gauges, summary table and full dashboard figure."""

    def __init__(self, symbol: str = "ASSET"):
        self.symbol = symbol.upper()
        self.logger = logging.getLogger(__name__)
        self.risk_metrics: Dict[str, Any] = {}
        self.technical_data: Optional[pd.DataFrame] = None
        self.predictions_df: Optional[pd.DataFrame] = None
        self.recommendation: str = "NEUTRE"
        self.score: float = 5.0

    def load_data(
        self,
        technical_data: Optional[pd.DataFrame] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Injecte les données avant génération du dashboard."""
        self.technical_data = technical_data.copy() if technical_data is not None else pd.DataFrame()
        self.predictions_df = predictions_df.copy() if predictions_df is not None else pd.DataFrame()
        self.risk_metrics = risk_metrics or {}
        if self.technical_data is not None and not self.technical_data.empty:
            try:
                # set current price if available
                self.current_price = float(self.technical_data["Close"].iloc[-1])
            except Exception:
                self.current_price = 0.0

    # --- MINI & COMPARISON VIEWS ---

    def create_mini_dashboard(self) -> go.Figure:
        """Compact quick-view showing price and optional IA predictions."""
        data = self.technical_data
        preds = self.predictions_df
        if data is None or data.empty:
            return self._create_empty_fig("Données historiques indisponibles")

        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(x=data.index, y=data["Close"], mode="lines", name="Prix",
                         line=dict(color="#1f77b4", width=2))
        )

        if preds is not None and not preds.empty:
            # Expect predictions_df indexed by date and column 'Predicted_Close' or similar
            pred_col = "Predicted_Close" if "Predicted_Close" in preds.columns else preds.columns[0]
            fig.add_trace(
                go.Scattergl(x=preds.index, y=preds[pred_col], mode="lines+markers", name="IA",
                             line=dict(color="#d62728", width=2, dash="dash"))
            )

        fig.update_layout(title=f"{self.symbol} - Vue Rapide", template="plotly_white", height=400)
        return fig

    @staticmethod
    def create_comparison_dashboard(symbols: List[str], data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
        """Compare multiple symbols (base 100 performance, ATR, RSI)."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Performance Rel.", "Volatilité (ATR)", "RSI", "Volume Norm."),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        colors = px.colors.qualitative.Set3
        for idx, s in enumerate(symbols):
            df = data_dict.get(s)
            if df is None or df.empty:
                continue
            color = colors[idx % len(colors)]
            if "Close" in df.columns and len(df) > 0:
                norm = df["Close"] / df["Close"].iloc[0] * 100
                fig.add_trace(go.Scattergl(x=df.index, y=norm, name=s, line=dict(color=color)), row=1, col=1)
            if "ATR" in df.columns:
                fig.add_trace(go.Scattergl(x=df.index, y=df["ATR"], showlegend=False, line=dict(color=color)), row=1, col=2)
            if "RSI" in df.columns:
                fig.add_trace(go.Scattergl(x=df.index, y=df["RSI"], showlegend=False, line=dict(color=color)), row=2, col=1)
        fig.update_layout(title="Comparaison Multi-Actions", height=900, template="plotly_white")
        return fig

    # --- GAUGES & TABLE HELPERS ---

    def _format_color_square(self, color: str) -> str:
        return f'<span style="color:{color}; font-size:20px;">■</span>'

    def _get_status_color(self, value: float, bands: List[tuple], colors: List[str]) -> str:
        for i, (low, high) in enumerate(bands):
            if low <= value < high:
                return colors[i]
        return colors[-1] if value >= bands[-1][1] else colors[0]

    def _add_sharpe_gauge(self, fig: go.Figure, row: int, col: int) -> None:
        sr = float(self.risk_metrics.get("sharpe_ratio", 0.0))
        bands = getattr(settings, "SHARPE_BANDS", [(-2, 0), (0, 1), (1, 2), (2, 3)])
        colors = getattr(settings, "SHARPE_COLORS", ["#d9534f", "#f0ad4e", "#5cb85c", "#2b7a2b"])
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sr,
                title={"text": "Sharpe Ratio", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [bands[0][0], bands[-1][1]]},
                    "bar": {"color": "darkblue"},
                    "steps": [{"range": b, "color": c} for b, c in zip(bands, colors)],
                },
            ),
            row=row,
            col=col,
        )

    def _add_var_gauge(self, fig: go.Figure, row: int, col: int) -> None:
        var = abs(float(self.risk_metrics.get("value_at_risk", 0.0)) * 100.0)
        bands = getattr(settings, "VAR_BANDS", [(0, 2), (2, 5), (5, 10), (10, 50)])
        colors = getattr(settings, "VAR_COLORS", ["#2b7a2b", "#5cb85c", "#f0ad4e", "#d9534f"])
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=var,
                number={"suffix": "%"},
                title={"text": "VaR (95%) %", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, bands[-1][1]]},
                    "bar": {"color": "darkred"},
                    "steps": [{"range": b, "color": c} for b, c in zip(bands, colors)],
                },
            ),
            row=row,
            col=col,
        )

    def _add_summary_table(self, fig: go.Figure, row: int, col: int) -> None:
        sr = float(self.risk_metrics.get("sharpe_ratio", 0.0))
        sr_color = self._get_status_color(sr, getattr(settings, "SHARPE_BANDS", [(-2, 0), (0, 1), (1, 2), (2, 3)]),
                                         getattr(settings, "SHARPE_COLORS", ["#d9534f", "#f0ad4e", "#5cb85c", "#2b7a2b"]))
        var_val = abs(float(self.risk_metrics.get("value_at_risk", 0.0)) * 100.0)
        var_color = self._get_status_color(var_val, getattr(settings, "VAR_BANDS", [(0, 2), (2, 5), (5, 10), (10, 50)]),
                                           getattr(settings, "VAR_COLORS", ["#2b7a2b", "#5cb85c", "#f0ad4e", "#d9534f"]))

        rows = [
            ["<b>Sharpe Ratio</b>", f"{sr:.2f}", f"{self._format_color_square(sr_color)}"],
            ["<b>VaR (95%)</b>", f"{var_val:.2f}%", f"{self._format_color_square(var_color)}"],
            ["---", "---", "---"],
            ["SCORE FINAL", f"<b>{self.score}/10</b>", f"<b>{self.recommendation}</b>"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=["Indicateur", "Valeur", "Interprétation"], fill_color="paleturquoise", align="left"),
                cells=dict(values=list(zip(*rows)), fill_color="lavender", align="left"),
            ),
            row=row,
            col=col,
        )

    # --- FULL DASHBOARD CREATION ---

    def create_full_dashboard(self) -> go.Figure:
        """Construit et renvoie la figure complète (prix, gauges, table)."""
        data = self.technical_data
        risk = self.risk_metrics or {}
        if data is None or data.empty:
            return self._create_empty_fig("Données historiques indisponibles")

        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[[{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]],
            vertical_spacing=0.08,
            subplot_titles=(f"Analyse {self.symbol}", "Sharpe Ratio", "VaR (95%)", "Synthèse Risque"),
        )

        # Price trace (main)
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Prix", line=dict(color="#1f77b4")), row=1, col=1)

        # Gauges
        self._add_sharpe_gauge(fig, row=2, col=1)
        self._add_var_gauge(fig, row=2, col=2)

        # Summary table
        self._add_summary_table(fig, row=3, col=1)

        fig.update_layout(height=1200, template="plotly_white", showlegend=False)
        return fig

    # alias for compatibility
    def create_dashboard(self) -> go.Figure:
        return self.create_full_dashboard()

    # --- UTILS ---

    def _create_empty_fig(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title=message, template="plotly_white", height=300)
        return fig