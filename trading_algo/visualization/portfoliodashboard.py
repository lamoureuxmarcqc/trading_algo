import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Optional

from trading_algo.portfolio.portfolio import Portfolio
from trading_algo.portfolio.portfoliomanager import PortfolioManager
from trading_algo.risk.risk_manager import RiskManager
from trading_algo.strategy.market_regime_engine import MarketRegimeEngine
from trading_algo.strategy.factor_engine import FactorEngine
from trading_algo.strategy.position_sizing import PositionSizingEngine
from trading_algo.strategy.risk_overlay import RiskOverlay

logger = logging.getLogger(__name__)


class PortfolioDashboard:
    """
    Dashboard stratégique avancé — version stable production.

    Robustesse:
    - Tolérance aux données manquantes
    - Aucun crash Dash
    - Fallback intelligent
    """

    def __init__(
        self,
        portfolio: Portfolio,
        portfolio_manager: Optional[PortfolioManager] = None,
        theme: str = "plotly_dark"
    ):
        self.portfolio = portfolio
        self.manager = portfolio_manager
        self.theme = theme
        self.last_regime = None

        # Engines
        self.regime_engine = MarketRegimeEngine()
        self.factor_engine = FactorEngine()
        self.sizing_engine = PositionSizingEngine()
        self.risk_overlay = RiskOverlay()
        self.risk_manager = RiskManager()

    # =========================================================
    # 🔹 1. STRATEGIC DASHBOARD (CORE)
    # =========================================================
    def create_strategic_report(
        self,
        macro_data: Dict,
        market_indices: pd.DataFrame,
        fundamentals_map: Dict
    ) -> go.Figure:

        try:
            if not self.portfolio or not self.portfolio.positions:
                return self._create_empty_fig("Aucun portefeuille chargé")

            # --- REGIME ---
            try:
                regime, regime_score = self.regime_engine.compute_regime(
                    macro_data, market_indices
                )
            except Exception:
                regime, regime_score = "NEUTRAL", 50

            # --- QUALITY ---
            quality_scores = {}
            for t in self.portfolio.positions.keys():
                fundamentals = fundamentals_map.get(t, {})
                try:
                    score = self.factor_engine.compute_quality_score(fundamentals)
                except Exception:
                    score = 0
                quality_scores[t] = score

            # --- TARGET ALLOCATION ---
            ranked = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)

            try:
                target_alloc = self.sizing_engine.size_positions(ranked, regime)
            except Exception:
                target_alloc = {t: 1 / len(ranked) for t, _ in ranked} if ranked else {}

            # --- RISK OVERLAY ---
            try:
                perf_data = {
                    'performance': {
                        'total_pnl_pct': self.portfolio.calculate_performance({}).get("total_pnl_pct", 0)
                    }
                }
                final_target = self.risk_overlay.apply(target_alloc, perf_data)
            except Exception:
                final_target = target_alloc

            # --- CURRENT ---
            try:
                current_alloc = self.portfolio.get_allocation()
            except Exception:
                current_alloc = {}

            all_tickers = sorted(set(final_target.keys()) | set(current_alloc.keys()))

            # =========================================================
            # 🔹 FIGURE
            # =========================================================
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "table"}]
                ],
                subplot_titles=(
                    "Market Regime",
                    "Quality Scores",
                    "Allocation (Current vs AI)",
                    "Rebalancing Actions"
                ),
                vertical_spacing=0.12
            )

            # --- REGIME ---
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=regime_score,
                title={'text': regime},
                gauge={
                    'axis': {'range': [-50, 100]},
                    'bar': {'color': "#00e5ff"}
                }
            ), row=1, col=1)

            # --- QUALITY ---
            if quality_scores:
                fig.add_trace(go.Bar(
                    x=list(quality_scores.keys()),
                    y=list(quality_scores.values()),
                    marker_color="#26a69a",
                    name="Quality"
                ), row=1, col=2)

            # --- ALLOCATION ---
            fig.add_trace(go.Bar(
                x=all_tickers,
                y=[final_target.get(t, 0) * 100 for t in all_tickers],
                name="AI Target"
            ), row=2, col=1)

            fig.add_trace(go.Bar(
                x=all_tickers,
                y=[current_alloc.get(t, 0) * 100 for t in all_tickers],
                name="Current"
            ), row=2, col=1)

            # --- TABLE ---
            table_data = []
            for t in all_tickers:
                target = final_target.get(t, 0)
                actual = current_alloc.get(t, 0)
                diff = target - actual

                action = "REBALANCE" if abs(diff) > 0.05 else "KEEP"

                table_data.append([
                    t,
                    f"{actual:.1%}",
                    f"{target:.1%}",
                    action
                ])

            if table_data:
                fig.add_trace(go.Table(
                    header=dict(
                        values=["Ticker", "Current", "Target", "Action"],
                        fill_color="#263238",
                        font=dict(color="white")
                    ),
                    cells=dict(values=list(zip(*table_data)))
                ), row=2, col=2)

            fig.update_layout(
                template=self.theme,
                height=850,
                title=f"Strategic Advisor — {datetime.now().strftime('%Y-%m-%d')}",
                showlegend=True
            )

            self._check_and_export(regime, fig)

            return fig

        except Exception as e:
            logger.exception(f"Strategic report error: {e}")
            return self._create_empty_fig("Erreur stratégique")

    # =========================================================
    # 🔹 PERFORMANCE
    # =========================================================
    def create_performance_chart(self) -> go.Figure:

        try:
            if not self.portfolio.performance_history:
                return self._create_empty_fig("Aucune donnée performance")

            df = pd.DataFrame(self.portfolio.performance_history)

            df['date'] = pd.to_datetime(df.get('timestamp', df.get('date')))
            df = df.sort_values('date')

            df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce").ffill()

            rolling_max = df["total_value"].cummax()
            df["drawdown"] = (df["total_value"] - rolling_max) / rolling_max * 100

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3]
            )

            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['total_value'],
                fill="tozeroy",
                name="Equity"
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['drawdown'],
                fill="tozeroy",
                name="Drawdown"
            ), row=2, col=1)

            fig.update_layout(
                template=self.theme,
                height=600,
                hovermode="x unified"
            )

            return fig

        except Exception as e:
            logger.exception(e)
            return self._create_empty_fig("Erreur performance")

    # =========================================================
    # 🔹 VISUAL REPORT
    # =========================================================
    def create_visual_report(self) -> go.Figure:

        try:
            tickers = list(self.portfolio.positions.keys())
            prices = self.manager.get_market_prices(tickers) if self.manager else {}

            df_summary = self.portfolio.get_summary(prices)
            alloc = self.portfolio.get_allocation(prices)

            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "domain"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )

            # Allocation
            if alloc:
                fig.add_trace(go.Pie(
                    labels=list(alloc.keys()),
                    values=list(alloc.values()),
                    hole=0.4
                ), row=1, col=1)

            # P&L
            if not df_summary.empty:
                fig.add_trace(go.Bar(
                    x=df_summary["Ticker"],
                    y=df_summary["P&L %"] * 100
                ), row=1, col=2)

                fig.add_trace(go.Bar(
                    x=df_summary["Ticker"],
                    y=df_summary["P&L"]
                ), row=2, col=2)

            fig.update_layout(template=self.theme, height=850)

            return fig

        except Exception as e:
            logger.exception(e)
            return self._create_empty_fig("Erreur visual report")

    # =========================================================
    # 🔹 UTILS
    # =========================================================
    def _check_and_export(self, regime: str, fig: go.Figure):
        try:
            if self.last_regime and self.last_regime != regime:
                os.makedirs("reports", exist_ok=True)
                filename = f"reports/regime_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
                fig.write_html(filename)
                logger.warning(f"🚨 Regime change → report saved: {filename}")
            self.last_regime = regime
        except Exception:
            pass

    def _create_empty_fig(self, msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=msg,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=18)
        )
        fig.update_layout(template=self.theme)
        return fig