"""
PortfolioDashboard - Couche de visualisation uniquement (16 avril 2026)
Responsabilité : Générer les figures Plotly et les composants UI à partir des données fournies.
Aucune logique métier lourde ici (déléguée à PortfolioManager).
"""
import os
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Optional, Any
from dash import Input, Output, State, html, dcc


from trading_algo.portfolio.portfolio import Portfolio
from trading_algo.portfolio.portfoliomanager import PortfolioManager

logger = logging.getLogger(__name__)

class PortfolioDashboard:
    """
    Couche de présentation / visualisation du portefeuille.
    Cette classe ne fait que transformer les données en graphiques et composants Dash.
    """

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        portfolio_manager: Optional[PortfolioManager] = None,
        theme: str = "plotly_white"   # Changé en plotly_white pour meilleure lisibilité par défaut
    ):
        self.portfolio = portfolio
        self.manager = portfolio_manager
        self.theme = theme
        self.last_regime = None

    # =========================================================
    # 1. RAPPORT STRATÉGIQUE PRINCIPAL (Conseiller IA)
    # =========================================================
    def create_strategic_report(
        self,
        macro_data: Dict = None,
        market_indices: pd.DataFrame = None,
        fundamentals_map: Dict = None
    ) -> go.Figure:
        """Crée le graphique principal du tableau de bord stratégique."""
        if not self.portfolio or not getattr(self.portfolio, 'positions', None):
            return self._create_empty_fig("Aucun portefeuille chargé ou aucune position.")

        macro_data = macro_data or {}
        if market_indices is None or (isinstance(market_indices, pd.DataFrame) and market_indices.empty):
            market_indices = pd.DataFrame()
        fundamentals_map = fundamentals_map or {}

        try:
            # Les données métier (régime, quality, allocation cible) doivent venir du Manager
            # Ici on simule ou on appelle le manager si disponible
            regime = "NEUTRAL"
            regime_score = 50
            quality_scores = {}
            target_alloc = {}
            current_alloc = self.portfolio.get_allocation() if hasattr(self.portfolio, 'get_allocation') else {}

            if self.manager:
                try:
                    regime_info = self.manager.get_market_regime(macro_data)
                    regime = regime_info.get("regime", "NEUTRAL")
                    regime_score = regime_info.get("score", 50)

                    # Quality scores et allocation cible via le manager
                    quality_scores = self.manager.get_quality_scores(self.portfolio, fundamentals_map)
                    target_alloc = self.manager.get_target_allocation(self.portfolio, model="buffett")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'appel au manager dans strategic report: {e}")

            # Préparation des données pour le graphique
            all_tickers = sorted(set(target_alloc.keys()) | set(current_alloc.keys()))

            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "table"}]],
                subplot_titles=("Régime de Marché", "Scores de Qualité", 
                               "Allocation Cible vs Actuelle", "Actions de Rééquilibrage"),
                vertical_spacing=0.14
            )

            # Gauge Régime
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=regime_score,
                title={"text": regime},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#00e5ff"}},
                number={"font": {"size": 32}}
            ), row=1, col=1)

            # Quality Scores
            if quality_scores:
                fig.add_trace(go.Bar(
                    x=list(quality_scores.keys()),
                    y=list(quality_scores.values()),
                    marker_color="#26a69a",
                    name="Quality Score"
                ), row=1, col=2)

            # Allocation Comparaison
            fig.add_trace(go.Bar(
                x=all_tickers,
                y=[target_alloc.get(t, 0) * 100 for t in all_tickers],
                name="Cible IA",
                marker_color="#1e88e5"
            ), row=2, col=1)

            fig.add_trace(go.Bar(
                x=all_tickers,
                y=[current_alloc.get(t, 0) * 100 for t in all_tickers],
                name="Actuelle",
                marker_color="#ef5350"
            ), row=2, col=1)

            # Tableau des actions
            table_data = []
            for t in all_tickers:
                target = target_alloc.get(t, 0)
                actual = current_alloc.get(t, 0)
                diff = target - actual
                action = "REBALANCE" if abs(diff) > 0.05 else "MAINTENIR"
                table_data.append([t, f"{actual:.1%}", f"{target:.1%}", action])

            if table_data:
                fig.add_trace(go.Table(
                    header=dict(values=["Ticker", "Actuel", "Cible", "Action"],
                                fill_color="#263238", font=dict(color="white")),
                    cells=dict(values=list(zip(*table_data)),
                               align="left",
                               fill_color="#f8f9fa")
                ), row=2, col=2)

            fig.update_layout(
                template=self.theme,
                height=860,
                title=f"Strategic Advisor — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            self._check_and_export(regime, fig)
            return fig

        except Exception as e:
            logger.exception(f"Erreur create_strategic_report: {e}")
            return self._create_empty_fig("Erreur lors de la génération du rapport stratégique")

    # =========================================================
    # 2. RAPPORT VISUEL / TECHNIQUE
    # =========================================================
    def create_visual_report(self) -> go.Figure:
        """Rapport visuel (pie + barres P&L)."""
        if not self.portfolio or not getattr(self.portfolio, 'positions', None):
            return self._create_empty_fig("Aucune position à afficher")

        try:
            prices = self.manager.get_market_prices(list(self.portfolio.positions.keys())) if self.manager else {}
            df_summary = self.portfolio.get_summary(prices) if hasattr(self.portfolio, 'get_summary') else pd.DataFrame()
            alloc = self.portfolio.get_allocation(prices) if hasattr(self.portfolio, 'get_allocation') else {}

            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "domain"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]],
                subplot_titles=("Répartition du Portefeuille", "P&L par Position (%)", 
                               "P&L Absolu ($)", "Performance")
            )

            # Pie Allocation
            if alloc:
                fig.add_trace(go.Pie(
                    labels=list(alloc.keys()),
                    values=list(alloc.values()),
                    hole=0.45,
                    textinfo="label+percent"
                ), row=1, col=1)

            # Barres P&L
            if not df_summary.empty:
                fig.add_trace(go.Bar(
                    x=df_summary.get("Ticker", []),
                    y=df_summary.get("P&L %", []) * 100,
                    marker_color=df_summary.get("P&L %", []).apply(lambda x: "green" if x > 0 else "red")
                ), row=1, col=2)

                fig.add_trace(go.Bar(
                    x=df_summary.get("Ticker", []),
                    y=df_summary.get("P&L", []),
                    marker_color="#66bb6a"
                ), row=2, col=1)

            fig.update_layout(
                template=self.theme,
                height=820,
                showlegend=False
            )
            return fig

        except Exception as e:
            logger.exception(f"Erreur create_visual_report: {e}")
            return self._create_empty_fig("Erreur lors de la génération du rapport visuel")

    # =========================================================
    # 3. RENDU DES RÉSULTATS D'OPTIMISATION (UI)
    # =========================================================
    def render_optimization_results(self, results: Dict[str, Any]) -> html.Div:
        """Prépare l'interface des résultats d'optimisation (appelé par le callback)."""
        if not results:
            return html.Div("Aucun résultat d'optimisation disponible.", className="alert alert-warning")

        try:
            children = []

            # Graphique Backtest
            if "backtest_values" in results and isinstance(results["backtest_values"], pd.Series):
                fig_backtest = go.Figure(go.Scatter(
                    x=results["backtest_values"].index,
                    y=results["backtest_values"],
                    mode="lines",
                    fill="tozeroy",
                    name="Valeur du portefeuille"
                ))
                fig_backtest.update_layout(title="Backtest - Évolution de la valeur", height=400, template=self.theme)
                children.append(html.H5("Courbe de Backtest"))
                children.append(dcc.Graph(figure=fig_backtest))

            # Métriques
            if "total_return" in results:
                metrics = [
                    ("Rendement total", f"{results.get('total_return', 0):.2%}"),
                    ("Rendement annualisé", f"{results.get('annualized_return', 0):.2%}"),
                    ("Volatilité", f"{results.get('volatility', 0):.2%}"),
                    ("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}"),
                    ("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}"),
                    ("Capital final", f"{results.get('final_capital', 0):,.0f} $"),
                ]
                table_rows = [html.Tr([html.Td(label), html.Td(value)]) for label, value in metrics]
                metrics_table = dbc.Table([html.Tbody(table_rows)], bordered=True, striped=True, size="sm")
                children.append(html.H5("Métriques de performance", className="mt-4"))
                children.append(metrics_table)

            # Poids optimaux
            if "optimal_weights" in results:
                weights = results["optimal_weights"]
                weights_df = pd.DataFrame(list(weights.items()), columns=["Ticker", "Poids"])
                weights_df["Poids"] = weights_df["Poids"].apply(lambda x: f"{x:.2%}")
                weights_table = dbc.Table.from_dataframe(weights_df, striped=True, bordered=True, hover=True, size="sm")
                children.append(html.H5("Pondérations optimales", className="mt-4"))
                children.append(weights_table)

            return html.Div(children, className="mt-3")

        except Exception as e:
            logger.exception(f"Erreur render_optimization_results: {e}")
            return html.Div(f"Erreur d'affichage des résultats : {str(e)}", className="alert alert-danger")

    # =========================================================
    # UTILS
    # =========================================================
    def _create_empty_fig(self, message: str) -> go.Figure:
        """Figure vide avec message explicatif."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#6c757d")
        )
        fig.update_layout(template=self.theme, height=600)
        return fig

    def _check_and_export(self, regime: str, fig: go.Figure):
        """Export optionnel en cas de changement de régime (debug/production)."""
        try:
            if self.last_regime and self.last_regime != regime:
                os.makedirs("reports", exist_ok=True)
                filename = f"reports/regime_change_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
                fig.write_html(filename)
                logger.info(f"Rapport régime sauvegardé : {filename}")
            self.last_regime = regime
        except Exception:
            pass