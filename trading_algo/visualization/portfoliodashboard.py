# trading_algo/visualization/portfoliodashboard.py
"""
PortfolioDashboard - Tableau de bord stratégique pour institutionnels.
Style : contraste élevé, couleurs sobres, aucune dépendance aux thèmes Plotly par défaut.
Fonctionnalités :
- Vue stratégique (régime de marché, scores qualité, allocation cible vs actuelle, actions)
- Vue technique (allocation circulaire, P&L, performance cumulée)
- Tableau des positions détaillé
- Résultats d'optimisation et simulation Monte Carlo
- Intégration avec PortfolioIntelligenceEngine (scoring, recommandations)
"""
import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Optional, Any, List
import plotly.express as px

from dash import html, dcc
import dash_bootstrap_components as dbc

from trading_algo.portfolio.portfolio import Portfolio
from trading_algo.portfolio.portfoliomanager import PortfolioManager

logger = logging.getLogger(__name__)

# =========================================================================
# PALETTE INSTITUTIONNELLE (contraste élevé)
# =========================================================================
COLORS = {
    "primary": "#0A2540",       # Bleu marine (texte principal)
    "secondary": "#425466",     # Gris bleu (en-têtes)
    "success": "#2E7D32",       # Vert
    "danger": "#C62828",        # Rouge
    "warning": "#ED6C02",       # Orange
    "info": "#0288D1",          # Bleu info
    "background": "#F8F9FC",    # Fond général très clair
    "card_bg": "#FFFFFF",       # Blanc pour les cartes
    "text_muted": "#5F6B7A",    # Texte secondaire
    "border": "#E0E4E8",        # Bordures
    "grid": "#E9ECEF",          # Grille des graphiques
    "plot_bg": "#FFFFFF",       # Fond des graphiques (blanc)
}


class PortfolioDashboard:
    """
    Dashboard complet pour la gestion de portefeuille.
    """

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        portfolio_manager: Optional[PortfolioManager] = None,
        theme: str = "none"  # Thème désactivé, on contrôle tout.
    ):
        self.portfolio = portfolio
        self.manager = portfolio_manager
        self.theme = theme
        self.last_regime = None

    # ------------------------------------------------------------------
    # 1. RAPPORT STRATÉGIQUE (avec scoring et allocation intelligente)
    # ------------------------------------------------------------------
    def create_strategic_report(
        self,
        macro_data: Dict = None,
        market_indices: pd.DataFrame = None,
        fundamentals_map: Dict = None
    ) -> go.Figure:
        """
        Graphique stratégique 2x2 :
        - Jauge du régime de marché
        - Scores de qualité (Buffett / Factor)
        - Allocation cible vs actuelle (barres)
        - Recommandations de rééquilibrage (tableau)
        """
        if not self.portfolio or not getattr(self.portfolio, 'positions', None):
            return self._create_empty_fig("Aucun portefeuille chargé")

        macro_data = macro_data or {}
        market_indices = market_indices if market_indices is not None else pd.DataFrame()
        fundamentals_map = fundamentals_map or {}

        try:
            # --- Récupération des données stratégiques via le manager ---
            regime = "NEUTRAL"
            regime_score = 50
            quality_scores = {}
            target_alloc = {}
            current_alloc = self.portfolio.get_allocation() if hasattr(self.portfolio, 'get_allocation') else {}
            recommendations = []

            if self.manager:
                try:
                    # Régime de marché
                    regime_info = self.manager.get_market_regime(macro_data)
                    regime = regime_info.get("regime", "NEUTRAL")
                    regime_score = regime_info.get("score", 50)

                    # Scoring et allocation via le moteur d'intelligence
                    if hasattr(self.manager, 'intelligence') and self.manager.intelligence:
                        quality_scores = self.manager.intelligence.compute_scores(fundamentals_map)
                        target_alloc = self.manager.intelligence.build_allocation(quality_scores)
                        recommendations = self.manager.intelligence.generate_recommendations(target_alloc)
                    else:
                        # Fallback simple
                        quality_scores = self.manager.get_quality_scores(fundamentals_map)
                        target_alloc = self.manager.get_target_allocation(self.portfolio, model="buffett")
                except Exception as e:
                    logger.warning(f"Erreur appel intelligence: {e}")

            # Tous les tickers concernés
            all_tickers = sorted(set(target_alloc.keys()) | set(current_alloc.keys()))

            # Construction du graphique 2x2
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "table"}]],
                subplot_titles=(
                    "<b>Régime de Marché</b>",
                    "<b>Scores de Qualité</b>",
                    "<b>Allocation (%) – Cible vs Actuelle</b>",
                    "<b>Recommandations de Rééquilibrage</b>"
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.12,
                row_heights=[0.45, 0.55]
            )

            # ---------- 1. Jauge du régime ----------
            gauge_color = {
                "BULL": "#2E7D32", "BULLISH": "#2E7D32",
                "BEAR": "#C62828", "BEARISH": "#C62828",
                "NEUTRAL": "#ED6C02"
            }.get(regime, "#0A2540")

            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=float(regime_score),
                title={"text": regime.title(), "font": {"size": 16, "color": COLORS["primary"]}},
                delta={"reference": 50, "increasing": {"color": COLORS["success"]}, "decreasing": {"color": COLORS["danger"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["primary"]},
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": COLORS["plot_bg"],
                    "borderwidth": 1,
                    "bordercolor": COLORS["border"],
                    "steps": [
                        {"range": [0, 30], "color": "#FFEBEE"},
                        {"range": [30, 70], "color": "#FFF8E1"},
                        {"range": [70, 100], "color": "#E8F5E9"}
                    ],
                    "threshold": {"line": {"color": COLORS["danger"], "width": 2}, "thickness": 0.75, "value": 50}
                },
                number={"font": {"size": 40, "color": COLORS["primary"]}}
            ), row=1, col=1)

            # ---------- 2. Scores de qualité (barres horizontales) ----------
            if quality_scores:
                sorted_scores = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
                tickers_q = [x[0] for x in sorted_scores]
                values_q = [x[1] for x in sorted_scores]
                colors_q = [COLORS["success"] if v >= 0.5 else COLORS["warning"] if v >= 0 else COLORS["danger"] for v in values_q]
                fig.add_trace(go.Bar(
                    x=values_q, y=tickers_q, orientation='h',
                    marker_color=colors_q,
                    text=[f"{v:.2f}" for v in values_q],
                    textposition='outside',
                    textfont=dict(color=COLORS["primary"], size=11),
                    name="Score",
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>"
                ), row=1, col=2)
                fig.update_xaxes(title_text="Score", range=[min(values_q)-0.2, max(values_q)+0.2],
                                 tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"], row=1, col=2)
                fig.update_yaxes(title_text="", tickfont=dict(color=COLORS["primary"]), row=1, col=2)

            # ---------- 3. Allocation (barres groupées) ----------
            fig.add_trace(go.Bar(
                x=all_tickers, y=[target_alloc.get(t, 0) * 100 for t in all_tickers],
                name="Allocation cible", marker_color=COLORS["primary"],
                text=[f"{target_alloc.get(t, 0)*100:.1f}%" for t in all_tickers],
                textposition="outside", textfont=dict(color=COLORS["primary"])
            ), row=2, col=1)

            fig.add_trace(go.Bar(
                x=all_tickers, y=[current_alloc.get(t, 0) * 100 for t in all_tickers],
                name="Allocation actuelle", marker_color=COLORS["warning"],
                text=[f"{current_alloc.get(t, 0)*100:.1f}%" for t in all_tickers],
                textposition="outside", textfont=dict(color=COLORS["primary"])
            ), row=2, col=1)

            fig.update_xaxes(title_text="Actif", tickangle=-30, row=2, col=1,
                             tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"])
            fig.update_yaxes(title_text="Allocation (%)", row=2, col=1,
                             tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"])

            # ---------- 4. Tableau des recommandations ----------
            if recommendations:
                table_rows = []
                for rec in recommendations:
                    ticker = rec.get("ticker", "")
                    action = rec.get("action", "")
                    delta = rec.get("delta", 0) * 100  # en %
                    if action == "BUY":
                        action_text = f"🔹 ACHETER   (+{delta:.1f}%)"
                    elif action == "SELL":
                        action_text = f"🔸 VENDRE    ({delta:.1f}%)"
                    else:
                        action_text = action
                    table_rows.append([ticker, action_text])
                fig.add_trace(go.Table(
                    header=dict(
                        values=["<b>Ticker</b>", "<b>Action recommandée</b>"],
                        fill_color=COLORS["secondary"],
                        font=dict(color="white", size=12),
                        align="center"
                    ),
                    cells=dict(
                        values=[list(zip(*table_rows))[0] if table_rows else [], list(zip(*table_rows))[1] if table_rows else []],
                        align="left",
                        font=dict(color=COLORS["primary"], size=11),
                        fill_color=COLORS["plot_bg"]
                    )
                ), row=2, col=2)
            else:
                fig.add_annotation(
                    text="Aucune recommandation pour le moment",
                    xref="x domain", yref="y domain", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=12, color=COLORS["text_muted"]),
                    row=2, col=2
                )

            # Mise en page générale
            fig.update_layout(
                template=None,
                height=880,
                title=dict(
                    text=f"📊 Strategic Advisor — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    font=dict(size=20, color=COLORS["primary"]),
                    x=0.5
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                    font=dict(color=COLORS["primary"])
                ),
                margin=dict(l=60, r=40, t=100, b=60),
                plot_bgcolor=COLORS["plot_bg"],
                paper_bgcolor=COLORS["background"],
                font=dict(color=COLORS["primary"])
            )
            self._check_and_export(regime, fig)
            return fig

        except Exception as e:
            logger.exception("Erreur create_strategic_report")
            return self._create_empty_fig("Erreur lors de la génération du rapport stratégique")

    # ------------------------------------------------------------------
    # 2. RAPPORT TECHNIQUE / VISUEL
    # ------------------------------------------------------------------
    def create_visual_report(self) -> go.Figure:
        """Camembert d'allocation, P&L en barres et performance relative."""
        if not self.portfolio or not getattr(self.portfolio, 'positions', None):
            return self._create_empty_fig("Aucune position à afficher")

        try:
            prices = self.manager.get_market_prices(list(self.portfolio.positions.keys())) if self.manager else {}
            df_summary = self.portfolio.get_summary(prices) if hasattr(self.portfolio, 'get_summary') else pd.DataFrame()
            alloc = self.portfolio.get_allocation(prices) if hasattr(self.portfolio, 'get_allocation') else {}

            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "domain"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]],
                subplot_titles=(
                    "<b>Répartition du Portefeuille</b>",
                    "<b>P&L par Position (%)</b>",
                    "<b>P&L Absolu ($)</b>",
                    "<b>Performance cumulée (%)</b>"
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.12
            )

            # Camembert
            if alloc:
                fig.add_trace(go.Pie(
                    labels=list(alloc.keys()), values=list(alloc.values()),
                    hole=0.45, textinfo="label+percent", textposition="auto",
                    marker=dict(colors=px.colors.qualitative.Set2),
                    hoverinfo="label+percent+value",
                    pull=[0.05 if v == max(alloc.values()) else 0 for v in alloc.values()]
                ), row=1, col=1)

            if not df_summary.empty:
                tickers = df_summary.get("Ticker", df_summary.index.tolist())
                pnl_pct = df_summary.get("P&L %", pd.Series([0]*len(tickers))).astype(float) * 100
                pnl_abs = df_summary.get("P&L", pd.Series([0]*len(tickers))).astype(float)

                # P&L %
                colors_pct = [COLORS["success"] if v > 0 else COLORS["danger"] for v in pnl_pct]
                fig.add_trace(go.Bar(
                    x=tickers, y=pnl_pct, marker_color=colors_pct,
                    text=[f"{v:+.2f}%" for v in pnl_pct], textposition="outside",
                    textfont=dict(color=COLORS["primary"]), name="P&L %"
                ), row=1, col=2)
                fig.update_yaxes(title_text="P&L (%)", tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"], row=1, col=2)

                # P&L absolu
                colors_abs = [COLORS["success"] if v > 0 else COLORS["danger"] for v in pnl_abs]
                fig.add_trace(go.Bar(
                    x=tickers, y=pnl_abs, marker_color=colors_abs,
                    text=[f"{v:+,.0f} $" for v in pnl_abs], textposition="outside",
                    textfont=dict(color=COLORS["primary"]), name="P&L $"
                ), row=2, col=1)
                fig.update_yaxes(title_text="P&L ($)", tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"], row=2, col=1)

                # Cumul
                cumul = pnl_pct.cumsum()
                fig.add_trace(go.Scatter(
                    x=tickers, y=cumul, mode="lines+markers",
                    name="Cumul P&L %", line=dict(color=COLORS["primary"], width=2),
                    marker=dict(size=6, color=COLORS["primary"])
                ), row=2, col=2)
                fig.update_yaxes(title_text="Cumul (%)", tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"], row=2, col=2)
                fig.update_xaxes(title_text="Actif", tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"], row=2, col=2)

            fig.update_layout(
                template=None, height=850,
                title=dict(text="📈 Analyse Technique", font=dict(size=18, color=COLORS["primary"]), x=0.5),
                showlegend=False,
                margin=dict(l=60, r=40, t=80, b=60),
                plot_bgcolor=COLORS["plot_bg"],
                paper_bgcolor=COLORS["background"],
                font=dict(color=COLORS["primary"])
            )
            return fig

        except Exception as e:
            logger.exception("Erreur create_visual_report")
            return self._create_empty_fig("Erreur lors de la génération du rapport visuel")

    # ------------------------------------------------------------------
    # 3. TABLEAU DES POSITIONS (complet et lisible)
    # ------------------------------------------------------------------
    def render_positions_table(self, prices: Optional[Dict[str, float]] = None,
                               preferred_order: Optional[List[str]] = None) -> html.Div:
        """Tableau détaillé des positions avec mise en forme monétaire et couleurs conditionnelles."""
        if not self.portfolio:
            return dbc.Alert("Aucun portefeuille chargé", color="info")

        try:
            prices = prices or {}
            df = self.portfolio.get_summary(prices) if hasattr(self.portfolio, 'get_summary') else pd.DataFrame()
            if df is None or df.empty:
                return dbc.Alert("Aucune position ouverte.", color="warning")

            # Mapping des colonnes (robuste)
            col_map = {
                "Ticker": ["ticker", "symbol", "Ticker", "Symbol"],
                "Qty": ["qty", "quantity", "shares", "Qty"],
                "Prix moyen": ["avg_price", "prix_moyen", "avgprice", "price_avg", "Prix moyen"],
                "Prix actuel": ["price", "current_price", "prix_actuel", "Prix actuel"],
                "Valeur": ["market_value", "value", "valeur", "marketvalue", "Valeur"],
                "P&L": ["pnl", "pl", "P&L", "profit_loss"],
                "P&L %": ["pnl_pct", "pnl_percent", "pnl%", "P&L %", "pnl_pct"],
                "Poids %": ["weight", "weight_pct", "allocation", "Poids %"]
            }

            def find_col(columns, candidates):
                for cand in candidates:
                    if cand in columns:
                        return cand
                    for col in columns:
                        if col.lower() == cand.lower():
                            return col
                return None

            source = {}
            for display, candidates in col_map.items():
                found = find_col(df.columns, candidates)
                if found:
                    source[display] = found

            if "Ticker" not in source and df.index.name is not None:
                df = df.reset_index()
                source["Ticker"] = df.columns[0]

            default_order = ["Ticker", "Qty", "Prix moyen", "Prix actuel", "Valeur", "P&L", "P&L %", "Poids %"]
            final_cols = [c for c in default_order if c in source]

            rename = {v: k for k, v in source.items()}
            render_df = df.rename(columns=rename)
            render_df = render_df[[c for c in default_order if c in render_df.columns]]

            # Formatage
            def fmt_currency(x):
                try:
                    if pd.isna(x):
                        return ""
                    return f"{x:,.2f} $"
                except:
                    return str(x)

            def fmt_pct(x, dec=2):
                try:
                    return f"{float(x)*100:.{dec}f}%"
                except:
                    return str(x)

            def fmt_qty(x):
                try:
                    if float(x).is_integer():
                        return f"{int(x)}"
                    return f"{x:,.4f}"
                except:
                    return str(x)

            styled = render_df.copy()
            for col in styled.columns:
                if col in ["P&L %", "Poids %"]:
                    styled[col] = styled[col].apply(lambda v: fmt_pct(v))
                elif col in ["Valeur", "P&L", "Prix moyen", "Prix actuel"]:
                    styled[col] = styled[col].apply(lambda v: fmt_currency(v))
                elif col == "Qty":
                    styled[col] = styled[col].apply(fmt_qty)
                else:
                    styled[col] = styled[col].apply(lambda v: str(v) if pd.notna(v) else "")

            # Construction du tableau avec Bootstrap
            table_header = html.Thead(
                html.Tr([html.Th(col, style={"fontWeight": "bold", "backgroundColor": COLORS["secondary"], "color": "white"}) for col in styled.columns])
            )
            table_body = []
            for idx, row in styled.iterrows():
                row_style = {}
                if "P&L" in render_df.columns and idx in render_df.index:
                    pnl_val = render_df.loc[idx, "P&L"]
                    try:
                        if float(pnl_val) > 0:
                            row_style = {"backgroundColor": "#E8F5E9"}
                        elif float(pnl_val) < 0:
                            row_style = {"backgroundColor": "#FFEBEE"}
                    except:
                        pass
                cells = [html.Td(row[col], style={"textAlign": "right" if col in ["Valeur","P&L","P&L %","Poids %"] else "left"}) for col in styled.columns]
                table_body.append(html.Tr(cells, style=row_style))

            table = dbc.Table(
                [table_header, html.Tbody(table_body)],
                striped=False, bordered=True, hover=True, size="sm", responsive=True,
                className="table-sm positions-table"
            )
            return html.Div(table, className="p-3 bg-white rounded shadow-sm")

        except Exception as e:
            logger.exception("Erreur render_positions_table")
            return dbc.Alert(f"Erreur affichage positions: {e}", color="danger")

    # ------------------------------------------------------------------
    # 4. RÉSULTATS D'OPTIMISATION (KPIs)
    # ------------------------------------------------------------------
    def render_optimization_results(self, results: Dict[str, Any]) -> html.Div:
        if not results:
            return dbc.Alert("Aucun résultat d'optimisation disponible.", color="warning")

        try:
            components = []
            if "backtest_values" in results and isinstance(results["backtest_values"], pd.Series):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results["backtest_values"].index, y=results["backtest_values"],
                    mode="lines", fill="tozeroy", line=dict(color=COLORS["primary"], width=2),
                    name="Valeur portefeuille"
                ))
                fig.update_layout(
                    title="Backtest - Évolution de la valeur", height=400,
                    template=None, margin=dict(l=40, r=40, t=60, b=40),
                    plot_bgcolor=COLORS["plot_bg"], paper_bgcolor=COLORS["background"],
                    font=dict(color=COLORS["primary"])
                )
                components.append(html.H5("Courbe de Backtest", className="mt-3"))
                components.append(dcc.Graph(figure=fig))

            if "total_return" in results:
                metrics = [
                    ("Rendement total", f"{results.get('total_return', 0):.2%}", "success"),
                    ("Rendement annualisé", f"{results.get('annualized_return', 0):.2%}", "primary"),
                    ("Volatilité", f"{results.get('volatility', 0):.2%}", "warning"),
                    ("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}", "info"),
                    ("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}", "danger"),
                    ("Capital final", f"{results.get('final_capital', 0):,.0f} $", "secondary"),
                ]
                cards = dbc.Row([
                    dbc.Col(
                        dbc.Card(dbc.CardBody([
                            html.H6(label, className="card-subtitle mb-2 text-muted"),
                            html.H4(value, className=f"card-title text-{color}")
                        ]), className="shadow-sm h-100"), width=4, className="mb-3"
                    ) for label, value, color in metrics
                ])
                components.append(html.H5("Métriques de performance", className="mt-4"))
                components.append(cards)

            if "optimal_weights" in results:
                w = results["optimal_weights"]
                w_df = pd.DataFrame(list(w.items()), columns=["Ticker", "Poids"])
                w_df["Poids"] = w_df["Poids"].apply(lambda x: f"{x:.2%}")
                w_table = dbc.Table.from_dataframe(w_df, striped=True, bordered=True, hover=True, size="sm")
                components.append(html.H5("Pondérations optimales", className="mt-4"))
                components.append(w_table)

            return html.Div(components, className="p-3")

        except Exception as e:
            logger.exception("Erreur render_optimization_results")
            return dbc.Alert(f"Erreur d'affichage : {e}", color="danger")

    # ------------------------------------------------------------------
    # 5. MONTE CARLO (visible)
    # ------------------------------------------------------------------
    def render_monte_carlo_results(self, sim_result: Dict[str, Any]) -> html.Div:
        try:
            if not sim_result or "error" in sim_result:
                err = sim_result.get("error", "Simulation non disponible")
                return dbc.Alert(f"⚠️ {err}", color="warning")

            paths = sim_result.get("paths_preview", [])
            if not paths:
                return dbc.Alert("Aucune trajectoire générée.", color="warning")

            arr = np.array(paths)
            timeframe = sim_result.get("timeframe", arr.shape[0])
            n_preview = min(arr.shape[1], 50)

            fig = go.Figure()
            x = list(range(1, timeframe + 1))

            # Trajectoires individuelles (gris foncé visible)
            for i in range(n_preview):
                fig.add_trace(go.Scatter(
                    x=x, y=arr[:, i], mode="lines",
                    line=dict(width=0.6, color="#555555"),
                    showlegend=False, opacity=0.35
                ))

            median = np.median(arr, axis=1)
            p5 = np.percentile(arr, 5, axis=1)
            p95 = np.percentile(arr, 95, axis=1)

            fig.add_trace(go.Scatter(x=x, y=median, mode="lines", name="Médiane",
                                     line=dict(color=COLORS["primary"], width=3)))
            fig.add_trace(go.Scatter(x=x, y=p5, mode="lines", name="P5 (pessimiste)",
                                     line=dict(color=COLORS["danger"], width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", name="P95 (optimiste)",
                                     line=dict(color=COLORS["success"], width=2, dash="dash"),
                                     fill='tonexty', fillcolor='rgba(46,125,50,0.1)'))

            fig.update_layout(
                title=f"Simulation Monte Carlo — {sim_result.get('n_simulations', '?')} trajectoires",
                xaxis_title="Jours", yaxis_title="Indice (base 100)",
                template=None,  # height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor=COLORS["plot_bg"], paper_bgcolor=COLORS["background"],
                font=dict(color=COLORS["primary"]),
                xaxis=dict(tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"]),
                yaxis=dict(tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"])
            )

            metrics = sim_result.get("metrics", {})
            metric_items = [
                dbc.ListGroupItem(f"📉 VaR 95% : <strong>{metrics.get('var_95', 0):.2%}</strong>"),
                dbc.ListGroupItem(f"📊 CVaR 95% : <strong>{metrics.get('cvar_95', 0):.2%}</strong>"),
                dbc.ListGroupItem(f"📈 Probabilité de profit : <strong>{metrics.get('prob_profit', 0):.2%}</strong>"),
            ]
            metrics_list = dbc.ListGroup(metric_items, flush=True, className="mt-2")

            return html.Div([
                dcc.Graph(figure=fig),
                html.H5("Résumé des risques", className="mt-3"),
                metrics_list
            ], className="p-3 bg-white rounded shadow-sm")

        except Exception as e:
            logger.exception("Erreur render_monte_carlo_results")
            return dbc.Alert(f"Erreur affichage simulation: {e}", color="danger")

    # ------------------------------------------------------------------
    # 6. UTILITAIRES
    # ------------------------------------------------------------------
    def _create_empty_fig(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text_muted"])
        )
        fig.update_layout(template=None, height=600,
                          plot_bgcolor=COLORS["plot_bg"],
                          paper_bgcolor=COLORS["background"])
        return fig

    def _check_and_export(self, regime: str, fig: go.Figure) -> None:
        try:
            if self.last_regime and self.last_regime != regime:
                os.makedirs("reports", exist_ok=True)
                filename = f"reports/regime_change_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
                fig.write_html(filename)
                logger.info(f"Rapport régime sauvegardé : {filename}")
            self.last_regime = regime
        except Exception:
            pass