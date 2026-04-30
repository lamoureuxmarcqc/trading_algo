# trading_algo/visualization/market_dashboard.py
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Palette institutionnelle (identique à portfoliodashboard)
COLORS = {
    "primary": "#0A2540",       # Bleu marine (texte principal)
    "secondary": "#425466",     # Gris bleu
    "success": "#2E7D32",       # Vert institutionnel
    "danger": "#C62828",        # Rouge financier
    "warning": "#ED6C02",       # Orange
    "info": "#0288D1",          # Bleu information
    "background": "#F8F9FC",    # Fond très clair
    "card_bg": "#FFFFFF",       # Cartes blanches
    "text_muted": "#5F6B7A",    # Texte secondaire
    "border": "#E0E4E8",        # Bordures douces
    "grid": "#E9ECEF",          # Lignes de grille
    "plot_bg": "#FFFFFF",       # Fond des graphiques (blanc pur)
}

# Palette de couleurs pour les courbes des indices (visibles sur fond blanc)
LINE_COLORS = ["#1E88E5", "#D32F2F", "#F57C00", "#2E7D32", "#8E24AA", "#00ACC1", "#6C757D"]


class MarketDashboard:
    """
    Tableau de bord marché : indices, secteurs, top movers, macro, fx, commodités.
    Génère une figure Plotly avec sous-graphiques (2x2) – style institutionnel.
    """

    def __init__(self):
        self.indices: Dict[str, pd.DataFrame] = {}
        self.sector_perf: pd.DataFrame = pd.DataFrame()
        self.top_movers: pd.DataFrame = pd.DataFrame()
        self.commodities: Dict[str, Any] = {}
        self.currencies: Dict[str, Any] = {}
        self.macro: Dict[str, Any] = {}
        self.period_label = "1M"

    def load_data(self, **kwargs) -> None:
        """Charge les données depuis le MarketManager ou le cache."""
        self.indices = kwargs.get('indices', {})
        self.sector_perf = kwargs.get('sector_perf', pd.DataFrame())
        self.top_movers = kwargs.get('top_movers', pd.DataFrame())
        self.commodities = kwargs.get('commodities', {})
        self.currencies = kwargs.get('currencies', {})
        self.macro = kwargs.get('macro', {})
        self.period_label = kwargs.get('period_label', "1M")

        if isinstance(self.sector_perf, list):
            self.sector_perf = pd.DataFrame(self.sector_perf)
        if isinstance(self.top_movers, list):
            self.top_movers = pd.DataFrame(self.top_movers)

        if not self.sector_perf.empty and 'perf' in self.sector_perf.columns:
            self.sector_perf['perf'] = pd.to_numeric(self.sector_perf['perf'], errors='coerce').fillna(0)

        if not self.top_movers.empty and 'perf' in self.top_movers.columns:
            self.top_movers['perf'] = pd.to_numeric(self.top_movers['perf'], errors='coerce').fillna(0)

        if not self.top_movers.empty and 'symbol' not in self.top_movers.columns:
            logger.warning("top_movers manque la colonne 'symbol'")
            self.top_movers = pd.DataFrame()

        logger.info(f"Market Dashboard prêt : {len(self.indices)} indices, {len(self.sector_perf)} secteurs, {len(self.top_movers)} top movers.")

    def _normalize_indices(self) -> Dict[str, Tuple[pd.Series, float]]:
        """Normalise chaque indice à base 100 et calcule la performance."""
        normalized = {}
        for name, df in self.indices.items():
            if not isinstance(df, pd.DataFrame) or df.empty or 'Close' not in df.columns:
                continue
            close = df['Close'].dropna()
            if len(close) < 2:
                continue
            base = close.iloc[0]
            if base == 0:
                continue
            norm = (close / base) * 100
            perf = ((close.iloc[-1] / base) - 1) * 100
            normalized[name] = (norm, perf)
        return normalized

    def _prepare_macro_rows(self) -> List[Tuple[str, str, str]]:
        """Construit les lignes pour la table macro."""
        rows: List[Tuple[str, str, str]] = []

        def add_items(prefix: str, source: Dict) -> None:
            for key, value in source.items():
                if isinstance(value, dict):
                    val = value.get('price', value.get('value', value.get('rate', 'N/A')))
                    change = value.get('change', '-')
                    if isinstance(change, (int, float)):
                        change_str = f"{change:+.2f}%" if change != 0 else "0.00%"
                    else:
                        change_str = str(change)
                    rows.append((f"{prefix} {key}", str(val), change_str))
                else:
                    rows.append((f"{prefix} {key}", str(value), "-"))

        add_items("📊 Macro", self.macro)
        add_items("💱 FX", self.currencies)
        add_items("📦 Comm.", self.commodities)

        if not rows:
            rows = [("Aucune donnée", "-", "-")]
        return rows

    def create_figure(self) -> go.Figure:
        """Crée la figure principale 2x2 avec style institutionnel (pas de template par défaut)."""
        indices_norm = self._normalize_indices()
        has_sectors = not self.sector_perf.empty and 'sector' in self.sector_perf.columns and 'perf' in self.sector_perf.columns
        has_movers = not self.top_movers.empty and 'symbol' in self.top_movers.columns and 'perf' in self.top_movers.columns

        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.55, 0.45],
            row_heights=[0.5, 0.5],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            specs=[
                [{"type": "xy"}, {"type": "treemap" if has_sectors else "scatter"}],
                [{"type": "table"}, {"type": "table"}]
            ],
            subplot_titles=(
                f"📈 Comparaison Indices ({self.period_label})",
                "📊 Force Relative des Secteurs" if has_sectors else "Aucune donnée secteur",
                "🔥 Top 10 Movers (S&P 500)" if has_movers else "Aucun mouvement significatif",
                "🌐 Macro, FX & Commodities"
            )
        )

        # ---- 1. Graphe des indices normalisés (courbes colorées) ----
        for i, (name, (norm_series, perf)) in enumerate(indices_norm.items()):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            fig.add_trace(go.Scatter(
                x=norm_series.index,
                y=norm_series,
                name=name,
                line=dict(width=2, color=color),
                hovertemplate=f"<b>{name}</b><br>Valeur: %{{y:.2f}}<br>Perf: {perf:+.2f}%<extra></extra>"
            ), row=1, col=1)

        if not indices_norm:
            fig.add_annotation(
                text="Aucune donnée d'indice disponible",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLORS["text_muted"]),
                row=1, col=1
            )

        # ---- 2. Treemap des secteurs ----
        if has_sectors:
            sizes = np.abs(self.sector_perf['perf']) + 0.5
            fig.add_trace(go.Treemap(
                labels=self.sector_perf['sector'],
                parents=[""] * len(self.sector_perf),
                values=sizes,
                marker=dict(
                    colors=self.sector_perf['perf'],
                    colorscale='RdYlGn',
                    cmid=0,
                    showscale=True,
                    colorbar=dict(title="Performance %", thickness=15, x=1.05, tickfont=dict(color=COLORS["primary"]))
                ),
                text=self.sector_perf['perf'].apply(lambda x: f"{x:+.2f}%"),
                textinfo="label+text",
                hoverinfo="label+text+value",
                textfont=dict(color=COLORS["primary"])  # texte visible
            ), row=1, col=2)
        else:
            fig.add_annotation(
                text="Secteurs non disponibles",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLORS["text_muted"]),
                row=1, col=2
            )

        # ---- 3. Tableau des top movers ----
        if has_movers:
            top = self.top_movers.head(10)
            movers_headers = ["Symbole", "Perf %", "Secteur"]
            movers_data = [
                top['symbol'].tolist(),
                top['perf'].apply(lambda x: f"{x:+.2f}%").tolist(),
                top.get('sector', [''] * len(top)).tolist()
            ]
            fig.add_trace(go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in movers_headers],
                    fill_color=COLORS["secondary"],
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=movers_data,
                    fill_color=COLORS["plot_bg"],
                    font=dict(color=COLORS["primary"], size=11),
                    align='left',
                    height=28
                )
            ), row=2, col=1)
        else:
            fig.add_annotation(
                text="Aucun mouvement détecté",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLORS["text_muted"]),
                row=2, col=1
            )

        # ---- 4. Tableau macro consolidé ----
        macro_rows = self._prepare_macro_rows()
        if macro_rows:
            headers = ["Indicateur", "Valeur", "Variation"]
            col0, col1, col2 = zip(*macro_rows)
            fig.add_trace(go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in headers],
                    fill_color=COLORS["secondary"],
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=[list(col0), list(col1), list(col2)],
                    fill_color=COLORS["plot_bg"],
                    font=dict(color=COLORS["primary"], size=11),
                    align='left',
                    height=28
                )
            ), row=2, col=2)
        else:
            fig.add_annotation(
                text="Données macro non disponibles",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLORS["text_muted"]),
                row=2, col=2
            )

        # ---- Mise en page (sans template Plotly) ----
        fig.update_layout(
            height=900,
            template=None,  # désactivé pour éviter blanc sur blanc
            margin=dict(t=80, b=40, l=40, r=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font=dict(color=COLORS["primary"])),
            hovermode="x unified",
            plot_bgcolor=COLORS["plot_bg"],
            paper_bgcolor=COLORS["background"],
            font=dict(color=COLORS["primary"])
        )

        # Axes du premier graphique
        fig.update_xaxes(title_text="Date", row=1, col=1,
                         tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"])
        fig.update_yaxes(title_text="Index (base 100)", row=1, col=1,
                         tickfont=dict(color=COLORS["primary"]), gridcolor=COLORS["grid"])

        return fig