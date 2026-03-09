import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_algo.risk.risk_manager import RiskManager


class SymbolDashboard:
    """
    Dashboard détaillé pour un symbole individuel.
    """

    def __init__(self, symbol, portfolio, portfolio_manager):
        self.symbol = symbol
        self.portfolio = portfolio
        self.manager = portfolio_manager

    def load_data(self):
        """
        Charge les données historiques du symbole via le data_extractor.
        """
        extractor = self.manager.data_extractor(self.symbol)
        df = extractor.get_historical_data(period="1y")

        # Calcul ATR si disponible
        if "High" in df.columns and "Low" in df.columns and "Close" in df.columns:
            df["H-L"] = df["High"] - df["Low"]
            df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
            df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
            df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
            df["ATR"] = df["TR"].rolling(14).mean()

        return df

    def compute_risk_metrics(self, df, benchmark_returns=None):
        """
        Calcule les métriques de risque via RiskManager.
        """
        returns = df["Close"].pct_change().dropna()

        metrics = {
            "sharpe": RiskManager.calculate_sharpe_ratio(returns),
            "max_drawdown": RiskManager.calculate_max_drawdown(df["Close"]),
            "var_95": RiskManager.calculate_value_at_risk(returns, 0.95),
        }

        if benchmark_returns is not None:
            metrics["beta"] = RiskManager.calculate_beta(returns, benchmark_returns)
        else:
            metrics["beta"] = None

        # ATR-based stop-loss / take-profit
        stop, take = RiskManager.calculate_atr_levels(df)
        metrics["stop_loss"] = stop
        metrics["take_profit"] = take

        # Position sizing
        account_value = self.portfolio.get_total_value()
        if stop:
            metrics["position_size"] = RiskManager.suggest_position_size(
                account_value, df["Close"].iloc[-1], stop
            )
        else:
            metrics["position_size"] = None

        return metrics

    def create_dashboard(self):
        """
        Crée un dashboard complet pour le symbole.
        """
        df = self.load_data()
        if df.empty:
            return None

        # Benchmark returns (ex: SP500)
        benchmark = "^GSPC"
        try:
            bench_df = self.manager.data_extractor(benchmark).get_historical_data(
                start=df.index.min(), end=df.index.max()
            )
            benchmark_returns = bench_df["Close"].pct_change().dropna()
        except:
            benchmark_returns = None

        metrics = self.compute_risk_metrics(df, benchmark_returns)

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f"Prix de {self.symbol}",
                "ATR",
                "Rendements journaliers",
                "Distribution des rendements",
                "Métriques de risque",
                "Niveaux recommandés"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "table"}, {"type": "table"}]
            ]
        )

        # 1. Prix
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"),
            row=1, col=1
        )

        # 2. ATR
        if "ATR" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["ATR"], mode="lines", name="ATR", line=dict(color="orange")),
                row=1, col=2
            )

        # 3. Rendements
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"].pct_change(), mode="lines", name="Returns"),
            row=2, col=1
        )

        # 4. Histogramme
        fig.add_trace(
            go.Histogram(x=df["Close"].pct_change(), nbinsx=30),
            row=2, col=2
        )

        # 5. Tableau des métriques
        fig.add_trace(
            go.Table(
                header=dict(values=["Métrique", "Valeur"], fill_color="lightgrey"),
                cells=dict(values=[
                    ["Sharpe", "Max Drawdown", "VaR 95%", "Beta"],
                    [
                        f"{metrics['sharpe']:.2f}",
                        f"{metrics['max_drawdown']:.2%}",
                        f"{metrics['var_95']:.2%}",
                        f"{metrics['beta']:.2f}" if metrics["beta"] else "-"
                    ]
                ])
            ),
            row=3, col=1
        )

        # 6. Stop-loss / Take-profit / Position size
        fig.add_trace(
            go.Table(
                header=dict(values=["Paramètre", "Valeur"], fill_color="lightgrey"),
                cells=dict(values=[
                    ["Stop-Loss", "Take-Profit", "Position Size"],
                    [
                        f"{metrics['stop_loss']:.2f}" if metrics["stop_loss"] else "-",
                        f"{metrics['take_profit']:.2f}" if metrics["take_profit"] else "-",
                        f"{metrics['position_size']:.2f}" if metrics["position_size"] else "-"
                    ]
                ])
            ),
            row=3, col=2
        )

        fig.update_layout(
            title=f"Dashboard du symbole : {self.symbol}",
            height=1200,
            showlegend=False
        )

        return fig
