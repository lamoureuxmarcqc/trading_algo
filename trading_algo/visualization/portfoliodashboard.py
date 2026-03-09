from trading_algo.portfolio.portfolio import Portfolio
from trading_algo.portfolio.portfoliomanager import PortfolioManager  # ou via __init__
import pandas as pd

"""
Module de dashboard pour le portefeuille
À ajouter dans votre fichier portfoliodashboard.py existant
"""

class PortfolioDashboard:
    """Dashboard pour visualiser le portefeuille"""
    
    def __init__(self, portfolio, portfolio_manager=None):
        self.portfolio = portfolio
        self.manager = portfolio_manager
    
    def create_portfolio_dashboard(self, market_prices=None):
        """Crée un dashboard complet du portefeuille"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if market_prices is None and self.manager:
            tickers = list(self.portfolio.positions.keys())
            market_prices = self.manager.get_market_prices(tickers)
        
        # Obtenir les données
        df_summary = self.portfolio.get_summary(market_prices or {})
        performance = self.portfolio.calculate_performance(market_prices or {})
        allocation = self.portfolio.get_allocation(market_prices or {})
        
        # Créer la figure avec sous-graphiques
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Allocation du Portefeuille', 
                'Performance par Position',
                'Évolution de la Valeur',
                'Distribution des P&L',
                'Historique des Transactions',
                'Métriques de Risque'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'table'}, {'type': 'indicator'}]
            ]
        )
        
        # 1. Graphique en camembert de l'allocation
        if allocation:
            labels = []
            values = []
            for k, v in allocation.items():
                if v > 0.01:  # Ignorer les petites allocations
                    labels.append(k if k != 'cash' else 'Liquidités')
                    values.append(v * 100)
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, hole=0.4),
                row=1, col=1
            )
        
        # 2. Barres de performance par position
        if not df_summary.empty:
            fig.add_trace(
                go.Bar(
                    x=df_summary['Ticker'],
                    y=df_summary['P&L %'],
                    name='P&L %',
                    marker_color=['green' if x >= 0 else 'red' for x in df_summary['P&L %']]
                ),
                row=1, col=2
            )
        
        # 3. Évolution de la valeur (si historique disponible)
        if hasattr(self.portfolio, 'performance_history') and self.portfolio.performance_history:
            dates = [p.get('date', i) for i, p in enumerate(self.portfolio.performance_history)]
            values = [p.get('total_value', 0) for p in self.portfolio.performance_history]
            
            fig.add_trace(
                go.Scatter(x=dates, y=values, mode='lines', name='Valeur'),
                row=2, col=1
            )
        
        # 4. Histogramme des P&L
        if not df_summary.empty:
            fig.add_trace(
                go.Histogram(x=df_summary['P&L %'], nbinsx=15, name='Distribution P&L'),
                row=2, col=2
            )
        
        # 5. Tableau des transactions récentes
        transactions = self.portfolio.get_transaction_history()[-10:]  # 10 dernières
        if transactions:
            table_data = []
            for t in transactions[-5:]:  # Limiter à 5 pour la lisibilité
                table_data.append([
                    t.ticker,
                    t.order_type,
                    f"{t.quantity:.2f}",
                    f"${t.limit_price:.2f}" if t.limit_price else '-',
                    t.status,
                    t.execution_time.strftime('%Y-%m-%d') if t.execution_time else '-'
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Ticker', 'Type', 'Qté', 'Prix', 'Statut', 'Date'],
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=list(zip(*table_data)) if table_data else [[]],
                        fill_color='lavender',
                        align='left'
                    )
                ),
                row=3, col=1
            )
        
        # 6. Indicateurs de performance
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=performance['total_value'],
                title={"text": "Valeur Totale"},
                delta={'reference': performance['total_value'] - performance['total_pnl']},
                number={'prefix': "$"}
            ),
            row=3, col=2
        )
        
        # Mise en page
        fig.update_layout(
            title_text=f"Dashboard du Portefeuille: {self.portfolio.name}",
            height=1200,
            showlegend=False
        )
        
        return fig
    
    def create_performance_chart(self, benchmark_ticker='^GSPC'):
        """Crée un graphique comparant la performance du portefeuille à un benchmark"""
        import plotly.graph_objects as go
        
        if not self.portfolio.performance_history:
            return None
        
        # Extraire les données
        df = pd.DataFrame(self.portfolio.performance_history)
        
        # Normaliser à 100
        initial_value = df['total_value'].iloc[0]
        df['portfolio_normalized'] = df['total_value'] / initial_value * 100
        
        fig = go.Figure()
        
        # Portefeuille
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['portfolio_normalized'],
            mode='lines',
            name='Portefeuille',
            line=dict(color='blue', width=2)
        ))
        
        # Benchmark (si disponible)
        if benchmark_ticker and self.manager:
            try:
                extractor = self.manager.data_extractor(benchmark_ticker)
                benchmark_data = extractor.get_historical_data(
                    start=df['date'].min(),
                    end=df['date'].max()
                )
                if not benchmark_data.empty:
                    benchmark_normalized = benchmark_data['Close'] / benchmark_data['Close'].iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=benchmark_data.index,
                        y=benchmark_normalized,
                        mode='lines',
                        name=f'Benchmark ({benchmark_ticker})',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
            except Exception:
                pass
        
        fig.update_layout(
            title='Performance du Portefeuille vs Benchmark',
            xaxis_title='Date',
            yaxis_title='Performance (Base 100)',
            hovermode='x unified'
        )
        
        return fig