# trading_algo/web_dashboard/callbacks/portfolio_callbacks.py
from dash import Input, Output, State, callback, no_update
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash import html, dcc
from trading_algo.portfolio.portfoliomanager import PortfolioManager
from trading_algo.data.data_extraction import StockDataExtractor

def register_portfolio_callbacks(app):
    
    @app.callback(
        Output('portfolio-selector', 'options'),
        Input('refresh-interval', 'n_intervals')
    )
    def update_portfolio_list(n):
        manager = PortfolioManager(StockDataExtractor)
        portfolios = manager.list_portfolios()
        return [{'label': name, 'value': name} for name in portfolios]
    
    @app.callback(
        [Output('portfolio-summary', 'children'),
         Output('portfolio-allocation-chart', 'figure'),
         Output('portfolio-performance-chart', 'figure'),
         Output('positions-pnl-chart', 'figure'),
         Output('portfolio-data-store', 'data')],
        Input('load-portfolio-btn', 'n_clicks'),
        State('portfolio-selector', 'value')
    )
    def load_portfolio(n_clicks, portfolio_name):
        if not portfolio_name:
            return no_update
        
        manager = PortfolioManager(StockDataExtractor)
        portfolio = manager.load_portfolio(portfolio_name)
        if not portfolio:
            return "Portefeuille non trouvé", go.Figure(), go.Figure(), go.Figure(), {}
        
        analysis = manager.analyze_portfolio()
        perf = analysis['performance']
        
        # Résumé textuel
        summary = html.Div([
            html.P(f"Valeur totale: ${perf['total_value']:,.2f}"),
            html.P(f"Liquidités: ${perf['cash']:,.2f}"),
            html.P(f"Investi: ${perf['invested']:,.2f}"),
            html.P(f"P&L total: ${perf['total_pnl']:,.2f} ({perf['total_pnl_pct']:.2f}%)"),
            html.P(f"Positions: {perf['num_positions']}"),
        ])
        
        # Graphique d'allocation (camembert)
        alloc = analysis['allocation']
        if alloc:
            labels = list(alloc.keys())
            values = list(alloc.values())
            fig_alloc = px.pie(values=values, names=labels, title="Allocation actuelle")
        else:
            fig_alloc = go.Figure()
        
        # Graphique de performance (si historique disponible)
        if hasattr(portfolio, 'performance_history') and portfolio.performance_history:
            df = pd.DataFrame(portfolio.performance_history)
            fig_perf = px.line(df, x='date', y='total_value', title="Évolution de la valeur")
        else:
            fig_perf = go.Figure()
        
        # Graphique des P&L par position
        positions_data = []
        for ticker, data in perf['positions'].items():
            positions_data.append({
                'Ticker': ticker,
                'P&L %': data['unrealized_pnl_pct'],
                'P&L $': data['unrealized_pnl']
            })
        if positions_data:
            df_pnl = pd.DataFrame(positions_data)
            fig_pnl = px.bar(df_pnl, x='Ticker', y='P&L %', 
                             color='P&L %', color_continuous_scale='RdYlGn',
                             title="P&L non réalisé par position (%)")
        else:
            fig_pnl = go.Figure()
        
        return summary, fig_alloc, fig_perf, fig_pnl, analysis
    
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value')
    )
    def render_content(tab):
        from web_dashboard.layouts.portfolio_layout import portfolio_layout
        from web_dashboard.layouts.symbol_layout import symbol_layout
        
        if tab == 'tab-portfolio':
            return portfolio_layout()
        elif tab == 'tab-symbol':
            return symbol_layout()