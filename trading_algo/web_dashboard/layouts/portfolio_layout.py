# trading_algo/web_dashboard/layouts/portfolio_layout.py
from dash import html, dcc
import plotly.graph_objects as go

def portfolio_layout():
    return html.Div([
        html.H1("Portefeuille Global", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H3("Sélectionner un portefeuille"),
                dcc.Dropdown(
                    id='portfolio-selector',
                    placeholder='Choisissez un portefeuille...'
                ),
                html.Button('Charger', id='load-portfolio-btn', n_clicks=0),
            ], className='four columns'),
            
            html.Div([
                html.H3("Résumé"),
                html.Div(id='portfolio-summary')
            ], className='four columns'),
            
            html.Div([
                html.H3("Actions rapides"),
                html.Button('Rééquilibrer', id='rebalance-btn', n_clicks=0),
                html.Button('Exporter PDF', id='export-pdf-btn', n_clicks=0),
            ], className='four columns'),
        ], className='row'),
        
        html.Hr(),
        
        html.Div([
            dcc.Graph(id='portfolio-allocation-chart'),
        ], className='row'),
        
        html.Div([
            dcc.Graph(id='portfolio-performance-chart'),
        ], className='row'),
        
        html.Div([
            dcc.Graph(id='positions-pnl-chart'),
        ], className='row'),
        
        dcc.Store(id='portfolio-data-store'),
        dcc.Interval(id='refresh-interval', interval=60*1000)  # Rafraîchissement toutes les minutes
    ])