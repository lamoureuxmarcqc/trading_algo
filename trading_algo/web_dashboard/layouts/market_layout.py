# python trading_algo/web_dashboard/layouts/market_layout.py
"""
Dash layout for Market Overview tab.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

def market_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='market-period-dropdown', options=[
                {'label': '1M', 'value': '1M'},
                {'label': '3M', 'value': '3M'},
                {'label': '6M', 'value': '6M'},
                {'label': '1Y', 'value': '1Y'},
                {'label': '5Y', 'value': '5Y'},
            ], value='1M', style={'width': '200px'}), width=3),
            dbc.Col(dbc.Button("Refresh", id='market-refresh-btn', color='primary'), width=2),
            dbc.Col(dbc.Button("Snapshot", id='market-snapshot-btn', color='secondary'), width=2),
            dbc.Col(html.Div(id='market-snapshot-output', children=""), width=3),
        ], align='center', className='mb-2'),
        dbc.Row([
            dbc.Col(dcc.Loading(id='market-loading', type='circle', children=[
                dcc.Graph(id='market-overview-fig')
            ]), width=12)
        ])
    ])