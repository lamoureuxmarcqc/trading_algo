# python trading_algo/web_dashboard/layouts/market_layout.py
"""
Dash layout for Market Overview tab (enhanced controls: sector filter, download, auto-refresh).
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
            dbc.Col(dbc.Button("Download top movers", id='market-download-btn', color='info'), width=2),
            dbc.Col(dcc.Download(id='market-download'), width=2),
            dbc.Col(dcc.Checklist(options=[{'label': 'Auto-refresh', 'value': 'on'}], value=['on'], id='market-auto-refresh', inline=True), width=1),
        ], align='center', className='mb-2'),

        # NEW: Market health summary row (small cards)
        dbc.Row([
            dbc.Col(html.Div(id='market-health', children=[], className="mb-3"), width=12)
        ]),

        dbc.Row([
            dbc.Col(dcc.Loading(id='market-loading', type='circle', children=[
                dcc.Graph(id='market-overview-fig')
            ]), width=12)
        ]),
        dbc.Row([
            dbc.Col(html.Div([
                html.Label("Filter top movers by sector:"),
                dcc.Dropdown(id='market-sector-filter', options=[{'label': 'All', 'value': 'ALL'}], value='ALL', clearable=False, style={'width': '250px'})
            ]), width=4)
        ], className='mt-2 mb-2'),
        # DEBUG: show raw cache contents for troubleshooting
        dbc.Row([
            dbc.Col(html.Hr(), width=12),
            dbc.Col(html.H5("Debug: market_overview cache (server)"), width=12),
            dbc.Col(html.Pre(id='market-cache-raw', style={'whiteSpace': 'pre-wrap', 'maxHeight': '240px', 'overflowY': 'auto', 'backgroundColor': '#f8f9fa', 'padding': '8px', 'borderRadius': '4px'}), width=12)
        ], className='mt-3')
    ])