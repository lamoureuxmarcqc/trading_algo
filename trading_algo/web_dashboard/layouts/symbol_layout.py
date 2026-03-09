# trading_algo/web_dashboard/layouts/symbol_layout.py
from dash import html, dcc

def symbol_layout():
    return html.Div([
        html.H1("Analyse par Symbole", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H3("Symbole"),
                dcc.Input(id='symbol-input', type='text', placeholder='ex: AAPL'),
            ], className='six columns'),
            
            html.Div([
                html.H3("Période"),
                dcc.Dropdown(
                    id='period-dropdown',
                    options=[
                        {'label': '1 mois', 'value': '1mo'},
                        {'label': '3 mois', 'value': '3mo'},
                        {'label': '6 mois', 'value': '6mo'},
                        {'label': '1 an', 'value': '1y'},
                        {'label': '2 ans', 'value': '2y'},
                        {'label': '5 ans', 'value': '5y'},
                    ],
                    value='1y'
                ),
            ], className='six columns'),
        ], className='row'),
        
        html.Div([
            html.Button('Analyser', id='analyze-btn', n_clicks=0),
        ], style={'textAlign': 'center', 'margin': '20px'}),
        
        html.Div(id='symbol-analysis-results')
    ])