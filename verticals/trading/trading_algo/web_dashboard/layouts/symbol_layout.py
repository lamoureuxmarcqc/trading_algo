from dash import html, dcc
import dash_bootstrap_components as dbc

def symbol_layout():
    """Layout déclaratif pour l'onglet 'Analyse Symbole'."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Analyse Technique Avancée", className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Symbole Boursier", html_for="symbol-input"),
                                dbc.Input(id='symbol-input', placeholder="ex: NVDA, BTC-USD...", type="text"),
                            ], width=4),
                            dbc.Col([
                                dbc.Label("Période", html_for="period-dropdown"),
                                dcc.Dropdown(
                                    id='period-dropdown',
                                    options=[
                                        {'label': '1 Mois', 'value': '1mo'},
                                        {'label': '3 Mois', 'value': '3mo'},
                                        {'label': '6 Mois', 'value': '6mo'},
                                        {'label': '1 An', 'value': '1y'},
                                        {'label': '5 Ans', 'value': '5y'},
                                    ],
                                    value='1y',
                                    clearable=False,
                                    style={"minWidth": "140px"}
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Br(),
                                dbc.Button("Lancer l'Analyse", id='analyze-btn', color="primary", className="w-100"),
                            ], width=4),
                        ], className="align-items-end")
                    ])
                ], className="shadow-sm border-0 mb-4"),
                
                # Zone où le résultat (cards + graphique) sera injecté par le callback
                html.Div(id='symbol-analysis-results')
            ], width=12)
        ])
    ], fluid=True, className="mt-3")
