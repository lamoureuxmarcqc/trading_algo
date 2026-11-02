"""
Dash layout for Market Overview tab - Version corrigée et améliorée (16 avril 2026)
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

def market_layout():
    """Layout complet et robuste pour l'onglet Macro & Risques."""
    return dbc.Container(
        fluid=True,
        className="p-0",
        children=[
            # =========================================================
            # HEADER & CONTROLS
            # =========================================================
            dbc.Row(
                className="mb-4 align-items-center",
                children=[
                    dbc.Col(
                        html.H2("🌍 Macro & Risques du Marché", className="fw-bold text-dark mb-0"),
                        md=6
                    ),
                    dbc.Col(
                        className="text-end",
                        md=6,
                        children=[
                            dbc.ButtonGroup(
                                children=[
                                    dbc.Button("Refresh", id='market-refresh-btn', color='primary', size="sm"),
                                    dbc.Button("Snapshot", id='market-snapshot-btn', color='secondary', size="sm"),
                                    dbc.Button("Download Report", id='market-download-btn', color='info', size="sm"),
                                ]
                            ),
                            dcc.Download(id='market-download'),
                        ]
                    ),
                ]
            ),

            # =========================================================
            # AUTO-REFRESH & PERIOD
            # =========================================================
            dbc.Row(
                className="mb-3 g-3 align-items-center",
                children=[
                    dbc.Col(
                        dbc.InputGroup([
                            dbc.InputGroupText("Période"),
                            dcc.Dropdown(
                                id='market-period-dropdown',
                                options=[
                                    {'label': '1 Mois', 'value': '1M'},
                                    {'label': '3 Mois', 'value': '3M'},
                                    {'label': '6 Mois', 'value': '6M'},
                                    {'label': '1 An', 'value': '1Y'},
                                    {'label': '5 Ans', 'value': '5Y'},
                                ],
                                value='1M',
                                clearable=False,
                                style={'width': '160px'}
                            ),
                        ]),
                        md=3
                    ),
                    dbc.Col(
                        dcc.Checklist(
                            id='market-auto-refresh',
                            options=[{'label': ' Auto-refresh toutes les 30s', 'value': 'on'}],
                            value=['on'],
                            inline=True,
                            className="ms-3"
                        ),
                        md=3
                    ),
                    dbc.Col(width=6),  # espace
                ]
            ),

            # =========================================================
            # MARKET HEALTH SUMMARY (utilisé par le callback)
            # =========================================================
            html.Div(
                id='market-health',
                className="mb-4",
                children=[html.Div("Chargement des indicateurs de santé du marché...", className="text-muted")]
            ),

            # =========================================================
            # MAIN GRAPH
            # =========================================================
            dbc.Card(
                className="shadow-sm mb-4",
                children=[
                    dbc.CardHeader("Vue d'ensemble du marché"),
                    dbc.CardBody(
                        dcc.Loading(
                            id='market-loading',
                            type='circle',
                            color="primary",
                            children=[
                                dcc.Graph(
                                    id='market-overview-fig',
                                    style={'height': '680px'},   # Hauteur forcée obligatoire
                                    config={'displayModeBar': True, 'scrollZoom': True}
                                )
                            ]
                        )
                    )
                ]
            ),

            # =========================================================
            # SECTOR FILTER
            # =========================================================
            dbc.Row(
                className="mb-4",
                children=[
                    dbc.Col(
                        md=5,
                        children=[
                            html.Label("Filtrer les Top Movers par secteur :", className="fw-bold text-muted small"),
                            dcc.Dropdown(
                                id='market-sector-filter',
                                options=[{'label': 'Tous les secteurs', 'value': 'ALL'}],
                                value='ALL',
                                clearable=False,
                                className="shadow-sm"
                            )
                        ]
                    ),
                ]
            ),

            # =========================================================
            # DEBUG SECTION (utile pour le diagnostic)
            # =========================================================
            dbc.Card(
                className="shadow-sm",
                children=[
                    dbc.CardHeader(
                        html.H5("Debug — Contenu brut du cache market", className="mb-0"),
                        className="bg-light"
                    ),
                    dbc.CardBody(
                        html.Pre(
                            id='market-cache-raw',
                            style={
                                'whiteSpace': 'pre-wrap',
                                'maxHeight': '260px',
                                'overflowY': 'auto',
                                'backgroundColor': '#f8f9fa',
                                'padding': '12px',
                                'borderRadius': '6px',
                                'fontSize': '0.85rem'
                            }
                        )
                    )
                ]
            ),
        ]
    )