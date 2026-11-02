"""
Portfolio Layout - Version corrigée et complète (16 avril 2026)
Toutes les fonctionnalités originales sont conservées.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

# =========================================================
# 🔹 KPI CARD COMPONENT
# =========================================================
def _create_kpi_card(
    title: str,
    element_id: str,
    icon_class: str,
    color: str = "primary"
) -> dbc.Col:
    return dbc.Col(
        xs=12, sm=6, md=3,
        children=[
            dbc.Card(
                className=f"shadow-sm border-0 border-start border-{color} border-5 h-100",
                children=[
                    dbc.CardBody(
                        className="p-3",
                        children=[
                            html.Div(
                                className="mb-2 d-flex align-items-center",
                                children=[
                                    html.I(className=f"{icon_class} me-2 text-{color}"),
                                    html.Span(title, className="text-muted small fw-bold text-uppercase"),
                                ],
                            ),
                            html.H3(
                                id=element_id,
                                className="mb-0 fw-bold",
                                children="---"
                            ),
                        ],
                    )
                ],
            )
        ],
    )


# =========================================================
# 🔹 MAIN PORTFOLIO LAYOUT - VERSION COMPLÈTE
# =========================================================
def portfolio_layout():
    return dbc.Container(
        fluid=True,
        className="p-4",
        children=[
            # Stores & Timers
            dcc.Store(id='portfolio-data-store', storage_type="memory"),
            dcc.Store(id='allocation-model-store', data='buffett'),
            dcc.Store(id='optimization-store', storage_type="memory"),
            dcc.Interval(id='portfolio-refresh-timer', interval=60 * 1000, n_intervals=0),

            # Header
            dbc.Row(className="mb-4 align-items-center", children=[
                dbc.Col(md=5, children=[
                    html.H1("Command Center", className="fw-bold text-dark mb-0"),
                    html.P("Intelligence Artificielle & Gestion des Risques", className="text-muted"),
                ]),
                dbc.Col(md=7, children=[
                    dbc.InputGroup(className="shadow-sm", children=[
                        dcc.Dropdown(
                            id='portfolio-selector',
                            placeholder="Sélectionner un portefeuille...",
                            className="flex-grow-1",
                            clearable=True,
                        ),
                        dbc.Button("Charger", id='load-portfolio-btn', color="primary", className="me-2"),
                        dcc.Dropdown(
                            id='allocation-model',
                            options=[
                                {'label': 'Buffett Concentré', 'value': 'buffett'},
                                {'label': 'Risk Parity', 'value': 'risk_parity'},
                                {'label': 'Equal Weight', 'value': 'equal_weight'},
                                {'label': 'Optimisation (Max Sharpe)', 'value': 'optimization'}
                            ],
                            value='buffett',
                            className="flex-grow-1",
                            clearable=False,
                            style={'minWidth': '160px'}
                        ),
                    ]),
                ]),
            ]),

            # KPI Dashboard
            dbc.Row(className="mb-4 g-3", children=[
                _create_kpi_card("Valeur Totale", "kpi-total-value", "bi bi-wallet2"),
                _create_kpi_card("Performance J", "kpi-return", "bi bi-graph-up", "success"),
                _create_kpi_card("Volatilité Ann.", "kpi-volatility", "bi bi-lightning", "warning"),
                _create_kpi_card("Sharpe Ratio", "kpi-sharpe", "bi bi-trophy", "info"),
            ]),

            # Main Analytics
            dbc.Row(children=[
                dbc.Col(lg=8, children=[
                    dbc.Card(className="shadow-sm mb-4", children=[
                        dbc.CardHeader(className="bg-white", children=[
                            dbc.Tabs(id="card-tabs", active_tab="tab-strat", children=[
                                dbc.Tab(label="Conseiller Stratégique (IA)", tab_id="tab-strat"),
                                dbc.Tab(label="Analyse Technique (Asset)", tab_id="tab-tech"),
                                dbc.Tab(label="Allocation cible", tab_id="tab-alloc"),
                                dbc.Tab(label="⚙️ Optimisation", tab_id="tab-opt"),
                            ]),
                        ]),
                        dbc.CardBody(style={"minHeight": "820px"}, children=[
                            dcc.Loading(type="dot", color="primary", children=[
                                # Stratégique
                                html.Div(id='strat-view', children=[
                                    dcc.Graph(id='portfolio-main-dashboard', style={'height': '780px'})
                                ]),
                                # Technique
                                html.Div(id='tech-view', style={"display": "none"}, children=[
                                    dcc.Graph(id='technical-analysis-graph', style={'height': '780px'})
                                ]),
                                # Allocation cible
                                html.Div(id='alloc-view', style={"display": "none"}, children=[
                                    dcc.Graph(id='allocation-graph', style={'height': '580px'}),
                                    dbc.Button("Appliquer le rééquilibrage", id="rebalance-btn",
                                               color="warning", className="mt-3 w-100"),
                                ]),
                                # Optimisation - SECTION COMPLÈTE
                                html.Div(id='opt-view', style={"display": "none"}, children=[
                                    html.H5("Paramètres de l'optimisation", className="mb-3"),

                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Tickers (séparés par virgules)"),
                                            dcc.Input(id="opt-tickers", type="text",
                                                      value="CNR.TO, BN.TO, CNQ.TO, XIU.TO, CSU.TO, RY.TO, SHOP.TO",
                                                      className="form-control mb-2")
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Label("Ticker obligataire (optionnel)"),
                                            dcc.Input(id="opt-bond-ticker", type="text",
                                                      value="AGG", className="form-control mb-2")
                                        ], md=6),
                                    ]),

                                    dbc.Row([
                                        dbc.Col([dbc.Label("Date de début"), dcc.Input(id="opt-start-date", type="text", value="2018-01-01", className="form-control mb-2")], md=3),
                                        dbc.Col([dbc.Label("Date de fin"), dcc.Input(id="opt-end-date", type="text", value="2025-01-01", className="form-control mb-2")], md=3),
                                        dbc.Col([dbc.Label("Horizon (années)"), dcc.Input(id="opt-horizon", type="number", value=4, className="form-control mb-2")], md=2),
                                        dbc.Col([
                                            dbc.Label("Fréquence rééquilibrage"),
                                            dcc.Dropdown(id="opt-rebalance-freq", options=[
                                                {"label": "Mensuel (ME)", "value": "ME"},
                                                {"label": "Trimestriel (QE)", "value": "QE"},
                                                {"label": "Annuel (YE)", "value": "YE"}
                                            ], value="ME", className="mb-2")
                                        ], md=4),
                                    ]),

                                    dbc.Row([
                                        dbc.Col([dbc.Label("Capital initial ($)"), dcc.Input(id="opt-capital", type="number", value=5_000_000, className="form-control mb-2")], md=3),
                                        dbc.Col([dbc.Label("Objectif ($)"), dcc.Input(id="opt-objectif", type="number", value=50_000_000, className="form-control mb-2")], md=3),
                                        dbc.Col([dbc.Label("Frais transaction (%)"), dcc.Input(id="opt-transaction-cost", type="number", value=0.001, step=0.001, className="form-control mb-2")], md=3),
                                        dbc.Col([dbc.Label("Taux sans risque"), dcc.Input(id="opt-risk-free-rate", type="number", value=0.02, step=0.01, className="form-control mb-2")], md=3),
                                    ]),

                                    dbc.Button("Lancer l'optimisation", id="run-optimization-btn",
                                               color="primary", className="mt-2 mb-4 w-100"),

                                    html.Div(id="optimization-results-container", className="mt-3")
                                ]),
                            ])
                        ]),
                    ]),

                    html.H4("Positions Actives", className="fw-bold mb-3 mt-4"),
                    html.Div(id='positions-table', className="bg-white rounded shadow-sm p-3"),
                ]),

                # RIGHT SIDE
                dbc.Col(lg=4, children=[
                    dbc.Card(className="shadow-sm mb-4", children=[
                        dbc.CardHeader("Régime de Marché"),
                        dbc.CardBody(id="market-regime-div", children=[html.Div("Chargement...", className="text-center text-muted")]),
                    ]),
                    dbc.Card(className="shadow-sm mb-4 bg-dark text-white", children=[
                        dbc.CardBody([
                            html.H5("Simulation Prédictive", className="text-info mb-3"),
                            dbc.Button("Lancer Monte-Carlo", id="run-monte-carlo-btn", color="info", className="w-100"),
                            dcc.Loading(children=[html.Div(id="monte-carlo-container", className="mt-3")]),
                        ])
                    ]),
                    dbc.Card(className="shadow-sm", children=[
                        dbc.CardHeader("Indicateurs de Risque", className="fw-bold"),
                        dbc.CardBody(id='risk-metrics-div', className="p-3"),
                    ]),
                ]),
            ]),
        ],
    )