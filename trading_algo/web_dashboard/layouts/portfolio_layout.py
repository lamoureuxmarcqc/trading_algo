from dash import html, dcc
import dash_bootstrap_components as dbc

# =========================================================
# 🔹 KPI CARD COMPONENT (REUSABLE / PRODUCTION READY)
# =========================================================
def _create_kpi_card(
    title: str,
    element_id: str,
    icon_class: str,
    color: str = "primary"
) -> dbc.Col:
    """
    Génère une carte KPI standardisée.

    Design:
    - Lisible
    - Rapide à mettre à jour via callbacks
    - Compatible dark/light mode futur
    """
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
                                    html.Span(
                                        title,
                                        className="text-muted small fw-bold text-uppercase"
                                    ),
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
# 🔹 MAIN LAYOUT
# =========================================================
def portfolio_layout():
    """
    Layout principal du Command Center Portfolio.

    Architecture:
    - Header stratégique
    - KPI temps réel
    - Dashboard IA / Technique
    - Positions
    - Risk + Monte Carlo

    Prêt pour:
    - AI advisor
    - Optimisation portefeuille
    - Allocation dynamique
    """

    return dbc.Container(
        fluid=True,
        className="p-4",
        children=[

            # =========================================================
            # 🔹 STORES & TIMERS (CRITICAL FOR DASH STABILITY)
            # =========================================================
            dcc.Store(id='portfolio-data-store', storage_type="memory"),

            dcc.Interval(
                id='portfolio-refresh-timer',
                interval=60 * 1000,  # 60 sec
                n_intervals=0
            ),

            # =========================================================
            # 🔹 HEADER
            # =========================================================
            dbc.Row(
                className="mb-4 align-items-center",
                children=[
                    dbc.Col(
                        md=6,
                        children=[
                            html.H1(
                                "Command Center",
                                className="fw-bold text-dark mb-0"
                            ),
                            html.P(
                                "Intelligence Artificielle & Gestion des Risques",
                                className="text-muted"
                            ),
                        ],
                    ),

                    dbc.Col(
                        md=6,
                        children=[
                            dbc.InputGroup(
                                className="shadow-sm",
                                children=[
                                    dcc.Dropdown(
                                        id='portfolio-selector',
                                        placeholder="Sélectionner un portefeuille...",
                                        className="flex-grow-1",
                                        clearable=True,
                                    ),
                                    dbc.Button(
                                        "Charger",
                                        id='load-portfolio-btn',
                                        color="primary"
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # =========================================================
            # 🔹 KPI DASHBOARD
            # =========================================================
            dbc.Row(
                className="mb-4 g-3",
                children=[
                    _create_kpi_card("Valeur Totale", "kpi-total-value", "bi bi-wallet2"),
                    _create_kpi_card("Performance J", "kpi-return", "bi bi-graph-up", "success"),
                    _create_kpi_card("Volatilité Ann.", "kpi-volatility", "bi bi-lightning", "warning"),
                    _create_kpi_card("Sharpe Ratio", "kpi-sharpe", "bi bi-trophy", "info"),
                ],
            ),

            # =========================================================
            # 🔹 MAIN ANALYTICS SECTION
            # =========================================================
            dbc.Row(
                children=[

                    # ================= LEFT SIDE =================
                    dbc.Col(
                        lg=8,
                        children=[

                            dbc.Card(
                                className="shadow-sm mb-4",
                                children=[

                                    # ---- Tabs IA / Technique ----
                                    dbc.CardHeader(
                                        className="bg-white",
                                        children=[
                                            dbc.Tabs(
                                                id="card-tabs",
                                                active_tab="tab-strat",
                                                children=[
                                                    dbc.Tab(
                                                        label="Conseiller Stratégique (IA)",
                                                        tab_id="tab-strat"
                                                    ),
                                                    dbc.Tab(
                                                        label="Analyse Technique (Asset)",
                                                        tab_id="tab-tech"
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),

                                    # ---- Body ----
                                    dbc.CardBody(
                                        children=[
                                            dcc.Loading(
                                                type="dot",
                                                children=[

                                                    # STRATEGY VIEW
                                                    html.Div(
                                                        id='strat-view',
                                                        style={"display": "block"},
                                                        children=[
                                                            dcc.Graph(
                                                                id='portfolio-main-dashboard',
                                                                style={'height': '800px'}
                                                            )
                                                        ],
                                                    ),

                                                    # TECHNICAL VIEW
                                                    html.Div(
                                                        id='tech-view',
                                                        style={"display": "none"},
                                                        children=[
                                                            dcc.Graph(
                                                                id='technical-analysis-graph',
                                                                style={'height': '800px'}
                                                            )
                                                        ],
                                                    ),
                                                ],
                                            )
                                        ]
                                    ),
                                ],
                            ),

                            # ---- Positions ----
                            html.H4(
                                "Positions Actives",
                                className="fw-bold mb-3 mt-4"
                            ),

                            html.Div(
                                id='positions-table',
                                className="bg-white rounded shadow-sm p-2"
                            ),
                        ],
                    ),

                    # ================= RIGHT SIDE =================
                    dbc.Col(
                        lg=4,
                        children=[

                            # ---- Monte Carlo ----
                            dbc.Card(
                                className="shadow-sm mb-4 bg-dark text-white",
                                children=[
                                    dbc.CardBody(
                                        children=[
                                            html.H5(
                                                "Simulation Prédictive",
                                                className="text-info mb-3"
                                            ),

                                            dbc.Button(
                                                "Lancer Monte-Carlo",
                                                id="run-monte-carlo-btn",
                                                color="info",
                                                className="w-100"
                                            ),

                                            dcc.Loading(
                                                children=[
                                                    html.Div(
                                                        id="monte-carlo-container",
                                                        className="mt-3"
                                                    )
                                                ]
                                            ),
                                        ]
                                    )
                                ],
                            ),

                            # ---- Risk Metrics ----
                            dbc.Card(
                                className="shadow-sm",
                                children=[
                                    dbc.CardHeader(
                                        "Indicateurs de Risque",
                                        className="fw-bold"
                                    ),
                                    dbc.CardBody(
                                        id='risk-metrics-div',
                                        className="p-2"
                                    ),
                                ],
                            ),
                        ],
                    ),
                ]
            ),
        ],
    )