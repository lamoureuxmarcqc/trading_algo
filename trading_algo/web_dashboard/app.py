# trading_algo/web_dashboard/app.py
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au chemin pour pouvoir importer trading_algo
sys.path.insert(0, str(Path(__file__).parent.parent))

from dash import Dash, html, dcc
from trading_algo.web_dashboard.layouts.portfolio_layout import portfolio_layout
from trading_algo.web_dashboard.layouts.symbol_layout import symbol_layout
from trading_algo.callbacks.portfolio_callbacks import register_portfolio_callbacks
from trading_algo.callbacks.symbol_callbacks import register_symbol_callbacks

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Tabs(id="main-tabs", value="tab-portfolio", children=[
        dcc.Tab(label="Portefeuille global", value="tab-portfolio"),
        dcc.Tab(label="Analyse par symbole", value="tab-symbol"),
    ]),
    html.Div(id="tab-content")
])

# Enregistrement des callbacks
register_portfolio_callbacks(app)
register_symbol_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)