# python trading_algo/callbacks/market_callbacks.py
from dash import Input, Output, State, html
from trading_algo.visualization.market_dashboard import MarketDashboard
from trading_algo.data.data_extraction import StockDataExtractor, MacroDataExtractor
import pandas as pd
import logging
import os
import json
import datetime

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.getcwd(), "cache", "market_overview.json")
SNAPSHOT_DIR = os.path.join(os.getcwd(), "dashboards")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def _read_cache():
    try:
        if not os.path.exists(CACHE_FILE):
            return None
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed reading market cache")
        return None

def _reconstruct_indices(indices_cache):
    """
    Convert indices JSON to DataFrame dict expected by MarketDashboard.create_figure
    """
    res = {}
    try:
        for sym, d in (indices_cache or {}).items():
            dates = pd.to_datetime(d.get('dates', []))
            closes = d.get('closes', [])
            if len(dates) != len(closes):
                continue
            df = pd.DataFrame({'Close': closes}, index=dates)
            res[sym] = df
    except Exception:
        logger.exception("Failed reconstructing indices")
    return res

def register_market_callbacks(app):
    @app.callback(
        Output('market-overview-fig', 'figure'),
        Input('market-refresh-btn', 'n_clicks'),
        State('market-period-dropdown', 'value'),
    )
    def update_market_overview(n_clicks, period):
        """
        Prefer cached snapshot produced by background job. If absent use a lightweight live fetch.
        """
        try:
            cache = _read_cache()
            if cache:
                indices = _reconstruct_indices(cache.get('indices'))
                sector_perf = pd.DataFrame(cache.get('sector_perf', []))
                top_gainers = pd.DataFrame(cache.get('top_gainers', []))
                top_losers = pd.DataFrame(cache.get('top_losers', []))
                macro = cache.get('macro', {})
            else:
                # fallback: do a small live fetch
                sde = StockDataExtractor()
                macro = MacroDataExtractor().get_all_macro_data()
                indices = {}
                for symbol in ['^GSPC', '^IXIC', '^DJI', '^FTSE', '^GDAXI', '^GSPTSE']:
                    try:
                        df = sde.get_historical_data(symbol=symbol, period='3mo', interval='1d')
                        indices[symbol] = df
                    except Exception as e:
                        logger.debug(f"index fetch error {symbol}: {e}")
                sector_perf = pd.DataFrame({'sector': ['Tech','Health','Financials'], 'perf': [1.2, -0.5, 0.8]})
                top_gainers = pd.DataFrame({'symbol':['AAPL','TSLA'],'perf':[2.3,-3.1],'volume':[1_000_000,2_000_000]})
                top_losers = top_gainers.copy()

            md = MarketDashboard()
            md.load_data(indices=indices, sector_perf=sector_perf, top_movers=top_gainers, macro=macro, commodities=None, currencies=None, period_label=period)
            fig = md.create_figure()
            return fig
        except Exception as e:
            logger.exception("Failed to update market overview")
            return go.Figure()

    @app.callback(
        Output('market-snapshot-output', 'children'),
        Input('market-snapshot-btn', 'n_clicks'),
        State('market-period-dropdown', 'value'),
    )
    def snapshot_market(n_clicks, period):
        """
        Save an HTML snapshot of the market figure and return a download link.
        Uses cached data to build the figure so snapshot is fast and deterministic.
        """
        if not n_clicks:
            return ""
        try:
            cache = _read_cache() or {}
            indices = _reconstruct_indices(cache.get('indices'))
            sector_perf = pd.DataFrame(cache.get('sector_perf', []))
            top_gainers = pd.DataFrame(cache.get('top_gainers', []))
            macro = cache.get('macro', {})

            md = MarketDashboard()
            md.load_data(indices=indices, sector_perf=sector_perf, top_movers=top_gainers, macro=macro, commodities=None, currencies=None, period_label=period)
            fig = md.create_figure()

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_snapshot_{ts}.html"
            path = os.path.join(SNAPSHOT_DIR, filename)
            fig.write_html(path, include_plotlyjs="cdn")
            link = html.A("Download snapshot", href=f"/{path}", target="_blank", className="btn btn-link")
            return link
        except Exception:
            logger.exception("Failed to create snapshot")
            return html.Div("Snapshot failed", className="text-danger")