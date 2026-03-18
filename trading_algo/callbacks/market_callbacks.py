# python trading_algo/callbacks/market_callbacks.py
from dash import Input, Output, State, html, dcc
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
                # fallback: try to align by length
                closes = closes[-len(dates):] if len(closes) >= len(dates) else closes
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
        Output('market-cache-raw', 'children'),
        Input('market-refresh-btn', 'n_clicks'),
        Input('market-period-dropdown', 'value'),
        Input('market-sector-filter', 'value'),
        State('market-auto-refresh', 'value'),
    )
    def update_market_overview(n_clicks, period, sector_filter, auto_refresh_val):
        """
        Prefer cached snapshot produced by background job. If absent use a lightweight live fetch.
        Filters top movers by sector (sector_filter='ALL' means no filter).
        """
        try:
            # if auto-refresh is turned off and this was triggered by interval/auto, respect it (handled by caller)
            cache = _read_cache()
            macro = {}
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
                top_gainers = pd.DataFrame({'symbol':['AAPL','TSLA'],'perf':[2.3,-3.1],'sector':['Technology','Automotive'],'volume':[1_000_000,2_000_000]})
                top_losers = top_gainers.copy()

            # Filter top movers by sector if requested
            if isinstance(top_gainers, pd.DataFrame) and sector_filter and sector_filter != 'ALL':
                try:
                    top_gainers = top_gainers[top_gainers['sector'] == sector_filter]
                except Exception:
                    pass

            md = MarketDashboard()
            md.load_data(indices=indices, sector_perf=sector_perf, top_movers=top_gainers, macro=macro, commodities=None, currencies=None, period_label=period)
            fig = md.create_figure()

            # prepare raw cache pretty-print for debug
            raw_pretty = json.dumps(cache or {}, indent=2, ensure_ascii=False)
            if len(raw_pretty) > 50000:
                raw_pretty = raw_pretty[:50000] + "\n\n...truncated..."
            return fig, raw_pretty
        except Exception as e:
            logger.exception("Failed to update market overview")
            return {}, "Error building figure"

    @app.callback(
        Output('market-download', 'data'),
        Input('market-download-btn', 'n_clicks'),
        State('market-sector-filter', 'value'),
        prevent_initial_call=True
    )
    def download_top_movers(n_clicks, sector_filter):
        """
        Create a CSV of the current top movers (filtered by sector) and send it to the user.
        """
        try:
            cache = _read_cache() or {}
            top = pd.DataFrame(cache.get('top_gainers', []))
            if sector_filter and sector_filter != 'ALL' and not top.empty:
                top = top[top['sector'] == sector_filter]
            if top.empty:
                # return a small CSV placeholder
                csv = "symbol,perf,sector,volume\n"
                return dcc.send_data_frame(lambda df: df.to_csv(index=False), pd.DataFrame(), filename=f"top_movers_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            return dcc.send_data_frame(top.to_csv, filename=f"top_movers_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        except Exception:
            logger.exception("Failed to prepare top movers CSV")
            return None

    # Populate sector filter options from cache on interval (keep it fast)
    @app.callback(
        Output('market-sector-filter', 'options'),
        Input('market-cache-raw', 'children'),
    )
    def populate_sector_filter(_raw):
        try:
            cache = _read_cache() or {}
            sectors = set()
            for t in cache.get('top_gainers', []) + cache.get('top_losers', []):
                if isinstance(t, dict):
                    s = t.get('sector')
                    if s:
                        sectors.add(s)
            options = [{'label': 'All', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in sorted(sectors)]
            return options
        except Exception:
            return [{'label': 'All', 'value': 'ALL'}]