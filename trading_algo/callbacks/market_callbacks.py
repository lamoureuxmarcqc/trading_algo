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
        Create an Excel workbook with multiple sheets (indices, sector_perf, top_gainers, top_losers, macro).
        The function writes into the provided buffer via dcc.send_bytes.
        """
        try:
            cache = _read_cache() or {}
            # rebuild small DataFrames
            top = pd.DataFrame(cache.get('top_gainers', []))
            losers = pd.DataFrame(cache.get('top_losers', []))
            sector_df = pd.DataFrame(cache.get('sector_perf', []))
            macro = cache.get('macro', {})

            # apply sector filter if requested
            if sector_filter and sector_filter != 'ALL' and not top.empty:
                top = top[top['sector'] == sector_filter]

            # indices sheet: reconstruct indices dict -> DataFrame (one sheet per index)
            indices_cache = cache.get('indices', {}) or {}

            import io
            def _to_excel_bytes(buffer):
                # buffer is a file-like object provided by Dash
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    # Top movers
                    if not top.empty:
                        top.to_excel(writer, sheet_name='Top Gainers', index=False)
                    else:
                        pd.DataFrame(columns=['symbol','perf','sector','volume']).to_excel(writer, sheet_name='Top Gainers', index=False)
                    # Top losers
                    if not losers.empty:
                        losers.to_excel(writer, sheet_name='Top Losers', index=False)
                    # Sector perf
                    if not sector_df.empty:
                        sector_df.to_excel(writer, sheet_name='Sector Perf', index=False)
                    # Macro: write as key/value
                    if isinstance(macro, dict) and macro:
                        macro_items = []
                        for k, v in macro.items():
                            macro_items.append({'indicator': k, 'value': str(v)})
                        pd.DataFrame(macro_items).to_excel(writer, sheet_name='Macro', index=False)
                    # Indices: each index as its own sheet (or a combined sheet)
                    for sym, idx in indices_cache.items():
                        try:
                            # ensure safe sheet name
                            sheet_name = str(sym)[:31]
                            df_idx = None
                            dates = idx.get('dates', [])
                            closes = idx.get('closes', [])
                            if len(dates) == len(closes) and dates:
                                df_idx = pd.DataFrame({'Date': dates, 'Close': closes})
                            elif closes:
                                df_idx = pd.DataFrame({'Close': closes})
                            if df_idx is not None:
                                df_idx.to_excel(writer, sheet_name=f"Idx_{sheet_name}", index=False)
                        except Exception:
                            # ignore per-index failure
                            logger.debug(f"Failed writing index sheet for {sym}")
                # writer.close() implicitly called

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_export_{ts}.xlsx"
            return dcc.send_bytes(_to_excel_bytes, filename=filename)
        except Exception:
            logger.exception("Failed to prepare top movers Excel export")
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