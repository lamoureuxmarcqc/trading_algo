import asyncio
import pandas as pd
import numpy as np
import os
import json
import io
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from trading_algo.data.data_extraction import StockDataExtractor, MacroDataExtractor
from trading_algo.strategy.market_regime_engine import MarketRegimeEngine
from trading_algo.screening.actions_sp500 import get_sp500_symbols
from trading_algo import settings

logger = logging.getLogger(__name__)


class MarketManager:
    """
    MarketManager centralise la logique métier liée au contexte de marché :
    - lecture/normalisation du cache
    - agrégation indices / secteurs / top movers
    - orchestration des appels I/O asynchrones
    - export des rapports de marché
    """

    def __init__(self, cache_service: Optional[Any] = None):
        """
        :param cache_service: instance optionnelle de cache (Redis/file). Si None, on utilise directement le fichier JSON.
        """
        self.cache = cache_service
        self.extractor = StockDataExtractor()
        self.macro_extractor = MacroDataExtractor()
        self.regime_engine = MarketRegimeEngine()

    # ----------------------------
    # Cache reading & reconstruction
    # ----------------------------
    def read_cache(self, cache_override: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retourne le contenu du cache. Priorité : cache_override > cache_service (Redis) > fichier JSON.
        """
        if cache_override:
            return cache_override

        # Try using provided cache service (if it exposes get_data or similar)
        if self.cache:
            try:
                data = self.cache.get_data()
                if isinstance(data, dict) and data:
                    return data
            except Exception as e:
                logger.debug("Cache service read failed: %s", e)

        # Fallback on file specified in settings
        cache_file = getattr(settings, "MARKET_CACHE_FILE", os.path.join(os.getcwd(), "cache", "market_overview.json"))
        try:
            if not os.path.exists(cache_file):
                logger.debug("Cache file not found: %s", cache_file)
                return {}
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.exception("Failed to read market cache file %s: %s", cache_file, e)
            return {}

    @staticmethod
    def reconstruct_indices(indices_cache: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Convertit la structure JSON du cache en dict of DataFrame indexées par date.
        Accepte clés flexibles ('dates'/'index' et 'closes'/'Close').
        """
        res: Dict[str, pd.DataFrame] = {}
        if not isinstance(indices_cache, dict):
            return res

        for sym, d in indices_cache.items():
            try:
                if not isinstance(d, dict):
                    continue
                dates = pd.to_datetime(d.get("dates") or d.get("index") or [])
                closes = d.get("closes") or d.get("Close") or []
                if len(dates) > 0 and len(closes) > 0:
                    min_len = min(len(dates), len(closes))
                    df = pd.DataFrame({"Close": closes[-min_len:]}, index=dates[-min_len:])
                    df.index = pd.to_datetime(df.index)
                    res[sym] = df.sort_index()
            except Exception as e:
                logger.debug("Failed reconstructing index %s: %s", sym, e)
        return res

    # ----------------------------
    # High-level overview builder
    # ----------------------------
    def build_overview(
        self,
        store_data: Optional[Dict] = None,
        period_label: str = "1M",
        sector_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Traite le cache (ou le store donné) et renvoie un dict structuré:
        {
            'indices': Dict[str, DataFrame],
            'sector_perf': pd.DataFrame,
            'top_movers': pd.DataFrame,
            'macro': Dict,
            'market_indicators': Dict,
            'commodities': Dict,
            'currencies': Dict,
            'period_label': str
        }

        Cette méthode utilise prioritairement les données du cache si disponibles,
        puis complète avec des appels à `MacroDataExtractor` et `StockDataExtractor` si nécessaire.
        """
        cache = self.read_cache(cache_override=store_data)

        # Base empty return structure
        overview = {
            "indices": {},
            "sector_perf": pd.DataFrame(),
            "top_movers": pd.DataFrame(),
            "macro": {},
            "market_indicators": {},
            "commodities": {},
            "currencies": {},
            "period_label": period_label
        }

        if not cache:
            # No cache: attempt best-effort live retrieval of macro & market indicators (non-blocking best-effort)
            try:
                overview["macro"] = self.macro_extractor.get_all_macro_data() or {}
            except Exception as e:
                logger.debug("Macro extractor failed: %s", e)
                overview["macro"] = {}

            try:
                overview["market_indicators"] = self.extractor.get_market_indicators() or {}
            except Exception as e:
                logger.debug("Market indicators fetch failed: %s", e)
                overview["market_indicators"] = {}

            # Try to fetch indices summary quickly (non-blocking fallback to cached _fetch_indices)
            try:
                indices_data = self._fetch_indices()
                # _fetch_indices returns dict with 'closes' lists; convert to DataFrames for UI consistency
                reconstructed = {}
                for sym, val in indices_data.items():
                    closes = val.get("closes", []) or []
                    # Use simple date index near now for display (best-effort)
                    if closes:
                        idx = pd.date_range(end=pd.Timestamp.now(), periods=len(closes), freq="D")
                        reconstructed[sym] = pd.DataFrame({"Close": closes}, index=idx)
                overview["indices"] = reconstructed
            except Exception:
                overview["indices"] = {}

            # commodities & currencies from macro if present
            macro = overview.get("macro", {})
            overview["commodities"] = macro.get("commodities", {}) if isinstance(macro, dict) else {}
            overview["currencies"] = macro.get("currencies", {}) if isinstance(macro, dict) else {}

            return overview

        # When cache exists, reconstruct structured parts
        try:
            overview["indices"] = self.reconstruct_indices(cache.get("indices", {}))
        except Exception as e:
            logger.debug("Failed reconstructing indices from cache: %s", e)
            overview["indices"] = {}

        try:
            overview["sector_perf"] = pd.DataFrame(cache.get("sector_perf", []) or [])
            if not overview["sector_perf"].empty and "perf" in overview["sector_perf"].columns:
                overview["sector_perf"]["perf"] = pd.to_numeric(overview["sector_perf"]["perf"], errors="coerce").fillna(0)
        except Exception as e:
            logger.debug("Failed parsing sector_perf: %s", e)
            overview["sector_perf"] = pd.DataFrame()

        try:
            top_gainers = pd.DataFrame(cache.get("top_gainers", []) or [])
            top_losers = pd.DataFrame(cache.get("top_losers", []) or [])
            # Apply sector filter if requested
            if sector_filter and sector_filter != "ALL" and not top_gainers.empty and "sector" in top_gainers.columns:
                top_gainers = top_gainers[top_gainers["sector"] == sector_filter]
            overview["top_movers"] = top_gainers if not top_gainers.empty else (top_losers if not top_losers.empty else pd.DataFrame())
        except Exception as e:
            logger.debug("Failed preparing top movers: %s", e)
            overview["top_movers"] = pd.DataFrame()

        # Macro section: prefer stored macro, else try live fetch
        try:
            if cache.get("macro"):
                overview["macro"] = cache.get("macro", {})
            else:
                overview["macro"] = self.macro_extractor.get_all_macro_data() or {}
        except Exception as e:
            logger.debug("Macro data retrieval failed: %s", e)
            overview["macro"] = {}

        # Market indicators: prefer cached market indicators if present, else live
        try:
            mi = cache.get("market_indicators") or {}
            if not mi:
                mi = self.extractor.get_market_indicators() or {}
            overview["market_indicators"] = mi
        except Exception as e:
            logger.debug("Market indicators retrieval failed: %s", e)
            overview["market_indicators"] = {}

        # Commodities & currencies: try to pull from macro or market_indicators
        overview["commodities"] = overview["macro"].get("commodities", {}) if isinstance(overview["macro"], dict) else {}
        overview["currencies"] = overview["macro"].get("currencies", {}) if isinstance(overview["macro"], dict) else {}
        # finalize
        overview["period_label"] = period_label

        return overview

    # ----------------------------
    # Export utilities
    # ----------------------------
    def export_market_report(self, cache_override: Optional[Dict] = None, sector_filter: Optional[str] = None) -> Optional[bytes]:
        """
        Génère un fichier Excel en mémoire (bytes) depuis le cache / live data.
        Sheets fournis :
          - Top Gainers
          - Top Losers
          - Sector Performance
          - Macro Data
          - Market Indicators
          - Commodities
          - Currencies
          - Indices Summary
        """
        cache = self.read_cache(cache_override=cache_override)
        if not cache:
            # attempt to build overview live if no cache
            overview = self.build_overview(store_data={}, sector_filter=sector_filter)
        else:
            overview = self.build_overview(store_data=cache, sector_filter=sector_filter)

        top = pd.DataFrame(cache.get("top_gainers", [])) if cache else overview.get("top_movers", pd.DataFrame())
        losers = pd.DataFrame(cache.get("top_losers", [])) if cache else pd.DataFrame()
        sector_df = pd.DataFrame(cache.get("sector_perf", [])) if cache else overview.get("sector_perf", pd.DataFrame())
        macro = overview.get("macro", {}) or {}
        market_indicators = overview.get("market_indicators", {}) or {}
        commodities = overview.get("commodities", {}) or {}
        currencies = overview.get("currencies", {}) or {}
        indices = overview.get("indices", {}) or {}

        # Apply sector filter to top if requested
        if not top.empty and sector_filter and sector_filter != "ALL" and "sector" in top.columns:
            top = top[top["sector"] == sector_filter]

        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                if not top.empty:
                    top.to_excel(writer, sheet_name="Top Gainers", index=False)
                if not losers.empty:
                    losers.to_excel(writer, sheet_name="Top Losers", index=False)
                if not sector_df.empty:
                    sector_df.to_excel(writer, sheet_name="Sector Performance", index=False)

                # Macro: flatten dict into DataFrame
                if macro:
                    try:
                        macro_df = pd.json_normalize(macro)
                        macro_df.to_excel(writer, sheet_name="Macro Data", index=False)
                    except Exception:
                        # fallback: key/value pairs
                        pd.DataFrame(list(macro.items()), columns=["Indicator", "Value"]).to_excel(writer, sheet_name="Macro Data", index=False)

                # Market indicators
                if market_indicators:
                    try:
                        mi_df = pd.json_normalize(market_indicators)
                        mi_df.to_excel(writer, sheet_name="Market Indicators", index=False)
                    except Exception:
                        pd.DataFrame(list(market_indicators.items()), columns=["Indicator", "Value"]).to_excel(writer, sheet_name="Market Indicators", index=False)

                # Commodities
                if commodities:
                    try:
                        pd.DataFrame.from_dict(commodities, orient="index").reset_index().rename(columns={"index": "Commodity"}).to_excel(writer, sheet_name="Commodities", index=False)
                    except Exception:
                        pd.DataFrame(list(commodities.items()), columns=["Commodity", "Value"]).to_excel(writer, sheet_name="Commodities", index=False)

                # Currencies
                if currencies:
                    try:
                        pd.DataFrame.from_dict(currencies, orient="index").reset_index().rename(columns={"index": "Pair"}).to_excel(writer, sheet_name="Currencies", index=False)
                    except Exception:
                        pd.DataFrame(list(currencies.items()), columns=["Pair", "Value"]).to_excel(writer, sheet_name="Currencies", index=False)

                # Indices summary: compile last price / change if available
                indices_rows = []
                for sym, df in indices.items():
                    try:
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            last = float(df["Close"].iloc[-1])
                            first = float(df["Close"].iloc[0]) if len(df) > 0 else last
                            change = ((last - first) / first * 100) if first != 0 else 0.0
                            indices_rows.append({"Symbol": sym, "Last": last, "Change_%": change})
                    except Exception:
                        continue
                if indices_rows:
                    pd.DataFrame(indices_rows).to_excel(writer, sheet_name="Indices Summary", index=False)

            output.seek(0)
            return output.getvalue()
        except Exception:
            logger.exception("Failed to export market report to Excel")
            return None

    # ----------------------------
    # Async refresh orchestration (unchanged)
    # ----------------------------
    async def refresh_all_market_data(self) -> bool:
        """
        Existing orchestration that pulls fresh data from extractors and populates cache via self.cache.save_data(...)
        """
        logger.info("🚀 Mise à jour globale du contexte marché lancée...")
        start_time = datetime.now()
        try:
            indices_task = asyncio.to_thread(self._fetch_indices)
            macro_task = asyncio.to_thread(self.macro_extractor.get_all_macro_data)
            indices_data, macro_data = await asyncio.gather(indices_task, macro_task)
            sector_perf, top_movers = await self._analyze_market_components_async()
            regime, score = self.regime_engine.compute_regime(macro_data, indices_data)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "indices": indices_data,
                "macro": macro_data,
                "sector_perf": sector_perf,
                "top_gainers": top_movers,
                "regime": {"status": regime, "score": score}
            }
            if self.cache:
                try:
                    self.cache.save_data(payload)
                except Exception:
                    logger.warning("Cache service failed to save payload; skipping.")
            # always also save locally
            try:
                cache_file = getattr(settings, "MARKET_CACHE_FILE", os.path.join(os.getcwd(), "cache", "market_overview.json"))
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except Exception:
                logger.exception("Failed to write local cache file")
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("✅ Market Cache mis à jour en %.2fs.", duration)
            return True
        except Exception as e:
            logger.exception("❌ Échec critique du MarketManager: %s", e)
            return False

    def _fetch_indices(self) -> Dict[str, Any]:
        """Récupère les données récentes pour les tickers de référence (VIX, SP500, etc.)"""
        results = {}
        # Récupération des symboles depuis settings ou fallback sécurisé
        index_symbols = getattr(settings, "index_symbols", ["^GSPC", "^VIX", "BTC-USD", "GC=F"])

        for sym in index_symbols:
            try:
                df = self.extractor.get_historical_data(sym, period='3mo')
                if df is not None and len(df) >= 2:
                    last_close = float(df['Close'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2])
                    results[sym] = {
                        'closes': df['Close'].dropna().tolist(),
                        'last_price': round(last_close, 2),
                        'change_pct': round(((last_close / prev_close) - 1) * 100, 2)
                    }
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération de l'indice {sym}: {e}")
        return results

    async def _analyze_market_components_async(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyse les composants du marché en parallèle pour extraire les performances sectorielles.
        """
        # On récupère les symboles (Top 50 pour la performance du scan)
        all_symbols = get_sp500_symbols() or ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META']
        target_symbols = all_symbols[:50]

        # Création des tâches asynchrones pour chaque action
        tasks = [self._process_single_stock(s) for s in target_symbols]
        results = await asyncio.gather(*tasks)

        # Filtrage des None (erreurs de fetch)
        valid_results = [r for r in results if r is not None]

        # Agrégation par secteurs
        sector_map = {}
        for item in valid_results:
            sector = item['sector']
            sector_map.setdefault(sector, []).append(item['perf'])

        sector_results = [
            {'sector': k, 'perf': round(sum(v) / len(v), 2)}
            for k, v in sector_map.items()
        ]

        # Top 10 Movers (Gagnants)
        top_movers = sorted(valid_results, key=lambda x: x['perf'], reverse=True)[:10]

        return sector_results, top_movers

    async def _process_single_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Traite un symbole individuel : récupère historique + secteur.
        Helper pour la parallélisation.
        """
        try:
            # Exécution dans un thread car yfinance/pandas bloquent l'event loop
            df = await asyncio.to_thread(self.extractor.get_historical_data, symbol, '1mo')
            if df is None or len(df) < 5:
                return None

            perf = float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100)

            # Récupération du secteur via les fondamentaux
            funda = await asyncio.to_thread(self.extractor.get_fundamental_data_fallback, symbol)
            sector = funda.get('profile', {}).get('sector', "Other")

            return {
                'symbol': symbol,
                'perf': round(perf, 2),
                'sector': sector
            }
        except Exception:
            # On logue peu ici pour ne pas polluer la console pendant le scan massif
            return None

    def get_market_health(self, cache_override: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retourne un résumé structuré de la santé du marché pour l'UI.
        Priorité : cache_override > cache service > live extraction.
        Structure renvoyée:
        {
            'score': float (0-100),
            'label': str ('Bullish'|'Bearish'|'Neutral'),
            'description': str,
            'vix': float | None,
            'put_call_ratio': float | None,
            'advance_decline': float | None,
            'indices': { 'sp500': float, 'nasdaq': float, 'dow': float },
            'raw': dict (market_indicators full)
        }
        """
        try:
            cache = self.read_cache(cache_override=cache_override)
            market_indicators = {}
            if cache and isinstance(cache, dict):
                # prefer any cached market_indicators or top-level regime info
                market_indicators = cache.get("market_indicators") or cache.get("macro", {}).get("market_indicators", {}) or {}
            if not market_indicators:
                # fallback to live quick extraction (best-effort)
                try:
                    market_indicators = self.extractor.get_market_indicators() or {}
                except Exception as e:
                    logger.debug("Live market indicators fetch failed: %s", e)
                    market_indicators = {}

            # If market_indicators contains overall_sentiment created by extractor, use it
            overall = market_indicators.get("overall_sentiment") if isinstance(market_indicators, dict) else None

            if overall and isinstance(overall, dict):
                score = float(overall.get("score_out_of_100", overall.get("score", 50)))
                label = overall.get("label", "Neutral")
                description = overall.get("description", "")
            else:
                # compute health score on best-effort basis using extractor helper if possible
                try:
                    score = float(self.extractor._compute_market_health_score(market_indicators))  # best-effort, private helper
                except Exception:
                    score = float(market_indicators.get("score", 50) if isinstance(market_indicators, dict) else 50)
                label = "Bullish" if score > 70 else "Bearish" if score < 30 else "Neutral"
                description = ""

            # gather a few helpful indicators for quick UI display
            vix = None
            put_call = None
            adv_decl = None
            try:
                vix = market_indicators.get("VIX", {}).get("price") if isinstance(market_indicators, dict) else None
                put_call = market_indicators.get("Equity Put/Call Ratio", {}).get("price") if isinstance(market_indicators, dict) else None
                adv_decl = market_indicators.get("NYSE Advance/Decline Line", {}).get("change") if isinstance(market_indicators, dict) else None
            except Exception:
                pass

            indices = {
                "sp500": market_indicators.get("S&P 500", {}).get("change"),
                "nasdaq": market_indicators.get("NASDAQ", {}).get("change"),
                "dow": market_indicators.get("Dow Jones", {}).get("change")
            }

            return {
                "score": score,
                "label": label,
                "description": description,
                "vix": vix,
                "put_call_ratio": put_call,
                "advance_decline": adv_decl,
                "indices": indices,
                "raw": market_indicators
            }

        except Exception as e:
            logger.exception("Failed to compute market health: %s", e)
            return {
                "score": 50,
                "label": "Neutral",
                "description": "Erreur calcul santé marché",
                "vix": None,
                "put_call_ratio": None,
                "advance_decline": None,
                "indices": {},
                "raw": {}
            }


# --- Bloc de test rapide ---
if __name__ == "__main__":
    class MockCache:
        def save_data(self, data): print("Data Saved to Mock Cache")

    async def test():
        manager = MarketManager(MockCache())
        await manager.refresh_all_market_data()

    asyncio.run(test())