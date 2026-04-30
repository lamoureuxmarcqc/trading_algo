# trading_algo/market/market_manager.py
"""
MarketManager - Orchestration des données de marché (indices, secteurs, macro, santé).
Responsabilités :
- Lecture/écriture du cache (Redis/fichier)
- Construction d'une vue agrégée pour le dashboard
- Rafraîchissement asynchrone des données
- Calcul de la santé du marché (via MarketHealthAnalyzer)
- Export Excel du rapport de marché

Dépendances : utilise les extracteurs refactorisés (data_extraction)
"""

import asyncio
import json
import logging
import os
import io
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from trading_algo import settings
from trading_algo.data.data_extraction import (
    StockDataExtractor,
    MacroDataExtractor,
    FundamentalExtractor,
    MarketHealthAnalyzer,
    SentimentExtractor,
    _yf_history_with_fallback
)
from trading_algo.screening.actions_sp500 import get_sp500_symbols

logger = logging.getLogger(__name__)


class MarketManager:
    """
    Gestionnaire central du contexte de marché.
    """

    def __init__(self, cache_service: Optional[Any] = None):
        """
        :param cache_service: objet optionnel avec méthodes get_data() / save_data()
        """
        self.cache = cache_service
        self.price_extractor = StockDataExtractor()
        self.macro_extractor = MacroDataExtractor()
        self.fundamental_extractor = FundamentalExtractor()
        self.health_analyzer = MarketHealthAnalyzer(self.macro_extractor)
        self.sentiment_extractor = SentimentExtractor()  # pour usage futur

        # Cache interne pour les symboles S&P500 (évite rechargement trop fréquent)
        self._sp500_symbols_cache: Optional[List[str]] = None
        self._sp500_symbols_timestamp: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Lecture du cache (Redis / fichier)
    # ------------------------------------------------------------------
    def read_cache(self, cache_override: Optional[Dict] = None) -> Dict[str, Any]:
        """Retourne le contenu du cache. Priorité : cache_override > cache_service > fichier."""
        if cache_override:
            return cache_override

        if self.cache:
            try:
                data = self.cache.get_data()
                if isinstance(data, dict) and data:
                    return data
            except Exception as e:
                logger.debug("Cache service read failed: %s", e)

        cache_file = getattr(settings, "MARKET_CACHE_FILE",
                             os.path.join(os.getcwd(), "cache", "market_overview.json"))
        try:
            if not os.path.exists(cache_file):
                return {}
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.exception("Failed to read market cache file %s: %s", cache_file, e)
            return {}

    def save_cache(self, payload: Dict[str, Any]) -> None:
        """Écrit le payload dans le cache (service + fichier)."""
        if self.cache:
            try:
                self.cache.save_data(payload)
            except Exception as e:
                logger.warning("Cache service save failed: %s", e)

        cache_file = getattr(settings, "MARKET_CACHE_FILE",
                             os.path.join(os.getcwd(), "cache", "market_overview.json"))
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.exception("Failed to write local cache file: %s", e)

    # ------------------------------------------------------------------
    # Reconstruction des DataFrames depuis le cache JSON
    # ------------------------------------------------------------------
    @staticmethod
    def reconstruct_indices(indices_cache: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Convertit la structure JSON en DataFrames indexés par date."""
        result = {}
        if not isinstance(indices_cache, dict):
            return result

        for sym, data in indices_cache.items():
            try:
                if not isinstance(data, dict):
                    continue
                closes = data.get("closes") or data.get("Close") or []
                raw_dates = data.get("dates") or data.get("index") or []
                if not closes:
                    continue
                if raw_dates:
                    dates = pd.to_datetime(raw_dates)
                else:
                    dates = pd.date_range(end=pd.Timestamp.now(), periods=len(closes), freq="B")
                min_len = min(len(dates), len(closes))
                df = pd.DataFrame({"Close": closes[-min_len:]}, index=dates[-min_len:])
                df.index = pd.to_datetime(df.index)
                result[sym] = df.sort_index()
            except Exception as e:
                logger.debug("Failed reconstructing index %s: %s", sym, e)
        return result

    # ------------------------------------------------------------------
    # Construction de la vue agrégée pour l'UI
    # ------------------------------------------------------------------
    def build_overview(
        self,
        store_data: Optional[Dict] = None,
        period_label: str = "1M",
        sector_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Construit un dictionnaire structuré contenant :
          - indices: Dict[str, pd.DataFrame]
          - sector_perf: pd.DataFrame
          - top_movers: pd.DataFrame
          - macro: Dict
          - market_indicators: Dict
          - commodities: Dict
          - currencies: Dict
          - period_label: str
        """
        cache = self.read_cache(store_data)
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

        # Si pas de cache, on essaie de récupérer en direct (best effort)
        if not cache:
            try:
                overview["macro"] = self.macro_extractor.get_all_macro_data() or {}
            except Exception:
                overview["macro"] = {}
            try:
                overview["market_indicators"] = self.price_extractor.get_market_indicators() or {}
            except Exception:
                overview["market_indicators"] = {}
            # Indices : récupération simple via _fetch_indices
            try:
                indices_data = self._fetch_indices()
                reconstructed = {}
                for sym, val in indices_data.items():
                    closes = val.get("closes", [])
                    if closes:
                        idx = pd.date_range(end=pd.Timestamp.now(), periods=len(closes), freq="D")
                        reconstructed[sym] = pd.DataFrame({"Close": closes}, index=idx)
                overview["indices"] = reconstructed
            except Exception:
                overview["indices"] = {}
            overview["commodities"] = overview["macro"].get("commodities", {})
            overview["currencies"] = overview["macro"].get("currencies", {})
            return overview

        # Reconstruction depuis le cache
        try:
            overview["indices"] = self.reconstruct_indices(cache.get("indices", {}))
        except Exception:
            overview["indices"] = {}

        try:
            sector_df = pd.DataFrame(cache.get("sector_perf", []) or [])
            if not sector_df.empty and "perf" in sector_df.columns:
                sector_df["perf"] = pd.to_numeric(sector_df["perf"], errors="coerce").fillna(0)
            overview["sector_perf"] = sector_df
        except Exception:
            overview["sector_perf"] = pd.DataFrame()

        try:
            top_gainers = pd.DataFrame(cache.get("top_gainers", []) or [])
            top_losers = pd.DataFrame(cache.get("top_losers", []) or [])
            if sector_filter and sector_filter != "ALL" and not top_gainers.empty and "sector" in top_gainers.columns:
                top_gainers = top_gainers[top_gainers["sector"] == sector_filter]
            overview["top_movers"] = top_gainers if not top_gainers.empty else top_losers
        except Exception:
            overview["top_movers"] = pd.DataFrame()

        # Macro
        macro = cache.get("macro") or {}
        if not macro:
            try:
                macro = self.macro_extractor.get_all_macro_data() or {}
            except Exception:
                macro = {}
        overview["macro"] = macro

        # Market indicators
        mi = cache.get("market_indicators") or {}
        if not mi:
            try:
                mi = self.price_extractor.get_market_indicators() or {}
            except Exception:
                mi = {}
        overview["market_indicators"] = mi

        overview["commodities"] = macro.get("commodities", {})
        overview["currencies"] = macro.get("currencies", {})
        return overview

    # ------------------------------------------------------------------
    # Export Excel (rapport de marché)
    # ------------------------------------------------------------------
    def export_market_report(self, cache_override: Optional[Dict] = None,
                             sector_filter: Optional[str] = None) -> Optional[bytes]:
        """Génère un fichier Excel en mémoire (bytes)."""
        cache = self.read_cache(cache_override)
        overview = self.build_overview(store_data=cache, sector_filter=sector_filter)

        # Récupération des données brutes depuis le cache si possible
        top_gainers = pd.DataFrame(cache.get("top_gainers", [])) if cache else overview["top_movers"]
        top_losers = pd.DataFrame(cache.get("top_losers", [])) if cache else pd.DataFrame()
        sector_df = overview["sector_perf"]
        macro = overview["macro"]
        market_indicators = overview["market_indicators"]
        commodities = overview["commodities"]
        currencies = overview["currencies"]
        indices = overview["indices"]

        if not top_gainers.empty and sector_filter and sector_filter != "ALL" and "sector" in top_gainers.columns:
            top_gainers = top_gainers[top_gainers["sector"] == sector_filter]

        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                if not top_gainers.empty:
                    top_gainers.to_excel(writer, sheet_name="Top Gainers", index=False)
                if not top_losers.empty:
                    top_losers.to_excel(writer, sheet_name="Top Losers", index=False)
                if not sector_df.empty:
                    sector_df.to_excel(writer, sheet_name="Sector Performance", index=False)

                # Macro
                if macro:
                    try:
                        pd.json_normalize(macro).to_excel(writer, sheet_name="Macro Data", index=False)
                    except Exception:
                        pd.DataFrame(list(macro.items()), columns=["Indicator", "Value"]).to_excel(writer, sheet_name="Macro Data", index=False)

                # Market indicators
                if market_indicators:
                    try:
                        pd.json_normalize(market_indicators).to_excel(writer, sheet_name="Market Indicators", index=False)
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

                # Indices summary
                indices_rows = []
                for sym, df in indices.items():
                    try:
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            last = float(df["Close"].iloc[-1])
                            first = float(df["Close"].iloc[0]) if len(df) > 1 else last
                            change = ((last - first) / first * 100) if first != 0 else 0.0
                            indices_rows.append({"Symbol": sym, "Last": last, "Change_%": change})
                    except Exception:
                        continue
                if indices_rows:
                    pd.DataFrame(indices_rows).to_excel(writer, sheet_name="Indices Summary", index=False)

            output.seek(0)
            return output.getvalue()
        except Exception as e:
            logger.exception("Failed to export market report: %s", e)
            return None

    # ------------------------------------------------------------------
    # Rafraîchissement asynchrone complet
    # ------------------------------------------------------------------
    async def refresh_all_market_data(self) -> bool:
        """Orchestration asynchrone : indices, macro, indicateurs, secteurs, top movers."""
        logger.info("🚀 Market data refresh started...")
        start = datetime.now()
        try:
            # Tâches I/O parallèles
            indices_task = asyncio.to_thread(self._fetch_indices)
            macro_task = asyncio.to_thread(self.macro_extractor.get_all_macro_data)
            indicators_task = asyncio.to_thread(self.price_extractor.get_market_indicators)
            indices_data, macro_data, market_indicators = await asyncio.gather(
                indices_task, macro_task, indicators_task
            )

            # Analyse des composants (secteurs + top movers) – aussi en parallèle
            sector_perf, top_movers = await self._analyze_market_components_async()

            # Régime de marché (utilise les données macro et indices)
            regime, score = await asyncio.to_thread(
                self._compute_regime_sync, macro_data, indices_data
            )

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "indices": indices_data,
                "macro": macro_data,
                "market_indicators": market_indicators or {},
                "sector_perf": sector_perf,
                "top_gainers": top_movers,
                "regime": {"status": regime, "score": score}
            }
            self.save_cache(payload)

            elapsed = (datetime.now() - start).total_seconds()
            logger.info("✅ Market cache updated in %.2f seconds", elapsed)
            return True
        except Exception as e:
            logger.exception("❌ Market refresh failed: %s", e)
            return False

    def _compute_regime_sync(self, macro_data: Dict, indices_data: Dict) -> Tuple[str, float]:
        """Calcul synchrone du régime de marché (délégation au moteur)."""
        # Le moteur peut être importé ici pour éviter dépendance circulaire
        from trading_algo.strategy.market_regime_engine import MarketRegimeEngine
        engine = MarketRegimeEngine()
        return engine.compute_regime(macro_data, indices_data)

    # ------------------------------------------------------------------
    # Récupération des indices de référence
    # ------------------------------------------------------------------
    def _fetch_indices(self) -> Dict[str, Any]:
        """Récupère les données récentes pour les indices majeurs."""
        results = {}
        symbols = getattr(settings, "index_symbols", ["^GSPC", "^IXIC", "^DJI", "^VIX", "BTC-USD", "GC=F"])
        for sym in symbols:
            try:
                df = self.price_extractor.get_historical_data(sym, period="3mo")
                if df is None or df.empty or len(df) < 2:
                    continue
                closes = df["Close"].dropna()
                dates = pd.to_datetime(closes.index).strftime("%Y-%m-%d").tolist()
                last = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                results[sym] = {
                    "dates": dates,
                    "closes": closes.tolist(),
                    "last_price": round(last, 2),
                    "change_pct": round(((last / prev) - 1) * 100, 2)
                }
            except Exception as e:
                logger.warning("Failed to fetch index %s: %s", sym, e)
        return results

    # ------------------------------------------------------------------
    # Analyse asynchrone des composants (secteurs, top movers)
    # ------------------------------------------------------------------
    async def _analyze_market_components_async(self) -> Tuple[List[Dict], List[Dict]]:
        """Calcule les performances sectorielles et les top movers (gainers)."""
        symbols = await self._get_top_symbols_async()
        # Traitement parallèle des symboles
        tasks = [self._process_single_stock(sym) for sym in symbols]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r is not None]

        if not valid:
            return [], []

        # Agrégation par secteur
        sector_map: Dict[str, List[float]] = {}
        for item in valid:
            sector = item["sector"]
            sector_map.setdefault(sector, []).append(item["perf"])
        sector_perf = [{"sector": k, "perf": round(sum(v)/len(v), 2)} for k, v in sector_map.items()]

        # Top 10 gainers (perf la plus élevée)
        top_movers = sorted(valid, key=lambda x: x["perf"], reverse=True)[:10]
        return sector_perf, top_movers

    async def _get_top_symbols_async(self) -> List[str]:
        """Retourne une liste de symboles représentatifs (S&P500 ou fallback)."""
        # Cache interne de 1 heure
        if (self._sp500_symbols_cache is not None and
            self._sp500_symbols_timestamp is not None and
            (datetime.now() - self._sp500_symbols_timestamp).seconds < 3600):
            return self._sp500_symbols_cache[:50]  # top 50 pour performance

        try:
            # Exécution dans un thread car get_sp500_symbols peut être bloquant
            symbols = await asyncio.to_thread(get_sp500_symbols)
            if symbols and len(symbols) > 0:
                self._sp500_symbols_cache = symbols
                self._sp500_symbols_timestamp = datetime.now()
                return symbols[:50]
        except Exception as e:
            logger.warning("Failed to get S&P500 symbols: %s", e)

        fallback = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "BRK-B", "JPM", "V"]
        return fallback

    async def _process_single_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Traite un symbole : performance récente + secteur."""
        try:
            # Récupération de l'historique (thread)
            df = await asyncio.to_thread(self.price_extractor.get_historical_data, symbol, "1mo")
            if df is None or len(df) < 5:
                return None
            perf = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

            # Récupération du secteur via FundamentalExtractor (thread)
            funda = await asyncio.to_thread(self.fundamental_extractor.get_fundamental_data, symbol)
            sector = funda.get("profile", {}).get("sector", "Other")
            return {
                "symbol": symbol,
                "perf": round(perf, 2),
                "sector": sector
            }
        except Exception:
            # Silencieux pour ne pas polluer les logs
            return None

    # ------------------------------------------------------------------
    # Santé du marché (pour l'UI)
    # ------------------------------------------------------------------
    def get_market_health(self, cache_override: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retourne un résumé structuré de la santé du marché :
          - score (0-100)
          - label (Bullish/Bearish/Neutral)
          - description
          - vix, put_call_ratio, advance_decline
          - indices (sp500, nasdaq, dow)
          - raw (données brutes)
        """
        try:
            cache = self.read_cache(cache_override)
            market_indicators = {}
            if cache and isinstance(cache, dict):
                market_indicators = cache.get("market_indicators") or {}

            # Si pas d'indicateurs, on essaie de les récupérer en direct
            if not market_indicators:
                try:
                    market_indicators = self.price_extractor.get_market_indicators() or {}
                except Exception:
                    market_indicators = {}

            # Utiliser l'analyseur de santé
            health = self.health_analyzer.compute_market_health(market_indicators)
            score = health.get("score_out_of_100", 50)
            label = health.get("label", "Neutral")
            description = health.get("description", "")

            # Extraire quelques indicateurs utiles
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
                "sp500": market_indicators.get("S&P 500", {}).get("change") if isinstance(market_indicators, dict) else None,
                "nasdaq": market_indicators.get("NASDAQ", {}).get("change") if isinstance(market_indicators, dict) else None,
                "dow": market_indicators.get("Dow Jones", {}).get("change") if isinstance(market_indicators, dict) else None,
            }

            return {
                "score": float(score),
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
                "description": "Erreur de calcul",
                "vix": None,
                "put_call_ratio": None,
                "advance_decline": None,
                "indices": {},
                "raw": {}
            }


# ------------------------------------------------------------------
# Test rapide (si exécuté directement)
# ------------------------------------------------------------------
if __name__ == "__main__":
    class DummyCache:
        def get_data(self):
            return {}
        def save_data(self, data):
            print("DummyCache saved")

    async def test():
        mgr = MarketManager(DummyCache())
        await mgr.refresh_all_market_data()
        health = mgr.get_market_health()
        print("Market health:", health["label"], health["score"])

    asyncio.run(test())