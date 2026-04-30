# trading_algo/portfolio/portfoliomanager.py
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timedelta

from .portfolio import Portfolio, Order, Position
from trading_algo.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


# =========================================================================
# MOTEUR D'INTELLIGENCE (défini AVANT PortfolioManager)
# =========================================================================
class PortfolioIntelligenceEngine:
    """Moteur de scoring, allocation et recommandations."""

    def __init__(self, manager):
        self.manager = manager

    def compute_scores(self, fundamentals_map: Dict) -> Dict[str, float]:
        scores = {}
        for ticker in self.manager.current_portfolio.positions.keys():
            f = fundamentals_map.get(ticker, {})
            data = self.manager._get_historical_data(ticker)
            vol = data['Close'].pct_change().std() * np.sqrt(252) if data is not None else 0.2
            scores[ticker] = self._buffett_score(f, vol)
        return scores

    def _buffett_score(self, f, vol):
        roe = f.get("roe", 0)
        roic = f.get("roic", 0)
        fcf = f.get("fcf_margin", 0)
        growth = f.get("revenue_growth", 0)
        debt = f.get("debt_to_equity", 1)
        pe = f.get("pe_ratio", 25)

        quality = (roe * 0.25 + roic * 0.25 + fcf * 0.2 + growth * 0.15)
        penalty = (debt * 0.1 + vol * 0.15 + pe / 30)
        return float(quality - penalty)

    def build_allocation(self, scores: Dict[str, float]) -> Dict[str, float]:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        total = sum(max(s, 0.01) for _, s in ranked)
        alloc = {t: max(s, 0.01) / total for t, s in ranked}
        alloc["cash"] = max(0, 1 - sum(alloc.values()))
        return alloc

    def generate_recommendations(self, target_alloc: Dict[str, float]) -> List[Dict]:
        orders = []
        prices = self.manager.get_market_prices(list(target_alloc.keys()))
        portfolio = self.manager.current_portfolio
        total_value = sum(
            pos.current_value(prices.get(t, 0))
            for t, pos in portfolio.positions.items()
        )
        for ticker, target_weight in target_alloc.items():
            pos = portfolio.positions.get(ticker)
            current_value = pos.current_value(prices.get(ticker, 0)) if pos else 0
            current_weight = current_value / total_value if total_value else 0
            diff = target_weight - current_weight
            if abs(diff) > 0.02:
                orders.append({
                    "ticker": ticker,
                    "action": "BUY" if diff > 0 else "SELL",
                    "delta": round(diff, 4)
                })
        return orders

    def portfolio_score(self, scores: Dict, weights: Dict) -> float:
        return sum(scores.get(t, 0) * weights.get(t, 0) for t in weights)


class PortfolioManager:
    """
    Gestionnaire central : fait le pont entre l'extraction de données,
    le stockage des portefeuilles et les algorithmes de décision.
    """

    def __init__(self, data_extractor_class, portfolios_dir: str = "portfolios"):
        self.data_extractor_class = data_extractor_class

        # Structure de dossiers robuste : /data/portfolios
        self.base_dir = Path(os.getcwd()) / "data" / portfolios_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.current_portfolio: Optional[Portfolio] = None
        self.current_portfolio_name: Optional[str] = None
        self.risk_manager = RiskManager()
        self.intelligence = PortfolioIntelligenceEngine(self)  # ✅ engine défini

        # Cache de session pour les données historiques
        self._market_data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=6)

        logger.info(f"PortfolioManager opérationnel sur : {self.base_dir}")

    # =========================================================
    # 1. PERSISTANCE & GESTION DES FICHIERS
    # =========================================================
    def create_portfolio(self, name: str, initial_cash: float) -> Portfolio:
        portfolio = Portfolio(cash=initial_cash, name=name)
        self.current_portfolio = portfolio
        self.current_portfolio_name = name
        self.save_portfolio()
        logger.info(f"Nouveau portefeuille créé : {name} (cash: {initial_cash:.2f} $)")
        return portfolio

    def list_portfolios(self) -> List[str]:
        candidates = [
            self.base_dir,
            Path.cwd() / "portfolios",
            Path.cwd() / "data" / "portfolios",
            Path.home() / ".trading_algo" / "portfolios"
        ]
        patterns = ["*.json", "*.portfolio.json", "*.jsonl"]
        seen = set()
        results: List[str] = []
        for d in candidates:
            try:
                if not d.exists():
                    continue
                for pattern in patterns:
                    for f in sorted(d.glob(pattern), key=lambda p: p.name):
                        name = f.stem
                        if name not in seen:
                            seen.add(name)
                            results.append(name)
            except Exception as e:
                logger.debug(f"Erreur scan {d}: {e}")
        if not results and self.base_dir.exists():
            try:
                for f in sorted(self.base_dir.iterdir(), key=lambda p: p.name):
                    if f.is_file() and f.suffix in ('.json', '.jsonl'):
                        name = f.stem
                        if name not in seen:
                            seen.add(name)
                            results.append(name)
            except Exception:
                pass
        return sorted(results)

    def load_portfolio(self, name: str) -> Optional[Portfolio]:
        if self.current_portfolio is not None and self.current_portfolio_name == name:
            logger.info(f"Portefeuille '{name}' servi depuis le cache mémoire")
            return self.current_portfolio
        filepath = self.base_dir / f"{name}.json"
        if not filepath.exists():
            logger.error(f"Portefeuille introuvable : {filepath}")
            return None
        try:
            self.current_portfolio = Portfolio.load_from_file(str(filepath))
            self.current_portfolio_name = name
            logger.info(f"Portefeuille '{name}' chargé (Cash: {self.current_portfolio.cash:.2f}$)")
            return self.current_portfolio
        except Exception as e:
            logger.exception(f"Erreur fatale lors du chargement de {name}: {e}")
            return None

    def save_portfolio(self):
        if self.current_portfolio:
            filepath = self.base_dir / f"{self.current_portfolio.name}.json"
            try:
                self.current_portfolio.save_to_file(str(filepath))
                logger.debug(f"Sauvegarde réussie : {self.current_portfolio.name}")
            except Exception as e:
                logger.error(f"Échec de la sauvegarde : {e}")

    # =========================================================
    # 2. MARKET DATA & ANALYTICS
    # =========================================================
    def get_market_prices(self, tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            return {}
        valid_tickers = [t for t in tickers if not t.endswith(('-C', '-P', '.C', '.P'))]
        if not valid_tickers:
            logger.warning("Aucun ticker valide après filtrage")
            return {t: 0.0 for t in tickers}
        extractor = self.data_extractor_class()
        try:
            prices = extractor.get_bulk_prices(valid_tickers)
            return {t: prices.get(t, 0.0) for t in tickers}
        except Exception as e:
            logger.warning(f"Bulk fetch échoué, fallback parallèle : {e}")
            def _fetch(t):
                try:
                    df = extractor.get_historical_data(t, period="1d")
                    return t, float(df['Close'].iloc[-1]) if not df.empty else None
                except Exception:
                    return t, None
            with ThreadPoolExecutor(max_workers=min(10, len(valid_tickers))) as executor:
                results = dict(executor.map(_fetch, valid_tickers))
                prices = {t: p for t, p in results.items() if p is not None}
                return {t: prices.get(t, 0.0) for t in tickers}

    def _is_cache_valid(self, ticker: str) -> bool:
        if ticker not in self._market_data_cache:
            return False
        ts = self._cache_timestamp.get(ticker)
        if ts is None:
            return False
        return datetime.now() - ts < self._cache_ttl

    def _get_historical_data(self, ticker: str, period: str = "3y") -> Optional[pd.DataFrame]:
        if self._is_cache_valid(ticker):
            return self._market_data_cache[ticker]
        extractor = self.data_extractor_class()
        try:
            df = extractor.get_historical_data(ticker, period=period)
            if df is not None and not df.empty:
                self._market_data_cache[ticker] = df
                self._cache_timestamp[ticker] = datetime.now()
                return df
        except Exception as e:
            logger.debug(f"Erreur historique pour {ticker}: {e}")
        return None

    def _calculate_advanced_risk(self, tickers: List[str], prices: Dict[str, float]) -> Dict:
        if not tickers:
            return {"status": "vide"}
        returns_map = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._get_historical_data, t, "3y"): t for t in tickers}
            for future in futures:
                t = futures[future]
                df = future.result()
                if df is not None and 'Close' in df.columns:
                    returns_map[t] = df['Close'].pct_change()
        if not returns_map:
            return {"status": "données_insuffisantes"}
        df_returns = pd.DataFrame(returns_map).fillna(0)
        total_val = sum(
            self.current_portfolio.positions[t].current_value(prices.get(t, 0))
            for t in df_returns.columns if t in self.current_portfolio.positions
        )
        if total_val <= 0:
            return {"status": "valeur_nulle"}
        weights = np.array([
            self.current_portfolio.positions[t].current_value(prices.get(t, 0)) / total_val
            for t in df_returns.columns
        ])
        portfolio_returns = df_returns.dot(weights)
        return self.risk_manager.risk_report(portfolio_returns)

    # =========================================================
    # 3. ANALYSE GLOBALE (utilise l'engine)
    # =========================================================
    def run_full_analysis(self, fundamentals_map: Dict) -> Dict:
        scores = self.intelligence.compute_scores(fundamentals_map)
        target_alloc = self.intelligence.build_allocation(scores)
        recommendations = self.intelligence.generate_recommendations(target_alloc)
        prices = self.get_market_prices(list(self.current_portfolio.positions.keys()))
        current_alloc = self.current_portfolio.get_allocation(prices)
        portfolio_score = self.intelligence.portfolio_score(scores, current_alloc)
        return {
            "scores": scores,
            "target_allocation": target_alloc,
            "current_allocation": current_alloc,
            "recommendations": recommendations,
            "portfolio_score": portfolio_score
        }

    # =========================================================
    # 4. COMPATIBILITÉ DASHBOARD (méthodes legacy)
    # =========================================================
    def get_quality_scores(self, fundamentals_map: Optional[Dict[str, Dict]] = None) -> Dict[str, float]:
        if not self.current_portfolio:
            return {}
        fundamentals_map = fundamentals_map or {}
        if hasattr(self, 'intelligence') and self.intelligence:
            return self.intelligence.compute_scores(fundamentals_map)
        # Fallback simple
        scores = {}
        for ticker in self.current_portfolio.positions.keys():
            scores[ticker] = 0.5
        return scores

    def get_target_allocation(self, portfolio: Portfolio, model: str = "buffett") -> Dict[str, float]:
        if model == "buffett" and hasattr(self, 'intelligence') and self.intelligence:
            # Simuler un scoring basique pour obtenir une allocation
            fake_scores = {t: 1.0 for t in portfolio.positions.keys()}
            return self.intelligence.build_allocation(fake_scores)
        tickers = list(portfolio.positions.keys())
        if not tickers:
            return {}
        alloc = {t: 1.0 / len(tickers) for t in tickers}
        return alloc

    def get_market_regime(self, market_data: Any) -> Dict:
        try:
            extractor = self.data_extractor_class()
            score = extractor._compute_market_health_score(market_data)
        except Exception:
            score = 50
        regimes = [(70, "BULL"), (40, "NEUTRAL"), (0, "BEAR")]
        regime = next(r for threshold, r in regimes if score >= threshold)
        return {"score": score, "regime": regime, "action": "EXPAND" if regime == "BULL" else "DEFEND"}

    # =========================================================
    # 5. MONTE CARLO
    # =========================================================
    def run_monte_carlo_simulation(self, n_simulations: int = 500, timeframe: int = 252) -> Dict[str, Any]:
        """
        Version professionnelle et robuste de la simulation Monte Carlo.
        - Téléchargement parallèle des données
        - Nettoyage et alignement des séries
        - Calcul des poids du portefeuille
        - Simulation Monte Carlo avancée (bootstrap + vol stochastique)
        - Calcul des métriques
        """

        from trading_algo.analytics.simulation import run_monte_carlo, calculate_simulation_metrics
        import numpy as np
        import pandas as pd

        # 1. Vérifications de base
        if not self.current_portfolio:
            return {"error": "no_portfolio"}

        tickers = list(self.current_portfolio.positions.keys())
        if not tickers:
            return {"error": "no_positions"}

        # 2. Téléchargement parallèle des données
        returns_map = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._get_historical_data, t, "3y"): t for t in tickers}

            for future in futures:
                t = futures[future]
                df = future.result()

                if df is None or df.empty or "Close" not in df.columns:
                    continue

                # Nettoyage : resample daily + ffill
                df = df.resample("1D").ffill()

                # Rendements propres
                ret = df["Close"].pct_change().dropna()
                if not ret.empty:
                    returns_map[t] = ret

        if not returns_map:
            return {"error": "no_historical_data"}

        # 3. Construction du DataFrame aligné
        df_returns = pd.DataFrame(returns_map).dropna(how="all")
        df_returns = df_returns.ffill().dropna()

        # 4. Valeur du portefeuille et poids
        prices = self.get_market_prices(tickers)
        total_val = sum(
            self.current_portfolio.positions[t].current_value(prices.get(t, None))
            for t in df_returns.columns
            if t in self.current_portfolio.positions and prices.get(t, None) is not None
        )

        if total_val <= 0:
            return {"error": "portfolio_value_zero"}

        weights = np.array([
            self.current_portfolio.positions[t].current_value(prices[t]) / total_val
            for t in df_returns.columns
        ])

        # 5. Simulation Monte Carlo robuste
        paths = run_monte_carlo(
            weights=weights,
            returns=df_returns,
            n_simulations=n_simulations,
            timeframe=timeframe,
            block_size=20,
            use_stochastic_vol=True,
            vol_kappa=0.15,
            vol_theta=1.0,
            vol_sigma=0.3
        )

        # 6. Calcul des métriques
        metrics = calculate_simulation_metrics(paths)

        # 7. Preview limité pour l’UI
        n_preview = min(50, paths.shape[1])
        preview = paths[:, :n_preview].tolist()

        return {
            "metrics": metrics,
            "paths_preview": preview,
            "n_simulations": n_simulations,
            "timeframe": timeframe,
            "tickers": list(df_returns.columns)
        }


    def analyze_portfolio(self, include_risk: bool = True) -> Dict[str, Any]:
        if not self.current_portfolio:
            return {'error': 'Aucun portefeuille actif'}
        tickers = list(self.current_portfolio.positions.keys())
        market_prices = self.get_market_prices(tickers)
        prices_for_calc = {
            t: (market_prices.get(t) or pos.average_price)
            for t, pos in self.current_portfolio.positions.items()
        }
        performance = self.current_portfolio.calculate_performance(prices_for_calc)
        allocation = self.current_portfolio.get_allocation(prices_for_calc)
        risk_metrics = {}
        if include_risk:
            risk_metrics = self._calculate_advanced_risk(tickers, prices_for_calc)
        else:
            risk_metrics = self._quick_risk_estimate(tickers, prices_for_calc)
        return {
            'performance': performance,
            'allocation': allocation,
            'risk_metrics': risk_metrics,
            'market_prices': prices_for_calc,
            'timestamp': pd.Timestamp.now().isoformat()
        }

    def _quick_risk_estimate(self, tickers: List[str], prices: Dict[str, float]) -> Dict:
        try:
            returns_map = {}
            for t in tickers:
                df = self._get_historical_data(t, period="1y")
                if df is not None and not df.empty and 'Close' in df.columns:
                    returns_map[t] = df['Close'].pct_change().dropna()
            if not returns_map:
                return {'volatility': 0.0, 'sharpe_ratio': 0.0, 'value_at_risk': 0.0}
            df_returns = pd.DataFrame(returns_map).fillna(0)
            total_val = sum(
                self.current_portfolio.positions[t].current_value(prices.get(t, 0))
                for t in df_returns.columns if t in self.current_portfolio.positions
            )
            if total_val <= 0:
                return {'volatility': 0.0, 'sharpe_ratio': 0.0, 'value_at_risk': 0.0}
            weights = np.array([
                self.current_portfolio.positions[t].current_value(prices.get(t, 0)) / total_val
                for t in df_returns.columns
            ])
            portfolio_returns = df_returns.dot(weights)
            vol = float(portfolio_returns.std() * np.sqrt(252))
            sharpe = float(self.risk_manager.calculate_sharpe_ratio(portfolio_returns))
            var = float(self.risk_manager.calculate_value_at_risk(portfolio_returns, 0.95))
            return {'volatility': vol, 'sharpe_ratio': sharpe, 'value_at_risk': var}
        except Exception as e:
            logger.debug(f"Quick risk estimate failed: {e}")
            return {'volatility': 0.0, 'sharpe_ratio': 0.0, 'value_at_risk': 0.0}

    # =========================================================
    # 6. MÉTHODES SUPPLÉMENTAIRES
    # =========================================================
    def compute_correlation_matrix(self, tickers: List[str]) -> pd.DataFrame:
        returns = {}
        for t in tickers:
            df = self._get_historical_data(t, "1y")
            if df is not None:
                returns[t] = df["Close"].pct_change()
        df_returns = pd.DataFrame(returns).dropna()
        return df_returns.corr()

    def generate_rebalance_orders(self, target_alloc: Dict[str, float]) -> List[Dict]:
        orders = []
        prices = self.get_market_prices(list(target_alloc.keys()))
        total_value = sum(
            pos.current_value(prices.get(t, 0))
            for t, pos in self.current_portfolio.positions.items()
        )
        for ticker, target_weight in target_alloc.items():
            current_value = self.current_portfolio.positions.get(ticker, Position()).current_value(prices.get(ticker, 0))
            current_weight = current_value / total_value if total_value else 0
            diff = target_weight - current_weight
            if abs(diff) > 0.02:
                orders.append({
                    "ticker": ticker,
                    "action": "BUY" if diff > 0 else "SELL",
                    "delta_weight": round(diff, 4)
                })
        return orders