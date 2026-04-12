import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Imports internes
from .portfolio import Portfolio, Order, Position
from trading_algo.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

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
        self.risk_manager = RiskManager()

        # Cache de session pour les données historiques (accélère le Risk Analysis)
        self._market_data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"PortfolioManager opérationnel sur : {self.base_dir}")

    # =========================================================
    # 1. PERSISTENCE & GESTION DES FICHIERS
    # =========================================================

    def create_portfolio(self, name: str, initial_cash: float) -> Portfolio:
        """Crée un nouveau portefeuille et l'enregistre immédiatement."""
        portfolio = Portfolio(cash=initial_cash, name=name)
        self.current_portfolio = portfolio
        self.save_portfolio()
        return portfolio

    def list_portfolios(self) -> List[str]:
        """Retourne la liste des noms de portefeuilles disponibles (sans .json)."""
        return [f.stem for f in self.base_dir.glob("*.json")]

    def load_portfolio(self, name: str) -> Optional[Portfolio]:
        """Charge un portefeuille spécifique depuis le disque."""
        filepath = self.base_dir / f"{name}.json"
        
        if not filepath.exists():
            logger.error(f"Portefeuille introuvable : {filepath}")
            return None
            
        try:
            self.current_portfolio = Portfolio.load_from_file(str(filepath))
            logger.info(f"Portefeuille '{name}' chargé (Cash: {self.current_portfolio.cash:.2f}$)")
            return self.current_portfolio
        except Exception as e:
            logger.exception(f"Erreur fatale lors du chargement de {name}: {e}")
            return None

    def save_portfolio(self):
        """Sauvegarde l'état du portefeuille actif."""
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
        """Récupère les derniers prix de clôture de manière optimisée."""
        if not tickers: return {}
        
        extractor = self.data_extractor_class()
        try:
            # On tente le batch fetch (plus rapide)
            return extractor.get_bulk_prices(tickers)
        except Exception:
            # Fallback en parallèle si le bulk échoue
            def _fetch(t):
                df = extractor.get_historical_data(t, period="1d")
                return t, float(df['Close'].iloc[-1]) if not df.empty else None

            with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
                results = dict(executor.map(_fetch, tickers))
                return {t: p for t, p in results.items() if p is not None}

    def analyze_portfolio(self) -> Dict[str, Any]:
        """Analyse complète : Performance + Allocation + Risque."""
        if not self.current_portfolio:
            return {'error': 'Aucun portefeuille actif'}

        tickers = list(self.current_portfolio.positions.keys())
        market_prices = self.get_market_prices(tickers)

        # On utilise le prix actuel ou le prix moyen si le marché est fermé/indisponible
        prices_for_calc = {
            t: (market_prices.get(t) or pos.average_price) 
            for t, pos in self.current_portfolio.positions.items()
        }

        performance = self.current_portfolio.calculate_performance(prices_for_calc)
        allocation = self.current_portfolio.get_allocation(prices_for_calc)
        risk_metrics = self._calculate_advanced_risk(tickers, prices_for_calc)

        return {
            'performance': performance,
            'allocation': allocation,
            'risk_metrics': risk_metrics,
            'market_prices': prices_for_calc,
            'timestamp': pd.Timestamp.now().isoformat()
        }

    def _calculate_advanced_risk(self, tickers: List[str], prices: Dict[str, float]) -> Dict:
        """Calcule les métriques de risque (VaR, Sharpe, Vol) via le RiskManager."""
        if not tickers: return {"status": "vide"}
        
        extractor = self.data_extractor_class()
        returns_map = {}

        # Mise à jour du cache historique en parallèle (3 ans de données pour la stabilité)
        missing = [t for t in tickers if t not in self._market_data_cache]
        if missing:
            with ThreadPoolExecutor(max_workers=5) as executor:
                def _load(t): return t, extractor.get_historical_data(t, period="3y")
                for t, df in executor.map(_load, missing):
                    if df is not None and not df.empty:
                        self._market_data_cache[t] = df

        for t in tickers:
            df = self._market_data_cache.get(t)
            if df is not None:
                returns_map[t] = df['Close'].pct_change()

        if not returns_map: return {"status": "données_insuffisantes"}

        # Calcul des poids du portefeuille
        df_returns = pd.DataFrame(returns_map).fillna(0)
        total_val = sum(self.current_portfolio.positions[t].current_value(prices.get(t, 0)) for t in df_returns.columns)
        
        if total_val <= 0: return {"status": "valeur_nulle"}

        weights = np.array([self.current_portfolio.positions[t].current_value(prices.get(t, 0)) / total_val for t in df_returns.columns])

        # Rendements pondérés du portefeuille global
        portfolio_returns = df_returns.dot(weights)
        return self.risk_manager.risk_report(portfolio_returns)

    # =========================================================
    # 3. MODÈLE STRATÉGIQUE (BUFFETT)
    # =========================================================

    def run_buffett_rebalance(self, fundamentals_map: Dict[str, Dict]) -> Tuple[Dict, Dict]:
        """
        Calcule les scores Buffett et génère une allocation cible.
        """
        if not self.current_portfolio: return {}, {}

        scores = {}
        for ticker in self.current_portfolio.positions.keys():
            f = fundamentals_map.get(ticker, {})
            
            # Extraction de la volatilité pour le calcul du score (pénalité de risque)
            data = self._market_data_cache.get(ticker)
            vol = data['Close'].pct_change().std() * np.sqrt(252) if data is not None else 0.20
            
            scores[ticker] = self._compute_buffett_score(f, vol)

        target_alloc = self._build_tiered_allocation(scores)
        return target_alloc, scores

    def _compute_buffett_score(self, fundamentals: Dict, volatility: float) -> float:
        """Modèle mathématique de sélection d'actifs 'Value'."""
        # Extraction avec valeurs par défaut prudentes
        metrics = {
            "roe": fundamentals.get("roe", 0.10),
            "margin": fundamentals.get("gross_margin", 0.20),
            "debt": fundamentals.get("debt_to_equity", 1.5),
            "moat": fundamentals.get("moat_score", 0.4)
        }
        
        quality = (metrics["roe"] * 0.3 + metrics["margin"] * 0.3 + metrics["moat"] * 0.2)
        risk_penalty = (min(metrics["debt"], 4.0) * 0.1 + volatility * 0.1)
        
        return float(quality - risk_penalty)

    def _build_tiered_allocation(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Transforme les scores en pourcentages d'allocation cibles."""
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # On ne garde que le top 10 pour la concentration Buffett
        top = sorted_assets[:10]
        
        # Tiers d'allocation décroissants
        tiers = [0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05]
        
        alloc = {ticker: tiers[i] for i, (ticker, _) in enumerate(top)}
        total = sum(alloc.values())
        alloc["cash"] = round(1.0 - total, 2)
        
        return alloc

    # --- MARKET CONTEXT ---

    def get_market_regime(self, market_data: Any) -> Dict:
        """Détermine si nous sommes en Risk-On ou Risk-Off."""
        try:
            extractor = self.data_extractor_class()
            score = extractor._compute_market_health_score(market_data)
        except:
            score = 50 

        regimes = [(70, "BULL"), (40, "NEUTRAL"), (0, "BEAR")]
        regime = next(r for threshold, r in regimes if score >= threshold)
        
        return {"score": score, "regime": regime, "action": "EXPAND" if regime == "BULL" else "DEFEND"}