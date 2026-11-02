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

# Import des routines d'optimisation (wrapper dans strategy)
from trading_algo.strategy.portfolio_optimizer import (
    optimize_portfolio as _optimize_portfolio,
    backtest_strategy as _backtest_strategy,
    build_portfolio_metadata as _build_portfolio_metadata
)

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

        # Stocke le dernier résultat d'optimisation (poids, backtest, meta)
        self._last_optimization: Optional[Dict[str, Any]] = None

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
    # 5. OPTIMISATION DE PORTEFEUILLE (intégration portfolio_optimizer)
    # =========================================================
    def optimize_current_portfolio(
        self,
        include_bonds: Optional[bool] = None,
        bond_ticker: Optional[str] = None,
        history_period: str = "5y",
        rebalance_freq: str = "ME",
        transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        """
        Lance l'optimisation (max Sharpe) sur le portefeuille courant en utilisant
        les routines définies dans `strategy.portfolio_optimizer`.
        Retourne dict avec poids optimaux, métadonnées et résultats de backtest.
        Non bloquant — en cas d'erreur on retourne 'error' dans le dict.
        """
        if not self.current_portfolio:
            logger.error("Aucun portefeuille chargé pour optimisation")
            return {"error": "no_portfolio_loaded"}

        tickers = list(self.current_portfolio.positions.keys())
        if not tickers:
            logger.error("Portefeuille vide")
            return {"error": "empty_portfolio"}

        # Récupération des historiques et construction d'un DataFrame de prix alignés
        price_map = {}
        for t in tickers:
            try:
                df = self._get_historical_data(t, period=history_period)
                if df is None or df.empty or 'Close' not in df.columns:
                    logger.warning(f"Historique insuffisant pour {t}")
                    continue
                price_map[t] = df['Close'].rename(t)
            except Exception as e:
                logger.debug(f"Erreur historique {t}: {e}")

        if not price_map:
            return {"error": "no_price_data"}

        prices = pd.concat(price_map.values(), axis=1, join='inner').dropna()
        if prices.empty:
            return {"error": "aligned_prices_empty"}

        # Rendements annualisés
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Métadonnées via fonction utilitaire existante (secteur, pays...)
        try:
            meta = _build_portfolio_metadata(tickers)
        except Exception as e:
            logger.debug("Impossible de construire les meta via build_portfolio_metadata: %s", e)
            meta = pd.DataFrame(index=tickers)

        # Décider inclusion obligations si paramètre non fourni
        if include_bonds is None:
            include_bonds = any(t for t in tickers if t.upper().startswith('AGG') or t.upper().endswith(('BND', 'TLT', 'XBB.TO')))

        try:
            weights = _optimize_portfolio(mean_returns, cov_matrix, meta, tickers, include_bonds, bond_ticker)
            optimal_series = pd.Series(weights, index=tickers)
        except Exception as e:
            logger.exception("Erreur lors de l'optimisation: %s", e)
            return {"error": f"optimization_failed: {e}"}

        # Backtest simple du portefeuille optimisé
        try:
            port_value = _backtest_strategy(returns, optimal_series, rebalance_freq, transaction_cost)
            # port_value is a Series (index dates)
            total_return = port_value.iloc[-1] - 1
            annualized_return = (1 + total_return) ** (252 / len(port_value)) - 1 if len(port_value) else np.nan
            daily_vol = port_value.pct_change().std() * np.sqrt(252) if len(port_value) else np.nan
            sharpe = (annualized_return - 0.02) / daily_vol if daily_vol and daily_vol != 0 else np.nan
            max_drawdown = (port_value / port_value.cummax() - 1).min() if len(port_value) else np.nan
            backtest_metrics = {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "annualized_vol": float(daily_vol),
                "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
                "max_drawdown": float(max_drawdown)
            }
        except Exception as e:
            logger.exception("Erreur lors du backtest: %s", e)
            port_value = None
            backtest_metrics = {"error": str(e)}

        result = {
            "tickers": tickers,
            "optimal_weights": optimal_series.to_dict(),
            "optimal_series": optimal_series,
            "backtest": {
                "metrics": backtest_metrics,
                "series": port_value
            },
            "meta": meta.to_dict() if isinstance(meta, pd.DataFrame) else meta
        }

        # Stocker dernier résultat d'optimisation dans l'objet manager pour réutilisation
        self._last_optimization = result
        return result

    def apply_optimized_allocation(self, threshold: float = 0.02) -> Dict[str, Any]:
        """
        Génère des ordres pour rapprocher le portefeuille des poids optimaux
        calculés par `optimize_current_portfolio`.
        threshold : poids minimum de différence pour émettre un ordre.
        Ne passe pas d'ordres réels — retourne la liste d'ordres à exécuter.
        """
        opt = getattr(self, "_last_optimization", None)
        if not opt:
            return {"error": "no_optimization_available"}
        optimal = opt.get("optimal_weights", {})
        prices = self.get_market_prices(list(self.current_portfolio.positions.keys()))
        current_alloc = self.current_portfolio.get_allocation(prices)
        orders = []
        total_value = sum(self.current_portfolio.positions[t].current_value(prices.get(t, 0)) for t in self.current_portfolio.positions)
        for t, target_w in optimal.items():
            cur_w = current_alloc.get(t, 0.0)
            diff = target_w - cur_w
            if abs(diff) >= threshold:
                action = "BUY" if diff > 0 else "SELL"
                orders.append({
                    "ticker": t,
                    "action": action,
                    "target_weight": target_w,
                    "current_weight": cur_w,
                    "delta_weight": diff,
                    "notional": float(diff * total_value)
                })
        return {"orders": orders, "total_value": float(total_value)}

    # =========================================================
    # 6. MONTE CARLO (déjà existante)
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

    def analyze_portfolio(self, include_risk: bool = True) -> Dict[str, Any]:
        """
        Return a JSON-serializable analysis summary used by the dashboard.
        Converts pandas/numpy/timestamp types to native Python types to avoid
        Dash serialization errors.
        """
        if not self.current_portfolio:
            logger.debug("analyze_portfolio: no portfolio loaded")
            return {}

        try:
            tickers = list(self.current_portfolio.positions.keys())
            market_prices = self.get_market_prices(tickers)

            # Performance from Portfolio (may contain numpy types)
            performance = self.current_portfolio.calculate_performance(market_prices)

            # Best-effort update history (non-fatal)
            try:
                self.current_portfolio.update_history(market_prices)
            except Exception:
                logger.debug("update_history failed during analyze_portfolio", exc_info=True)

            # Risk metrics (may include pandas Series / Timestamp keys)
            risk_metrics = {}
            if include_risk and tickers:
                try:
                    risk_metrics = self._calculate_advanced_risk(tickers, market_prices) or {}
                except Exception as e:
                    logger.debug(f"analyze_portfolio: risk calc failed: {e}", exc_info=True)
                    risk_metrics = {}

            # -----------------------
            # SANITIZE FOR JSON
            # -----------------------
            def _to_py(x):
                # convert numpy scalars, pandas types to native python types
                if isinstance(x, (np.generic,)):
                    return x.item()
                if isinstance(x, (pd.Timestamp, pd.DatetimeIndex)):
                    return str(x)
                return x

            # sanitize performance numbers and positions
            perf_clean = {}
            for k, v in (performance or {}).items():
                if k == "positions" and isinstance(v, dict):
                    pos_clean = {}
                    for t, p in v.items():
                        if isinstance(p, dict):
                            p_clean = {}
                            for pk, pv in p.items():
                                if isinstance(pv, (np.generic, pd.Timestamp)):
                                    p_clean[pk] = _to_py(pv)
                                else:
                                    try:
                                        # cast numeric-like to float/int, leave others
                                        if isinstance(pv, (int, float)):
                                            p_clean[pk] = float(pv) if isinstance(pv, float) else int(pv)
                                        else:
                                            p_clean[pk] = pv
                                    except Exception:
                                        p_clean[pk] = pv
                            pos_clean[t] = p_clean
                        else:
                            pos_clean[t] = p
                    perf_clean["positions"] = pos_clean
                else:
                    # numeric fields cast to float when possible
                    if isinstance(v, (np.generic,)):
                        perf_clean[k] = _to_py(v)
                    else:
                        perf_clean[k] = v
            # ensure top-level numeric fields are native numbers
            for fld in ("total_value", "cash", "total_pnl", "total_pnl_pct"):
                if fld in perf_clean:
                    try:
                        perf_clean[fld] = float(perf_clean[fld])
                    except Exception:
                        pass

            # sanitize market_prices values
            prices_clean = {}
            for t, val in (market_prices or {}).items():
                try:
                    prices_clean[t] = float(val) if val is not None else None
                except Exception:
                    prices_clean[t] = val

            # sanitize risk_metrics: especially returns_series keys (Timestamp) -> str
            if risk_metrics:
                rm = dict(risk_metrics)  # shallow copy
                rs = rm.get("returns_series")
                if rs is not None:
                    try:
                        # Convert to pandas.Series then map to iso strings and floats
                        s = pd.Series(rs)
                        s = s.dropna().astype(float)
                        rm["returns_series"] = {str(idx): float(v) for idx, v in s.items()}
                    except Exception:
                        # Fallback: try iterating dict
                        try:
                            new_rs = {}
                            for k, v in dict(rs).items():
                                new_rs[str(k)] = float(v) if v is not None else None
                            rm["returns_series"] = new_rs
                        except Exception:
                            rm["returns_series"] = {}
                # ensure numeric metrics are native types
                for k, v in rm.items():
                    if k != "returns_series":
                        if isinstance(v, (np.generic,)):
                            rm[k] = _to_py(v)
                risk_metrics = rm
            # --- existing code above ---
            # sanitize risk_metrics ... (existing)
            # -----------------------
            # CORRELATION MATRIX (daily returns, aligned)
            # -----------------------
            corr_dict = {}
            try:
                # build returns map
                returns_map = {}
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = {executor.submit(self._get_historical_data, t, "3y"): t for t in tickers}
                    for fut in futures:
                        t = futures[fut]
                        df = fut.result()
                        if df is None or df.empty or "Close" not in df.columns:
                            continue
                        ser = df["Close"].pct_change().dropna()
                        if not ser.empty:
                            returns_map[t] = ser

                if returns_map:
                    df_returns = pd.DataFrame(returns_map).dropna(how="all").ffill().dropna()
                    if not df_returns.empty:
                        corr = df_returns.corr().fillna(0)
                        # convert to plain python float dict
                        corr_dict = {str(r): {str(c): float(corr.at[r, c]) for c in corr.columns} for r in corr.index}
            except Exception:
                logger.debug("Failed to compute correlation matrix", exc_info=True)

            return {
                "performance": perf_clean,
                "risk_metrics": risk_metrics,
                "market_prices": prices_clean,
                "correlation_matrix": corr_dict
            }

        except Exception as e:
            logger.exception("analyze_portfolio failed: %s", e)
            return {
                "performance": {
                    "total_value": 0.0,
                    "cash": getattr(self.current_portfolio, "cash", 0.0),
                    "total_pnl": 0.0,
                    "total_pnl_pct": 0.0,
                    "positions": {}
                },
                "risk_metrics": {},
                "market_prices": {}
            }
