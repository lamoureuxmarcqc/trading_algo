import os
"""
Module de gestion avancée du portefeuille
"""
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .portfolio import Portfolio, Order



class PortfolioManager:
    """Gestionnaire de portefeuille avec fonctionnalités avancées"""
    
    def __init__(self, data_extractor, portfolios_dir: str = "portfolios"):
        self.data_extractor = data_extractor
        self.portfolios_dir = portfolios_dir
        self.current_portfolio: Optional[Portfolio] = None
        os.makedirs(portfolios_dir, exist_ok=True)
    
    def create_portfolio(self, name: str, initial_cash: float) -> Portfolio:
        """Crée un nouveau portefeuille"""
        portfolio = Portfolio(cash=initial_cash, name=name)
        self.current_portfolio = portfolio
        return portfolio
    
    def load_portfolio(self, name: str) -> Optional[Portfolio]:
        """Charge un portefeuille existant"""
        filepath = os.path.join(self.portfolios_dir, f"{name}.json")
        if os.path.exists(filepath):
            self.current_portfolio = Portfolio.load_from_file(filepath)
            return self.current_portfolio
        return None
    
    def save_current_portfolio(self):
        """Sauvegarde le portefeuille courant"""
        if self.current_portfolio:
            filepath = os.path.join(
                self.portfolios_dir, 
                f"{self.current_portfolio.name}.json"
            )
            self.current_portfolio.save_to_file(filepath)
    
    def list_portfolios(self) -> List[str]:
        """Liste tous les portefeuilles disponibles"""
        files = os.listdir(self.portfolios_dir)
        return [f.replace('.json', '') for f in files if f.endswith('.json')]

    def _convert_to_yahoo_symbol(self, ticker: str) -> str:
        """Convertit un symbole avec suffixe -C ou -U au format Yahoo Finance"""
        if ticker.endswith('-C'):
            base = ticker[:-2]  # Enlève le suffixe -C
            # Remplacer le point par un tiret pour les classes d'actions
            base = base.replace('.', '-')
            return f"{base}.TO"  # Ajoute .TO pour la bourse canadienne
        elif ticker.endswith('-U'):
            base = ticker[:-2]  # Enlève le suffixe -U
            # Remplacer le point par un tiret (ex: BRK.B -> BRK-B)
            base = base.replace('.', '-')
            return base  # Pas de suffixe pour les actions US
        else:
            # Si le symbole n'a pas de suffixe, on remplace quand même les points
            return ticker.replace('.', '-')  

    def get_market_prices(self, tickers: List[str]) -> Dict[str, float]:
        prices = {}
        for ticker in tickers:
            yahoo_ticker = self._convert_to_yahoo_symbol(ticker)   # <-- ici
            try:
                extractor = self.data_extractor(yahoo_ticker)
                data = extractor.get_historical_data(period="1d")
                if data.empty:
                    data = extractor.get_historical_data(period="1mo")
                if not data.empty:
                    prices[ticker] = data['Close'].iloc[-1]
                else:
                    prices[ticker] = None
                    print(f"⚠️ Aucune donnée pour {ticker} (converti en {yahoo_ticker})")
            except Exception as e:
                print(f"⚠️ Erreur pour {ticker} (converti en {yahoo_ticker}): {e}")
                prices[ticker] = None
        return prices   

    def analyze_portfolio(self) -> Dict:
        if not self.current_portfolio:
            return {'error': 'Aucun portefeuille chargé'}
    
        tickers = list(self.current_portfolio.positions.keys())
        market_prices = self.get_market_prices(tickers)
    
        # Remplacer les prix manquants par le prix d'achat moyen
        for ticker, pos in self.current_portfolio.positions.items():
            if market_prices.get(ticker) is None:
                market_prices[ticker] = pos.average_price
                print(f"ℹ️ Utilisation du prix d'achat moyen pour {ticker}: {pos.average_price}")
    
        # Calculer la performance
        performance = self.current_portfolio.calculate_performance(market_prices)
    
       
        # Analyse de diversification
        allocation = self.current_portfolio.get_allocation(market_prices)
        
        # Calcul du beta du portefeuille
        beta_portfolio = 0
        for ticker, position in self.current_portfolio.positions.items():
            if ticker in market_prices:
                weight = allocation.get(ticker, 0)
                # Récupérer le beta de l'action (simulé ici)
                beta_stock = 1.0  # À remplacer par une vraie valeur
                beta_portfolio += weight * beta_stock
        
        # Corrélation entre les positions
        correlations = {}
        if len(tickers) > 1:
            try:
                # Récupérer les données historiques
                data_dict = {}
                for ticker in tickers[:5]:  # Limiter pour la performance
                    extractor = self.data_extractor(ticker)
                    data = extractor.get_historical_data(period="6mo")
                    if not data.empty:
                        data_dict[ticker] = data['Close']
                
                if len(data_dict) > 1:
                    df = pd.DataFrame(data_dict)
                    correlations = df.pct_change().corr().to_dict()
            except Exception as e:
                print(f"⚠️ Erreur calcul corrélation: {e}")
        
        # Métriques de risque avancées
        risk_metrics = self._calculate_risk_metrics(market_prices)
        
        return {
            'performance': performance,
            'allocation': allocation,
            'beta_portfolio': beta_portfolio,
            'correlations': correlations,
            'risk_metrics': risk_metrics,
            'market_prices': market_prices
        }
    
    def _calculate_risk_metrics(self, market_prices: Dict[str, float]) -> Dict:
        """Calcule des métriques de risque avancées"""
        if not self.current_portfolio:
            return {}
        
        # Simulation de Monte Carlo simplifiée
        total_value = self.current_portfolio.total_value(market_prices)
        
        # VaR historique (simulée)
        var_95 = total_value * 0.02  # 2% de perte potentielle
        var_99 = total_value * 0.03  # 3% de perte potentielle
        
        # Stress test - scénarios de marché
        stress_scenarios = {
            'market_crash_-20': total_value * 0.8,
            'tech_bubble_burst': total_value * 0.85,
            'recession': total_value * 0.9,
            'inflation_spike': total_value * 0.95
        }
        
        return {
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'stress_test_scenarios': stress_scenarios,
            'concentration_risk': self._calculate_concentration_risk(market_prices)
        }
    
    def _calculate_concentration_risk(self, market_prices: Dict[str, float]) -> float:
        """Calcule le risque de concentration (indice Herfindahl)"""
        allocation = self.current_portfolio.get_allocation(market_prices)
        
        if not allocation:
            return 0
        
        # Exclure le cash
        weights = [w for k, w in allocation.items() if k != 'cash']
        
        if not weights:
            return 0
        
        # Indice de Herfindahl-Hirschman
        hhi = sum(w**2 for w in weights)
        
        # Normaliser entre 0 et 1
        n = len(weights)
        if n > 1:
            hhi_normalized = (hhi - 1/n) / (1 - 1/n)
        else:
            hhi_normalized = 1
        
        return hhi_normalized
    
    def suggest_rebalance(
        self, 
        target_allocation: Dict[str, float],
        min_trade_pct: float = 0.01
    ) -> Tuple[List[Order], Dict]:
        """Suggère un rééquilibrage du portefeuille"""
        if not self.current_portfolio:
            return [], {'error': 'Aucun portefeuille chargé'}
        
        tickers = list(self.current_portfolio.positions.keys())
        market_prices = self.get_market_prices(tickers)
        
        # Générer les ordres
        orders = self.current_portfolio.generate_rebalance_orders(
            target_allocation,
            market_prices,
            min_trade_pct=min_trade_pct
        )
        
        # Calculer l'impact
        current_allocation = self.current_portfolio.get_allocation(market_prices)
        total_value = self.current_portfolio.total_value(market_prices)
        
        impact = {
            'current_allocation': current_allocation,
            'target_allocation': target_allocation,
            'orders_count': len(orders),
            'total_trade_value': sum(order.value for order in orders),
            'total_trade_pct': sum(order.value for order in orders) / total_value if total_value > 0 else 0
        }
        
        return orders, impact
    
    def backtest_strategy(
        self, 
        strategy_func,
        initial_cash: float,
        start_date: str,
        end_date: str,
        tickers: List[str]
    ) -> Dict:
        """Backtest d'une stratégie sur le portefeuille"""
        
        # Récupérer les données historiques
        data_dict = {}
        for ticker in tickers:
            extractor = self.data_extractor(ticker)
            data = extractor.get_historical_data(start=start_date, end=end_date)
            if not data.empty:
                data_dict[ticker] = data
        
        if not data_dict:
            return {'error': 'Pas de données disponibles'}
        
        # Créer un DataFrame avec les dates communes
        dates = None
        for ticker, data in data_dict.items():
            if dates is None:
                dates = set(data.index)
            else:
                dates = dates.intersection(set(data.index))
        
        dates = sorted(list(dates))
        
        # Simuler le portefeuille
        portfolio = Portfolio(cash=initial_cash)
        equity_curve = []
        
        for i, date in enumerate(dates):
            prices = {}
            for ticker in tickers:
                if ticker in data_dict and date in data_dict[ticker].index:
                    prices[ticker] = data_dict[ticker].loc[date, 'Close']
            
            # Exécuter la stratégie
            if i > 0:
                prev_date = dates[i-1]
                prev_prices = {}
                for ticker in tickers:
                    if ticker in data_dict and prev_date in data_dict[ticker].index:
                        prev_prices[ticker] = data_dict[ticker].loc[prev_date, 'Close']
                
                signals = strategy_func(prices, prev_prices, portfolio)
                
                # Exécuter les ordres
                for signal in signals:
                    if signal['type'] == 'buy' and signal['ticker'] in prices:
                        portfolio.add_position(
                            signal['ticker'],
                            signal['quantity'],
                            prices[signal['ticker']]
                        )
                    elif signal['type'] == 'sell' and signal['ticker'] in prices:
                        try:
                            portfolio.remove_position(
                                signal['ticker'],
                                signal['quantity'],
                                prices[signal['ticker']]
                            )
                        except ValueError:
                            pass
            
            # Enregistrer la valeur du portefeuille
            equity_curve.append({
                'date': date,
                'value': portfolio.total_value(prices)
            })
        
        # Calculer les métriques de performance
        df = pd.DataFrame(equity_curve)
        df['returns'] = df['value'].pct_change()
        
        total_return = (df['value'].iloc[-1] / initial_cash - 1) * 100
        annual_return = total_return / (len(dates) / 252) if len(dates) > 0 else 0
        volatility = df['returns'].std() * np.sqrt(252) * 100
        sharpe = (df['returns'].mean() / df['returns'].std() * np.sqrt(252)) if df['returns'].std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'equity_curve': df,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': df['value'].iloc[-1] if not df.empty else initial_cash
        }