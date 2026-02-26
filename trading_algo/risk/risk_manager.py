import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

class RiskManager:
    """
    Calcule des métriques de risque avancées et suggère des niveaux de stop-loss/take-profit
    pour une gestion optimisée des risques dans les investissements en actions.
    """
    
    @staticmethod
    def calculate_atr_levels(data: pd.DataFrame, atr_multiplier_stop: float = 2.0, 
                             atr_multiplier_take: float = 3.0) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcule un stop-loss et un take-profit basés sur l'ATR.
        
        Args:
            data: DataFrame contenant les colonnes 'ATR' et 'Close'.
            atr_multiplier_stop: Multiplicateur pour le stop-loss (par défaut 2.0).
            atr_multiplier_take: Multiplicateur pour le take-profit (par défaut 3.0).
        
        Returns:
            Tuple de (stop-loss, take-profit), ou (None, None) si les données sont invalides.
        """
        if 'ATR' not in data.columns or data.empty:
            return None, None
        last_atr = data['ATR'].iloc[-1]
        last_close = data['Close'].iloc[-1]
        stop_loss = float(last_close - atr_multiplier_stop * last_atr)
        take_profit = float(last_close + atr_multiplier_take * last_atr)
        return stop_loss, take_profit
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calcule le ratio de Sharpe annualisé.
        
        Args:
            returns: Série des rendements journaliers.
            risk_free_rate: Taux sans risque annualisé (par défaut 0.02).
        
        Returns:
            Ratio de Sharpe, ou 0.0 si l'écart-type est nul.
        """
        if returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() - risk_free_rate / 252
        return float(excess_returns / returns.std() * np.sqrt(252))
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calcule le drawdown maximum.
        
        Args:
            equity_curve: Série de la courbe d'équité (valeurs cumulées).
        
        Returns:
            Drawdown maximum (valeur négative minimale).
        """
        if equity_curve.empty:
            return 0.0
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return float(drawdown.min())
    
    @staticmethod
    def calculate_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcule la Value at Risk (VaR) historique.
        
        Args:
            returns: Série des rendements journaliers.
            confidence_level: Niveau de confiance (par défaut 0.95 pour 95%).
        
        Returns:
            VaR (perte potentielle maximale avec le niveau de confiance donné).
        """
        if returns.empty:
            return 0.0
        sorted_returns = returns.sort_values()
        var_index = int((1 - confidence_level) * len(sorted_returns))
        return float(sorted_returns.iloc[var_index])
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calcule le bêta de l'action par rapport au marché.
        
        Args:
            stock_returns: Série des rendements de l'action.
            market_returns: Série des rendements du marché (ex. indice de référence).
        
        Returns:
            Valeur du bêta, ou 0.0 si les données sont insuffisantes.
        """
        if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
            return 0.0
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return float(covariance / market_variance) if market_variance != 0 else 0.0
    
    @staticmethod
    def suggest_position_size(account_balance: float, entry: float, stop: float, 
                              risk_percentage: float = 0.01) -> float:
        """
        Sugère la taille de position basée sur le risque toléré.
        
        Args:
            account_balance: Solde du compte.
            entry: Prix d'entrée.
            stop: Niveau de stop-loss.
            risk_percentage: Pourcentage de risque par trade (par défaut 1%).
        
        Returns:
            Taille de position suggérée (nombre d'actions).
        """
        risk_amount = account_balance * risk_percentage
        risk_per_share = abs(entry - stop)
        return float(risk_amount / risk_per_share) if risk_per_share > 0 else 0.0
    
    @staticmethod
    def risk_reward_ratio(entry: float, stop: float, target: float) -> float:
        """
        Calcule le ratio risque/récompense.
        
        Args:
            entry: Prix d'entrée.
            stop: Niveau de stop-loss.
            target: Niveau de take-profit.
        
        Returns:
            Ratio risque/récompense, ou 0.0 si le risque est nul.
        """
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return reward / risk if risk > 0 else 0.0