"""
Module de gestion des risques pour l'analyse boursière
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class RiskManager:
    """Calcule des métriques de risque et suggère des niveaux de stop-loss/take-profit"""
    
    @staticmethod
    def calculate_atr_stop(data: pd.DataFrame, atr_multiplier: float = 2.0) -> Optional[float]:
        """Calcule un stop-loss basé sur l'ATR"""
        if 'ATR' not in data.columns or data.empty:
            return None
        last_atr = data['ATR'].iloc[-1]
        last_close = data['Close'].iloc[-1]
        return float(last_close - atr_multiplier * last_atr)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcule le ratio de Sharpe annualisé"""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() - risk_free_rate / 252
        return float(excess_returns / returns.std() * np.sqrt(252))
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calcule le drawdown maximum"""
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return float(drawdown.min())
    
    @staticmethod
    def risk_reward_ratio(entry: float, stop: float, target: float) -> float:
        """Calcule le ratio risque/récompense"""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return reward / risk if risk > 0 else 0.0