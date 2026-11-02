import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from scipy import stats

class RiskManager:
    """
    Gestionnaire de risques purement calculatoire.
    Ne contient aucune logique de récupération de données (Data Agnostic).
    """

    # ----------------------------------------------------------------------
    # Métriques de Performance & Risque
    # ----------------------------------------------------------------------

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, 
                               periods_per_year: int = 252) -> float:
        # Nettoyage des données pour éviter les erreurs stats
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if clean_returns.empty or clean_returns.std() == 0:
            return 0.0
            
        adj_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = clean_returns - adj_rf
        return float(excess_returns.mean() / clean_returns.std() * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, 
                                 periods_per_year: int = 252) -> float:
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if clean_returns.empty:
            return 0.0
            
        adj_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns_avg = clean_returns.mean() - adj_rf
        
        downside_returns = clean_returns[clean_returns < 0]
        if downside_returns.empty:
            return 0.0
            
        downside_std = downside_returns.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
            
        return float(excess_returns_avg / downside_std * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Calcule le MDD à partir des rendements.
        Correction du bug des pourcentages aberrants.
        """
        if returns.empty:
            return 0.0
            
        # Construction de la courbe d'équité (base 100 pour la stabilité)
        equity_curve = (1 + returns).cumprod()
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # Le drawdown est une valeur négative (ex: -0.15 pour -15%)
        mdd = float(drawdown.min())
        return max(mdd, -1.0) # On ne peut pas perdre plus de 100% (levier exclu)

    @staticmethod
    def calculate_value_at_risk(returns: pd.Series, confidence_level: float = 0.95, 
                                 method: str = "historical") -> float:
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if clean_returns.empty:
            return 0.0
        
        alpha = 1 - confidence_level
        if method == "historical":
            return float(clean_returns.quantile(alpha))
        elif method == "parametric":
            return float(stats.norm.ppf(alpha, clean_returns.mean(), clean_returns.std()))
        return 0.0

    # ----------------------------------------------------------------------
    # Analyse de Corrélation & Bêta
    # ----------------------------------------------------------------------

    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        if stock_returns.empty or market_returns.empty:
            return 1.0 # Neutre par défaut
            
        combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(combined) < 5: return 1.0
        
        cov_matrix = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])
        market_variance = cov_matrix[1, 1]
        
        if market_variance == 0: return 1.0
        return float(cov_matrix[0, 1] / market_variance)

    # ----------------------------------------------------------------------
    # Diagnostics Avancés
    # ----------------------------------------------------------------------

    @staticmethod
    def ulcer_index(returns: pd.Series) -> float:
        """Mesure le stress (profondeur et durée des drawdowns)."""
        clean_returns = returns.dropna()
        if clean_returns.empty: return 0.0
        
        wealth_index = (1 + clean_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return float(np.sqrt((drawdowns**2).mean()))

    def risk_report(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Génère un rapport de risque complet et propre pour l'interface Dash.
        """
        # 1. Nettoyage initial des rendements
        clean_returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

        if clean_returns.empty or (clean_returns == 0).all():
            return {
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 
                'volatility': 0.0, 'var_95': 0.0, 'ulcer_index': 0.0
            }

        # 2. Calculs des métriques
        vol_ann = float(clean_returns.std() * np.sqrt(252))
        mdd = self.calculate_max_drawdown(clean_returns)
        sharpe = self.calculate_sharpe_ratio(clean_returns)
        
        # VaR Fallback (on tente historique, sinon paramétrique)
        var_val = self.calculate_value_at_risk(clean_returns, 0.95, "historical")
        if np.isnan(var_val) or var_val == 0:
            var_val = self.calculate_value_at_risk(clean_returns, 0.95, "parametric")

        # 3. Construction du dictionnaire final
        report = {
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(self.calculate_sortino_ratio(clean_returns), 2),
            'max_drawdown': round(mdd, 4),  # Garder en décimal pour le traitement UI
            'volatility': round(vol_ann, 4),
            'var_95': round(float(var_val), 4),
            'ulcer_index': round(self.ulcer_index(clean_returns), 4),
            'returns_series': clean_returns.to_dict()
        }

        if market_returns is not None:
            report['beta'] = round(self.calculate_beta(clean_returns, market_returns), 2)

        return report

    # ----------------------------------------------------------------------
    # Sorties et Tailles de Position
    # ----------------------------------------------------------------------

    @staticmethod
    def suggest_position_size(account_balance: float, entry: float, stop: float,
                               risk_percentage: float = 0.01) -> Dict[str, float]:
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return {"units": 0.0, "total_risk_cash": 0.0, "position_value": 0.0}
            
        risk_cash = account_balance * risk_percentage
        units = risk_cash / risk_per_share
        return {
            "units": float(units),
            "total_risk_cash": float(risk_cash),
            "position_value": float(units * entry)
        }

    @staticmethod
    def risk_reward_ratio(entry: float, stop: float, target: float) -> float:
        """
        Calcule le ratio Risk/Reward donné :
          - reward = (target - entry) / entry
          - risk   = abs(entry - stop) / entry
          - ratio  = reward / risk (si risk > 0)
        Retourne float. Si risk == 0 retourne float('inf') pour signaler ratio infini.
        """
        try:
            if entry is None or stop is None or target is None:
                return 0.0
            if entry == 0:
                return 0.0
            reward = (target - entry) / entry
            risk = abs(entry - stop) / entry
            if risk == 0:
                return float('inf') if reward > 0 else 0.0
            return float(reward / risk)
        except Exception:
            return 0.0

    @staticmethod
    def calculate_atr_levels(df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calcule ATR et niveaux de sortie sur un DataFrame OHLC."""
        if df is None or len(df) < period + 1:
            return {"atr": 0.0, "stop_loss": 0.0, "take_profit": 0.0}

        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        current_price = df['Close'].iloc[-1]
        
        return {
            "atr": round(float(atr), 4),
            "stop_loss": round(float(current_price - (atr * 2)), 2),
            "take_profit": round(float(current_price + (atr * 4)), 2)
        }