import numpy as np
import pandas as pd

def run_monte_carlo(weights, returns, n_simulations=1000, timeframe=252):
    """
    Simule n trajectoires futures du portefeuille.
    - weights: array des poids des actifs
    - returns: DataFrame des rendements historiques
    """
    if returns.empty or len(weights) == 0:
        return np.zeros((timeframe, n_simulations))
    
    # Calcul des paramètres statistiques
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Rendement et Volatilité attendus du portefeuille (annualisés)
    port_return = np.sum(mean_returns * weights) * timeframe
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(timeframe)
    
    # Simulation des rendements Quotidiens (Mouvement Brownien Géométrique)
    # On génère une matrice (jours x simulations)
    daily_returns = np.random.normal(
        port_return / timeframe, 
        port_vol / np.sqrt(timeframe), 
        (timeframe, n_simulations)
    )
    
    # Transformation en trajectoires de prix (base 100)
    # np.exp(cumsum) permet de simuler la capitalisation composée
    price_paths = np.exp(np.cumsum(daily_returns, axis=0)) * 100
    
    return price_paths

def calculate_simulation_metrics(paths):
    """
    Extrait les probabilités et la Value-at-Risk des trajectoires.
    """
    if paths.size == 0:
        return {"var_95": 0, "cvar_95": 0, "prob_profit": 0}

    # On analyse la valeur à la dernière ligne (fin de l'horizon)
    final_returns = (paths[-1, :] - 100) / 100
    
    # Value at Risk 95% (le 5ème percentile des pires résultats)
    var_95 = np.percentile(final_returns, 5)
    
    # Conditional VaR (Moyenne des pertes dans le scénario catastrophe)
    cvar_95 = final_returns[final_returns <= var_95].mean()
    
    # Probabilité que le rendement soit > 0
    prob_profit = (final_returns > 0).mean()
    
    return {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "prob_profit": prob_profit
    }