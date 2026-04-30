import numpy as np
import pandas as pd


# ============================================================
#  Monte Carlo Robuste (Bootstrap + Vol Stochastique + Corrélations)
# ============================================================

def run_monte_carlo(
    weights: np.ndarray,
    returns: pd.DataFrame,
    n_simulations: int = 1000,
    timeframe: int = 252,
    block_size: int = 20,
    use_stochastic_vol: bool = True,
    vol_kappa: float = 0.10,
    vol_theta: float = 1.0,
    vol_sigma: float = 0.15
):
    """
    Monte Carlo robuste pour portefeuilles multi-actifs.
    Combine :
    - bootstrap par blocs (capturer les régimes)
    - corrélations dynamiques via Cholesky
    - volatilité stochastique (Heston-like simplifié)
    - diffusion lognormale

    Retourne :
        paths : matrice (timeframe x n_simulations)
    """

    if returns.empty or len(weights) == 0:
        return np.zeros((timeframe, n_simulations))

    # Nettoyage : suppression des rendements aberrants
    q_low = returns.quantile(0.001)
    q_high = returns.quantile(0.999)

    # IMPORTANT : axis=1 pour appliquer colonne par colonne
    returns = returns.clip(lower=q_low, upper=q_high, axis=1)

    # Statistiques multi-actifs
    mu = returns.mean().values
    cov = returns.cov().values

    # Décomposition de Cholesky pour corrélations
    chol = np.linalg.cholesky(cov)

    n_assets = returns.shape[1]
    paths = np.zeros((timeframe, n_simulations))

    # Préparation bootstrap
    n_blocks = timeframe // block_size + 2
    idx = np.arange(len(returns))

    for sim in range(n_simulations):

        # 1. Bootstrap par blocs
        sampled_returns = []
        for _ in range(n_blocks):
            start = np.random.randint(0, len(idx) - block_size)
            sampled_returns.append(returns.iloc[start:start + block_size].values)

        sampled_returns = np.vstack(sampled_returns)[:timeframe]

        # 2. Corrélations dynamiques
        correlated_noise = sampled_returns @ chol.T

        # 3. Volatilité stochastique (Heston-like simplifié)
        if use_stochastic_vol:
            vol = np.ones(timeframe)
            for t in range(1, timeframe):
                vol[t] = abs(
                    vol[t - 1]
                    + vol_kappa * (vol_theta - vol[t - 1])
                    + vol_sigma * np.random.randn()
                )

            # Anti-explosion : clamp la vol
            vol = np.clip(vol, 0.5, 3.0)

            correlated_noise = correlated_noise * vol[:, None]

        # 4. Rendement du portefeuille
        portfolio_returns = correlated_noise @ weights

        # 5. Simulation lognormale (base 100)
        price_path = 100 * np.exp(np.cumsum(portfolio_returns))
        paths[:, sim] = price_path

    return paths


# ============================================================
#  Métriques avancées
# ============================================================

def calculate_simulation_metrics(paths: np.ndarray) -> dict:
    """
    Calcule des métriques avancées :
    - médiane
    - percentiles
    - VaR / CVaR
    - probabilité de perte / gain
    - drawdown médian
    """

    if paths.size == 0:
        return {
            "median": 0,
            "p5": 0,
            "p95": 0,
            "mean": 0,
            "std": 0,
            "var_95": 0,
            "cvar_95": 0,
            "prob_loss": 0,
            "prob_gain_10pct": 0,
            "max_drawdown_median_path": 0
        }

    final_values = paths[-1, :]          # niveaux (ex : 100, 110, 90)
    final_returns = (final_values - 100) / 100.0   # rendements (ex : 0.10, -0.05, etc.)

    # Percentiles sur les NIVEAUX (pour le graphique)
    median = float(np.median(final_values))
    p5 = float(np.percentile(final_values, 5))
    p95 = float(np.percentile(final_values, 95))

    # VaR / CVaR sur les RENDEMENTS
    var_95 = float(np.percentile(final_returns, 5))
    cvar_95 = float(final_returns[final_returns <= var_95].mean())

    # Probabilités sur les RENDEMENTS
    prob_loss = float(np.mean(final_returns < 0))
    prob_gain_10pct = float(np.mean(final_returns > 0.10))

    # Drawdown sur la trajectoire médiane (en niveaux)
    median_index = np.argsort(final_values)[len(final_values) // 2]
    median_path = paths[:, median_index]
    running_max = np.maximum.accumulate(median_path)
    drawdowns = 1 - median_path / running_max
    max_dd = float(np.max(drawdowns))

    return {
        "median": median,
        "p5": p5,
        "p95": p95,
        "mean": float(np.mean(final_values)),
        "std": float(np.std(final_values)),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "prob_loss": prob_loss,
        "prob_gain_10pct": prob_gain_10pct,
        "max_drawdown_median_path": max_dd
    }

