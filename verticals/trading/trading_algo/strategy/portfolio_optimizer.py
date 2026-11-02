#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimisation de portefeuille actions + obligations (optionnel)
avec analyse de corrélation, PCA, backtest et rééquilibrage dynamique.
Utilise les modules existants : trading.data.data_extraction (fetch_stock_data, get_stock_overview)
et trading_algo.logging_config.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import des modules internes
from trading_algo.data.data_extraction import fetch_stock_data, get_stock_overview
from trading_algo.logging_config import init_logging

# --------------------------------------------------------------
# Initialisation du logging
# --------------------------------------------------------------
init_logging(level=os.getenv("LOG_LEVEL", "INFO"), logfile=os.getenv("LOG_FILE", None))
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Paramètres globaux (modifiables)
# --------------------------------------------------------------
HORIZON_ANS = 4                     # horizon d'investissement en années
CAPITAL_INITIAL = 5_000_000         # capital de départ
OBJECTIF = 50_000_000               # objectif visé

# Liste des actions (exemple)
TICKERS_ACTIONS = ["CNR.TO", "BN.TO", "CNQ.TO", "XIU.TO", "CSU.TO", "RY.TO", "SHOP.TO"]
BOND_TICKER = "AGG"                 # ETF obligataire (US Aggregate) - remplacer par XBB.TO pour Canada

START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
REBALANCE_FREQ = "ME"                # ME = mensuel (fin de mois), QE = trimestriel, YE = annuel
TRANSACTION_COST = 0.001            # 0.1% par rééquilibrage

# --------------------------------------------------------------
# 1. Déterminer si on inclut les obligations (horizon < 5 ans)
# --------------------------------------------------------------
if HORIZON_ANS < 5:
    INCLURE_OBLIGATIONS = True
    logger.warning("Horizon < 5 ans : ajout d'obligations pour réduire le risque.")
else:
    INCLURE_OBLIGATIONS = False

if INCLURE_OBLIGATIONS:
    TICKERS = TICKERS_ACTIONS + [BOND_TICKER]
else:
    TICKERS = TICKERS_ACTIONS

# --------------------------------------------------------------
# 2. Récupération des prix via fetch_stock_data
# --------------------------------------------------------------
def get_prices_from_fetch(tickers, start_date, end_date):
    """
    Récupère les prix de clôture ajustés pour une liste de tickers
    entre deux dates, en utilisant fetch_stock_data.
    Retourne un DataFrame (index=datetime, colonnes=tickers).
    """
    price_data = {}
    for symbol in tickers:
        try:
            # On récupère le maximum de données puis on filtre par dates
            data = fetch_stock_data(symbol, period="max", include_technicals=False)
            if 'history_json' in data and data['history_json']['dates']:
                dates = pd.to_datetime(data['history_json']['dates'])
                closes = data['history_json']['closes']
                df = pd.DataFrame({'date': dates, symbol: closes}).set_index('date')
                # Filtrer selon la plage demandée
                df = df.loc[start_date:end_date]
                if not df.empty:
                    price_data[symbol] = df[symbol]
                else:
                    logger.warning(f"Pas de données dans la période pour {symbol}")
            else:
                logger.warning(f"Données historiques manquantes pour {symbol}")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de {symbol}: {e}")
    if not price_data:
        raise ValueError("Aucune donnée de prix récupérée")
    prices = pd.concat(price_data, axis=1)
    # Supprimer les lignes avec des NaN
    prices = prices.dropna()
    return prices

# --------------------------------------------------------------
# 3. Métadonnées (secteur, pays) via get_stock_overview
# --------------------------------------------------------------
def build_portfolio_metadata(tickers):
    """Construit un DataFrame avec secteur, pays, industrie pour chaque ticker."""
    metadata = {}
    for symbol in tickers:
        try:
            overview = get_stock_overview(symbol)
            metadata[symbol] = {
                "Secteur": overview.get("sector", "Inconnu"),
                "Pays": overview.get("country", "Inconnu"),
                "Industrie": overview.get("industry", "Inconnu"),
                "Nom": overview.get("name", "N/A"),
                "Beta": overview.get("beta", np.nan),
                "Cap_boursière": overview.get("market_cap", "N/A"),
            }
        except Exception as e:
            logger.error(f"Erreur pour {symbol}: {e}")
            metadata[symbol] = {"Secteur": "Inconnu", "Pays": "Inconnu", "Industrie": "Inconnu"}
    return pd.DataFrame(metadata).T

# --------------------------------------------------------------
# 4. Fonction d'optimisation (max Sharpe avec contraintes)
# --------------------------------------------------------------
def optimize_portfolio(mean_returns, cov_matrix, meta, tickers, include_bonds, bond_ticker=None):
    """
    Calcule les poids optimaux maximisant le ratio de Sharpe.
    Contraintes : pas de vente à découvert, somme=1,
    secteur <=30%, pays <=40%, obligations (si inclus) entre 20% et 40%.
    """
    n = len(tickers)
    # Contraintes sectorielles
    unique_sectors = meta["Secteur"].unique()
    sector_constraints = []
    for sec in unique_sectors:
        if sec == "Inconnu":
            continue
        mask = meta["Secteur"] == sec
        idx = [i for i, val in enumerate(mask) if val]
        if idx:
            sector_constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=idx: 0.30 - sum(w[idx])
            })
    # Contraintes géographiques
    unique_countries = meta["Pays"].unique()
    country_constraints = []
    for c in unique_countries:
        if c == "Inconnu":
            continue
        mask = meta["Pays"] == c
        idx = [i for i, val in enumerate(mask) if val]
        if idx:
            country_constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=idx: 0.40 - sum(w[idx])
            })
    # Contrainte obligations
    bond_constraints = []
    if include_bonds and bond_ticker in tickers:
        bond_idx = tickers.index(bond_ticker)
        bond_constraints.append({
            "type": "ineq",
            "fun": lambda w: w[bond_idx] - 0.20   # min 20%
        })
        bond_constraints.append({
            "type": "ineq",
            "fun": lambda w: 0.40 - w[bond_idx]   # max 40%
        })
    # Fonction objectif : -Sharpe
    def neg_sharpe(w, mean_returns, cov_matrix, risk_free_rate=0.02):
        port_ret = np.sum(mean_returns * w)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return - (port_ret - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    constraints += sector_constraints + country_constraints + bond_constraints
    bounds = tuple((0, 1) for _ in range(n))
    initial_guess = np.array([1/n] * n)

    result = minimize(neg_sharpe, initial_guess, args=(mean_returns, cov_matrix),
                      method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        logger.warning("L'optimisation n'a pas convergé : " + result.message)
    return result.x

# --------------------------------------------------------------
# 5. Backtest avec rééquilibrage dynamique
# --------------------------------------------------------------
def backtest_strategy(returns, optimal_weights, rebalance_freq='ME', transaction_cost=0.001):
    """
    Simule l'évolution du portefeuille avec rééquilibrage périodique.
    returns : DataFrame des rendements quotidiens
    optimal_weights : Series des poids cibles (index = tickers)
    rebalance_freq : 'ME', 'QE', 'YE'
    transaction_cost : frais en proportion de la valeur échangée
    Retourne une Series de la valeur du portefeuille (normalisée à 1 au début).
    """
    # Déterminer les dates de rééquilibrage
    if rebalance_freq == 'ME':
        rebalance_dates = returns.resample('ME').first().index
    elif rebalance_freq == 'QE':
        rebalance_dates = returns.resample('QE').first().index
    elif rebalance_freq == 'YE':
        rebalance_dates = returns.resample('YE').first().index
    else:
        raise ValueError("Fréquence non supportée. Utilisez 'ME', 'QE' ou 'YE'.")
    
    current_weights = optimal_weights.copy()
    current_value = 1.0
    portfolio_value = pd.Series(index=returns.index, dtype=float)
    
    for date in returns.index:
        if date in rebalance_dates:
            # Appliquer les coûts de transaction
            current_value *= (1 - transaction_cost)
            current_weights = optimal_weights.copy()
        # Rendement du jour
        daily_ret = returns.loc[date]
        port_ret = np.sum(current_weights * daily_ret)
        current_value *= (1 + port_ret)
        portfolio_value[date] = current_value
        # Mise à jour des poids après le rendement (drift)
        new_weights = current_weights * (1 + daily_ret) / (1 + port_ret)
        current_weights = new_weights / new_weights.sum()
    return portfolio_value

# --------------------------------------------------------------
# 6. Fonction principale
# --------------------------------------------------------------
def main():
    logger.info("=== Optimisation de portefeuille ===")
    logger.info(f"Horizon: {HORIZON_ANS} ans | Capital: {CAPITAL_INITIAL:,.0f} $ | Objectif: {OBJECTIF:,.0f} $")
    logger.info(f"Inclusion des obligations: {INCLURE_OBLIGATIONS}")
    
    # Récupération des prix
    logger.info("Récupération des prix historiques via fetch_stock_data...")
    prices = get_prices_from_fetch(TICKERS, START_DATE, END_DATE)
    logger.info(f"Données récupérées du {prices.index[0].date()} au {prices.index[-1].date()}")
    
    # Rendements
    returns = prices.pct_change().dropna()
    
    # Métadonnées
    logger.info("Récupération des métadonnées (secteur, pays)...")
    meta = build_portfolio_metadata(TICKERS)
    
    # ----------------------------------------------------------
    # Matrice de corrélation
    # ----------------------------------------------------------
    corr = returns.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
    plt.title("Matrice de corrélation des rendements")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    logger.info("Heatmap sauvegardée: correlation_matrix.png")
    plt.close()
    
    # ----------------------------------------------------------
    # Analyse PCA
    # ----------------------------------------------------------
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)
    pca = PCA(n_components=min(5, len(TICKERS)))
    pca.fit(returns_scaled)
    explained_variance = pca.explained_variance_ratio_
    logger.info("PCA - Variance expliquée :")
    for i, var in enumerate(explained_variance):
        logger.info(f"  PC{i+1}: {var:.2%}")
    logger.info(f"  Cumulée: {np.cumsum(explained_variance)[-1]:.2%}")
    loadings = pd.DataFrame(pca.components_.T,
                            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
                            index=TICKERS)
    logger.debug("Loadings:\n" + loadings.round(3).to_string())
    market_factor = loadings["PC1"].abs().sort_values(ascending=False)
    logger.info("Titres les plus corrélés au facteur marché (PC1) :\n" + market_factor.head(5).to_string())
    
    # ----------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    optimal_weights = optimize_portfolio(mean_returns, cov_matrix, meta, TICKERS,
                                         INCLURE_OBLIGATIONS, BOND_TICKER)
    optimal_series = pd.Series(optimal_weights, index=TICKERS)
    logger.info("Pondérations optimales (max Sharpe) :")
    for ticker, w in optimal_series[optimal_series > 0.01].sort_values(ascending=False).items():
        logger.info(f"  {ticker}: {w:.2%}")
    
    # ----------------------------------------------------------
    # Backtest
    # ----------------------------------------------------------
    logger.info(f"Lancement du backtest avec rééquilibrage {REBALANCE_FREQ}...")
    port_value = backtest_strategy(returns, optimal_series, REBALANCE_FREQ, TRANSACTION_COST)
    
    total_return = port_value.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(port_value)) - 1
    daily_vol = port_value.pct_change().std() * np.sqrt(252)
    sharpe = (annualized_return - 0.02) / daily_vol if daily_vol != 0 else np.nan
    max_drawdown = (port_value / port_value.cummax() - 1).min()
    
    logger.info("Résultats du backtest :")
    logger.info(f"  Rendement total: {total_return:.2%}")
    logger.info(f"  Rendement annualisé: {annualized_return:.2%}")
    logger.info(f"  Volatilité annualisée: {daily_vol:.2%}")
    logger.info(f"  Ratio de Sharpe: {sharpe:.2f}")
    logger.info(f"  Max drawdown: {max_drawdown:.2%}")
    
    # Graphique de la courbe de valeur
    plt.figure(figsize=(12,5))
    plt.plot(port_value.index, port_value, linewidth=1.5)
    plt.title(f"Backtest - Portefeuille optimisé (rééquilibrage {REBALANCE_FREQ})")
    plt.xlabel("Date")
    plt.ylabel("Valeur (capital initial = 1)")
    plt.grid(True)
    plt.savefig("backtest_portfolio.png")
    logger.info("Graphique sauvegardé: backtest_portfolio.png")
    plt.close()
    
    # ----------------------------------------------------------
    # Projection vers l'objectif 50 M$
    # ----------------------------------------------------------
    capital_final_simule = CAPITAL_INITIAL * port_value.iloc[-1]
    logger.info(f"Capital initial : {CAPITAL_INITIAL:,.0f} $")
    logger.info(f"Capital final simulé (historique) : {capital_final_simule:,.0f} $")
    if capital_final_simule >= OBJECTIF:
        logger.info("✅ Objectif de 50 M$ atteignable selon le backtest.")
    else:
        logger.warning("⚠️ Objectif non atteint. Envisagez d'augmenter le levier, d'allonger l'horizon, ou d'ajouter des actifs plus risqués.")
    
    logger.info("=== Fin de l'analyse ===")

# --------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------
if __name__ == "__main__":
    main()