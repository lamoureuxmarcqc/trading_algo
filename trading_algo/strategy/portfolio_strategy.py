import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from trading_algo.data.data_extraction import get_stock_overview
from trading_algo.logging_config import init_logging

# --------------------------------------------------------------
# 0. Initialisation du logging
# --------------------------------------------------------------
init_logging(level=os.getenv("LOG_LEVEL", "INFO"), logfile=os.getenv("LOG_FILE", None))
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# 1. Paramètres généraux et horizon
# --------------------------------------------------------------
HORIZON_ANS = 4   # exemple : 4 ans (<5 ans => ajout obligations)
CAPITAL_INITIAL = 5_000_000
OBJECTIF = 50_000_000

if HORIZON_ANS < 5:
    INCLURE_OBLIGATIONS = True
    logger.warning("Horizon < 5 ans : ajout d'obligations pour réduire le risque.")
else:
    INCLURE_OBLIGATIONS = False

# Liste des actions (exemple)
TICKERS_ACTIONS = ["CNR.TO", "BN.TO", "CNQ.TO", "XIU.TO", "CSU.TO", "RY.TO", "SHOP.TO"]
# Ajouter un ETF obligataire si nécessaire
BOND_TICKER = "AGG"   # iShares Core US Aggregate Bond ETF (ou XBB.TO pour Canada)

if INCLURE_OBLIGATIONS:
    TICKERS = TICKERS_ACTIONS + [BOND_TICKER]
else:
    TICKERS = TICKERS_ACTIONS

# Période d'analyse (pour les rendements historiques)
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"

# --------------------------------------------------------------
# 2. Récupération des données et métadonnées
# --------------------------------------------------------------
def build_portfolio_metadata(tickers):
    """Utilise get_stock_overview pour extraire secteur, pays, etc."""
    metadata = {}
    for symbol in tickers:
        try:
            overview = get_stock_overview(symbol)
            metadata[symbol] = {
                "Secteur": overview["sector"],
                "Pays": overview["country"],
                "Industrie": overview["industry"],
                "Nom": overview["name"],
                "Beta": overview["beta"],
                "Cap_boursière": overview["market_cap"],
            }
        except Exception as e:
            logger.error(f"Erreur pour {symbol}: {e}")
            metadata[symbol] = {"Secteur": "Inconnu", "Pays": "Inconnu", "Industrie": "Inconnu"}
    return pd.DataFrame(metadata).T

logger.info("Téléchargement des prix et calcul des rendements...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Adj Close"]
returns = data.pct_change().dropna()
meta = build_portfolio_metadata(TICKERS)

# --------------------------------------------------------------
# 3. Matrice de corrélation + PCA (facteurs cachés)
# --------------------------------------------------------------
corr = returns.corr()

# Visualisation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation - Actions + Obligations (optionnel)")
plt.tight_layout()
plt.savefig("correlation_matrix_with_bonds.png")
logger.info("Heatmap de corrélation sauvegardée sous 'correlation_matrix_with_bonds.png'")
plt.close()

# PCA sur les rendements normalisés
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)
pca = PCA(n_components=min(5, len(TICKERS)))
pca.fit(returns_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

logger.info("Analyse PCA - Facteurs cachés de corrélation :")
for i, var in enumerate(explained_variance):
    logger.info(f"Composante {i+1}: {var:.2%} de variance expliquée")
logger.info(f"Cumulé à {len(explained_variance)} composantes: {cumulative_variance[-1]:.2%}")

loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=TICKERS)
logger.debug("Loadings des premières composantes (facteurs cachés) :\n" + loadings.round(3).to_string())

market_factor_stocks = loadings["PC1"].abs().sort_values(ascending=False)
logger.info("Titres les plus corrélés au facteur principal (marché) :\n" + market_factor_stocks.head(5).to_string())

# --------------------------------------------------------------
# 4. Calcul des rendements/volatilités annualisés (avec obligations)
# --------------------------------------------------------------
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

if INCLURE_OBLIGATIONS:
    bond_corr_with_stocks = corr[BOND_TICKER].drop(BOND_TICKER).mean()
    logger.info(f"Corrélation moyenne {BOND_TICKER} avec actions: {bond_corr_with_stocks:.2f}")

# --------------------------------------------------------------
# 5. Optimisation avec contraintes (secteurs, pays, obligations)
# --------------------------------------------------------------
unique_sectors = meta["Secteur"].unique()
sector_constraints = []
for sec in unique_sectors:
    if sec == "Inconnu":
        continue
    mask = meta["Secteur"] == sec
    idx = [i for i, val in enumerate(mask) if val]
    if len(idx) > 0:
        sector_constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=idx: 0.30 - sum(w[idx])
        })

unique_countries = meta["Pays"].unique()
country_constraints = []
for c in unique_countries:
    if c == "Inconnu":
        continue
    mask = meta["Pays"] == c
    idx = [i for i, val in enumerate(mask) if val]
    if len(idx) > 0:
        country_constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=idx: 0.40 - sum(w[idx])
        })

bond_constraints = []
if INCLURE_OBLIGATIONS:
    bond_idx = TICKERS.index(BOND_TICKER)
    bond_constraints.append({
        "type": "ineq",
        "fun": lambda w: w[bond_idx] - 0.20   # min 20%
    })
    bond_constraints.append({
        "type": "ineq",
        "fun": lambda w: 0.40 - w[bond_idx]   # max 40%
    })

def neg_sharpe(w, mean_returns, cov_matrix, risk_free_rate=0.02):
    port_ret = np.sum(mean_returns * w)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = (port_ret - risk_free_rate) / port_vol
    return -sharpe

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
constraints += sector_constraints
constraints += country_constraints
constraints += bond_constraints

bounds = tuple((0, 1) for _ in TICKERS)

initial_guess = np.array([1/len(TICKERS)] * len(TICKERS))
result = minimize(neg_sharpe, initial_guess, args=(mean_returns, cov_matrix),
                  method="SLSQP", bounds=bounds, constraints=constraints)

optimal_weights = result.x
optimal_series = pd.Series(optimal_weights, index=TICKERS)

logger.info("Pondérations optimales (max Sharpe) :")
for ticker, w in optimal_series[optimal_series > 0.01].sort_values(ascending=False).items():
    logger.info(f"{ticker}: {w:.2%}")

# --------------------------------------------------------------
# 6. Backtest avec rééquilibrage dynamique
# --------------------------------------------------------------
def backtest_strategy(returns, optimal_weights, rebalance_freq='M', transaction_cost=0.001):
    weights = pd.DataFrame(index=returns.index, columns=returns.columns)
    portfolio_value = pd.Series(index=returns.index, dtype=float)

    if rebalance_freq == 'M':
        rebalance_dates = returns.resample('M').first().index
    elif rebalance_freq == 'Q':
        rebalance_dates = returns.resample('Q').first().index
    elif rebalance_freq == 'Y':
        rebalance_dates = returns.resample('Y').first().index
    else:
        raise ValueError("fréquence non supportée")

    current_weights = optimal_weights.copy()
    current_value = 1.0

    for date in returns.index:
        if date in rebalance_dates:
            current_value *= (1 - transaction_cost)
            current_weights = optimal_weights.copy()

        daily_ret = returns.loc[date]
        port_ret = np.sum(current_weights * daily_ret)
        current_value *= (1 + port_ret)
        portfolio_value[date] = current_value

        new_weights = current_weights * (1 + daily_ret) / (1 + port_ret)
        current_weights = new_weights / new_weights.sum()

    return portfolio_value

logger.info("Lancement du backtest avec rééquilibrage mensuel...")
port_value = backtest_strategy(returns, optimal_series, rebalance_freq='M', transaction_cost=0.001)

total_return = port_value.iloc[-1] - 1
annualized_return = (1 + total_return) ** (252 / len(port_value)) - 1
daily_vol = port_value.pct_change().std() * np.sqrt(252)
sharpe = (annualized_return - 0.02) / daily_vol
max_drawdown = (port_value / port_value.cummax() - 1).min()

logger.info("Résultats du backtest (rééquilibrage mensuel) :")
logger.info(f"Rendement total : {total_return:.2%}")
logger.info(f"Rendement annualisé : {annualized_return:.2%}")
logger.info(f"Volatilité annualisée : {daily_vol:.2%}")
logger.info(f"Ratio de Sharpe : {sharpe:.2f}")
logger.info(f"Max drawdown : {max_drawdown:.2%}")

plt.figure(figsize=(12,5))
plt.plot(port_value.index, port_value, label="Portefeuille optimal (rééquilibré)")
plt.title("Backtest du portefeuille avec rééquilibrage dynamique")
plt.xlabel("Date")
plt.ylabel("Valeur (capital initial = 1)")
plt.legend()
plt.grid(True)
plt.savefig("backtest_portfolio.png")
logger.info("Graphique du backtest sauvegardé sous 'backtest_portfolio.png'")
plt.close()

# --------------------------------------------------------------
# 7. Projection vers 50 M$ avec le backtest
# --------------------------------------------------------------
capital_final_simule = CAPITAL_INITIAL * port_value.iloc[-1]
logger.info(f"Capital initial : {CAPITAL_INITIAL:,.0f} $")
logger.info(f"Capital final simulé (historique) : {capital_final_simule:,.0f} $")
if capital_final_simule >= OBJECTIF:
    logger.info("✅ Objectif de 50 M$ atteignable selon le backtest.")
else:
    logger.warning("⚠️ Objectif non atteint. Envisagez d'augmenter le levier ou d'ajouter des actifs plus risqués.")