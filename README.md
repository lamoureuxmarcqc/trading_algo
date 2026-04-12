# Trading Algo – Prédiction et Trading Automatisé d’Actions

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-green)](https://github.com/lamoureuxmarcqc/trading_algo)

**Trading Algo** est une suite logicielle modulaire pour l'analyse financière quantitative. Il combine l'extraction de données massives, le machine learning (LSTM, CatBoost, etc.) et des visualisations interactives pour aider à la décision de trading sur le S&P 500.

---

## 📌 Sommaire
1. [Fonctionnalités](#-fonctionnalités)
2. [Structure du projet](#-structure-du-projet)
3. [Installation](#-installation)
4. [Utilisation](#-utilisation)
5. [Configuration](#-configuration-avancée)
6. [Tests](#-tests)
7. [Web Dashboard](#web-dashboard)
8. [Contributions](#-contribution)
9. [Licence](#-licence)

---

## ✨ Fonctionnalités

- 📈 **Extraction de données** : Intégration native avec `yfinance`, support de cache local pour éviter les limitations d'API.
- 🧠 **Intelligence Artificielle** : 
  - Modèles Deep Learning : **LSTM** (TensorFlow) pour les séries temporelles.
  - Modèles Classiques : **CatBoost**, **SVR**, **MLP**, et **Régression Linéaire**.
- 🔍 **Auto-ML Lite** : Recherche automatique du meilleur modèle via `find_best_model.py`.
- 📊 **Dashboards** : Génération automatique de rapports interactifs en HTML (Plotly/Dash).
- 💼 **Gestion de Portefeuille** : Outils de screening et de suivi des performances.

---

## 🗂️ Structure du Projet

```text
trading_algo/           # Package principal
├── data/               # Extraction et gestion des données boursières
├── models/             # Architecture des modèles et scripts d'entraînement
├── preprocessing/      # Pipelines de nettoyage et scaling
├── screening/          # Algorithmes de sélection (S&P 500)
├── visualization/      # Moteur de rendu des dashboards
├── web_dashboard/      # Interface Dash pour visualisation et contrôle
├── __init__.py
└── __main__.py         # Point d'entrée CLI
pyproject.toml          # Dépendances et configuration build
.env                    # Secrets et configurations (à créer)

---

## 🚀 Installation

### 1. Cloner et préparer l'environnement
```bash
git clone [https://github.com/lamoureuxmarcqc/trading_algo.git](https://github.com/lamoureuxmarcqc/trading_algo.git)
cd trading_algo
python -m venv venv
# Activation (Linux/Mac) : source venv/bin/activate | (Windows) : venv\Scripts\activate
```

### 2. Installation du package
```bash
pip install --upgrade pip
pip install -e .
```

### 3. Configuration des API
Copiez le fichier d'exemple et remplissez vos clés :
```bash
cp .env.example .env  # Si disponible, sinon créez un fichier .env
```
*Le système fonctionne en mode dégradé (données simulées) si les clés sont manquantes.*

---

## 🏁 Utilisation

L'outil s'utilise principalement via la commande globale `trading-algo`.

| Commande | Description |
| :--- | :--- |
| `trading-algo` | Lance le menu interactif. |
| `trading-algo AAPL --advanced` | Analyse complète de l'action avec modèle avancé. |
| `trading-algo MSFT --mode train` | Entraîne le modèle sur Microsoft. |
| `trading-algo AAPL,TSLA --mode compare` | Compare les performances de deux titres. |
| `trading-algo --mode screen` | Recherche les meilleures opportunités du S&P 500. |

---

## 📊 Sorties et Artefacts

Les résultats sont organisés comme suit :
- **`models_saved/`** : Contient les fichiers `.keras`, les scalers `.pkl` et les rapports JSON de performances.
- **`dashboards/`** : Rapports visuels interactifs au format `.html`.
- **`logs/`** : Historique détaillé des opérations et erreurs.

---

## 🧪 Tests

Pour garantir la stabilité, lancez la suite de tests :
```bash
# Lancer tous les tests
python -m unittest discover tests

# Test spécifique des imports
python trading_algo/test_imports.py
```

---

## Web Dashboard

Le projet inclut une interface Dash pour la visualisation et le contrôle du terminal de trading.

### Démarrage local
1. Créez et activez un environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.\.venv\Scripts\activate   # Windows PowerShell
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Variables d'environnement utiles :

- `DEBUG` : `True` ou `False` (par défaut `True` si non défini)
- `WEB_HOST` : adresse d'écoute (par défaut `127.0.0.1`)
- `WEB_PORT` : port du serveur (par défaut `8050`)
- `REDIS_URL` / `REDIS_URI` : URL Redis (optionnel) pour le cache

4. Lancer l'application :
```bash
python start.py
# ou
python -m trading_algo.web_dashboard.app
```

Le dashboard sera accessible à `http://{WEB_HOST}:{WEB_PORT}`.

### Points d'attention / Dépannage
- Si les onglets n'apparaissent pas : vérifier les logs du serveur pour des erreurs d'import au démarrage (modules lourds comme TensorFlow peuvent planter l'import des callbacks). Utiliser le journal pour identifier le module fautif.
- Messages d'avertissement React (ex: `defaultProps`) provenant de bibliothèques (dash-bootstrap-components) sont généralement sans gravité.
- Vérifier `assets/` et `assets/custom.css` si le style masque des composants (ex. `display: none` sur `.nav` ou `.tab`).
- Pour debug côté client : ouvrir DevTools → Network → filtrer `/_dash-layout` et `/_dash-update-component` pour voir les réponses 500 et le message d'erreur Python.

### Configuration du scheduler
- Intervalle de rafraîchissement configurable via `MARKET_REFRESH_INTERVAL_MIN` dans `trading_algo.settings`.
- En production, préférez un worker séparé (Celery/Redis) pour éviter le blocage du process web.

---

## 🤝 Contribution

1. Forkez le dépôt.
2. Créez votre branche : `git checkout -b feature/nom-de-la-feature`.
3. Assurez-vous que votre code respecte la norme **PEP8**.
4. Ouvrez une Pull Request.

---

## 📄 Licence

Distribué sous licence **MIT**. Voir `LICENSE` pour plus d'informations.

---

**Développé avec ❤️ par Marc Lamoureux** 🔗 [GitHub Profile](https://github.com/lamoureuxmarcqc)
