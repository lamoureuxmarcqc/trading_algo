Le README a été mis à jour pour refléter la nouvelle structure du projet et les changements d'installation.

```markdown
# Trading Algo – Prédiction et Trading Automatisé d’Actions

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-green)](https://github.com/lamoureuxmarcqc/trading_algo)

**Trading Algo** est une suite complète d’outils pour l’extraction de données financières, l’entraînement de modèles prédictifs (LSTM, SVR, MLP, CatBoost…) et la visualisation interactive des performances.  
Conçu pour analyser les actions du S&P 500, ce projet permet de backtester des stratégies, de comparer plusieurs modèles de machine learning et de générer des tableaux de bord automatiques.

---

## ✨ Fonctionnalités

- 📈 **Extraction de données** via Yahoo Finance (`yfinance`) et gestion de cache.
- 🧠 **Entraînement multi-modèles** :
  - Réseaux de neurones LSTM (Keras / TensorFlow)
  - Régresseurs : Support Vector, MLP, CatBoost, Régression Linéaire
- 🔍 **Recherche automatique du meilleur modèle** (`find_best_model.py`)
- 📊 **Tableaux de bord interactifs** générés avec Plotly / Dash
- 💾 **Sauvegarde** des modèles, scalers et métriques (JSON / images)
- ⚙️ **Configuration centralisée** via variables d’environnement (fichier `.env`)
- 🧪 **Tests d’intégration** pour valider les importations et le pipeline

---

## 🗂️ Structure du projet

```
projet_trading/
├── trading_algo/               # Package principal
│   ├── data/                   # Extraction et prétraitement
│   │   └── data_extraction.py
│   ├── models/                 # Modèles ML et entraînement
│   │   ├── stockmodeltrain.py
│   │   ├── find_best_model.py
│   │   └── stockpredictor.py
│   ├── preprocessing/          # Préparation des données
│   ├── screening/              # Screening des actions S&P500
│   │   └── actions_sp500.py
│   ├── visualization/          # Génération de graphiques et dashboards
│   │   └── dashboard.py
│   ├── __init__.py
│   └── __main__.py             # Point d'entrée principal
├── pyproject.toml              # Configuration moderne du projet (dépendances)
├── .env                        # Variables d'environnement (clés API)
├── .gitignore                  # Fichiers ignorés par Git
├── README.md                   # Ce fichier
└── ...
```

> **Note** : Les dossiers `checkpoints/`, `models_saved/`, `dashboards/`, `cache/` et les fichiers `.pyc` sont exclus du versionnement (via `.gitignore`).

---

## 🚀 Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/lamoureuxmarcqc/trading_algo.git
cd trading_algo
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3. Installer le package en mode développement
```bash
pip install -e .
```
Cette commande installe toutes les dépendances listées dans `pyproject.toml` et rend la commande `trading-algo` disponible dans l’environnement virtuel.

### 4. Configurer les variables d’environnement
Créez un fichier `.env` à la racine du projet (à partir de `.env.example` si fourni) et renseignez vos clés API :

```
FMP_API_KEY=votre_cle_fmp
POLYGON_API_KEY=votre_cle_polygon
TWITTER_X_BEARER=votre_bearer_token_twitter
NY_TIMES_API_KEY=votre_cle_nytimes
```

Si certaines clés ne sont pas disponibles, le programme utilisera des données simulées.

---

## 🏁 Utilisation

### Lancer l’analyse interactive d’une action
```bash
trading-algo
```
Sans argument, un menu interactif vous propose de choisir une action parmi les plus populaires ou d’entrer un symbole personnalisé.

### Analyser une action spécifique
```bash
trading-algo AAPL
```
L’analyse de base est lancée avec le modèle `StockModelTrain`.

### Mode avancé (avec `StockPredictor`)
```bash
trading-algo AAPL --advanced
```
Utilise le module avancé pour des prédictions plus détaillées.

### Autres options
```bash
trading-algo --help
```
Affiche toutes les options disponibles : `--period`, `--mode`, `--dashboard`, etc.

### Exemples
- Comparer plusieurs actions :  
  ```bash
  trading-algo AAPL,MSFT,GOOGL --mode compare
  ```
- Lancer le screening du S&P 500 :  
  ```bash
  trading-algo --mode screen
  ```
- Entraîner un modèle sans analyse :  
  ```bash
  trading-algo AAPL --mode train
  ```

---

## ⚙️ Configuration avancée

Le fichier `.env` supporte également des paramètres généraux :

```ini
DEBUG=True
LOG_LEVEL=INFO
CACHE_DIR=cache/
DATA_DIR=data/
MODELS_DIR=models_saved/
YF_CACHE=True
YF_CACHE_EXPIRE=3600
TRAIN_TEST_SPLIT=0.8
RANDOM_SEED=42
```

---

## 📊 Résultats et Métriques

Après chaque entraînement, les artefacts suivants sont sauvegardés dans `models_saved/<SYMBOLE>/` :
- Modèle au format `.keras`
- Scalers (feature/target) au format `.pkl`
- Graphiques d’entraînement (`.png`)
- Fichier JSON contenant les métriques

Les dashboards générés (mode `--dashboard` ou `--advanced`) sont placés dans `dashboards/` au format `.html` et `.png`.

---

## 🧪 Tests

Exécutez la suite de tests pour valider l’intégrité du projet :
```bash
python -m unittest discover tests
```
Ou lancez des scripts de test individuels :
```bash
python test_imports.py
python test_dashboard.py
```

---

## 🤝 Contribution

Les contributions sont les bienvenues !  
1. Forkez le projet  
2. Créez une branche (`git checkout -b feature/amazing-idea`)  
3. Committez vos changements (`git commit -m 'Add some amazing idea'`)  
4. Pushez (`git push origin feature/amazing-idea`)  
5. Ouvrez une Pull Request  

Merci de respecter les conventions PEP8 et d’ajouter des tests pour toute nouvelle fonctionnalité.

---

## 📄 Licence

Ce projet est sous licence **MIT**. Vous êtes libre de l’utiliser, le modifier et le distribuer, sous réserve de conserver la notice de droit d’auteur.  
Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- [yfinance](https://github.com/ranaroussi/yfinance) pour l’accès aux données boursières
- [TensorFlow / Keras](https://www.tensorflow.org/) pour les modèles LSTM
- [scikit-learn](https://scikit-learn.org/) pour les régresseurs classiques
- [Plotly Dash](https://plotly.com/dash/) pour la visualisation interactive
- [CatBoost](https://catboost.ai/) pour le gradient boosting

---

**Développé avec ❤️ par Marc Lamoureux**  
🔗 [https://github.com/lamoureuxmarcqc](https://github.com/lamoureuxmarcqc)
```

**Principales modifications apportées :**
- Structure du projet mise à jour : `trading_algo/` à la racine, plus de dossier `src/`.
- Installation : utilisation de `pip install -e .` (via `pyproject.toml`), création du fichier `.env`.
- Commande `trading-algo` expliquée avec des exemples concrets.
- Options de configuration dans `.env` listées.
- Mise à jour des chemins de sauvegarde (`models_saved/`, `dashboards/`).

Ce README correspond désormais à l’état actuel du projet.