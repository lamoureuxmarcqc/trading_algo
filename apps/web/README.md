# Web Cockpit

Frontend Next.js du cockpit institutionnel.

## Variables utiles

- `API_BASE_URL`: URL serveur pour les server components Next.js
- `NEXT_PUBLIC_API_BASE_URL`: URL publique si des composants client sont ajoutes ensuite

Exemple:

```text
API_BASE_URL=http://127.0.0.1:8000/api/v1
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

## Pages deja branchees a `api-core`

- `/dashboard`
- `/portfolio`
- `/risk`
- `/research`
- `/trading`
- `/admin`

## Console trading-algo

- `/research` embarque la console **Trading Algo / Quant command console**.
- Elle consomme `POST /api/v1/terminal/trading-algo` cote client via `NEXT_PUBLIC_API_BASE_URL`.
- Commandes supportees aujourd'hui:
  - `analyze`
  - `compare`
  - `screen`
- Resultats affiches: prix, tendance, recommandation, daily return, total return, volatilite 20 jours, Sharpe, VaR 95 et RSI.

## Strategie de resilience

- Chaque page tente d'abord l'API FastAPI.
- Le cockpit privilegie `GET /api/v1/terminal/snapshot` comme source agregee pour reduire le fan-out reseau.
- En cas d'echec, le cockpit bascule sur un snapshot fallback pour rester navigable.
- Le badge `Mode` dans l'UI indique si la page est en `live api` ou en `fallback snapshot`.

## Risque enrichi

- `/risk` affiche maintenant:
  - les stress tests historiques enrichis
  - le contexte macro par scenario
  - les impacts sur le portefeuille
  - un tableau de correlation portefeuille

## Construction de portefeuille

- `/portfolio` affiche maintenant la strategie `barbell` calculee par `api-core`
- la vue montre:
  - la repartition defensive / opportuniste / cash
  - la liste des allocations cibles
  - les deltas de rebalance par ligne
  - la justification de posture selon le regime courant

## Roadmap AI, backtesting et guardrails

La roadmap produit/technique est documentee dans [../../docs/ai-trading-model-integration-roadmap.md](/C:/Users/marc/source/projet_trading/docs/ai-trading-model-integration-roadmap.md).

Surfaces UI ciblees:

- `/research`
  - diagnostics modele par ticker
  - confidence score, model version, hit rate, MAE/RMSE
  - drawer **AI Explanation** avec top features SHAP/LIME ou fallback heuristique
- `/dashboard`
  - prediction bands et probabilistic cones sur les opportunites principales
  - warnings de modele stale ou drift detecte
- `/risk`
  - vue staleness/drift globale
  - exposition des positions aux signaux IA
- `/trading`
  - toggles de staging papier
  - seuil de buy probability
  - stop-loss/take-profit guardrails
  - file `Pending Orders`
- `/backtesting` ou onglet dedie
  - lookback configurable
  - PnL, max drawdown, win/loss ratio, trade count
  - comparaison benchmark

Regle produit: toute automatisation d'ordre commence en **paper trading only**. Aucun ordre live ne doit etre envoye depuis le cockpit sans confirmation explicite et permission separee.
