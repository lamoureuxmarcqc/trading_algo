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
