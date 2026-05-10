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
- En cas d'echec, le cockpit bascule sur un snapshot fallback pour rester navigable.
- Le badge `Mode` dans l'UI indique si la page est en `live api` ou en `fallback snapshot`.
