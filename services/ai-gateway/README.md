# AI Gateway

Point d'entrée unique pour tous les appels LLM de la LAMOUREUX AI PLATFORM.

## Rôle

- Abstraction modèle (Ollama, vLLM, providers distants)
- Enforcement de la classification avant tout appel réseau
- Routage intelligent (coût, latence, disponibilité)
- Fallback automatique
- Métriques (tokens, coût, latence par tenant)

## Contrat

```http
POST /v1/inference
```

## Règles

| Classification | Cibles autorisées |
|---|---|
| PUBLIC | LOCAL, LOCAL_GPU, REMOTE |
| INTERNAL | LOCAL, LOCAL_GPU, REMOTE |
| CONFIDENTIAL | LOCAL, LOCAL_GPU |
| HIGHLY_CONFIDENTIAL | LOCAL, LOCAL_GPU |
| RESTRICTED | LOCAL uniquement |

La règle est appliquée **avant** tout appel réseau.

## Lancer

```bash
uvicorn app.main:app --reload --port 8001
```
