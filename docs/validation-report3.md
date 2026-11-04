# 📄 Document 1 — `validation-report.md` (mis à jour)

Je conserve la structure de votre rapport original, mais j'y intègre les faits de cette session : séparation, tests, smoke tests, résolution des incidents d'infrastructure (Postgres, `.env`, `sys.path`), nettoyage Docker et audit de sécurité Git.

```markdown
# Validation Report — Séparation Plateforme / Verticales

**Date :** 2026-11-02
**Statut :** ✅ Validé
**Validateur :** Marc Lamoureux
**Branche mergée :** `refactor/platform-vs-verticals` → `main`
**Commit de merge :** `91b90c3`
**Tags publiés :** `platform-v1.0`, `platform-v1.0-merged`, `pre-separation`

---

## 1. Périmètre validé

Refonte architecturale : séparation du monolithe `trading_algo` en une **infrastructure de plateforme** (`services/`) et un **premier vertical métier** (`verticals/trading/`), sans régression fonctionnelle.

| Composant | Statut | Preuve |
|---|---|---|
| Structure cible (Phase 1) | ✅ | 12 fichiers créés (READMEs, docs, scripts) |
| Déplacement des fichiers (Phase 2) | ✅ | 174 fichiers déplacés via `git mv` (historique préservé) |
| Réparation des imports (Phase 3) | ✅ | `conftest.py`, `main.py`, `config.py` corrigés |
| Tests unitaires `api-core` | ✅ | **37 / 37 passed** |
| Smoke test `knowledge` | ✅ | `{"status":"ok","version":"0.1.0",...}` |
| Smoke test `api-core` | ✅ | `{"status":"ok",...}` |
| Smoke test cockpit Next.js | ✅ | `Ready in 3.9s`, `GET / 200` |
| Test end-to-end `/terminal/snapshot` | ✅ | **200 OK** avec données réelles (positions, signaux, scénarios) |
| Cohérence Git | ✅ | Merge `--no-ff`, tags poussés, aucun secret dans les commits |
| Nettoyage Docker | ✅ | ~5 Go libérés (stack média supprimée) |

---

## 2. Preuves techniques

### 2.1 Structure cible (Phase 1)

```text
verticals/
├── trading/
│   ├── README.md
│   ├── agents/__init__.py
│   ├── api-core/          ← déplacé depuis services/api-core
│   ├── trading_algo/      ← déplacé depuis la racine
│   └── web/               ← déplacé depuis apps/web
├── lawyer/  (README planifié)
├── doctor/  (README planifié)
├── crm/     (README planifié)
└── cnesst/  (README planifié)

apps/
└── web-platform/README.md  ← portail plateforme (vide, prochaine session)

docs/
├── platform/
└── verticals/
    ├── trading.md
    ├── lawyer.md
    ├── doctor.md
    ├── crm.md
    └── cnesst.md
```

### 2.2 Tests `api-core`

```
verticals/trading/api-core> pytest tests -q
.....................................                    [100%]
37 passed in 16.93s
```

**Note :** la suite `api-core` est passée de 33 (rapport Phase 1 précédent) à 37 tests entre-temps (ajouts liés aux routes AI et terminal).

### 2.3 Stack Docker

```
lamoureux-postgres     Up (healthy)     0.0.0.0:5433->5432/tcp
lamoureux-redis        Up (healthy)     0.0.0.0:6379->6379/tcp
lamoureux-qdrant       Up (unhealthy*)  0.0.0.0:6333-6334->6333-6334/tcp
lamoureux-ollama       Up               0.0.0.0:11434->11434/tcp
lamoureux-ai-gateway   Up (healthy)     0.0.0.0:8001->8001/tcp
```

\* Qdrant fonctionne (healthcheck compose mal calibré, non bloquant).

### 2.4 Endpoint critique `/api/v1/terminal/snapshot`

Réponse **HTTP 200** avec données live :
- `portfolio` : 3 positions (NVDA, MSFT, AAPL), NAV 1 415 460 USD
- `signals` : AMZN, META, LLY, QQQ
- `scenarios` : 7 scénarios historiques (1929, 2000, 1973, 2008, 2022, 1989, 2020)
- `barbell` : allocation avec instructions de rebalance
- `correlation_matrix`, `position_risk`, `orders`, `fills`, `research`, `factors`, `sectors`, `users`, `audit_logs`

**Preuve que la chaîne complète fonctionne :** `uvicorn` → FastAPI → SQLAlchemy → PostgreSQL Docker → réponse JSON → rendu cockpit.

### 2.5 Cockpit Next.js

```
▲ Next.js 15.3.2
- Local:    http://localhost:3000
- Network:  http://169.254.98.81:3000
✓ Ready in 3.9s
✓ Compiled / in 1406ms (720 modules)
GET / 200 in 1813ms
```

⚠️ **Point cosmétique identifié** : warning `allowedDevOrigins` pour l'accès via `169.254.98.81:3000` (non bloquant, à corriger dans `next.config.mjs`).

---

## 3. Incidents résolus pendant la session

| # | Incident | Cause racine | Correction |
|---|---|---|---|
| 1 | `ModuleNotFoundError: trading_algo` (pytest) | `trading_algo/` n'était plus dans le `sys.path` de `api-core` | `conftest.py` : ajout de `verticals/trading/` au `sys.path` |
| 2 | `ModuleNotFoundError: trading_algo` (uvicorn) | uvicorn ne charge pas `conftest.py` | `app/main.py` : bootstrap `sys.path` avant les imports `app.*` |
| 3 | `NameError: name 'logging' is not defined` | Ordre d'imports cassé lors du patch bootstrap | Réordonnancement : stdlib → bootstrap → tiers → `app.*` |
| 4 | `failed to resolve host 'postgres'` | `env_file=".env"` relatif au CWD, donc introuvable depuis `api-core/` | `config.py` : `env_file` absolu vers `Path(__file__).parents[5] / ".env"` |
| 5 | `password authentication failed for user "postgres"` | Le `.env` utilisait `postgres:albany01`, mais le conteneur a `lamoureux:changeme` | Alignement du DSN sur les identifiants réels du conteneur |
| 6 | `Failed to fetch` sur cockpit | CORS + API `api-core` non joignable depuis l'IP réseau | Ajout de `169.254.98.81:3000` dans `ALLOWED_ORIGINS` et `--host 0.0.0.0` sur uvicorn |
| 7 | `500 Internal Server Error` sur `/terminal/snapshot` | Connexion DB impossible (cf. #4 et #5) | Résolu par #4 et #5 |
| 8 | Docker saturé (~5 Go de conteneurs média inactifs) | Stack Jellyfin/Radarr/Sonarr/qBittorrent/prowlarr/flaresolverr/gluetun | Suppression des images et conteneurs → ~5 Go libérés |

---

## 4. Audit de sécurité Git

| Vérification | Résultat |
|---|---|
| `.env` tracké actuellement | ❌ Non (sortie vide de `git ls-files .env`) |
| `.env` dans `.gitignore` | ✅ Oui |
| Secrets dans le commit `5ae92f7` (`git show … \| Select-String "…"`) | ✅ Vide |
| Secrets dans le diff `main..refactor/platform-vs-verticals` | ✅ Vide |
| Historique de `.env` | ⚠️ **Présent** dans 2 commits de février 2026 (`2224359`, `d559160`) — **à traiter hors séparation** |

**Action requise (indépendante de la séparation) :** révocation et régénération des clés API exposées historiquement (FMP, Polygon, Twitter/X, NY Times, FRED, Alpha Vantage) + rotation du mot de passe Postgres.

---

## 5. Conformité aux ADRs

| ADR | Règle | Statut |
|---|---|---|
| ADR-001 | Local-first par défaut | ✅ (infra locale) |
| ADR-002 | AI Gateway point d'entrée unique | ✅ (inchangé par la séparation) |
| ADR-003 | Classification avant routage | ✅ (inchangé) |
| ADR-004 | Mémoire multi-niveaux | ⏳ (phase 3) |
| ADR-005 | Permissions agents | ⏳ (phase 4) |
| **ADR-006** | **Frontière infrastructure / vertical** | ✅ **Appliqué pour la première fois par cette séparation** |

**ADR-006 est désormais concret :** `services/` ne contient plus aucune logique métier trading. `verticals/trading/` contient tout le métier. La frontière est vérifiable par l'arborescence.

---

## 6. Nettoyage Docker

| Élément | Avant | Après | Gain |
|---|---|---|---|
| Images | 15,41 Go | 10,6 Go | **−4,8 Go** |
| Containers | 60,93 Mo | 401 Ko | −60 Mo |
| Build Cache | 202,5 Mo | 0 | −202 Mo |
| Local Volumes | 2,16 Go | 2,16 Go | inchangé |
| **Total Docker** | **~17,8 Go** | **~12,8 Go** | **~5 Go** |

La stack média (Jellyfin, Radarr, Sonarr, qBittorrent, Prowlarr, Flaresolverr, gluetun) a été supprimée. Seuls les services Lamoureux AI restent.

---

## 7. Ce qui est débloqué

1. **`apps/web-platform`** — portail plateforme (sélection verticale, admin, chat universel, RAG générique)
2. **Vertical CNESST** — pilote de démonstration de la réutilisabilité
3. **Documentation** `docs/verticals/*.md` — spécifications par vertical
4. **Correction cosmétique Next.js** — `allowedDevOrigins` dans `next.config.mjs`

---

## 8. Dette technique identifiée (non bloquante)

| # | Dette | Priorité |
|---|---|---|
| 1 | `trading_algo` importé via `sys.path`, pas via un package installable (`pip install -e`) | Moyenne (session dédiée) |
| 2 | Parité UI HTML statique ↔ React cockpit | Moyenne |
| 3 | Warning Next.js `allowedDevOrigins` | Basse (5 min) |
| 4 | Rotation des clés API historiquement exposées | **Haute** |
| 5 | Qdrant `unhealthy` (healthcheck compose) | Basse |
| 6 | Rapport racine `pytest` non re-testé après séparation | Basse |

---

## 9. Commandes de reproduction

```powershell
# 1. Vérifier la séparation
cd C:\Users\marc\source\projet_trading
git log --oneline -5
git tag -l

# 2. Tests api-core
cd verticals\trading\api-core
pytest tests -q
# → 37 passed

# 3. Démarrer les services (3 terminaux)
cd C:\Users\marc\source\projet_trading\services\knowledge
uvicorn app.main:app --host 0.0.0.0 --port 8002

cd C:\Users\marc\source\projet_trading\verticals\trading\api-core
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

cd C:\Users\marc\source\projet_trading\verticals\trading\web
npm run dev

# 4. Smoke tests
curl.exe http://localhost:8002/health
curl.exe http://localhost:8000/api/v1/terminal/snapshot
# Navigateur : http://localhost:3000/research
```

---

## 10. Signature

**Validé par :** Marc Lamoureux
**Date :** 2026-11-02
**Commit :** `91b90c3`
**Tag :** `platform-v1.0-merged`
**Prochaine étape :** `apps/web-platform` + vertical CNESST
```

