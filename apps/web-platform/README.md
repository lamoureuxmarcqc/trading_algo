# apps/web-platform — Portail de la LAMOUREUX AI PLATFORM

> **Rôle :** UI transversale de la plateforme. Permet de sélectionner un vertical,
> d'administrer les services génériques et d'utiliser le RAG sans logique métier.
>
> **Ce n'est pas** le cockpit trading. Le cockpit trading vit dans
> `verticals/trading/web/` et reste indépendant.

**Statut :** squelette à implémenter (Phase 13)
**Stack :** Next.js 15 (app router), TypeScript strict, Tailwind
**Dépend de :** `services/ai-gateway`, `services/knowledge`, `services/orchestrator`, `services/model-registry`, `services/self-improvement`

---

## 1. Principe directeur

Le portail est **agnostique du métier**. Il ne contient aucun mot de trading,
de droit, de santé ou d'administration. Il expose :

1. la liste des verticaux disponibles et leur statut ;
2. un chat/RAG générique (appel `services/knowledge`) ;
3. la gestion des agents (`services/orchestrator`) ;
4. le registre de modèles (`services/model-registry`) ;
5. les propositions d'auto-amélioration (`services/self-improvement`) ;
6. l'administration plateforme (tenants, classifications, santé des services).

Le portail **appelle les services via HTTP uniquement**. Aucun import Python,
aucun accès direct à PostgreSQL.

---

## 2. Pages

| Route | Rôle | Service appelé |
|---|---|---|
| `/` | Landing + grille de sélection des verticaux | statique + `/health` de chaque vertical |
| `/search` | RAG générique (question → réponse + citations) | `POST /api/v1/knowledge/query` |
| `/ingest` | Ingestion de documents (admin) | `POST /api/v1/knowledge/ingest` |
| `/agents` | Liste, config, invocation d'agents | `GET/POST /api/v1/agents/*` |
| `/models` | Registre de modèles, promotion, rollback | `GET/POST /api/v1/models/*` |
| `/improvement` | File de propositions, simulate, approve, reject, deploy | `/api/v1/improvement/*` |
| `/admin` | Santé des services, tenants, classifications, policy IA Gateway | `/health`, `/v1/policy` |

**Interdit** dans ce portail : toute route `/portfolio`, `/orders`, `/risk`,
`/signals`, `/trading`. Ces routes appartiennent au vertical trading.

---

## 3. Grille de sélection des verticaux (`/`)

Chaque carte affiche :

- nom du vertical (`Trading`, `Lawyer`, `Doctor`, `CRM`, `CNESST`)
- statut (`live` / `scaffold` / `planned`)
- lien sortant vers son UI
- lien vers sa doc (`docs/verticals/{nom}.md`)

Exemple de config statique (`lib/verticals.ts`) :

```ts
export type VerticalStatus = "live" | "scaffold" | "planned";

export interface Vertical {
  slug: string;
  name: string;
  description: string;
  status: VerticalStatus;
  uiUrl?: string;        // ex: http://localhost:3000
  healthUrl?: string;    // ex: http://localhost:8000/health
  docPath: string;       // ex: docs/verticals/trading.md
}

export const VERTICALS: Vertical[] = [
  {
    slug: "trading",
    name: "Trading",
    description: "Institutional quant + AI trading cockpit",
    status: "live",
    uiUrl: "http://localhost:3000",
    healthUrl: "http://localhost:8000/health",
    docPath: "docs/verticals/trading.md",
  },
  {
    slug: "cnesst",
    name: "CNESST",
    description: "Dossiers CNESST — pilote de réutilisabilité",
    status: "scaffold",
    uiUrl: "http://localhost:3001",
    healthUrl: "http://localhost:8010/health",
    docPath: "docs/verticals/cnesst.md",
  },
  { slug: "lawyer", name: "Lawyer", description: "Cabinet juridique (RESTRICTED)", status: "planned", docPath: "docs/verticals/lawyer.md" },
  { slug: "doctor", name: "Doctor", description: "Recherche médicale (RESTRICTED)", status: "planned", docPath: "docs/verticals/doctor.md" },
  { slug: "crm",    name: "CRM",    description: "Relation client (CONFIDENTIAL)", status: "planned", docPath: "docs/verticals/crm.md" },
];