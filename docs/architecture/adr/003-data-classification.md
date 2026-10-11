# ADR-003 — Classification des données avant routage

**Statut :** Accepted
**Date :** 2026-10-10

## Contexte

Une donnée sensible ne doit jamais être routée vers un provider distant, même en cas d'erreur de configuration. La classification doit être une propriété explicite de la requête, pas une inférence.

## Décision

Cinq niveaux : `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `HIGHLY_CONFIDENTIAL`, `RESTRICTED`. Le gateway applique la matrice de routage **avant tout appel réseau**. Une violation lève `PrivacyViolation` (HTTP 403).

## Matrice

| Classification | LOCAL | LOCAL_GPU | REMOTE |
|---|---|---|---|
| PUBLIC | ✅ | ✅ | ✅ |
| INTERNAL | ✅ | ✅ | ✅ |
| CONFIDENTIAL | ✅ | ✅ | ❌ |
| HIGHLY_CONFIDENTIAL | ✅ | ✅ | ❌ |
| RESTRICTED | ✅ | ❌ | ❌ |

## Conséquences

- Toute requête doit porter une classification explicite.
- Le client ne peut pas désactiver la classification.
- Le gateway peut refuser une requête sans l'exécuter.

## Règles

- Défaut applicatif : `INTERNAL`.
- Les documents importés héritent de la classification du workspace.
- `RESTRICTED` = données médicales, credentials, secrets.
