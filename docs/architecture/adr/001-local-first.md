# ADR-001 — Local-first par défaut

**Statut :** Accepted
**Date :** 2026-10-10
**Décideur :** Marc Lamoureux

## Contexte

La plateforme traite des données sensibles : financières, médicales, personnelles. Envoyer ces données à un provider distant crée un risque de fuite, de rétention, d'entraînement involontaire et de non-conformité.

## Décision

Toute donnée est traitée localement par défaut. Aucun appel réseau sortant n'est autorisé sans classification explicite et conforme à la politique. Le système doit fonctionner intégralement offline (dégradé mais fonctionnel).

## Conséquences

**Positives :**
- Confidentialité par construction
- Fonctionnement en environnement isolé (air-gap)
- Indépendance vis-à-vis des providers
- Coût marginal nul sur l'inférence locale

**Négatives :**
- Qualité dépendante des modèles locaux disponibles
- Besoin de GPU pour les gros modèles
- Coût matériel initial

## Alternatives rejetées

- **Cloud-first** : rejeté (risque de fuite, coût, dépendance).
- **Hybride par défaut** : rejeté (complexité, ambiguïté sur le routage).

## Règles

- Local est la cible par défaut.
- Remote est opt-in, jamais opt-out.
- `RESTRICTED` ne quitte jamais le local.
