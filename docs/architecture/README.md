# Architecture Institutionnelle

Ce dossier traduit le plan directeur v3.0 en artefacts exploitables dans le repo.

## Objectif

Faire converger `projet_trading` vers un Investment Operating System institutionnel:

- modulaire
- distribue
- observable
- resilient
- securise
- AI-ready

## Structure

- `adr/`: Architecture Decision Records versionnes.
- `api-governance.md`: standards HTTP, versioning et deprecation.
- `event-governance.md`: conventions Kafka/event-driven et compatibilite schema.
- `monorepo-target.md`: mapping entre le repo actuel et la cible institutionnelle.
- `platform-blueprint.md`: synthese executable de la cible d'architecture.

## Regles de gouvernance

- Toute decision structurante doit creer ou mettre a jour un ADR.
- Tout nouveau service doit exposer `GET /health`, `GET /ready` et `GET /metrics`.
- Toute API publique doit accepter et propager `X-Correlation-ID`.
- Toute nouvelle integration event-driven doit suivre `com.terminal.<domain>.<event>.v<version>`.
- Toute capacite infra doit etre gerable par code (`terraform`, `helm`, manifests versionnes).
