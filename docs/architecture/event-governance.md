# Event Governance

## Convention de nommage

Format obligatoire:

```text
com.terminal.<domain>.<event>.v<version>
```

Exemple:

```text
com.terminal.orders.created.v1
```

## Regles de compatibilite

- backward compatible obligatoire
- suppression de champs interdite
- renommage de champs interdit
- ajout de champs permis si optionnel
- changement majeur -> nouveau topic ou nouvelle version de contrat

## Formats de schema autorises

- Avro
- Protobuf
- JSON Schema

## Registry cible

- Confluent Schema Registry

## Evenements prioritaires a introduire

- `com.terminal.auth.session-created.v1`
- `com.terminal.portfolio.refreshed.v1`
- `com.terminal.orders.created.v1`
- `com.terminal.orders.cancelled.v1`
- `com.terminal.risk.scenario-evaluated.v1`
- `com.terminal.ai.signal-generated.v1`

## Strategie d'adoption

1. Commencer par publier les evenements de lecture/metier deja presents dans `api-core`.
2. Introduire un bus Kafka/Redpanda derriere une abstraction applicative.
3. Versionner les contrats avant d'ouvrir la consommation multi-services.

## Etat actuel du repo

- `services/api-core` expose maintenant un outbox persistant `event_outbox`.
- Les evenements critiques sont publies avec une enveloppe normalisee et consultables via `GET /api/v1/admin/events`.
- Un bus local en memoire est disponible pour des handlers applicatifs synchrones, en attendant un dispatcher brokerise.
- Un dispatcher applicatif traite les evenements `pending/failed`, marque les tentatives et conserve les erreurs de livraison.
- Le pilotage operatoire de l'outbox est expose via `GET /api/v1/admin/events/summary` et `POST /api/v1/admin/events/dispatch`.
