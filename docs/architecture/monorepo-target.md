# Monorepo Target

## Cible institutionnelle

```text
institutional-terminal/
├── apps/
├── services/
├── engines/
├── infrastructure/
├── shared/
├── docs/
├── tests/
└── .github/
```

## Mapping avec le repo actuel

| Cible | Repo actuel | Statut |
| --- | --- | --- |
| `apps/` | `apps/` | en place |
| `services/` | `services/` | en place |
| `engines/` | `trading_algo/` + `services/*-engine` | partiel |
| `infrastructure/` | `infra/` | nomenclature a aligner |
| `shared/` | `libs/` | nomenclature a aligner |
| `docs/` | `docs/` | en place |
| `tests/` | tests disperses a la racine | a consolider |

## Decision de transition

Le repo n'est pas renomme ni restructure massivement a ce stade.

La trajectoire retenue est:

1. stabiliser les standards d'architecture
2. documenter les decisions via ADR
3. migrer progressivement `infra -> infrastructure` et `libs -> shared`
4. isoler les moteurs quant dans `engines/`

Cette approche limite le risque de rupture tout en rendant la cible explicite.
