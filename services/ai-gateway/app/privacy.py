"""Enforcement de la classification avant tout appel réseau."""
from .models import DataClassification, ModelTarget


# Matrice de routage — source de vérité
ALLOWED_TARGETS: dict[DataClassification, set[ModelTarget]] = {
    DataClassification.PUBLIC: {ModelTarget.LOCAL, ModelTarget.LOCAL_GPU, ModelTarget.REMOTE},
    DataClassification.INTERNAL: {ModelTarget.LOCAL, ModelTarget.LOCAL_GPU, ModelTarget.REMOTE},
    DataClassification.CONFIDENTIAL: {ModelTarget.LOCAL, ModelTarget.LOCAL_GPU},
    DataClassification.HIGHLY_CONFIDENTIAL: {ModelTarget.LOCAL, ModelTarget.LOCAL_GPU},
    DataClassification.RESTRICTED: {ModelTarget.LOCAL},
}


class PrivacyViolation(Exception):
    """Levée quand une requête viole la politique de classification."""

    def __init__(self, classification: DataClassification, requested: ModelTarget):
        self.classification = classification
        self.requested = requested
        super().__init__(
            f"Classification {classification.value} interdit la cible {requested.value}"
        )


def allowed_targets(classification: DataClassification) -> set[ModelTarget]:
    return ALLOWED_TARGETS[classification]


def is_allowed(classification: DataClassification, target: ModelTarget) -> bool:
    if target == ModelTarget.AUTO:
        return True
    return target in ALLOWED_TARGETS[classification]


def enforce(classification: DataClassification, target: ModelTarget) -> None:
    """Lève PrivacyViolation si la cible n'est pas autorisée."""
    if not is_allowed(classification, target):
        raise PrivacyViolation(classification, target)


def resolve_auto(classification: DataClassification) -> ModelTarget:
    """Choisit une cible par défaut quand AUTO est demandé."""
    targets = ALLOWED_TARGETS[classification]
    # Préférence : LOCAL > LOCAL_GPU > REMOTE
    for preferred in (ModelTarget.LOCAL, ModelTarget.LOCAL_GPU, ModelTarget.REMOTE):
        if preferred in targets:
            return preferred
    raise PrivacyViolation(classification, ModelTarget.AUTO)
