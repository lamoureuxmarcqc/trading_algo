"""Memory : decisions (immuable) + facts (long-term).

Revision ID: mem_001
Revises:
Create Date: 2026-10-16
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "mem_001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── Table decisions (APPEND-ONLY) ───
    op.create_table(
        "decisions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False, index=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("evidence", JSONB, nullable=True),
        sa.Column("action", sa.Text, nullable=True),
        sa.Column("outcome", JSONB, nullable=True),
        sa.Column("evaluation", JSONB, nullable=True),
        sa.Column("model_version", sa.String(128), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("correlation_id", sa.String(128), nullable=True),
    )

    # Trigger : interdit UPDATE et DELETE
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_decision_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'decisions is append-only: DELETE interdit';
            END IF;

            IF TG_OP = 'UPDATE' THEN
                -- Autoriser uniquement si les colonnes immuables sont inchangees
                IF OLD.decision IS DISTINCT FROM NEW.decision
                   OR OLD.reason IS DISTINCT FROM NEW.reason
                   OR OLD.evidence IS DISTINCT FROM NEW.evidence
                   OR OLD.action IS DISTINCT FROM NEW.action
                   OR OLD.tenant_id IS DISTINCT FROM NEW.tenant_id
                   OR OLD.created_at IS DISTINCT FROM NEW.created_at
                   OR OLD.model_version IS DISTINCT FROM NEW.model_version
                   OR OLD.confidence IS DISTINCT FROM NEW.confidence
                   OR OLD.correlation_id IS DISTINCT FROM NEW.correlation_id
                THEN
                    RAISE EXCEPTION 'decisions est append-only: seuls outcome et evaluation peuvent etre ajoutes';
                END IF;

                -- Interdire de vider outcome/evaluation une fois remplis
                IF OLD.outcome IS NOT NULL AND NEW.outcome IS NULL THEN
                    RAISE EXCEPTION 'outcome deja enregistre, ne peut etre efface';
                END IF;
                IF OLD.evaluation IS NOT NULL AND NEW.evaluation IS NULL THEN
                    RAISE EXCEPTION 'evaluation deja enregistree, ne peut etre effacee';
                END IF;
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER decisions_immutable
        BEFORE UPDATE OR DELETE ON decisions
        FOR EACH ROW EXECUTE FUNCTION prevent_decision_mutation();
    """)

    # ─── Table facts (long-term memory, validée humainement) ───
    op.create_table(
        "facts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False, index=True),
        sa.Column("fact", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=True),
        sa.Column(
            "validated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("validated_by", sa.String(128), nullable=True),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("metadata", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS decisions_immutable ON decisions")
    op.execute("DROP FUNCTION IF EXISTS prevent_decision_mutation()")
    op.drop_table("facts")
    op.drop_table("decisions")
