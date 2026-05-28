"""event outbox and idempotency hardening

Revision ID: 0001_event_outbox_hardening
Revises:
Create Date: 2026-05-14 12:00:00
"""

from alembic import op


revision = "0001_event_outbox_hardening"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_fingerprint TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_idempotency_tenant_operation_key
                UNIQUE (tenant_id, operation, idempotency_key)
        );
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_idempotency_keys_tenant_operation
            ON idempotency_keys (tenant_id, operation);
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS event_outbox (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            event_name TEXT NOT NULL,
            topic TEXT NOT NULL,
            event_version INTEGER NOT NULL DEFAULT 1,
            tenant_id TEXT NOT NULL DEFAULT 'public',
            correlation_id TEXT,
            aggregate_type TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            delivery_status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            dispatched_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    op.execute(
        """
        ALTER TABLE event_outbox
            ADD COLUMN IF NOT EXISTS event_version INTEGER NOT NULL DEFAULT 1,
            ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'public',
            ADD COLUMN IF NOT EXISTS correlation_id TEXT,
            ADD COLUMN IF NOT EXISTS aggregate_type TEXT NOT NULL DEFAULT 'unknown',
            ADD COLUMN IF NOT EXISTS aggregate_id TEXT NOT NULL DEFAULT 'unknown',
            ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            ADD COLUMN IF NOT EXISTS delivery_status TEXT NOT NULL DEFAULT 'pending',
            ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS last_error TEXT,
            ADD COLUMN IF NOT EXISTS dispatched_at TIMESTAMPTZ;
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_event_outbox_topic_created_at
            ON event_outbox (topic, created_at DESC);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_event_outbox_delivery_status
            ON event_outbox (delivery_status, created_at ASC);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_event_outbox_correlation_id
            ON event_outbox (correlation_id);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_event_outbox_correlation_id;")
    op.execute("DROP INDEX IF EXISTS idx_event_outbox_delivery_status;")
    op.execute("DROP INDEX IF EXISTS idx_event_outbox_topic_created_at;")
    op.execute("DROP TABLE IF EXISTS event_outbox;")
    op.execute("DROP INDEX IF EXISTS idx_idempotency_keys_tenant_operation;")
    op.execute("DROP TABLE IF EXISTS idempotency_keys;")
