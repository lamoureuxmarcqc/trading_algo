"""add symbol market data controls

Revision ID: 0002_symbol_market_data_controls
Revises: 0001_event_outbox_hardening
Create Date: 2026-05-22 09:00:00
"""

from alembic import op


revision = "0002_symbol_market_data_controls"
down_revision = "0001_event_outbox_hardening"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE symbols
            ADD COLUMN IF NOT EXISTS market_data_ticker TEXT,
            ADD COLUMN IF NOT EXISTS market_data_enabled BOOLEAN NOT NULL DEFAULT TRUE;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE symbols
            DROP COLUMN IF EXISTS market_data_ticker,
            DROP COLUMN IF EXISTS market_data_enabled;
        """
    )
