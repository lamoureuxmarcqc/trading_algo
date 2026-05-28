from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import select


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.db.models import Account, Position, Symbol  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from import_portfolio_csv import infer_exchange, normalize_ticker  # noqa: E402


def main() -> None:
    with SessionLocal() as session:
        imported_positions = session.scalars(
            select(Position)
            .join(Account, Account.id == Position.account_id)
            .join(Symbol, Symbol.id == Position.symbol_id)
            .where(Account.broker_name == "Imported CSV")
        ).all()

        renamed_symbols = 0
        merged_symbols = 0
        moved_positions = 0
        deleted_symbols = 0

        for position in imported_positions:
            symbol = position.symbol
            target_ticker = normalize_ticker(symbol.ticker)
            if target_ticker == symbol.ticker:
                continue

            existing = session.scalar(select(Symbol).where(Symbol.ticker == target_ticker))
            if existing is None:
                symbol.ticker = target_ticker
                symbol.exchange = infer_exchange(target_ticker, symbol.currency)
                renamed_symbols += 1
                continue

            if existing.id != symbol.id:
                duplicate = session.scalar(
                    select(Position).where(
                        Position.account_id == position.account_id,
                        Position.symbol_id == existing.id,
                    )
                )
                if duplicate is None:
                    position.symbol_id = existing.id
                    moved_positions += 1
                else:
                    duplicate.quantity = position.quantity
                    duplicate.average_cost = position.average_cost
                    duplicate.market_price = position.market_price
                    duplicate.market_value = position.market_value
                    duplicate.unrealized_pnl = position.unrealized_pnl
                    duplicate.updated_at = position.updated_at
                    session.delete(position)
                    moved_positions += 1

                session.flush()
                still_used = session.scalar(select(Position).where(Position.symbol_id == symbol.id).limit(1))
                if still_used is None:
                    session.delete(symbol)
                    deleted_symbols += 1
                merged_symbols += 1

        session.commit()
        print(
            "COMMITTED: "
            f"renamed_symbols={renamed_symbols} "
            f"merged_symbols={merged_symbols} "
            f"moved_positions={moved_positions} "
            f"deleted_symbols={deleted_symbols}"
        )


if __name__ == "__main__":
    main()
