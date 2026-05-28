from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import re
import sys
import unicodedata

from sqlalchemy import select


SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.db.models import Account, Position, Symbol  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402


@dataclass(frozen=True)
class ImportedRow:
    account_number: str
    ticker: str
    name: str
    asset_class: str
    market_data_enabled: bool
    currency: str
    quantity: Decimal
    average_cost: Decimal
    market_price: Decimal | None
    market_value: Decimal | None
    unrealized_pnl: Decimal | None
    updated_at: datetime


def normalize_ticker(raw_ticker: str) -> str:
    ticker = raw_ticker.strip().upper()
    if ticker.endswith("-C"):
        return f"{ticker[:-2]}.TO"
    if ticker.endswith("-U"):
        return ticker[:-2]
    return ticker


def normalize_section_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.strip().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a brokerage portfolio CSV into api-core tables.")
    parser.add_argument("csv_path", type=Path, help="Path to the source CSV export")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and plan changes without committing them",
    )
    return parser.parse_args()


def parse_decimal(value: str) -> Decimal | None:
    cleaned = (value or "").strip()
    if not cleaned or cleaned in {"N/D", "ND", "-"}:
        return None
    cleaned = cleaned.replace("\xa0", "").replace(" ", "").replace(",", ".")
    return Decimal(cleaned)


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S.%f")
    return parsed.replace(tzinfo=timezone.utc)


def infer_asset_class(ticker: str, name: str) -> str:
    uppercase_name = name.upper()
    if "ETF" in uppercase_name:
        return "etf"
    if "BOND" in uppercase_name:
        return "bond"
    if "GOLD" in uppercase_name or "MINT" in uppercase_name:
        return "commodity"
    return "equity"


def infer_asset_class_from_section(section_label: str, ticker: str, name: str) -> str:
    if section_label == "fixed_income_cad":
        return "fixed_income"
    if section_label == "mutual_funds_cad":
        return "mutual_fund"
    return infer_asset_class(ticker, name)


def infer_market_data_enabled(asset_class: str, ticker: str, name: str) -> bool:
    if asset_class in {"fixed_income", "mutual_fund"}:
        return False
    uppercase_name = name.upper()
    if re.search(r"\d", ticker.replace(".TO", "")) and "ETF" not in uppercase_name:
        return False
    return True


def infer_exchange(ticker: str, currency: str) -> str | None:
    if ticker.endswith(".TO") or currency == "CAD":
        return "TSX"
    if currency == "USD":
        return "US"
    return None


def load_rows(csv_path: Path) -> list[ImportedRow]:
    raw_lines = csv_path.read_text(encoding="cp1252").splitlines()
    section_currency: str | None = None
    section_label: str | None = None
    rows: list[ImportedRow] = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped == "sep=;" or stripped.startswith("Compte;"):
            continue
        normalized_section = normalize_section_label(stripped)
        if normalized_section == "ACTIONS CAD":
            section_currency = "CAD"
            section_label = "equities_cad"
            continue
        if normalized_section == "ACTIONS USD":
            section_currency = "USD"
            section_label = "equities_usd"
            continue
        if normalized_section == "TITRES A REVENU FIXE CAD":
            section_currency = "CAD"
            section_label = "fixed_income_cad"
            continue
        if normalized_section == "FONDS COMMUNS CAD":
            section_currency = "CAD"
            section_label = "mutual_funds_cad"
            continue
        if not section_currency:
            continue

        columns = next(csv.reader([line], delimiter=";"))
        if len(columns) < 15:
            continue

        ticker = normalize_ticker(columns[1])
        name = columns[2].strip()
        asset_class = infer_asset_class_from_section(section_label or "", ticker, name)
        rows.append(
            ImportedRow(
                account_number=columns[0].strip(),
                ticker=ticker,
                name=name,
                asset_class=asset_class,
                market_data_enabled=infer_market_data_enabled(asset_class, ticker, name),
                currency=section_currency,
                quantity=parse_decimal(columns[3]) or Decimal("0"),
                average_cost=parse_decimal(columns[4]) or Decimal("0"),
                market_price=parse_decimal(columns[7]),
                market_value=parse_decimal(columns[11]),
                unrealized_pnl=parse_decimal(columns[13]),
                updated_at=parse_timestamp(columns[6]),
            )
        )

    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv_path)
    if not rows:
        raise SystemExit("No portfolio rows found in CSV.")

    with SessionLocal() as session:
        account_map: dict[str, Account] = {}
        symbol_map: dict[str, Symbol] = {}
        imported_keys: set[tuple[str, str]] = set()
        created_accounts = 0
        created_symbols = 0
        inserted_positions = 0
        updated_positions = 0
        deleted_positions = 0

        for row in rows:
            account = account_map.get(row.account_number)
            if account is None:
                account = session.scalar(select(Account).where(Account.account_number == row.account_number))
                if account is None:
                    account = Account(
                        broker_name="Imported CSV",
                        account_number=row.account_number,
                        account_type="brokerage",
                        base_currency=row.currency,
                        status="active",
                    )
                    session.add(account)
                    session.flush()
                    created_accounts += 1
                account_map[row.account_number] = account

            symbol = symbol_map.get(row.ticker)
            if symbol is None:
                symbol = session.scalar(select(Symbol).where(Symbol.ticker == row.ticker))
                if symbol is None:
                    symbol = Symbol(
                        ticker=row.ticker,
                        asset_class=row.asset_class,
                        exchange=infer_exchange(row.ticker, row.currency),
                        market_data_ticker=row.ticker if row.market_data_enabled else None,
                        market_data_enabled=row.market_data_enabled,
                        currency=row.currency,
                        is_active=True,
                    )
                    session.add(symbol)
                    session.flush()
                    created_symbols += 1
                else:
                    symbol.asset_class = row.asset_class
                    symbol.exchange = infer_exchange(row.ticker, row.currency)
                    symbol.market_data_enabled = row.market_data_enabled
                    symbol.market_data_ticker = row.ticker if row.market_data_enabled else None
                symbol_map[row.ticker] = symbol

            imported_keys.add((account.id, symbol.id))

            position = session.scalar(
                select(Position).where(Position.account_id == account.id, Position.symbol_id == symbol.id)
            )
            if position is None:
                position = Position(
                    account_id=account.id,
                    symbol_id=symbol.id,
                    quantity=row.quantity,
                    average_cost=row.average_cost,
                    market_price=row.market_price,
                    market_value=row.market_value,
                    unrealized_pnl=row.unrealized_pnl,
                    updated_at=row.updated_at,
                )
                session.add(position)
                inserted_positions += 1
            else:
                position.quantity = row.quantity
                position.average_cost = row.average_cost
                position.market_price = row.market_price
                position.market_value = row.market_value
                position.unrealized_pnl = row.unrealized_pnl
                position.updated_at = row.updated_at
                updated_positions += 1

        for account in account_map.values():
            existing_positions = session.scalars(select(Position).where(Position.account_id == account.id)).all()
            for existing in existing_positions:
                if (existing.account_id, existing.symbol_id) not in imported_keys:
                    session.delete(existing)
                    deleted_positions += 1

        if args.dry_run:
            session.rollback()
            mode = "DRY RUN"
        else:
            session.commit()
            mode = "COMMITTED"

        print(
            f"{mode}: rows={len(rows)} accounts={len(account_map)} "
            f"created_accounts={created_accounts} created_symbols={created_symbols} "
            f"inserted_positions={inserted_positions} updated_positions={updated_positions} "
            f"deleted_positions={deleted_positions}"
        )


if __name__ == "__main__":
    main()
